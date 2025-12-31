"""SharpPredict node for ComfyUI-Sharp."""

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Try to import ComfyUI folder_paths for output directory
try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

from ..utils.image import comfy_to_numpy_rgb, convert_focallength


# Global cache for encoded features (single image only)
_encode_cache = {
    "image_hash": None,
    "monodepth_output": None,
    "image_resized": None,
    "original_shape": None,
}


def _compute_image_hash(image_np: np.ndarray) -> str:
    """Compute hash of image for cache key."""
    return hashlib.sha256(image_np.tobytes()).hexdigest()[:16]


class SharpPredict:
    """Run SHARP inference to generate 3D Gaussians from a single image or batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SHARP_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "focal_length_mm": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 500.0,
                    "step": 0.1,
                    "tooltip": "Focal length in mm (35mm equivalent). 0 = auto (defaults to 30mm)."
                }),
                "output_prefix": ("STRING", {
                    "default": "sharp",
                    "tooltip": "Prefix for output PLY filename or folder name for batches."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "GAUSSIANS_3D", "EXTRINSICS", "INTRINSICS",)
    RETURN_NAMES = ("ply_path", "gaussians", "extrinsics", "intrinsics",)
    FUNCTION = "predict"
    CATEGORY = "SHARP"
    OUTPUT_NODE = True
    DESCRIPTION = "Generate 3D Gaussian Splatting PLY file(s) from image(s) using SHARP. Batch input creates a folder with numbered PLY files. Outputs both file path and direct Gaussians3D object for flexible downstream processing."

    @torch.no_grad()
    def predict(
        self,
        model: dict,
        image: torch.Tensor,
        focal_length_mm: float = 0.0,
        output_prefix: str = "sharp",
    ):
        """Run SHARP inference and save PLY file(s).

        For single image: saves {prefix}_{timestamp}.ply
        For batch: creates folder {prefix}_{timestamp}/ with 001.ply, 002.ply, etc.

        Features are cached per image - changing focal_length with same image is instant.
        """
        from sharp.utils.gaussians import save_ply, unproject_gaussians

        predictor = model["predictor"]
        device = torch.device(model["device"])

        # Handle batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        batch_size = image.shape[0]
        print(f"[SHARP] Processing {batch_size} image(s)")

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)

        # Determine output path(s)
        if batch_size == 1:
            # Single image: save directly as PLY file
            output_filename = f"{output_prefix}_{timestamp}.ply"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            is_batch = False
        else:
            # Multiple images: create folder
            folder_name = f"{output_prefix}_{timestamp}"
            output_folder = os.path.join(OUTPUT_DIR, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            output_path = output_folder
            is_batch = True

        all_ply_paths = []
        all_gaussians = []
        all_extrinsics = []
        all_intrinsics = []

        inference_start = time.time()

        for i in range(batch_size):
            # Extract single image from batch
            single_image = image[i:i+1]
            image_np = comfy_to_numpy_rgb(single_image)
            height, width = image_np.shape[:2]

            if i == 0:
                print(f"[SHARP] Image size: {width}x{height}")

            # Determine focal length in pixels
            if focal_length_mm > 0:
                f_px = convert_focallength(width, height, focal_length_mm)
            else:
                f_px = convert_focallength(width, height, 30.0)

            # Run inference with caching
            print(f"[SHARP] Running inference on image {i+1}/{batch_size}...")
            gaussians = self._predict_image_cached(predictor, image_np, f_px, device)

            # Determine output filename
            if is_batch:
                ply_filename = f"{i+1:03d}.ply"
                ply_path = os.path.join(output_folder, ply_filename)
            else:
                ply_path = output_path

            # Save PLY and get metadata
            _, metadata = save_ply(gaussians, f_px, (height, width), Path(ply_path))

            all_ply_paths.append(ply_path)
            all_gaussians.append(gaussians)
            all_extrinsics.append(metadata["extrinsic"])
            all_intrinsics.append(metadata["intrinsic"])

            print(f"[SHARP] Saved: {ply_path} ({metadata['num_gaussians']:,} gaussians)")

        inference_time = time.time() - inference_start
        print(f"[SHARP] Total inference time: {inference_time:.2f}s ({inference_time/batch_size:.2f}s per image)")

        # Return values
        if is_batch:
            # For batch: return folder path, first image's gaussians and camera params
            # (assuming all images have same camera - user can override)
            return (output_path, all_gaussians[0], all_extrinsics[0], all_intrinsics[0],)
        else:
            # For single image: return PLY path, gaussians and camera params
            return (output_path, all_gaussians[0], all_extrinsics[0], all_intrinsics[0],)

    def _predict_image_cached(
        self,
        predictor,
        image: np.ndarray,
        f_px: float,
        device: torch.device,
    ):
        """Predict Gaussians with caching of encoded features.

        The expensive encode step is cached per image.
        Changing focal_length reuses cached features (instant).
        """
        global _encode_cache
        from sharp.utils.gaussians import unproject_gaussians

        internal_shape = (1536, 1536)
        height, width = image.shape[:2]

        # Compute image hash for cache
        image_hash = _compute_image_hash(image)

        # Check cache
        if _encode_cache["image_hash"] == image_hash:
            # Cache hit - reuse encoded features
            print(f"[SHARP] Cache hit - reusing encoded features (focal_length change is instant)")
            monodepth_output = _encode_cache["monodepth_output"]
            image_resized_pt = _encode_cache["image_resized"]
        else:
            # Cache miss - need to encode
            print(f"[SHARP] Cache miss - running encoder (this is the slow part)...")

            # Clear old cache
            _encode_cache["image_hash"] = None
            _encode_cache["monodepth_output"] = None
            _encode_cache["image_resized"] = None
            _encode_cache["original_shape"] = None

            # Convert to tensor and normalize
            image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0

            # Resize to internal resolution
            image_resized_pt = F.interpolate(
                image_pt[None],
                size=(internal_shape[1], internal_shape[0]),
                mode="bilinear",
                align_corners=True,
            )

            # Encode (expensive)
            encode_start = time.time()
            monodepth_output, _ = predictor.encode(image_resized_pt)
            encode_time = time.time() - encode_start
            print(f"[SHARP] Encode time: {encode_time:.2f}s")

            # Update cache
            _encode_cache["image_hash"] = image_hash
            _encode_cache["monodepth_output"] = monodepth_output
            _encode_cache["image_resized"] = image_resized_pt
            _encode_cache["original_shape"] = (height, width)

        # Decode (cheap) - always run with current focal length
        disparity_factor = torch.tensor([f_px / width]).float().to(device)

        decode_start = time.time()
        gaussians_ndc = predictor.decode(monodepth_output, image_resized_pt, disparity_factor)
        decode_time = time.time() - decode_start
        print(f"[SHARP] Decode time: {decode_time:.2f}s")

        # Build intrinsics for unprojection
        intrinsics = (
            torch.tensor(
                [
                    [f_px, 0, width / 2, 0],
                    [0, f_px, height / 2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            .float()
            .to(device)
        )
        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= internal_shape[0] / width
        intrinsics_resized[1] *= internal_shape[1] / height

        # Convert Gaussians to metric space
        gaussians = unproject_gaussians(
            gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
        )

        return gaussians


NODE_CLASS_MAPPINGS = {
    "SharpPredict": SharpPredict,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpPredict": "SHARP Predict (Image to PLY)",
}
