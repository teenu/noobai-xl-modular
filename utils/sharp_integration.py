"""SHARP 3D model generation integration.

This module provides utilities for running Apple's SHARP model to convert
2D images into 3D Gaussian Splatting representations.
"""

import os
import subprocess
import shutil
from typing import Optional, Tuple
from config import logger, SHARP_CONFIG


def check_sharp_available() -> Tuple[bool, str]:
    """Check if SHARP is available in the system.

    Returns:
        Tuple of (is_available, message)
    """
    sharp_path = shutil.which("sharp")
    if sharp_path is None:
        return False, "SHARP command not found. Install with: pip install ml-sharp"

    try:
        result = subprocess.run(
            ["sharp", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, f"SHARP available at: {sharp_path}"
        else:
            return False, f"SHARP found but not working: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "SHARP command timed out"
    except Exception as e:
        return False, f"Error checking SHARP: {e}"


def get_sharp_checkpoint_path() -> Optional[str]:
    """Locate the SHARP checkpoint file.

    Checks default cache location and returns path if found.

    Returns:
        Path to checkpoint file or None if not found
    """
    checkpoint_path = os.path.join(
        SHARP_CONFIG.DEFAULT_CACHE_DIR,
        SHARP_CONFIG.CHECKPOINT_NAME
    )

    if os.path.exists(checkpoint_path):
        return checkpoint_path

    # Check if it might be in current directory
    if os.path.exists(SHARP_CONFIG.CHECKPOINT_NAME):
        return os.path.abspath(SHARP_CONFIG.CHECKPOINT_NAME)

    return None


def run_sharp_inference(
    input_image_path: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None
) -> Optional[str]:
    """Run SHARP inference to generate 3D Gaussian representation.

    Args:
        input_image_path: Path to input PNG image
        output_dir: Directory to save output .ply file
        checkpoint_path: Optional path to SHARP checkpoint (auto-detected if None)

    Returns:
        Path to generated .ply file or None on failure
    """
    # Validate input
    if not os.path.exists(input_image_path):
        logger.error(f"SHARP: Input image not found: {input_image_path}")
        return None

    # Get checkpoint
    if checkpoint_path is None:
        checkpoint_path = get_sharp_checkpoint_path()

    # Build output path with same base name
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]

    # Create a temp directory for SHARP output (it outputs to a folder)
    sharp_output_dir = os.path.join(output_dir, f".sharp_temp_{base_name}")
    os.makedirs(sharp_output_dir, exist_ok=True)

    try:
        # Build command
        cmd = [
            "sharp", "predict",
            "-i", input_image_path,
            "-o", sharp_output_dir
        ]

        # Add checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            cmd.extend(["-c", checkpoint_path])

        logger.info(f"SHARP: Running inference on {input_image_path}")
        logger.debug(f"SHARP command: {' '.join(cmd)}")

        # Run SHARP
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SHARP_CONFIG.TIMEOUT_SECONDS
        )

        if result.returncode != 0:
            logger.error(f"SHARP failed with code {result.returncode}: {result.stderr}")
            return None

        # Find the generated .ply file
        ply_files = [f for f in os.listdir(sharp_output_dir) if f.endswith('.ply')]

        if not ply_files:
            logger.error(f"SHARP: No .ply file generated in {sharp_output_dir}")
            return None

        # Move the .ply file to the output directory with proper name
        source_ply = os.path.join(sharp_output_dir, ply_files[0])
        dest_ply = os.path.join(output_dir, f"{base_name}.ply")

        shutil.move(source_ply, dest_ply)
        logger.info(f"SHARP: 3D model saved to {dest_ply}")

        return dest_ply

    except subprocess.TimeoutExpired:
        logger.error(f"SHARP: Inference timed out after {SHARP_CONFIG.TIMEOUT_SECONDS}s")
        return None
    except Exception as e:
        logger.error(f"SHARP: Unexpected error: {e}")
        return None
    finally:
        # Cleanup temp directory
        try:
            if os.path.exists(sharp_output_dir):
                shutil.rmtree(sharp_output_dir)
        except Exception as cleanup_error:
            logger.warning(f"SHARP: Failed to cleanup temp dir: {cleanup_error}")
