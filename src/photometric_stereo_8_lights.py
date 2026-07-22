"""
Photometric Stereo test for 8 illumination directions.

Expected input images:
    01_left.bmp
    01_right.bmp
    01_top.bmp
    01_bottom.bmp
    01_corner_top_left.bmp
    01_corner_top_right.bmp
    01_corner_bottom_left.bmp
    01_corner_bottom_right.bmp

Optional:
    01_none.bmp  -> ambient/dark image, subtracted from all images

Outputs:
    results/albedo.png
    results/normals_rgb.png
    results/depth.png
    results/depth.npy
    results/valid_mask.png

Install:
    pip install numpy opencv-python matplotlib
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuration
# ============================================================

IMAGE_DIR = Path("images/")
OUTPUT_DIR = IMAGE_DIR / "results"

FILE_NAMES = {
    "left": "16_left.bmp",
    "right": "16_right.bmp",
    "top": "16_top.bmp",
    "bottom": "16_bottom.bmp",
    "top_left": "16_corner_top_left.bmp",
    "top_right": "16_corner_top_right.bmp",
    "bottom_left": "16_corner_bottom_left.bmp",
    "bottom_right": "16_corner_bottom_right.bmp",
}

DARK_IMAGE_NAME = "16_none.bmp"

# Assumed elevation of all lights above the metal surface.
# 30 degrees is a reasonable starting value.
# Replace this later with calibrated light directions.
LIGHT_ELEVATION_DEG = 30.0

# Ignore pixels that are too dark in almost all images.
MIN_MEAN_INTENSITY = 4.0

# Ignore pixels that are saturated in one or more images.
SATURATION_LIMIT = 250.0

# Resize for a quick first test.
# Set to 1.0 for full 2048 x 1536 resolution.
SCALE = 0.5

# Small regularization improves numerical stability.
REGULARIZATION = 1e-6

# Optional smoothing of gradients before depth integration.
GRADIENT_BLUR_SIGMA = 1.0


# ============================================================
# Image loading
# ============================================================

def load_grayscale(path: Path, scale: float = 1.0) -> np.ndarray:
    """Load an image as grayscale float32."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    if scale != 1.0:
        image = cv2.resize(
            image,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    return image.astype(np.float32)


def load_image_stack() -> tuple[np.ndarray, list[str]]:
    """Load all eight illumination images and subtract the dark image."""
    names = list(FILE_NAMES.keys())
    images = []

    dark_path = IMAGE_DIR / DARK_IMAGE_NAME

    if dark_path.exists():
        dark = load_grayscale(dark_path, SCALE)
        print(f"Using dark image: {dark_path}")
    else:
        dark = None
        print("No dark image found. Ambient subtraction is skipped.")

    for name in names:
        path = IMAGE_DIR / FILE_NAMES[name]
        image = load_grayscale(path, SCALE)

        if dark is not None:
            image = np.clip(image - dark, 0.0, 255.0)

        images.append(image)

    stack = np.stack(images, axis=0)
    return stack, names


# ============================================================
# Light directions
# ============================================================

def direction_from_azimuth_elevation(
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """
    Convert azimuth/elevation into a unit light vector.

    Coordinate convention:
        +x = image right
        +y = image down
        +z = toward camera / surface normal

    Azimuth:
          0 deg = right
         90 deg = bottom
        180 deg = left
        270 deg = top
    """
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)

    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)

    vector = np.array([x, y, z], dtype=np.float32)
    return vector / np.linalg.norm(vector)


def build_light_matrix(elevation_deg: float) -> np.ndarray:
    """
    Construct assumed light directions for the eight positions.

    IMPORTANT:
    These are geometric assumptions only. For accurate reconstruction,
    calibrate the light directions using a matte calibration sphere.
    """
    azimuths = {
        "right": 0.0,
        "bottom_right": 45.0,
        "bottom": 90.0,
        "bottom_left": 135.0,
        "left": 180.0,
        "top_left": 225.0,
        "top": 270.0,
        "top_right": 315.0,
    }

    directions = {
        name: direction_from_azimuth_elevation(azimuth, elevation_deg)
        for name, azimuth in azimuths.items()
    }

    # Same order as FILE_NAMES.
    return np.stack([directions[name] for name in FILE_NAMES.keys()], axis=0)


# ============================================================
# Photometric stereo
# ============================================================

def estimate_normals_and_albedo(
    image_stack: np.ndarray,
    light_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve I = L g in least-squares sense.

    g = rho * n
    rho = albedo
    n = unit surface normal
    """
    number_of_images, height, width = image_stack.shape
    pixels = image_stack.reshape(number_of_images, -1)

    # Regularized pseudo-inverse:
    # g = (L^T L + lambda I)^-1 L^T I
    identity = np.eye(3, dtype=np.float32)
    inverse = np.linalg.inv(
        light_matrix.T @ light_matrix + REGULARIZATION * identity
    ) @ light_matrix.T

    g = inverse @ pixels

    albedo = np.linalg.norm(g, axis=0)
    normals = g / np.maximum(albedo, 1e-8)

    normals = normals.T.reshape(height, width, 3)
    albedo = albedo.reshape(height, width)

    mean_intensity = np.mean(image_stack, axis=0)
    saturated = np.any(image_stack >= SATURATION_LIMIT, axis=0)

    valid_mask = (
        (mean_intensity >= MIN_MEAN_INTENSITY)
        & (~saturated)
        & np.isfinite(albedo)
        & (albedo > 1e-8)
        & (normals[:, :, 2] > 0.05)
    )

    normals[~valid_mask] = 0.0
    albedo[~valid_mask] = 0.0

    return normals, albedo, valid_mask


# ============================================================
# Surface integration
# ============================================================

def frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Integrate gradients using the Frankot-Chellappa FFT method.

    p = dz/dx
    q = dz/dy
    """
    height, width = p.shape

    wx = 2.0 * np.pi * np.fft.fftfreq(width)
    wy = 2.0 * np.pi * np.fft.fftfreq(height)
    wx_grid, wy_grid = np.meshgrid(wx, wy)

    p_fft = np.fft.fft2(p)
    q_fft = np.fft.fft2(q)

    denominator = wx_grid**2 + wy_grid**2
    denominator[0, 0] = 1.0

    depth_fft = (
        -1j * wx_grid * p_fft - 1j * wy_grid * q_fft
    ) / denominator

    depth_fft[0, 0] = 0.0
    depth = np.real(np.fft.ifft2(depth_fft))

    return depth.astype(np.float32)


def normals_to_depth(
    normals: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Convert surface normals to gradients and integrate them."""
    nx = normals[:, :, 0]
    ny = normals[:, :, 1]
    nz = normals[:, :, 2]

    safe_nz = np.where(np.abs(nz) < 1e-6, 1e-6, nz)

    # For n proportional to [-dz/dx, -dz/dy, 1]
    p = -nx / safe_nz
    q = -ny / safe_nz

    p[~valid_mask] = 0.0
    q[~valid_mask] = 0.0

    if GRADIENT_BLUR_SIGMA > 0:
        p = cv2.GaussianBlur(p, (0, 0), GRADIENT_BLUR_SIGMA)
        q = cv2.GaussianBlur(q, (0, 0), GRADIENT_BLUR_SIGMA)

    depth = frankot_chellappa(p, q)

    if np.any(valid_mask):
        depth -= np.median(depth[valid_mask])

    depth[~valid_mask] = np.nan
    return depth


# ============================================================
# Saving and visualization
# ============================================================

def normalize_to_uint8(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Robustly normalize an image for visualization."""
    if mask is None:
        values = image[np.isfinite(image)]
    else:
        values = image[mask & np.isfinite(image)]

    if values.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    lower = np.percentile(values, lower_percentile)
    upper = np.percentile(values, upper_percentile)

    if upper <= lower:
        upper = lower + 1.0

    normalized = (image - lower) / (upper - lower)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0

    return np.round(normalized * 255.0).astype(np.uint8)


def save_results(
    normals: np.ndarray,
    albedo: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    albedo_uint8 = normalize_to_uint8(albedo, valid_mask)

    # Map normal components from [-1, 1] to [0, 255].
    normals_rgb = np.clip((normals + 1.0) * 127.5, 0, 255).astype(np.uint8)
    normals_rgb[~valid_mask] = 0

    depth_uint8 = normalize_to_uint8(depth, valid_mask)

    cv2.imwrite(str(OUTPUT_DIR / "albedo.png"), albedo_uint8)
    cv2.imwrite(
        str(OUTPUT_DIR / "normals_rgb.png"),
        cv2.cvtColor(normals_rgb, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(str(OUTPUT_DIR / "depth.png"), depth_uint8)
    cv2.imwrite(
        str(OUTPUT_DIR / "valid_mask.png"),
        valid_mask.astype(np.uint8) * 255,
    )
    np.save(OUTPUT_DIR / "depth.npy", depth)

    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


def show_results(
    normals: np.ndarray,
    albedo: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    normals_rgb = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)
    normals_rgb[~valid_mask] = 0.0

    albedo_view = normalize_to_uint8(albedo, valid_mask)
    depth_view = normalize_to_uint8(depth, valid_mask)

    plt.figure()
    plt.imshow(albedo_view, cmap="gray")
    plt.title("Estimated albedo")
    plt.axis("off")
    plt.tight_layout()

    plt.figure()
    plt.imshow(normals_rgb)
    plt.title("Estimated surface normals")
    plt.axis("off")
    plt.tight_layout()

    plt.figure()
    plt.imshow(depth_view, cmap="gray")
    plt.title("Relative depth")
    plt.axis("off")
    plt.tight_layout()

    plt.show()


# ============================================================
# Main
# ============================================================

def main() -> None:
    image_stack, names = load_image_stack()
    light_matrix = build_light_matrix(LIGHT_ELEVATION_DEG)

    print("\nImage order and assumed light vectors:")
    for name, vector in zip(names, light_matrix):
        print(
            f"{name:>12s}: "
            f"x={vector[0]: .3f}, "
            f"y={vector[1]: .3f}, "
            f"z={vector[2]: .3f}"
        )

    normals, albedo, valid_mask = estimate_normals_and_albedo(
        image_stack,
        light_matrix,
    )

    depth = normals_to_depth(normals, valid_mask)

    valid_fraction = 100.0 * np.mean(valid_mask)
    print(f"\nValid pixels: {valid_fraction:.1f} %")

    save_results(normals, albedo, depth, valid_mask)
    show_results(normals, albedo, depth, valid_mask)


if __name__ == "__main__":
    main()
