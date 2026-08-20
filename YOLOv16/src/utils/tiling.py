"""Shared tiling helpers used during dataset generation and inference."""

from collections.abc import Iterator

import numpy as np


def calculate_tile_positions(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: float,
) -> list[tuple[int, int]]:
    """Return top-left positions for overlapping square tiles.

    The last tile in each direction is shifted to the image border so the
    complete image is covered.
    """

    if tile_size <= 0:
        raise ValueError("tile_size must be greater than 0.")

    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in the range [0, 1).")

    stride = int(tile_size * (1 - overlap))

    if stride <= 0:
        raise ValueError("Overlap is too large.")

    def calculate_positions(length: int) -> list[int]:
        if length <= tile_size:
            return [0]

        positions = list(range(0, length - tile_size + 1, stride))
        final_position = length - tile_size

        if positions[-1] != final_position:
            positions.append(final_position)

        return positions

    x_positions = calculate_positions(image_width)
    y_positions = calculate_positions(image_height)

    return [(x, y) for y in y_positions for x in x_positions]


def iter_image_tiles(
    image: np.ndarray,
    tile_size: int,
    overlap: float,
) -> Iterator[tuple[int, int, int, int, int, np.ndarray]]:
    """Yield all tiles of an image with their full-image coordinates.

    Yields:
        tile_index, x1, y1, x2, y2, tile
    """

    image_height, image_width = image.shape[:2]

    positions = calculate_tile_positions(
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        overlap=overlap,
    )

    for tile_index, (x1, y1) in enumerate(positions):
        x2 = min(x1 + tile_size, image_width)
        y2 = min(y1 + tile_size, image_height)
        tile = image[y1:y2, x1:x2]

        yield tile_index, x1, y1, x2, y2, tile
