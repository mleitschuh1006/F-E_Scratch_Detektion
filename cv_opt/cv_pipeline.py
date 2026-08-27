from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def _odd(value: int, minimum: int = 1) -> int:
    if value <= 0:
        return 0
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _largest_component(binary_u8: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, 8)
    if n <= 1:
        return binary_u8
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def _outer_envelope(binary: np.ndarray) -> np.ndarray:
    """Fill the outer silhouette of the workpiece, including internal holes."""
    binary_u8 = binary.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(binary_u8, dtype=np.uint8) > 0

    largest = max(contours, key=cv2.contourArea)
    envelope = np.zeros_like(binary_u8)
    cv2.drawContours(envelope, [largest], -1, 255, thickness=cv2.FILLED)
    return envelope > 0


def build_roi(gray: np.ndarray, p: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Detect the metallic workpiece and define the valid inspection region."""
    raw = (gray > int(p["roi_min_gray"])).astype(np.uint8) * 255

    close_k = _odd(int(p["roi_close_kernel_px"]), minimum=3)
    if close_k >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)

    raw = _largest_component(raw)
    roi_u8 = raw.copy()

    erode_k = _odd(int(p["roi_erode_kernel_px"]), minimum=3)
    if erode_k >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
        roi_u8 = cv2.erode(roi_u8, kernel)

    # Optional extra exclusion for the OUTER workpiece boundary.
    # Hole removal is handled separately by detect_holes().
    extra = float(p.get("extra_boundary_exclusion_px", 0))
    if extra > 0:
        envelope = _outer_envelope(raw > 0)
        distance = cv2.distanceTransform(envelope.astype(np.uint8), cv2.DIST_L2, 5)
        roi_u8 = roi_u8 & ((distance > extra).astype(np.uint8) * 255)

    return raw > 0, roi_u8 > 0


def detect_holes(
    gray: np.ndarray,
    raw_roi: np.ndarray,
    p: dict[str, Any],
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    Detect circular holes with Hough circle detection and plausibility checks.

    HoughCircles proposes circular structures. Candidates are only accepted when
    their center is sufficiently dark and clearly darker than the surrounding
    ring. This helps distinguish real holes from arbitrary circular texture.

    Returns
    -------
    hole_mask:
        Boolean mask containing every accepted circle plus the configured margin.
    circles:
        List of (x, y, radius) in ORIGINAL image pixels.
    """
    cfg = p.get("hole_detection", {})
    if not bool(cfg.get("enabled", False)):
        return np.zeros_like(gray, dtype=bool), []

    scale = float(cfg.get("detection_scale", 0.5))
    scale = min(1.0, max(0.1, scale))

    blur_k_original = _odd(int(cfg.get("blur_kernel_px", 9)), minimum=3)
    blur_k_scaled = _odd(max(3, int(round(blur_k_original * scale))), minimum=3)

    if scale < 0.999:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray

    small = cv2.GaussianBlur(small, (blur_k_scaled, blur_k_scaled), 0)

    min_dist = max(1.0, float(cfg.get("min_distance_px", 25)) * scale)
    min_radius = max(1, int(round(float(cfg.get("min_radius_px", 8)) * scale)))
    max_radius = max(min_radius + 1, int(round(float(cfg.get("max_radius_px", 60)) * scale)))

    found = cv2.HoughCircles(
        small,
        cv2.HOUGH_GRADIENT,
        dp=float(cfg.get("dp", 1.2)),
        minDist=min_dist,
        param1=float(cfg.get("canny_threshold", 120)),
        param2=float(cfg.get("accumulator_threshold", 28)),
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if found is None:
        return np.zeros_like(gray, dtype=bool), []

    envelope = _outer_envelope(raw_roi)
    h, w = gray.shape

    center_factor = float(cfg.get("center_radius_factor", 0.55))
    ring_inner_factor = float(cfg.get("ring_inner_factor", 0.85))
    ring_outer_factor = float(cfg.get("ring_outer_factor", 1.30))
    center_max_gray = float(cfg.get("center_max_gray", 40))
    dark_gray_threshold = float(cfg.get("dark_gray_threshold", 40))
    min_dark_fraction = float(cfg.get("min_dark_fraction", 0.85))
    min_ring_contrast = float(cfg.get("min_ring_contrast", 50))

    accepted: list[tuple[int, int, int]] = []

    for sx, sy, sr in found[0]:
        x = int(round(float(sx) / scale))
        y = int(round(float(sy) / scale))
        r = int(round(float(sr) / scale))

        if not (0 <= x < w and 0 <= y < h):
            continue

        center_r = max(2.0, center_factor * r)

        # Work only in a small crop around the candidate. This keeps the
        # plausibility checks fast even on perforated parts with hundreds of holes.
        outer_r = max(center_r, ring_outer_factor * r)
        x0 = max(0, int(np.floor(x - outer_r - 1)))
        x1 = min(w, int(np.ceil(x + outer_r + 2)))
        y0 = max(0, int(np.floor(y - outer_r - 1)))
        y1 = min(h, int(np.ceil(y + outer_r + 2)))
        if x1 <= x0 or y1 <= y0:
            continue

        crop_gray = gray[y0:y1, x0:x1]
        crop_env = envelope[y0:y1, x0:x1]
        local_y, local_x = np.ogrid[y0:y1, x0:x1]
        d2 = (local_x - x) ** 2 + (local_y - y) ** 2
        center_sel = d2 <= center_r ** 2

        # Suppress circles created by the black image background / outer border.
        envelope_fraction = float(np.mean(crop_env[center_sel]))
        if envelope_fraction < 0.85:
            continue

        ring_sel = (d2 >= (ring_inner_factor * r) ** 2) & (d2 <= (ring_outer_factor * r) ** 2)
        center_values = crop_gray[center_sel]
        ring_values = crop_gray[ring_sel & crop_env]
        if center_values.size == 0 or ring_values.size == 0:
            continue

        center_median = float(np.median(center_values))
        ring_median = float(np.median(ring_values))
        dark_fraction = float(np.mean(center_values <= dark_gray_threshold))

        if center_median > center_max_gray:
            continue
        if dark_fraction < min_dark_fraction:
            continue
        if (ring_median - center_median) < min_ring_contrast:
            continue

        # Avoid duplicate detections of the same hole.
        duplicate = False
        for ax, ay, ar in accepted:
            min_sep = 0.6 * min(r, ar)
            if (x - ax) ** 2 + (y - ay) ** 2 < min_sep ** 2:
                duplicate = True
                break
        if not duplicate:
            accepted.append((x, y, r))

    margin = max(0, int(cfg.get("exclusion_margin_px", 8)))
    hole_mask_u8 = np.zeros_like(gray, dtype=np.uint8)
    for x, y, r in accepted:
        cv2.circle(hole_mask_u8, (x, y), r + margin, 255, thickness=cv2.FILLED)

    return hole_mask_u8 > 0, accepted


def contrast_stretch(gray: np.ndarray, roi: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Stretch gray values using percentiles calculated only inside the valid ROI."""
    values = gray[roi]
    if values.size == 0:
        values = gray.reshape(-1)

    low = float(np.percentile(values, p["stretch_low_percentile"]))
    high = float(np.percentile(values, p["stretch_high_percentile"]))

    if high <= low + 1e-6:
        stretched = gray.copy()
    else:
        stretched = np.clip(
            (gray.astype(np.float32) - low) * (255.0 / (high - low)),
            0, 255
        ).astype(np.uint8)

    clip_limit = float(p["clahe_clip_limit"])
    if clip_limit > 0:
        grid = int(p["clahe_grid_size"])
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
        stretched = clahe.apply(stretched)

    return stretched


def scratch_response(normalized: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Enhance local intensity deviations."""
    work = normalized

    blur_k = _odd(int(p["blur_kernel_px"]), minimum=3)
    if blur_k >= 3:
        work = cv2.GaussianBlur(work, (blur_k, blur_k), 0)

    feature_k = _odd(int(p["feature_kernel_px"]), minimum=3)
    mode = p["feature_mode"]

    if mode == "local_residual":
        background = cv2.GaussianBlur(work, (feature_k, feature_k), 0)
        return cv2.absdiff(work, background)

    if mode == "both_hat":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feature_k, feature_k))
        bright = cv2.morphologyEx(work, cv2.MORPH_TOPHAT, kernel)
        dark = cv2.morphologyEx(work, cv2.MORPH_BLACKHAT, kernel)
        return cv2.max(bright, dark)

    raise ValueError(f"Unknown feature_mode: {mode}")


def threshold_response(response: np.ndarray, roi: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Keep only the strongest responses inside the valid scratch ROI."""
    values = response[roi]
    if values.size == 0:
        return np.zeros_like(roi, dtype=bool)

    threshold = max(1.0, float(np.percentile(values, p["threshold_percentile"])))
    return (response >= threshold) & roi


def morphology(binary: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Morphological cleanup."""
    result = binary.astype(np.uint8)

    open_k = _odd(int(p["opening_kernel_px"]), minimum=3)
    if open_k >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    close_k = _odd(int(p["closing_kernel_px"]), minimum=3)
    if close_k >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result > 0


def filter_components(binary: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Remove connected components that do not look sufficiently scratch-like."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if n <= 1:
        return binary

    min_area = int(p["min_component_area_px"])
    min_elongation = float(p["min_component_elongation"])
    min_max_dim = int(p.get("min_component_max_dimension_px", 0))

    keep = np.zeros(n, dtype=bool)

    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        elongation = max(w, h) / max(1, min(w, h))
        max_dim = max(w, h)

        if area < min_area:
            continue
        if elongation < min_elongation:
            continue
        if min_max_dim > 0 and max_dim < min_max_dim:
            continue

        keep[label] = True

    return keep[labels]



def _principal_orientation(points_xy: np.ndarray) -> tuple[float, float, float]:
    """
    Estimate the dominant direction of a point cloud using PCA.

    Returns
    -------
    angle_deg:
        Dominant orientation in [0, 180).
    elongation:
        PCA-based elongation. 1.0 means compact/isotropic; larger means line-like.
    length_px:
        Extent of the point cloud along its dominant PCA axis.
    """
    if points_xy.shape[0] < 3:
        return 0.0, 1.0, 0.0

    pts = points_xy.astype(np.float32)
    mean = np.mean(pts, axis=0, keepdims=True)
    centered = pts - mean

    cov = np.cov(centered, rowvar=False)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return 0.0, 1.0, 0.0

    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]

    major = vectors[:, 0]
    angle = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)

    major_var = max(float(values[0]), 1e-6)
    minor_var = max(float(values[1]), 1e-6)
    elongation = float(np.sqrt(major_var / minor_var))

    projection = centered @ major
    length_px = float(np.max(projection) - np.min(projection))

    return angle, elongation, length_px


def _angle_difference_deg(a: float, b: float) -> float:
    """Smallest orientation difference for unoriented lines (0..90 degrees)."""
    diff = abs(float(a) - float(b)) % 180.0
    return min(diff, 180.0 - diff)


def filter_parallel_edge_components(
    binary: np.ndarray,
    raw_roi: np.ndarray,
    p: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """
    Remove detections that strongly resemble the OUTER workpiece edge.

    A connected prediction is removed only if ALL configured conditions hold:
      1. sufficiently much of the component lies near the outer boundary,
      2. the component is sufficiently elongated,
      3. the component is sufficiently long,
      4. its dominant direction is approximately parallel to the local
         outer-boundary tangent.

    This is deliberately different from a simple broad border crop:
    a scratch that merely crosses the border at a different angle should remain.

    Internal holes are not considered here. They are handled separately by
    detect_holes().
    """
    cfg = p.get("edge_parallel_filter", {})
    if not bool(cfg.get("enabled", False)):
        return binary.copy(), np.zeros_like(binary, dtype=bool), []

    envelope = _outer_envelope(raw_roi)
    envelope_u8 = envelope.astype(np.uint8)

    contours, _ = cv2.findContours(
        envelope_u8 * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return binary.copy(), np.zeros_like(binary, dtype=bool), []

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if contour.shape[0] < 5:
        return binary.copy(), np.zeros_like(binary, dtype=bool), []

    # For every pixel inside the outer silhouette:
    # approximate Euclidean distance to the outside / outer border.
    distance_to_outer = cv2.distanceTransform(envelope_u8, cv2.DIST_L2, 5)

    max_distance = float(cfg.get("max_distance_px", 80))
    min_near_fraction = float(cfg.get("min_near_edge_fraction", 0.80))
    max_angle_difference = float(cfg.get("max_angle_difference_deg", 12.0))
    min_elongation = float(cfg.get("min_component_elongation", 2.5))
    min_length = float(cfg.get("min_component_length_px", 40))
    tangent_window = max(3, int(cfg.get("contour_tangent_window_px", 35)))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), 8
    )
    if n <= 1:
        return binary.copy(), np.zeros_like(binary, dtype=bool), []

    result = binary.copy()
    rejected = np.zeros_like(binary, dtype=bool)
    diagnostics: list[dict[str, float]] = []

    contour_n = contour.shape[0]

    for label in range(1, n):
        ys, xs = np.where(labels == label)
        if xs.size < 3:
            continue

        component_xy = np.column_stack((xs, ys)).astype(np.float32)

        distances = distance_to_outer[ys, xs]
        near_fraction = float(np.mean(distances <= max_distance))

        # Cheap early exit: if the component is not predominantly near the edge,
        # there is no reason to calculate orientations.
        if near_fraction < min_near_fraction:
            continue

        component_angle, component_elongation, component_length = _principal_orientation(
            component_xy
        )

        if component_elongation < min_elongation:
            continue
        if component_length < min_length:
            continue

        cx, cy = centroids[label]
        delta = contour - np.array([cx, cy], dtype=np.float32)
        nearest_idx = int(np.argmin(np.sum(delta * delta, axis=1)))

        # Local contour tangent: PCA over a short arc around the closest boundary
        # point. CHAIN_APPROX_NONE gives approximately one contour point per pixel,
        # so the window parameter is easy to interpret in pixels.
        offsets = np.arange(-tangent_window, tangent_window + 1)
        indices = (nearest_idx + offsets) % contour_n
        local_contour = contour[indices]

        boundary_angle, _, _ = _principal_orientation(local_contour)
        angle_difference = _angle_difference_deg(component_angle, boundary_angle)

        if angle_difference > max_angle_difference:
            continue

        component_mask = labels == label
        result[component_mask] = False
        rejected[component_mask] = True

        diagnostics.append({
            "label": float(label),
            "area_px": float(stats[label, cv2.CC_STAT_AREA]),
            "near_edge_fraction": near_fraction,
            "component_angle_deg": component_angle,
            "boundary_angle_deg": boundary_angle,
            "angle_difference_deg": angle_difference,
            "component_elongation": component_elongation,
            "component_length_px": component_length,
        })

    return result, rejected, diagnostics


def filter_bright_evidence(
    binary: np.ndarray,
    gray: np.ndarray,
    p: dict[str, Any],
) -> np.ndarray:
    """
    Keep a predicted component only when it contains sufficient absolute
    BRIGHT local evidence in the unstretched gray image.

    This is deliberately a component-level gate: the original absolute
    local-residual mask is preserved for segmentation, but a component cannot
    survive purely because of dark deviations or because it belongs to the
    strongest percentile of an otherwise weak image.
    """
    cfg = p.get("bright_evidence_filter", {})
    if not bool(cfg.get("enabled", False)):
        return binary.copy()

    work = gray
    blur_k = _odd(int(p["blur_kernel_px"]), minimum=3)
    if blur_k >= 3:
        work = cv2.GaussianBlur(work, (blur_k, blur_k), 0)

    feature_k = _odd(int(p["feature_kernel_px"]), minimum=3)
    background = cv2.GaussianBlur(work, (feature_k, feature_k), 0)
    bright_response = cv2.subtract(work, background)

    min_gray_residual = float(cfg.get("min_gray_residual", 30))
    min_fraction = float(cfg.get("min_fraction", 0.05))
    min_pixels = int(cfg.get("min_pixels", 3))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), 8
    )
    if n <= 1:
        return binary.copy()

    keep = np.zeros(n, dtype=bool)
    for label in range(1, n):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        strong_bright_pixels = int(
            np.count_nonzero(bright_response[component] >= min_gray_residual)
        )
        strong_fraction = strong_bright_pixels / max(1, area)

        if strong_bright_pixels < min_pixels:
            continue
        if strong_fraction < min_fraction:
            continue

        keep[label] = True

    return keep[labels]

def segment_scratches(image: np.ndarray, p: dict[str, Any], return_debug: bool = False):
    """Complete classical-CV scratch detection pipeline."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    raw_roi, roi = build_roi(gray, p)

    # Keep the existing contrast normalization unchanged.
    normalized = contrast_stretch(gray, roi, p)

    # Detect geometrically circular holes independently from the scratch detector.
    hole_mask, circles = detect_holes(gray, raw_roi, p)
    scratch_roi = roi & (~hole_mask)

    response = scratch_response(normalized, p)
    thresholded = threshold_response(response, scratch_roi, p)
    after_morphology = morphology(thresholded, p)
    after_component_filter = filter_components(after_morphology, p)
    after_component_filter &= scratch_roi

    # Remove only edge-near components that are ALSO line-like and locally
    # parallel to the OUTER workpiece contour.
    final, edge_rejected_mask, edge_rejections = filter_parallel_edge_components(
        after_component_filter,
        raw_roi,
        p,
    )

    # Final minimal plausibility gate: a component must contain enough absolute
    # bright local evidence in the ORIGINAL gray image. This prevents the
    # percentile threshold from forcing weak structures into the result and
    # rejects components that are caused only by dark local deviations.
    final = filter_bright_evidence(final, gray, p)
    final &= scratch_roi

    if return_debug:
        return final, {
            "gray": gray,
            "raw_roi": raw_roi,
            "roi": roi,
            "hole_mask": hole_mask,
            "circles": circles,
            "scratch_roi": scratch_roi,
            "normalized": normalized,
            "response": response,
            "thresholded": thresholded,
            "after_morphology": after_morphology,
            "after_component_filter": after_component_filter,
            "edge_rejected_mask": edge_rejected_mask,
            "edge_rejections": edge_rejections,
            "final": final,
        }

    return final
