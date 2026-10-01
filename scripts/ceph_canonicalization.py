"""Coordinate geometry for reversible clinical-ceph previews.

Coordinates are pixel centers, x right and y down.  Matrices act on column
vectors [x, y, 1].  The inverse maps model-space points to the original PNG;
resampling and cropping of raster intensities themselves are not reversible.
"""

from __future__ import annotations

import numpy as np


MODEL_CENTER = 63.5


def translation(x: float, y: float) -> np.ndarray:
    out = np.eye(3, dtype=np.float64)
    out[:2, 2] = (x, y)
    return out


def orientation_matrix(width: int, height: int, horizontal: bool, vertical: bool) -> np.ndarray:
    """Explicit reflection about the source image's pixel-center midpoint."""
    if width <= 0 or height <= 0:
        raise ValueError("Source dimensions must be positive")
    out = np.eye(3, dtype=np.float64)
    if horizontal:
        out = translation(width - 1, 0) @ np.diag([-1.0, 1.0, 1.0]) @ out
    if vertical:
        out = translation(0, height - 1) @ np.diag([1.0, -1.0, 1.0]) @ out
    return out


def map_points(matrix: np.ndarray, points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Points must have shape (N, 2) and be finite")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ matrix.T)[:, :2]


def fit_similarity(source, target, weights=None) -> np.ndarray:
    """Weighted least-squares similarity, positive scale and no reflection."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 3:
        raise ValueError("At least three corresponding 2D landmarks are required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Landmark coordinates must be finite")
    weights = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(source),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Landmark weights must be positive and finite")
    weights /= weights.sum()
    sc, tc = np.sum(source * weights[:, None], axis=0), np.sum(target * weights[:, None], axis=0)
    s, t = source - sc, target - tc
    u, singular, vt = np.linalg.svd((s * weights[:, None]).T @ t)
    signs = np.diag([1.0, np.linalg.det(u @ vt)])
    rotation = u @ signs @ vt
    denominator = np.sum(weights * np.sum(s * s, axis=1))
    if denominator <= 1e-12:
        raise ValueError("Fit landmarks are coincident")
    scale = float(np.sum(singular * np.diag(signs)) / denominator)
    if scale <= 0:
        raise ValueError("Landmarks do not support a positive similarity scale")
    result = np.eye(3, dtype=np.float64)
    result[:2, :2] = scale * rotation.T
    result[:2, 2] = tc - result[:2, :2] @ sc
    return result


def fov_transform(center, fov_width: float, rotation_degrees: float = 0.0,
                  orientation=None) -> np.ndarray:
    """Map a square source FOV of given width to the 128-square model frame."""
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("FOV center must be two finite coordinates")
    if not np.isfinite(fov_width) or fov_width <= 0 or not np.isfinite(rotation_degrees):
        raise ValueError("FOV width must be positive and rotation finite")
    a = np.deg2rad(rotation_degrees)
    rotation = np.array([[np.cos(a), -np.sin(a), 0],
                         [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=np.float64)
    scale = np.diag([128 / fov_width, 128 / fov_width, 1.0])
    return translation(MODEL_CENTER, MODEL_CENTER) @ scale @ rotation @ translation(*-center) @ (
        np.eye(3) if orientation is None else orientation)


def scale_about_model_center(matrix: np.ndarray, factor: float) -> np.ndarray:
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Scale factor must be positive and finite")
    return translation(MODEL_CENTER, MODEL_CENTER) @ np.diag([factor, factor, 1.0]) @ (
        translation(-MODEL_CENTER, -MODEL_CENTER) @ matrix)


def pillow_inverse_coefficients(matrix: np.ndarray) -> tuple[float, ...]:
    inverse = np.linalg.inv(matrix)
    return (float(inverse[0, 0]), float(inverse[0, 1]), float(inverse[0, 2]),
            float(inverse[1, 0]), float(inverse[1, 1]), float(inverse[1, 2]))


def geometry_diagnostics(matrix: np.ndarray, image_size: tuple[int, int]) -> dict:
    linear = matrix[:2, :2]
    singular = np.linalg.svd(linear, compute_uv=False)
    if not np.allclose(singular[0], singular[1], atol=1e-10, rtol=1e-10):
        raise ValueError("Non-uniform scaling or shear is forbidden")
    inverse = np.linalg.inv(matrix)
    corners = map_points(inverse, [[-0.5, -0.5], [127.5, -0.5],
                                   [127.5, 127.5], [-0.5, 127.5]])
    width, height = image_size
    inside = (corners[:, 0] >= -0.5) & (corners[:, 0] <= width - 0.5) & (
        corners[:, 1] >= -0.5) & (corners[:, 1] <= height - 0.5)
    yy, xx = np.indices((128, 128))
    sample_points = map_points(inverse, np.column_stack((xx.ravel(), yy.ravel())))
    sampled_inside = ((sample_points[:, 0] >= 0) & (sample_points[:, 0] <= width - 1) &
                      (sample_points[:, 1] >= 0) & (sample_points[:, 1] <= height - 1))
    return {"model_px_per_source_px": float(singular.mean()),
            "source_fov_width_px": float(128 / singular.mean()),
            "determinant": float(np.linalg.det(linear)),
            "source_fov_corners": corners.tolist(),
            "all_fov_corners_inside_source": bool(inside.all()),
            "fraction_model_pixel_centers_outside_source": float(1 - sampled_inside.mean())}
