# Adapted from the collaborative Diffused-Remap project; see PSEUDO_LABELS.md.
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class RemapResult:
    image_raw: np.ndarray
    image_filled: np.ndarray
    depth_raw: np.ndarray
    valid_mask: np.ndarray
    filled_mask: np.ndarray


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    z = depth.reshape(-1)
    x = (xx.reshape(-1) - K[0, 2]) * z / K[0, 0]
    y = (yy.reshape(-1) - K[1, 2]) * z / K[1, 1]
    return np.column_stack((x, y, z))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def project(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points[:, 2]
    uvw = points @ K.T
    return uvw[:, :2] / z[:, None], z


def fill_holes_nearest(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid target pixels from the closest valid projected pixel."""
    if not valid.any():
        return array.copy()
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return array[tuple(nearest)]


def remap_infinite_depth(
    image: np.ndarray,
    K_frame: np.ndarray,
    K_event: np.ndarray,
    frame_to_event: np.ndarray,
    event_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Nature/DSEC-Det baseline: rotation-only (infinite-depth) image warp."""
    from scipy.ndimage import map_coordinates

    width, height = event_size
    yy, xx = np.indices((height, width), dtype=np.float64)
    event_pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    event_rays = event_pixels @ np.linalg.inv(K_event).T
    frame_rays = event_rays @ frame_to_event[:3, :3]
    projected = frame_rays @ K_frame.T
    source_x = projected[..., 0] / projected[..., 2]
    source_y = projected[..., 1] / projected[..., 2]
    valid = (
        np.isfinite(source_x) & np.isfinite(source_y) & (projected[..., 2] > 0)
        & (source_x >= 0) & (source_x <= image.shape[1] - 1)
        & (source_y >= 0) & (source_y <= image.shape[0] - 1)
    )
    warped = np.stack([
        map_coordinates(image[..., channel], [source_y, source_x], order=1, mode="constant", cval=0)
        for channel in range(image.shape[2])
    ], axis=-1).astype(image.dtype)
    warped[~valid] = 0
    return warped, valid


def remap_plane_homography(
    image: np.ndarray,
    K_frame: np.ndarray,
    K_event: np.ndarray,
    frame_to_event: np.ndarray,
    event_size: tuple[int, int],
    plane_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    plane_distance_m: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Previous alignment: plane-induced homography at 10 m by default."""
    from scipy.ndimage import map_coordinates

    normal = np.asarray(plane_normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    rotation = frame_to_event[:3, :3]
    translation = frame_to_event[:3, 3:4]
    homography = K_event @ (
        rotation + translation @ normal[None, :] / float(plane_distance_m)
    ) @ np.linalg.inv(K_frame)
    inverse = np.linalg.inv(homography)
    width, height = event_size
    yy, xx = np.indices((height, width), dtype=np.float64)
    target = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    source = target @ inverse.T
    source_x = source[..., 0] / source[..., 2]
    source_y = source[..., 1] / source[..., 2]
    valid = (
        np.isfinite(source_x) & np.isfinite(source_y) & (source[..., 2] > 0)
        & (source_x >= 0) & (source_x <= image.shape[1] - 1)
        & (source_y >= 0) & (source_y <= image.shape[0] - 1)
    )
    warped = np.stack([
        map_coordinates(image[..., channel], [source_y, source_x], order=1, mode="constant", cval=0)
        for channel in range(image.shape[2])
    ], axis=-1).astype(image.dtype)
    warped[~valid] = 0
    return warped, valid


def remap_frame_to_event(
    image: np.ndarray,
    frame_depth: np.ndarray,
    K_frame: np.ndarray,
    K_event: np.ndarray,
    frame_to_event: np.ndarray,
    event_size: tuple[int, int],
    fill_holes: bool = True,
) -> RemapResult:
    """Forward splat a rectified frame image into rectified event view with z-buffering."""
    image = np.asarray(image)
    depth = np.asarray(frame_depth)
    if image.shape[:2] != depth.shape:
        raise ValueError("image and frame_depth resolutions differ")
    points = backproject(depth, K_frame)
    event_points = transform_points(points, frame_to_event)
    uv, z = project(event_points, K_event)
    width, height = event_size
    safe_uv = np.where(np.isfinite(uv), uv, -1.0)
    x = np.rint(safe_uv[:, 0]).astype(np.int64)
    y = np.rint(safe_uv[:, 1]).astype(np.int64)
    source_valid = np.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 0)
    valid = source_valid & np.isfinite(z) & (z > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    source_indices = np.flatnonzero(valid)
    target_indices = y[valid] * width + x[valid]
    # Stable sort by target, then depth; the first occurrence is the closest.
    order = np.lexsort((z[valid], target_indices))
    targets_sorted = target_indices[order]
    first = np.r_[True, targets_sorted[1:] != targets_sorted[:-1]]
    chosen_source = source_indices[order[first]]
    chosen_target = targets_sorted[first]

    channels = image.shape[2:] or ()
    raw = np.zeros((height * width,) + channels, dtype=image.dtype)
    raw[chosen_target] = image.reshape((-1,) + channels)[chosen_source]
    raw = raw.reshape((height, width) + channels)
    depth_raw = np.full(height * width, np.nan, dtype=np.float32)
    depth_raw[chosen_target] = z[chosen_source]
    depth_raw = depth_raw.reshape(height, width)
    mask = np.isfinite(depth_raw)
    filled = fill_holes_nearest(raw, mask) if fill_holes else raw.copy()
    return RemapResult(raw, filled, depth_raw, mask, ~mask if fill_holes else np.zeros_like(mask))


def reproject_event_depth_to_frame(
    event_depth: np.ndarray,
    K_event: np.ndarray,
    K_frame: np.ndarray,
    event_to_frame: np.ndarray,
    frame_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer sparse event-camera metric depth into frame-camera coordinates."""
    dummy = np.asarray(event_depth, dtype=np.float32)
    points = transform_points(backproject(dummy, K_event), event_to_frame)
    uv, z = project(points, K_frame)
    width, height = frame_size
    safe_uv = np.where(np.isfinite(uv), uv, -1.0)
    x, y = np.rint(safe_uv[:, 0]).astype(int), np.rint(safe_uv[:, 1]).astype(int)
    valid = np.isfinite(dummy.reshape(-1)) & (dummy.reshape(-1) > 0) & np.isfinite(z) & (z > 0)
    valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
    target = y[valid] * width + x[valid]
    source = np.flatnonzero(valid)
    order = np.lexsort((z[valid], target))
    sorted_target = target[order]
    first = np.r_[True, sorted_target[1:] != sorted_target[:-1]]
    output = np.full(height * width, np.nan, dtype=np.float32)
    output[sorted_target[first]] = z[source[order[first]]]
    output = output.reshape(height, width)
    return output, np.isfinite(output)
