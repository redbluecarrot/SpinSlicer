"""Two-dimensional Snell ray tracing; lengths are in millimetres."""
import numpy as np


def refract_rays(v_in, n, n1, n2):
    """Return unit transmitted directions and a hit/TIR validity mask."""
    if not np.isfinite([n1, n2]).all() or n1 <= 0 or n2 <= 0:
        raise ValueError("Refractive indices must be finite and positive.")
    directions = np.asarray(v_in, dtype=np.float64)
    normals = np.asarray(n, dtype=np.float64)
    lengths = np.linalg.norm(directions, axis=1)
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(lengths) & np.isfinite(normal_lengths) & (lengths > 0) & (normal_lengths > 0)
    directions = np.divide(directions, lengths[:, None], out=np.zeros_like(directions), where=valid[:, None])
    normals = np.divide(normals, normal_lengths[:, None], out=np.zeros_like(normals), where=valid[:, None])
    cosine = -np.sum(directions * normals, axis=1)
    normals[cosine < 0] *= -1
    cosine = np.clip(np.abs(cosine), 0.0, 1.0)
    eta = n1 / n2
    discriminant = 1.0 - eta**2 * (1.0 - cosine**2)
    valid &= discriminant >= -32 * np.finfo(float).eps
    result = eta * directions + (eta * cosine - np.sqrt(np.maximum(discriminant, 0)))[:, None] * normals
    result[~valid] = 0
    return result, valid


def intersect_lines(p0, v0, p1, v1):
    """Intersect rays with segments; parallel lines return infinite distances."""
    delta = p1[None, :, :] - p0[:, None, :]
    determinant = v0[:, 0, None] * v1[None, :, 1] - v0[:, 1, None] * v1[None, :, 0]
    parallel = np.abs(determinant) < 1e-12
    denominator = np.where(parallel, 1.0, determinant)
    t = (delta[:, :, 0] * v1[None, :, 1] - delta[:, :, 1] * v1[None, :, 0]) / denominator
    u = (delta[:, :, 0] * v0[:, 1, None] - delta[:, :, 1] * v0[:, 0, None]) / denominator
    t[parallel] = np.inf
    return t, u


def get_square_segments(size, theta_rad):
    half = size / 2.0
    corners = np.array([[-half, half], [half, half], [half, -half], [-half, -half]])
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    corners = corners @ np.array([[c, s], [-s, c]])
    edges = np.roll(corners, -1, axis=0) - corners
    # Corners are clockwise: the left-hand edge normal points outward.
    normals = np.column_stack([-edges[:, 1], edges[:, 0]]) / size
    return corners, edges, normals


def _first_square_hit(origins, directions, size, angle, allow_zero=False):
    corners, edges, normals = get_square_segments(size, angle)
    t, u = intersect_lines(origins, directions, corners, edges)
    valid = (t >= (-1e-9 if allow_zero else 1e-9)) & (u >= -1e-12) & (u <= 1 + 1e-12)
    t[~valid] = np.inf
    sides = np.argmin(t, axis=1)
    distances = t[np.arange(len(origins)), sides]
    hit = np.isfinite(distances)
    points = origins + np.where(hit, np.maximum(distances, 0), 0)[:, None] * directions
    return points, normals[sides], hit


def _co_rotating(points, directions, valid, theta):
    c, s = np.cos(theta), np.sin(theta)
    inverse_rotation = np.array([[c, -s], [s, c]])
    points = points @ inverse_rotation
    directions = directions @ inverse_rotation
    points[~valid] = 0
    directions[~valid] = 0
    return points, directions, valid


def trace_cuvette(s_positions, theta_rad, L_out, L_in, n_air, n_glass, n_resin):
    """Air -> rotating square glass shell -> square resin region."""
    origins = np.column_stack([s_positions, np.full(len(s_positions), L_out)])
    directions = np.tile([0.0, -1.0], (len(s_positions), 1))
    glass, normals, hit = _first_square_hit(origins, directions, L_out, theta_rad)
    glass_directions, refracted = refract_rays(directions, normals, n_air, n_glass)
    hit &= refracted
    resin, normals, inner_hit = _first_square_hit(glass, glass_directions, L_in, theta_rad, allow_zero=True)
    resin_directions, refracted = refract_rays(glass_directions, normals, n_glass, n_resin)
    return _co_rotating(resin, resin_directions, hit & inner_hit & refracted, theta_rad)


def intersect_circle(p0, v0, R):
    """Return ordered distances to a circle, supporting non-unit directions."""
    a = np.sum(v0 * v0, axis=1)
    b = np.sum(p0 * v0, axis=1)
    c = np.sum(p0 * p0, axis=1) - R**2
    discriminant = b**2 - a * c
    valid = (a > 0) & np.isfinite(discriminant) & (discriminant >= 0)
    root = np.sqrt(np.maximum(discriminant, 0))
    entry = np.divide(-b - root, a, out=np.full_like(b, np.inf), where=valid)
    exit_ = np.divide(-b + root, a, out=np.full_like(b, np.inf), where=valid)
    return entry, exit_, valid


def trace_immersion_vial(s_positions, theta_rad, L_out, L_in, n_air, n_glass, n_liquid, n_resin):
    """Surrounding medium -> cylindrical glass wall -> resin.

    L_out/L_in are the vial's outer/inner diameters. A stationary immersion
    tank is entered at normal incidence, so its flat wall does not bend rays.
    For a vial in air pass n_liquid=n_air. Interface reflections are omitted.
    """
    origins = np.column_stack([s_positions, np.full(len(s_positions), L_out)])
    directions = np.tile([0.0, -1.0], (len(s_positions), 1))
    entry, _, valid = intersect_circle(origins, directions, L_out / 2)
    valid &= entry >= 0
    glass = origins + np.where(valid, entry, 0)[:, None] * directions
    glass_directions, transmitted = refract_rays(directions, glass / (L_out / 2), n_liquid, n_glass)
    valid &= transmitted
    entry, _, inner_hit = intersect_circle(glass, glass_directions, L_in / 2)
    inner_hit &= entry >= -1e-9
    resin = glass + np.where(inner_hit, np.maximum(entry, 0), 0)[:, None] * glass_directions
    resin_directions, transmitted = refract_rays(glass_directions, resin / (L_in / 2), n_glass, n_resin)
    return _co_rotating(resin, resin_directions, valid & inner_hit & transmitted, theta_rad)
