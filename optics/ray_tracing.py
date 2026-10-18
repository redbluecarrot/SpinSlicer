import numpy as np


def refract_rays(v_in, n, n1, n2):
    """
    Vectorized ray refraction.
    v_in: (N, 2) incident directions (normalized)
    n: (N, 2) surface normals pointing towards the incoming ray (normalized)
    n1, n2: refractive indices
    Returns: v_out (N, 2) refracted directions, valid (N,) boolean mask for total internal reflection.
    """
    eta = n1 / n2
    cos_theta_i = -np.sum(v_in * n, axis=1)

    # Check if rays hit from the back side of the normal, if so, flip normal
    flip_mask = cos_theta_i < 0
    n = np.where(flip_mask[:, None], -n, n)
    cos_theta_i = np.where(flip_mask, -cos_theta_i, cos_theta_i)

    sin2_theta_t = eta**2 * (1.0 - cos_theta_i**2)
    valid = sin2_theta_t <= 1.0

    cos_theta_t = np.sqrt(np.maximum(0.0, 1.0 - sin2_theta_t))
    v_out = np.zeros_like(v_in)
    v_out[valid] = eta * v_in[valid] + (eta * cos_theta_i[valid] - cos_theta_t[valid])[:, None] * n[valid]

    # Normalize just to be safe
    norms = np.linalg.norm(v_out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    v_out = v_out / norms

    return v_out, valid

def intersect_lines(p0, v0, p1, v1):
    """
    Intersect N rays with M line segments.
    p0: (N, 2) ray origins
    v0: (N, 2) ray directions
    p1: (M, 2) segment start points
    v1: (M, 2) segment directions (p2 - p1)
    Returns: 
        t: (N, M) distances along rays
        u: (N, M) parameters along segments (0 to 1)
    """
    # Cross product 2D: a x b = a_x*b_y - a_y*b_x
    # p0 + t*v0 = p1 + u*v1
    # t*v0 - u*v1 = p1 - p0
    # Solve for t and u using Cramer's rule

    N = p0.shape[0]
    M = p1.shape[0]

    dx = p1[:, 0][None, :] - p0[:, 0][:, None] # (N, M)
    dy = p1[:, 1][None, :] - p0[:, 1][:, None]

    det = v0[:, 0][:, None] * v1[:, 1][None, :] - v0[:, 1][:, None] * v1[:, 0][None, :]

    # Avoid division by zero
    det_safe = np.where(np.abs(det) < 1e-9, 1e-9, det)

    t = (dx * v1[:, 1][None, :] - dy * v1[:, 0][None, :]) / det_safe
    u = (dx * v0[:, 1][:, None] - dy * v0[:, 0][:, None]) / det_safe

    # If det == 0, lines are parallel, set t to inf
    t[np.abs(det) < 1e-9] = np.inf

    return t, u


def get_square_segments(size, theta_rad):
    """
    Returns points p1, v1 and outward normals for a rotated square.
    """
    half = size / 2.0
    corners = np.array([
        [-half, half],
        [half, half],
        [half, -half],
        [-half, -half]
    ])

    c, s = np.cos(theta_rad), np.sin(theta_rad)
    rot = np.array([[c, -s], [s, c]])
    corners_rot = np.dot(corners, rot.T)

    p1 = corners_rot
    p2 = np.roll(corners_rot, -1, axis=0)
    v1 = p2 - p1

    # Normals (outward)
    # v1 is direction of the edge. Normal is (v1_y, -v1_x)
    normals = np.column_stack([v1[:, 1], -v1[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= norms

    return p1, v1, normals

def trace_cuvette(s_positions, theta_rad, L_out, L_in, n_air, n_glass, n_resin):
    """
    Trace rays through a square cuvette.
    s_positions: (N,) array of ray positions on the projector
    Returns origins and directions in resin: p_resin, v_resin, and a valid mask.
    """
    N = len(s_positions)
    p0 = np.column_stack([s_positions, np.full_like(s_positions, L_out)]) # Start above the cuvette
    v0 = np.zeros((N, 2))
    v0[:, 1] = -1.0 # rays go down

    # Outer segments
    p1_out, v1_out, n_out = get_square_segments(L_out, theta_rad)

    # Inner segments
    p1_in, v1_in, n_in = get_square_segments(L_in, theta_rad)

    # 1. Intersect with outer segments
    t_out, u_out = intersect_lines(p0, v0, p1_out, v1_out)

    # We only care about intersections where t > 0 and 0 <= u <= 1
    valid_out_inter = (t_out > 1e-6) & (u_out >= 0) & (u_out <= 1)
    t_out[~valid_out_inter] = np.inf

    # Find the FIRST intersection (minimum t)
    hit_idx_out = np.argmin(t_out, axis=1)
    min_t_out = t_out[np.arange(N), hit_idx_out]

    hit_mask1 = min_t_out < np.inf

    # Update position
    p_glass = p0 + min_t_out[:, None] * v0

    # Refract air -> glass
    n_hit_out = n_out[hit_idx_out]
    v_glass, valid_ref1 = refract_rays(v0, n_hit_out, n_air, n_glass)

    hit_mask1 = hit_mask1 & valid_ref1

    # 2. Intersect with inner segments
    t_in, u_in = intersect_lines(p_glass, v_glass, p1_in, v1_in)
    valid_in_inter = (t_in > 1e-6) & (u_in >= 0) & (u_in <= 1)
    t_in[~valid_in_inter] = np.inf

    hit_idx_in = np.argmin(t_in, axis=1)
    min_t_in = t_in[np.arange(N), hit_idx_in]

    hit_mask2 = min_t_in < np.inf

    # Clamp inf values to 0 to avoid RuntimeWarning; these rays are
    # already masked out by hit_mask2 and will not affect the result.
    min_t_in_safe = np.where(hit_mask2, min_t_in, 0.0)
    p_resin = p_glass + min_t_in_safe[:, None] * v_glass

    # Refract glass -> resin
    # The normal of the inner segment pointing INTO the glass is exactly the outward normal n_in
    n_hit_in = n_in[hit_idx_in]
    v_resin, valid_ref2 = refract_rays(v_glass, n_hit_in, n_glass, n_resin)

    valid_mask = hit_mask1 & hit_mask2 & valid_ref2

    # Rotate p_resin and v_resin by -theta_rad to put them in the resin's co-rotating frame!
    c, s = np.cos(-theta_rad), np.sin(-theta_rad)
    rot_inv = np.array([[c, -s], [s, c]])
    p_resin = np.dot(p_resin, rot_inv.T)
    v_resin = np.dot(v_resin, rot_inv.T)

    return p_resin, v_resin, valid_mask


def intersect_circle(p0, v0, R):
    """
    Intersect rays with a circle centered at origin.
    p0: (N, 2)
    v0: (N, 2)
    R: float
    Returns:
        t1, t2: distances to intersections. t1 is entry, t2 is exit.
        valid: boolean mask
    """
    # |p0 + t*v0|^2 = R^2
    # t^2 + 2(p0 dot v0)t + |p0|^2 - R^2 = 0
    b = 2.0 * np.sum(p0 * v0, axis=1)
    c = np.sum(p0**2, axis=1) - R**2

    discriminant = b**2 - 4*c
    valid = discriminant >= 0

    sqrt_disc = np.sqrt(np.maximum(0, discriminant))
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    return t1, t2, valid

def trace_immersion_vial(s_positions, theta_rad, L_out, L_in, n_air, n_glass, n_liquid, n_resin):
    """
    Trace rays through a stationary square tank with immersion liquid and a rotating cylindrical vial.
    Cube is L_out x L_out. Vial has inner diameter L_in.
    Let's assume the vial has the same glass thickness as the cube: t = (L_out - L_in) / 2 ?
    Wait, L_in is the inner diameter of the vial. 
    Let's use a standard vial wall thickness, e.g., 2.0 mm (since it's not passed separately, 
    but we can just use the same logic: outer diameter is L_out? No, if it's a vial inside a cube, 
    the cube size is L_out, and the vial inner diameter is L_in. Let's assume the vial outer diameter 
    is L_in + 4.0 mm, or just assume the vial is just a thin cylinder, or maybe R_out = L_in/2 + 2.0.
    Wait, in slicing_engine, L_out is params.cuvette_outer_mm (which is diameter), and L_in is diameter - 2*glass_thickness.
    So R_out = L_out / 2, R_in = L_in / 2.
    The stationary cube has size L_out x L_out.
    """
    N = len(s_positions)
    p0 = np.column_stack([s_positions, np.full_like(s_positions, L_out/2.0)]) # Start at the front face of the cube
    v0 = np.zeros((N, 2))
    v0[:, 1] = -1.0 # rays go down

    # 1. Enter the cube (air -> glass -> liquid)
    # Front face of the cube is at y = L_out / 2. Normal is (0, 1).
    # Since rays are parallel to the normal (angle 0), there's no refraction!
    # Just a straight line.
    # The cube is filled with liquid. So rays enter liquid directly.
    # We'll just advance them into the liquid.
    p_liq = p0.copy()
    v_liq = v0.copy()

    # 2. Intersect with outer cylinder (vial)
    R_out = L_out / 2.0
    R_in = L_in / 2.0

    t1_out, t2_out, valid_out = intersect_circle(p_liq, v_liq, R_out)
    hit_mask1 = valid_out & (t1_out > -1e-6)

    p_glass_vial = p_liq + t1_out[:, None] * v_liq
    # Normal of circle at p is p / |p|
    n_vial_out = p_glass_vial / R_out

    # Refract liquid -> glass
    v_glass_vial, valid_ref1 = refract_rays(v_liq, n_vial_out, n_liquid, n_glass)
    hit_mask1 = hit_mask1 & valid_ref1

    # 3. Intersect with inner cylinder (resin)
    t1_in, t2_in, valid_in = intersect_circle(p_glass_vial, v_glass_vial, R_in)
    hit_mask2 = valid_in & (t1_in > 1e-6)

    p_resin = p_glass_vial + t1_in[:, None] * v_glass_vial
    n_vial_in = p_resin / R_in

    # Refract glass -> resin
    # Normal pointing OUT of the inner circle is n_vial_in.
    v_resin, valid_ref2 = refract_rays(v_glass_vial, n_vial_in, n_glass, n_resin)

    valid_mask = hit_mask1 & hit_mask2 & valid_ref2

    # Rotate p_resin and v_resin by -theta_rad!
    # Because the resin is rotating by theta, to compute dose on a stationary grid,
    # we need the rays in the resin's frame.
    c, s = np.cos(-theta_rad), np.sin(-theta_rad)
    rot_inv = np.array([[c, -s], [s, c]])
    p_resin = np.dot(p_resin, rot_inv.T)
    v_resin = np.dot(v_resin, rot_inv.T)

    return p_resin, v_resin, valid_mask
