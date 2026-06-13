import numpy as np
import shapely.geometry as sg

def generate_polygon_vertices(
    num_sides: int,
    image_size: int,
    regular: bool,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate vertices of a regular or irregular simple star-convex polygon.

    Args:
        num_sides: Number of sides (vertices).
        image_size: The target image size in pixels.
        regular: True for regular polygon, False for irregular.
        rng: A NumPy random Generator.

    Returns:
        np.ndarray: Vertices of shape (num_sides, 2).
    """
    cx = image_size / 2.0
    cy = image_size / 2.0
    
    # Base radius: 30% to 42% of image size
    r_base = image_size * rng.uniform(0.3, 0.42)
    theta_0 = rng.uniform(0.0, 2.0 * np.pi)
    d_theta = 2.0 * np.pi / num_sides
    
    vertices = []
    for k in range(num_sides):
        angle = theta_0 + k * d_theta
        r = r_base
        
        if not regular:
            # Add noise to radius and angle
            r = r_base + rng.uniform(-0.25 * r_base, 0.25 * r_base)
            angle = angle + rng.uniform(-0.15 * d_theta, 0.15 * d_theta)
            
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        vertices.append((x, y))
        
    return np.array(vertices)

def step_with_bounce(
    poly: sg.Polygon,
    curr_pos: tuple[float, float],
    velocity: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float], bool]:
    """
    Simulate a single step of the moving point, bouncing off the polygon if hit.

    Args:
        poly: The shapely Polygon boundary.
        curr_pos: Current position (x, y).
        velocity: Current velocity (vx, vy).

    Returns:
        tuple containing (new_pos, new_velocity, bounced_flag)
    """
    P_start = curr_pos
    P_end = (P_start[0] + velocity[0], P_start[1] + velocity[1])
    step_seg = sg.LineString([P_start, P_end])
    
    # Check intersection with the boundary
    inter = step_seg.intersection(poly.boundary)
    if inter.is_empty:
        return P_end, velocity, False
        
    # Find the intersection point closest to P_start
    if isinstance(inter, sg.Point):
        I = (inter.x, inter.y)
    elif isinstance(inter, (sg.MultiPoint, sg.GeometryCollection)):
        # Find closest point
        points = []
        if isinstance(inter, sg.MultiPoint):
            points = list(inter.geoms)
        else:
            # Extract points from GeometryCollection
            for geom in inter.geoms:
                if isinstance(geom, sg.Point):
                    points.append(geom)
                elif isinstance(geom, sg.LineString):
                    points.extend([sg.Point(c) for c in geom.coords])
        if not points:
            return P_end, velocity, False
        closest_p = min(points, key=lambda p: np.hypot(p.x - P_start[0], p.y - P_start[1]))
        I = (closest_p.x, closest_p.y)
    else:
        # Fallback if intersection format is unexpected
        return P_end, velocity, False

    # Find which edge of the polygon was hit to compute normal
    coords = list(poly.boundary.coords)
    hit_edge = None
    min_dist = float('inf')
    
    for i in range(len(coords) - 1):
        A, B = coords[i], coords[i+1]
        edge_seg = sg.LineString([A, B])
        dist = edge_seg.distance(sg.Point(I))
        if dist < min_dist:
            min_dist = dist
            hit_edge = (A, B)
            
    if hit_edge is None:
        # Fallback
        return P_end, velocity, False
        
    A, B = hit_edge
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    
    # Compute normal (nx, ny)
    nx, ny = -dy, dx
    length = np.hypot(nx, ny)
    if length > 0:
        nx /= length
        ny /= length
        
    # Ensure normal points inward
    test_p = sg.Point(I[0] + 1e-3 * nx, I[1] + 1e-3 * ny)
    if not poly.contains(test_p):
        nx = -nx
        ny = -ny
        
    # Reflect the velocity: V_reflected = V - 2 * (V . N) * N
    v_dot_n = velocity[0] * nx + velocity[1] * ny
    vx_ref = velocity[0] - 2.0 * v_dot_n * nx
    vy_ref = velocity[1] - 2.0 * v_dot_n * ny
    reflected_velocity = (vx_ref, vy_ref)
    
    # Calculate new position based on remaining fraction of the step
    d_total = np.hypot(velocity[0], velocity[1])
    d_1 = np.hypot(I[0] - P_start[0], I[1] - P_start[1])
    d_rem = max(0.0, d_total - d_1)
    
    if d_total > 0:
        new_pos = (
            I[0] + d_rem * (vx_ref / d_total),
            I[1] + d_rem * (vy_ref / d_total)
        )
    else:
        new_pos = I
        
    # Clamp new position if it went slightly out due to precision
    if not poly.contains(sg.Point(new_pos)):
        new_pos = (I[0] + 1e-2 * nx, I[1] + 1e-2 * ny)
        
    return new_pos, reflected_velocity, True

def simulate_trajectory(
    poly_vertices: np.ndarray,
    num_timesteps: int,
    velocity_set: list[tuple[float, float]],
    image_size: int,
    rng: np.random.Generator
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], int] | None:
    """
    Simulate a trajectory of an object inside the polygon with wall bouncing.
    
    Returns the path, the velocities at each step, and the collision label 
    for the next step.

    Args:
        poly_vertices: Vertices of the polygon (N, 2).
        num_timesteps: Number of rendered timesteps.
        velocity_set: List of possible velocity tuples (vx, vy).
        image_size: The image size.
        rng: A NumPy random Generator.

    Returns:
        tuple containing (path, velocities, label), or None if simulation failed.
    """
    poly = sg.Polygon(poly_vertices)
    bounds = poly.bounds
    
    for _ in range(150):
        # Sample starting point inside the polygon
        x0 = rng.uniform(bounds[0], bounds[2])
        y0 = rng.uniform(bounds[1], bounds[3])
        p0 = sg.Point(x0, y0)
        
        # Ensure starting point is well inside the polygon boundary
        if not poly.contains(p0) or poly.boundary.distance(p0) < 3.0:
            continue
            
        # Choose a random initial velocity
        v = velocity_set[rng.choice(len(velocity_set))]
        
        path = [(x0, y0)]
        velocities = [v]
        bounces = []
        
        # Simulate for num_timesteps steps
        # - Rendered timesteps: 0 to num_timesteps - 1 (length num_timesteps)
        # - The next step is from num_timesteps - 1 to num_timesteps
        for t in range(num_timesteps):
            curr_pos = path[-1]
            curr_vel = velocities[-1]
            next_pos, next_vel, bounced = step_with_bounce(poly, curr_pos, curr_vel)
            
            path.append(next_pos)
            velocities.append(next_vel)
            bounces.append(bounced)
            
        # The collision label (0 or 1) is whether a bounce occurred in the last simulated step
        label = 1 if bounces[-1] else 0
        
        # We return the path of length num_timesteps (representing t=0..num_timesteps-1)
        # and its corresponding velocities.
        # path has length num_timesteps + 1; we keep path[:-1] for rendered frames
        return path[:-1], velocities[:-1], label
        
    return None

def apply_label_noise(
    true_label: int,
    label_noise: float,
    rng: np.random.Generator
) -> int:
    """
    Apply label noise to collision prediction label (0 or 1).
    """
    if label_noise <= 0.0:
        return true_label
        
    if rng.uniform(0.0, 1.0) < label_noise:
        return 1 - true_label
        
    return true_label
