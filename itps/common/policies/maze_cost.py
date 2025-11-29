import numpy as np
import torch
import matplotlib.pyplot as plt

def get_maze_segments(maze: torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Calculates the boundary lines of a grid-based maze where walls are 1 and path is 0.
    Based on round() logic, boundaries exist at index +/- 0.5.
    
    This version merges contiguous segments into single lines.
    
    Returns:
        segments (list): A list of tuples (x1, y1, x2, y2) representing line segments.
                         x corresponds to ROW index, y corresponds to COLUMN index.
    """

    if type(maze) == torch.Tensor:
        maze = maze.detach().cpu().numpy()
    elif type(maze) == np.ndarray:
        pass
    else:
        raise ValueError(f"Invalid type for maze: {type(maze)}")

    segments = []
    rows, cols = maze.shape

    # --- 1. Horizontal Boundaries (Transitions between Rows) ---
    # We look for places where maze[r, c] != maze[r+1, c]
    # The boundary exists at x = r + 0.5
    
    # Calculate difference along axis 0 (rows)
    row_diff = np.diff(maze.astype(int), axis=0) 
    
    # Iterate over each row interface to find and merge segments
    for r in range(row_diff.shape[0]):
        # Find column indices where a transition occurs at this row level
        c_indices = np.where(row_diff[r, :] != 0)[0]
        
        if len(c_indices) == 0:
            continue
            
        # Group consecutive indices to merge segments
        # Logic: Find breaks where the difference between indices is > 1
        breaks = np.where(np.diff(c_indices) > 1)[0] + 1
        groups = np.split(c_indices, breaks)
        
        for group in groups:
            # The boundary is at fixed x
            x_pos = r + 0.5
            
            # The line spans from the start of the first col to the end of the last col in the group
            y_start = group[0] - 0.5
            y_end = group[-1] + 0.5
            
            segments.append((x_pos, y_start, x_pos, y_end))

    # --- 2. Vertical Boundaries (Transitions between Columns) ---
    # We look for places where maze[r, c] != maze[r, c+1]
    # The boundary exists at y = c + 0.5
    
    # Calculate difference along axis 1 (columns)
    col_diff = np.diff(maze.astype(int), axis=1)
    
    # Iterate over each column interface to find and merge segments
    for c in range(col_diff.shape[1]):
        # Find row indices where a transition occurs at this column level
        r_indices = np.where(col_diff[:, c] != 0)[0]
        
        if len(r_indices) == 0:
            continue
            
        # Group consecutive indices
        breaks = np.where(np.diff(r_indices) > 1)[0] + 1
        groups = np.split(r_indices, breaks)
        
        for group in groups:
            # The boundary is at fixed y
            y_pos = c + 0.5
            
            # The line spans from the start of the first row to the end of the last row in the group
            x_start = group[0] - 0.5
            x_end = group[-1] + 0.5
            
            segments.append((x_start, y_pos, x_end, y_pos))

    return torch.tensor(segments, dtype=torch.float32)

def calculate_maze_barrier_cost(trajectory: torch.Tensor, segments: torch.Tensor, epsilon: float = 1e-6, alpha: float = 10) -> torch.Tensor:
    """
    Calculates the log barrier cost for a batch of trajectories against maze walls.
    
    The logic implements distance-to-segment:
    1. If the point projects onto the segment (between endpoints), distance is perpendicular.
    2. If the point projects outside, distance is Euclidean to the nearest endpoint.
    
    Args:
        trajectory (torch.Tensor): Shape (B, T, 2) where last dim is (x, y).
        segments (torch.Tensor): Shape (N, 4) where last dim is (x1, y1, x2, y2).
        epsilon (float): Small value to prevent log(0).
        
    Returns:
        barrier_cost (torch.Tensor): Shape (B, T). Sum of -log(dist) for all edges.
    """
    # 1. Prepare Shapes for Broadcasting
    # We want to compare every trajectory point (B, T) against every Segment (N)
    
    # Trajectory: (B, T, 1, 2)
    traj_expanded = trajectory.unsqueeze(2) 
    
    # Segments: (1, 1, N, 4)
    segs_expanded = segments.view(1, 1, -1, 4)
    
    # Extract endpoints P1(x1, y1) and P2(x2, y2)
    # P1, P2 shape: (1, 1, N, 2)
    p1 = segs_expanded[..., :2]
    p2 = segs_expanded[..., 2:]
    
    # 2. Vectorized Point-to-Segment Distance
    # Vector representing the segment: AB = P2 - P1
    seg_vec = p2 - p1
    
    # Vector from P1 to the trajectory point: AP = Traj - P1
    pt_vec = traj_expanded - p1
    
    # Project AP onto AB to find position 't' along the segment
    # t = dot(AP, AB) / dot(AB, AB)
    # shape of seg_len_sq: (1, 1, N)
    seg_len_sq = torch.sum(seg_vec ** 2, dim=-1)
    
    # Avoid division by zero for zero-length segments (though maze segments shouldn't be 0)
    seg_len_sq = torch.clamp(seg_len_sq, min=epsilon)
    
    # Dot product: sum(AP * AB, dim=-1)
    dot_prod = torch.sum(pt_vec * seg_vec, dim=-1)
    
    # Normalized projection scalar t
    t = dot_prod / seg_len_sq
    
    # Clamp t to segment range [0, 1]
    # If t was < 0, it means the point is "before" P1 (closest is P1)
    # If t was > 1, it means the point is "after" P2 (closest is P2)
    # If 0 <= t <= 1, the point is "alongside" the segment (closest is projection)
    t_clamped = torch.clamp(t, 0.0, 1.0)
    
    # Calculate the nearest point on the segment
    # Nearest = P1 + t_clamped * (P2 - P1)
    # We need to unsqueeze t_clamped to (B, T, N, 1) for broadcasting against (..., 2)
    nearest_point = p1 + t_clamped.unsqueeze(-1) * seg_vec
    
    # 3. Calculate Euclidean Distance
    # Dist = ||Traj - Nearest||
    # diff shape: (B, T, N, 2)
    diff = traj_expanded - nearest_point
    dist = torch.norm(diff, dim=-1) # shape: (B, T, N)
    # print(dist.shape)
    # two_closest_walls = torch.topk(-dist, k=2, dim=-1)[0]

    
    # # 4. Calculate Log Barrier
    # # Cost = -log(distance)
    # # We add epsilon inside log to prevent -inf if distance is exactly 0
    # barrier_costs = -alpha * torch.log(-two_closest_walls + epsilon)
    # barrier_costs = torch.abs(alpha * 1/(two_closest_walls + epsilon))
    closest_wall = torch.min(dist, dim=-1)[0]
    # print(closest_wall.shape)
    barrier_costs = torch.nn.functional.sigmoid(-alpha * closest_wall)
    # print(barrier_costs.shape)
    # Sum costs from all walls for each state
    # Shape: (B, T)
    # total_barrier_cost = torch.sum(barrier_costs, dim=-1)
    total_barrier_cost = barrier_costs
    # print(total_barrier_cost.shape)
    return total_barrier_cost
    
def clamped_barrier_cost_gradient(trajectory: torch.Tensor, segments: torch.Tensor, epsilon: float = 1e-6, max_value: float = 100.0) -> torch.Tensor:
    """
    Calculates the gradient of the clamped barrier cost for a batch of trajectories against maze walls.
    
    Args:
        trajectory (torch.Tensor): Shape (B, T, 2) where last dim is (x, y).
        segments (torch.Tensor): Shape (N, 4) where last dim is (x1, y1, x2, y2).
        epsilon (float): Small value to prevent log(0).
    """
    cost = calculate_maze_barrier_cost(trajectory, segments, epsilon=epsilon)
    grad = torch.autograd.grad(cost, trajectory, grad_outputs=torch.ones_like(cost), create_graph=False, retain_graph=False)[0]
    clamped_grad = torch.clamp(grad, min=-max_value, max=max_value)
    return clamped_grad

def calculate_trajectory_intersection_cost(trajectory, segments, cost_type='euclidean', alpha=.3, epsilon=1e-6):
    """
    Calculates a cost for trajectory segments that cross maze walls.
    Drives state i+1 towards state i to resolve the intersection.

    Args:
        trajectory (torch.Tensor): Shape (B, T, 2).
        segments (torch.Tensor): Shape (N, 4).
        cost_type (str): 'euclidean' (dist to intersection) or 'fraction' (percentage of step clipped).
        epsilon (float): Stability term.

    Returns:
        intersection_cost (torch.Tensor): Shape (B, T-1).
    """
    # 1. Define Vectors
    # A = State i, B = State i+1
    # Shape: (B, T-1, 1, 2)
    A = trajectory[:, :-1, :].unsqueeze(2)
    B = trajectory[:, 1:, :].unsqueeze(2)
    
    # C = Wall Start, D = Wall End
    # Shape: (1, 1, N, 2)
    C = segments[:, :2].view(1, 1, -1, 2)
    D = segments[:, 2:].view(1, 1, -1, 2)

    # 2. Vector Differences
    # Traj Vector: AB = B - A
    AB = B - A
    # Wall Vector: CD = D - C
    CD = D - C
    # Connector: AC = C - A
    AC = C - A

    # 3. 2D Cross Product Analog (x1*y2 - y1*x2)
    # We define a helper for the 2D cross product
    def cross_2d(v1, v2):
        return v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]

    denom = cross_2d(AB, CD)
    
    # Avoid division by zero (parallel lines)
    # If denom is 0, t and u become huge, which we filter out later via masks
    denom = torch.where(torch.abs(denom) < epsilon, torch.tensor(epsilon), denom)

    # 4. Calculate Intersection Parameters t and u
    # t: fraction along Trajectory AB where intersection occurs (0 to 1)
    # u: fraction along Wall CD where intersection occurs (0 to 1)
    # Formula derived from A + t(AB) = C + u(CD)
    
    t = cross_2d(AC, CD) / denom
    u = cross_2d(AC, AB) / denom

    # 5. Determine Validity of Intersections
    # An intersection is valid if 0 <= t <= 1 AND 0 <= u <= 1
    # We use a slightly relaxed range (-eps, 1+eps) to handle grazing hits numerically
    valid_intersection = (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    
    # 6. Find Closest Intersection (Min t)
    # A single step might jump over multiple walls. We care about the *first* one (smallest t).
    
    # Set t of invalid intersections to infinity so min() ignores them
    t_filtered = torch.where(valid_intersection, t, torch.tensor(float('inf')))
    
    # Find min t along the Wall dimension (dim=-1)
    # t_min: (B, T-1), values are the fraction of the step before hitting the FIRST wall
    t_min, _ = torch.min(t_filtered, dim=-1)
    
    # Identify which steps actually hit a wall (t_min is not inf)
    has_collision = t_min < float('inf')
    
    # Replace inf with 1.0 (no collision = full step allowed) for cost calculation
    t_clamped = torch.where(has_collision, t_min, torch.tensor(1.0))
    
    # 7. Calculate Cost
    # We want to penalize the part of the step that exists AFTER the wall (from t to 1).
    
    if cost_type == 'euclidean':
        # Cost is distance from s_{i+1} to the intersection point P
        # P = A + t * AB
        # dist = || B - P || = || B - (A + t(B-A)) || = || (1-t)(B-A) ||
        # squared euclidean is usually cleaner for optimization
        step_len_sq = torch.sum(AB.squeeze(2)**2, dim=-1) # (B, T-1)
        cost = ((1.0 - t_clamped) ** 2) * step_len_sq * alpha
        
    elif cost_type == 'fraction':
        # Simply penalize the fraction of the step that is invalid.
        # If t=0.1 (hit early), penalty is (0.9)^2 = 0.81 (high).
        # If t=1.0 (no hit), penalty is 0.
        cost = (1.0 - t_clamped) ** 2
        
    else:
        raise ValueError("Unknown cost_type")

    # Only apply cost where collision occurred
    # (Though t_clamped=1.0 yields cost 0.0, explicit masking is safer for gradients)
    final_cost = torch.where(has_collision, cost, torch.tensor(0.0))
    
    return final_cost

def calculate_trajectory_intersection_cost_gradient(trajectory: torch.Tensor, segments: torch.Tensor, epsilon: float = 1e-6, max_value: float = 10.0) -> torch.Tensor:
    """
    Calculates the gradient of the trajectory intersection cost for a batch of trajectories against maze walls.
    
    Args:
        trajectory (torch.Tensor): Shape (B, T, 2) where last dim is (x, y).
        segments (torch.Tensor): Shape (N, 4) where last dim is (x1, y1, x2, y2).
        epsilon (float): Small value to prevent log(0).
    """
    cost = calculate_trajectory_intersection_cost(trajectory, segments, epsilon=epsilon)
    grad = torch.autograd.grad(cost, trajectory, grad_outputs=torch.ones_like(cost), create_graph=False, retain_graph=False)[0]
    clamped_grad = torch.clamp(grad, min=-max_value, max=max_value)
    return clamped_grad

def calculate_virtual_tail_cost(trajectory, segments, rot_scale=1.0, trans_scale=0, buffer=0.2, epsilon=1e-6):
    """
    Calculates a guidance cost by constructing a 'virtual tail'.
    
    1. Finds the EARLIEST step i->i+1 that crosses a wall.
    2. Calculates intersection point P on the wall.
    3. Adjusts P by 'buffer' distance towards state i to create Pivot Point P_adj.
    4. Takes the trajectory tail (points i+1 onward).
    5. Rotates the tail around P_adj to flatten it against the wall (scaled by rot_scale).
    6. Translates the tail towards state i (scaled by trans_scale).
    7. Returns MSE between actual tail and virtual tail.
    
    Args:
        trajectory (torch.Tensor): Shape (B, T, 2)
        segments (torch.Tensor): Shape (N, 4)
        rot_scale (float): 0.0 to 1.0 (or higher). 1.0 flattens completely parallel to wall.
        trans_scale (float): Scale factor for moving tail back towards state i.
        buffer (float): Distance to shift the intersection point back towards the safe state i.
        
    Returns:
        loss (torch.Tensor): Shape (B,). Scalar cost per trajectory.
    """
    batch_size, num_steps, _ = trajectory.shape
    device = trajectory.device
    
    # --- 1. Find Intersections (Vectorized) ---
    # A = State i, B = State i+1
    A = trajectory[:, :-1, :].unsqueeze(2) # (B, T-1, 1, 2)
    B = trajectory[:, 1:, :].unsqueeze(2)  # (B, T-1, 1, 2)
    
    # C = Wall Start, D = Wall End
    C = segments[:, :2].view(1, 1, -1, 2)  # (1, 1, N, 2)
    D = segments[:, 2:].view(1, 1, -1, 2)  # (1, 1, N, 2)

    AB = B - A
    CD = D - C
    AC = C - A

    def cross_2d(v1, v2):
        return v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]

    denom = cross_2d(AB, CD)
    denom = torch.where(torch.abs(denom) < epsilon, torch.tensor(epsilon, device=device), denom)

    # t: fraction along Trajectory AB (0 to 1)
    t = cross_2d(AC, CD) / denom
    # u: fraction along Wall CD (0 to 1)
    u = cross_2d(AC, AB) / denom

    # Valid intersection: t in [0,1], u in [0,1]
    valid_mask = (t >= -epsilon) & (t <= 1.0 + epsilon) & (u >= -epsilon) & (u <= 1.0 + epsilon)
    
    # --- 2. Process Each Trajectory in Batch ---
    # (Looping over batch is safer for complex geometric reconstruction per item)
    
    batch_losses = []
    
    for b in range(batch_size):
        # Filter collisions for this trajectory
        # t_vals: (T-1, N)
        t_vals = t[b]
        mask = valid_mask[b]
        
        # We need the EARLIEST step (smallest step index i) that has ANY collision.
        # Check if any wall is hit at each step
        step_hits_wall = torch.any(mask, dim=1) # (T-1,)
        
        # Get indices of steps that hit walls
        hit_indices = torch.nonzero(step_hits_wall, as_tuple=True)[0]
        
        if len(hit_indices) == 0:
            # No collision, zero loss
            batch_losses.append(torch.tensor(0.0, device=device, requires_grad=True))
            continue
            
        # Earliest collision step index
        i = hit_indices[0]
        
        # Within this step, find the wall that was hit earliest (smallest t)
        # mask[i] is shape (N,)
        valid_t_in_step = torch.where(mask[i], t_vals[i], torch.tensor(float('inf'), device=device))
        best_t, wall_idx = torch.min(valid_t_in_step, dim=0)
        
        # Ensure best_t is usable (clamp for numerical safety)
        best_t = torch.clamp(best_t, 0.0, 1.0)
        
        # --- 3. Construct Virtual Tail ---
        
        # Data for calculation (detached, so we don't backprop through the TARGET construction)
        # We want to pull the ACTIVE trajectory towards a FIXED target.
        curr_traj_tail = trajectory[b, i+1:, :] # Shape (Len_Tail, 2) - This has gradients!
        
        with torch.no_grad():
            s_i = trajectory[b, i, :]
            s_next = trajectory[b, i+1, :]
            
            # Intersection Point P
            # P = A + t * AB
            vec_step = s_next - s_i
            step_len = torch.norm(vec_step)
            
            # Safe normalization
            if step_len > epsilon:
                step_dir = vec_step / step_len
            else:
                step_dir = torch.zeros_like(vec_step)
                
            # Distance from s_i to intersection
            dist_to_int = best_t * step_len
            
            # Clamp distance so it doesn't go negative (past s_i)
            # This ensures p_adj is between s_i and the intersection point
            dist_adj = torch.clamp(dist_to_int - buffer, min=0.0)
            
            # Adjusted Pivot Point
            p_adj = s_i + step_dir * dist_adj
            
            # Wall Vector W
            w_start = segments[wall_idx, :2]
            w_end = segments[wall_idx, 2:]
            vec_wall = w_end - w_start
            
            # Vector from Adjusted Pivot to S_{i+1} 
            # (Basically represents the part of the step considered 'invalid' or 'penetrating' relative to the buffer)
            vec_penetration = s_next - p_adj
            
            # --- Rotation Logic ---
            # Angle of penetration vector
            angle_pen = torch.atan2(vec_penetration[1], vec_penetration[0])
            
            # Angle of wall vector
            angle_wall = torch.atan2(vec_wall[1], vec_wall[0])
            
            # We have two wall directions: angle_wall and angle_wall + pi.
            # We want the one that makes the SMALLER angle with the penetration vector.
            # Diff 1
            d1 = (angle_pen - angle_wall + np.pi) % (2*np.pi) - np.pi
            # Diff 2 (opposite direction)
            angle_wall_opp = angle_wall + np.pi
            d2 = (angle_pen - angle_wall_opp + np.pi) % (2*np.pi) - np.pi
            
            if torch.abs(d1) < torch.abs(d2):
                delta_angle = d1
            else:
                delta_angle = d2
            
            # Target rotation: We want to reduce this delta by rot_scale.
            # If we rotate the vector by -delta, it aligns with wall.
            rotation_angle = -delta_angle * rot_scale
            
            # Create Rotation Matrix
            c = torch.cos(rotation_angle)
            s = torch.sin(rotation_angle)
            R = torch.tensor([[c, -s], [s, c]], device=device)
            
            # --- Apply Rotation to whole tail ---
            # Center tail at P_adj
            tail_centered = curr_traj_tail.detach() - p_adj
            # Rotate: (N, 2) @ (2, 2) -> (N, 2)
            # Need transpose for matmul: (R @ v.T).T = v @ R.T
            tail_rotated = torch.matmul(tail_centered, R.T)
            
            # --- Translation Logic ---
            # Move closer to state i.
            # Direction: From P_adj towards S_i (which is -step_dir)
            vec_back = s_i - p_adj
            if torch.norm(vec_back) > epsilon:
                dir_back = vec_back / torch.norm(vec_back)
            else:
                dir_back = torch.zeros_like(vec_back)
                
            # Magnitude: scaled multiple of length of line segment (P_adj to S_{i+1})
            len_segment = torch.norm(vec_penetration)
            translation = dir_back * len_segment * trans_scale
            
            # --- Final Virtual Target ---
            virtual_tail = tail_rotated + p_adj + translation
        
        # --- 4. Compute Loss ---
        # MSE between Actual Tail (with grads) and Virtual Tail (detached)
        loss = torch.nn.functional.mse_loss(curr_traj_tail, virtual_tail)
        batch_losses.append(loss)
        
    return torch.stack(batch_losses)

def calculate_self_overlap_cost(trajectory: torch.Tensor, min_separation: float, ignore_nearby: int, alpha: float):
    """
    Penalize trajectory self-overlap/looping by applying a soft penalty when non-adjacent
    points come within `min_separation` of each other.

    Args:
        trajectory: (B, T, 2)
        min_separation: distance threshold in same units as trajectory (unnormalized)
        ignore_nearby: ignore pairs with |i-j| <= ignore_nearby (adjacent steps)
        alpha: sharpness for sigmoid or hinge
    Returns:
        per-step cost (B, T) or per-trajectory (B,) depending preference. We'll return per-step sum.
    """
    B, T, _ = trajectory.shape
    # pairwise distances: (B, T, T)
    diff = trajectory.unsqueeze(2) - trajectory.unsqueeze(1)  # (B, T, T, 2)
    dist = torch.norm(diff, dim=-1)  # (B, T, T)

    # build mask to exclude self and nearby indices
    idxs = torch.arange(T, device=trajectory.device)
    i = idxs.view(-1, 1)
    j = idxs.view(1, -1)
    mask = (torch.abs(i - j) > ignore_nearby).float()  # (T, T)

    # Compute soft hinge: cost = softplus(min_separation - dist)
    # Optionally scale with alpha: softplus(alpha * (min_sep - d)) / alpha
    inv = torch.nn.functional.softplus(alpha * (min_separation - dist)) / alpha  # (B, T, T)
    # zero out excluded pairs
    inv = inv * mask.unsqueeze(0)

    # Sum pairwise costs per point or per trajectory.
    per_point_cost = torch.sum(inv, dim=2)  # (B, T) -> cost for each point (sum over other points)
    # Option: average or clamp
    return per_point_cost

def calculate_virtual_tail_cost_gradient(trajectory: torch.Tensor, segments: torch.Tensor, epsilon: float = 1e-6, max_value: float = 1000.0) -> torch.Tensor:
    """
    Calculates the gradient of the virtual tail cost for a batch of trajectories against maze walls.
    
    Args:
        trajectory (torch.Tensor): Shape (B, T, 2) where last dim is (x, y).
        segments (torch.Tensor): Shape (N, 4) where last dim is (x1, y1, x2, y2).
        epsilon (float): Small value to prevent log(0).
    """
    cost = calculate_virtual_tail_cost(trajectory, segments, epsilon=epsilon)
    grad = torch.autograd.grad(cost, trajectory, grad_outputs=torch.ones_like(cost), create_graph=False, retain_graph=False, allow_unused=True)[0]
    if grad is None:
        grad = torch.zeros_like(trajectory)
    clamped_grad = torch.clamp(grad, min=-max_value, max=max_value)
    return clamped_grad

def calculate_self_overlap_cost_gradient(trajectory: torch.Tensor, min_separation: float = 1.0, ignore_nearby: int = 2, alpha: float = 10.0, max_value: float = 1000.0) -> torch.Tensor:
    cost = calculate_self_overlap_cost(trajectory, min_separation=min_separation, ignore_nearby=ignore_nearby, alpha=alpha)
    # sum cost over time to get scalar per batch element
    cost_scalar = cost.mean(dim=1)  # (B,)
    grad = torch.autograd.grad(cost_scalar, trajectory, grad_outputs=torch.ones_like(cost_scalar), create_graph=False, retain_graph=False, allow_unused=False)[0]
    if grad is None:
        return torch.zeros_like(trajectory)
    return torch.clamp(grad, min=-max_value, max=max_value)


def calculate_combo_trajectory_intersection_and_virtual_tail_cost_gradient(trajectory: torch.Tensor, segments: torch.Tensor, epsilon: float = 1e-6, max_value: float = 1000.0) -> torch.Tensor:
    intersection_grad = calculate_trajectory_intersection_cost_gradient(trajectory, segments, epsilon=epsilon, max_value=max_value)
    virtual_tail_grad = calculate_virtual_tail_cost_gradient(trajectory, segments, epsilon=epsilon, max_value=max_value)
    return intersection_grad + virtual_tail_grad


def calculate_combo_virtual_tail_and_self_overlap_cost_gradient(
    trajectory: torch.Tensor,
    segments: torch.Tensor,
    min_separation: float = 0.1,
    ignore_nearby: int = 2,
    alpha: float = 10.0,
    epsilon: float = 1e-6,
    max_value: float = 1000.0,
    virtual_weight: float = 1.0,
    self_weight: float = 0.1,
) -> torch.Tensor:
    """
    Combine the gradients from the virtual-tail cost and the self-overlap cost.

    This helper computes each component's gradient (already clamped by their
    respective helpers) and returns a weighted sum, clamped to `[-max_value, max_value]`.

    Args:
        trajectory: (B, T, 2) trajectory tensor requiring gradients.
        segments: (N, 4) maze segments tensor (used by virtual-tail cost).
        min_separation, ignore_nearby, alpha: parameters forwarded to self-overlap cost.
        epsilon: numerical stability forwarded to the virtual-tail component.
        max_value: final gradient clamp magnitude.
        virtual_weight: scalar weight for the virtual-tail gradient.
        self_weight: scalar weight for the self-overlap gradient.

    Returns:
        grad: (B, T, 2) combined gradient tensor clamped to [-max_value, max_value].
    """
    virtual_grad = calculate_virtual_tail_cost_gradient(trajectory, segments, epsilon=epsilon, max_value=max_value)
    self_grad = calculate_self_overlap_cost_gradient(trajectory, min_separation=min_separation, ignore_nearby=ignore_nearby, alpha=alpha, max_value=max_value)

    combined = virtual_weight * virtual_grad + self_weight * self_grad
    return torch.clamp(combined, min=-max_value, max=max_value)


def plot_barrier_heatmap(maze, segments_tensor, resolution=20):
    """
    Plots a heatmap of the barrier costs over the maze area.
    
    Args:
        maze (np.array): The maze array for dimensions.
        segments_tensor (torch.Tensor): The calculated wall segments.
        resolution (int): Points per unit grid cell for the heatmap.
    """
    rows, cols = maze.shape
    
    # Create a meshgrid of points covering the maze
    # We use indexing='ij' so x maps to rows and y maps to cols
    x = np.linspace(0, rows-1, rows * resolution)
    y = np.linspace(0, cols-1, cols * resolution)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Stack to create (Height, Width, 2) coordinates
    grid_points = np.stack([X, Y], axis=-1)
    
    # Flatten to (1, N, 2) to pass as a "trajectory" to the barrier function
    flat_points = grid_points.reshape(1, -1, 2)
    traj_tensor = torch.tensor(flat_points, dtype=torch.float32)
    
    # Calculate costs
    # Note: This might be memory intensive for very large mazes. 
    # For a 12x12 maze, it's fine.
    with torch.no_grad():
        costs = calculate_maze_barrier_cost(traj_tensor, segments_tensor)
        
    # Reshape back to grid dimensions
    cost_grid = costs.reshape(X.shape).numpy()
    
    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Heatmap
    # We cap the maximum cost for visualization because points ON the wall 
    # go to infinity, which washes out the color scale.
    vmax = np.percentile(cost_grid, 99) # Auto-scale to 95th percentile
    
    im = ax.imshow(cost_grid, origin='upper', cmap='hot', 
                   extent=[-0.5, cols-0.5, rows-0.5, -0.5], # Align to integer centers
                   vmax=vmax)
    
    plt.colorbar(im, label='Log Barrier Cost (-log(dist))')
    
    # Plot Wall Segments overlay
    segments_np = segments_tensor.numpy()
    for seg in segments_np:
        x1, y1, x2, y2 = seg
        # Plot (y, x) because matplotlib is (col, row)
        ax.plot([y1, y2], [x1, x2], 'c-', linewidth=1.5, alpha=0.7)

    ax.set_title(f"Barrier Cost Heatmap\n(Lighter = Higher Cost/Closer to Wall)")
    ax.set_xlabel("Y Coordinate (Column)")
    ax.set_ylabel("X Coordinate (Row)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    old_maze = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool)

    new_maze = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool)
    maze = new_maze ^ old_maze
    segments = get_maze_segments(maze)
    plot_barrier_heatmap(maze, segments)