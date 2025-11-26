import numpy as np
import matplotlib.pyplot as plt
import torch

def get_maze_segments(maze):
    """
    Calculates the boundary lines of a grid-based maze where walls are 1 and path is 0.
    Based on round() logic, boundaries exist at index +/- 0.5.
    
    This version merges contiguous segments into single lines.
    
    Returns:
        segments (list): A list of tuples (x1, y1, x2, y2) representing line segments.
                         x corresponds to ROW index, y corresponds to COLUMN index.
    """
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

    return np.array(segments)

def calculate_maze_barrier(trajectory, segments, epsilon=1e-6):
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
    
    # 4. Calculate Log Barrier
    # Cost = -log(distance)
    # We add epsilon inside log to prevent -inf if distance is exactly 0
    barrier_costs = -torch.log(dist + epsilon)
    
    # Sum costs from all walls for each state
    # Shape: (B, T)
    total_barrier_cost = torch.sum(barrier_costs, dim=-1)
    
    return total_barrier_cost

def clamped_barrier_cost_gradient(trajectory, segments, epsilon=1e-6, max_value=100.0):
    """
    Calculates the gradient of the clamped barrier cost for a batch of trajectories against maze walls.
    
    Args:
        trajectory (torch.Tensor): Shape (B, T, 2) where last dim is (x, y).
        segments (torch.Tensor): Shape (N, 4) where last dim is (x1, y1, x2, y2).
        epsilon (float): Small value to prevent log(0).
    """
    cost = calculate_maze_barrier(trajectory, segments, epsilon)
    grad = torch.autograd.grad(cost, trajectory, grad_outputs=torch.ones_like(cost), create_graph=False, retain_graph=False)
    clamped_grad = torch.clamp(grad, min=-max_value, max=max_value)
    return clamped_grad

def visualize_results(maze, segments):
    """
    Visualizes the maze and the calculated continuous edges.
    Note: Matplotlib plots (x, y) as (horizontal, vertical).
    Your coordinate system uses x=row (vertical), y=col (horizontal).
    We swap them in the plot command to match the visual orientation.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. Plot the grid (imshow treats index 0 as Y/Vertical, index 1 as X/Horizontal)
    # We use origin='upper' to match matrix indexing (0,0 at top left)
    ax.imshow(maze, cmap='Greys', origin='upper')
    
    # 2. Plot the calculated segments
    # Since plot takes (horizontal_axis, vertical_axis), we pass (y, x)
    for i, seg in enumerate(segments):
        x1, y1, x2, y2 = seg
        # Plotting formatted as (col_coords, row_coords)
        # Using different markers for start/end to show they are single lines now
        ax.plot([y1, y2], [x1, x2], color='red', linewidth=2, marker='.', markersize=5)

    # Decoration to prove the .5 alignment
    # Set ticks to show integer centers
    ax.set_xticks(np.arange(maze.shape[1]))
    ax.set_yticks(np.arange(maze.shape[0]))
    ax.grid(color='blue', linestyle='--', linewidth=0.5, alpha=0.3)
    
    ax.set_title(f"Maze Walls (Merged): {len(segments)} Segments")
    ax.set_xlabel("Y Coordinate (Column Index)")
    ax.set_ylabel("X Coordinate (Row Index)")
    
    plt.tight_layout()
    plt.show()

# --- Main Execution ---

# 1. Setup the Maze
maze = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]).astype(bool)

# 2. Get the continuous line segments
# Returns array of shape (N, 4) -> [x1, y1, x2, y2]
wall_segments_np = get_maze_segments(maze)

print(f"Found {len(wall_segments_np)} wall segments.")

# 3. Demonstration of Log Barrier Calculation
# Create dummy trajectory: Batch=2, Time=5, Coords=2
# We place some points near walls to see high cost
dummy_traj_np = np.array([
    # Batch 1: A path moving near the center (safe)
    [[1.5, 1.5], [1.5, 2.5], [1.5, 3.5], [1.5, 4.5], [2.5, 4.5]], 
    # Batch 2: A path very close to a wall at index 0 (unsafe)
    [[0.6, 1.5], [0.55, 1.5], [0.51, 1.5], [0.501, 1.5], [1.5, 1.5]] 
])

# Convert to Torch Tensors
traj_tensor = torch.tensor(dummy_traj_np, dtype=torch.float32)
segments_tensor = torch.tensor(wall_segments_np, dtype=torch.float32)

# Calculate Costs
costs = calculate_maze_barrier(traj_tensor, segments_tensor)

print("\n--- Log Barrier Costs (Higher is closer to wall) ---")
print("Batch 1 (Safe path):")
print(costs[0])
print("Batch 2 (Approaching wall x=0.5):")
print(costs[1])

# 4. Visualize
visualize_results(maze, wall_segments_np)