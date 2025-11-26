import numpy as np
import matplotlib.pyplot as plt

def get_maze_segments(maze):
    """
    Calculates the boundary lines of a grid-based maze where walls are 1 and path is 0.
    Based on round() logic, boundaries exist at index +/- 0.5.
    
    Returns:
        segments (list): A list of tuples (x1, y1, x2, y2) representing line segments.
                         x corresponds to ROW index, y corresponds to COLUMN index.
    """
    segments = []
    rows, cols = maze.shape

    # --- 1. Horizontal Boundaries (Transitions between Rows) ---
    # We look for places where maze[r, c] != maze[r+1, c]
    # The boundary exists at x = r + 0.5
    # The segment spans y from c - 0.5 to c + 0.5
    
    # Calculate difference along axis 0 (rows)
    # diff[r, c] is non-zero if row r and r+1 are different
    row_diff = np.diff(maze.astype(int), axis=0) 
    
    # Get coordinates where transitions happen
    r_indices, c_indices = np.nonzero(row_diff)
    
    for r, c in zip(r_indices, c_indices):
        # The boundary is between row r and r+1
        x_pos = r + 0.5
        
        # The line spans the width of column c
        y_start = c - 0.5
        y_end = c + 0.5
        
        segments.append((x_pos, y_start, x_pos, y_end))

    # --- 2. Vertical Boundaries (Transitions between Columns) ---
    # We look for places where maze[r, c] != maze[r, c+1]
    # The boundary exists at y = c + 0.5
    # The segment spans x from r - 0.5 to r + 0.5
    
    # Calculate difference along axis 1 (columns)
    col_diff = np.diff(maze.astype(int), axis=1)
    
    # Get coordinates where transitions happen
    r_indices, c_indices = np.nonzero(col_diff)
    
    for r, c in zip(r_indices, c_indices):
        # The boundary is between col c and c+1
        y_pos = c + 0.5
        
        # The line spans the height of row r
        x_start = r - 0.5
        x_end = r + 0.5
        
        segments.append((x_start, y_pos, x_end, y_pos))

    return np.array(segments)

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
    for seg in segments:
        x1, y1, x2, y2 = seg
        # Plotting formatted as (col_coords, row_coords)
        ax.plot([y1, y2], [x1, x2], color='red', linewidth=2, marker='o', markersize=2)

    # Decoration to prove the .5 alignment
    # Set ticks to show integer centers
    ax.set_xticks(np.arange(maze.shape[1]))
    ax.set_yticks(np.arange(maze.shape[0]))
    ax.grid(color='blue', linestyle='--', linewidth=0.5, alpha=0.3)
    
    ax.set_title("Maze Walls (Red Lines at indices +/- 0.5)")
    ax.set_xlabel("Y Coordinate (Column Index)")
    ax.set_ylabel("X Coordinate (Row Index)")
    
    # Invert Y axis to match matrix layout (if not already handled by imshow/origin)
    # Imshow handles it, but plotting lines over it requires care. 
    # With origin='upper', increasing Row index goes DOWN.
    
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
wall_segments = get_maze_segments(maze)

print(f"Found {len(wall_segments)} wall segments.")
print(f"Sample Segment (x1, y1, x2, y2): {wall_segments[0]}")
print(sorted(wall_segments, key=lambda x: x[0]))

# 3. Visualize
visualize_results(maze, wall_segments)