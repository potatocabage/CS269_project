# Maze Guidance for Diffusion Policies

This document details the implementation of maze-based guidance costs for diffusion policies, designed to prevent wall collisions during trajectory generation. The system integrates geometric cost functions into the diffusion sampling process, allowing the policy to avoid obstacles at inference time without retraining.

## 1. Maze Cost Functions

Located in `itps/common/policies/maze_cost.py`, these functions calculate costs and gradients based on the relationship between the predicted trajectory and the maze walls.

### Wall Representation
The maze walls are converted into a set of line segments $(x_1, y_1, x_2, y_2)$. These segments are derived automatically from the grid-based maze map using `get_maze_segments`.

### Cost Types

#### A. Barrier Cost (`calculate_maze_barrier_cost`)
Calculates a soft barrier potential field around walls.
- **Logic**: Computes the Euclidean distance from each point in the trajectory to the nearest wall segment.
- **Cost**: Uses a sigmoid-like function (or log barrier) to penalize points that get too close to walls.
- **Gradient**: Pushes points away from walls perpendicular to the wall surface.

#### B. Intersection Cost (`calculate_trajectory_intersection_cost`)
Penalizes trajectory segments that actually cross a wall.
- **Logic**: Checks for geometric intersection between every step vector $S_t \rightarrow S_{t+1}$ and every wall segment.
- **Cost**: If an intersection occurs at fraction $t$ of the step ($0 \le t \le 1$), the cost penalizes the remaining fraction $(1-t)^2$.
- **Gradient**: Effectively pulls the step back so it ends before the wall.

#### C. Virtual Tail Cost (`calculate_virtual_tail_cost`)
A more advanced guidance term that attempts to "fix" the trajectory geometry upon collision.
- **Logic**:
    1. Identifies the *earliest* collision point in the trajectory.
    2. Constructs a "Virtual Tail" by taking the trajectory after the collision and rotating/translating it to align parallel to the wall and shift it back into safe space.
    3. Calculates MSE between the actual trajectory tail and this "Virtual Tail".
- **Gradient**: Rotates and translates the entire future trajectory to match the safe virtual target, preserving the shape of the motion while correcting its direction.

#### D. Combo Cost
Combines Intersection and Virtual Tail costs for robust avoidance.

## 2. Integration with Diffusion Policy

The costs are integrated into the `DiffusionPolicy` class in `itps/common/policies/diffusion/modeling_diffusion.py`.

### Sampling Process (`DiffusionModel.conditional_sample`)
During the DDIM/DDPM denoising loop:
1. The model predicts the noise/clean sample.
2. A gradient is calculated from the selected maze cost function.
3. This gradient is subtracted from the model output (or intermediate sample) to steer the generation.

### Clean vs. Noisy Guidance
The implementation supports two modes of guidance via the `apply_cost_on_clean` flag:
1. **Noisy Guidance**: Calculates gradients on the noisy sample $x_t$. This is the standard classifier guidance approach but can be unstable for geometric constraints because $x_t$ is noisy.
2. **Clean Guidance (DPS style)**: Estimates $\hat{x}_0$ (clean trajectory) from $x_t$, calculates the cost on $\hat{x}_0$, and backpropagates the gradient to $x_t$. This provides much cleaner geometric gradients (walls are sharp in $x_0$ space).

## 3. Visualization Tool (`test_denoising_viz.py`)

A Pygame-based visualization script is provided to analyze the effect of these guidance costs in real-time.

### Features
- **3-Panel Display**:
    - **Left**: Baseline trajectory (MCW=0).
    - **Middle**: Guided trajectory (MCW=X). Shows the noisy step (grey), estimated clean step (green), and the gradient-adjusted clean step (red).
    - **Right**: Overlay of Baseline (Blue) and Guided (Red) for direct comparison.
- **Step-by-Step Replay**: visualizing the denoising process step-by-step.
- **Interactive Mode**: Click on the map to set the start position.

### Usage

**Basic Command:**
```bash
python itps/test_denoising_viz.py -p diffusion -mcw 10.0 -cg
```

**Arguments:**
- `-p, --policy`: Policy type (`diffusion`).
- `-mcw, --maze_cost_weight`: Weight for the maze cost (e.g., `10.0` or `100.0`).
- `-cg, --clean_guidance`: Enable Clean Guidance (DPS). Highly recommended for wall avoidance.
- `-i, --interactive`: Run in interactive mode (click to start).
- `-ni`: Number of inference steps (default 10).
- `-sv`: Save video filename (e.g., `test.mp4`).

**Example: Interactive Test with Clean Guidance**
```bash
python itps/test_denoising_viz.py -p diffusion -mcw 50.0 -cg -i
```

**Example: Save Video of Specific Coordinate**
```bash
python itps/test_denoising_viz.py -p diffusion -mcw 50.0 -cg -x 150 -y 200 -sv output.mp4
```

## 4. Key Implementation Details

- **Maze Segments**: `get_maze_segments` in `maze_cost.py` automatically merges contiguous wall blocks into long segments to reduce the number of distance checks (vectorization optimization).
- **Normalization**: The diffusion model operates in normalized space. The `maze_wall_cost_gradient` method handles the unnormalization of the trajectory to calculate physical distance costs, then propagates gradients back to the normalized space.
- **Gradient Clamping**: Gradients are clamped (e.g., in `clamped_barrier_cost_gradient`) to prevent exploding gradients from destabilizing the diffusion process.

