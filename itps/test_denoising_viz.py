import sys
import os
import argparse
import time
import json
import numpy as np
import pygame
import torch
import einops
from pathlib import Path
import matplotlib.pyplot as plt

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from common.utils.utils import seeded_context
from common.policies.utils import get_device_from_parameters
from interact_maze2d import MazeEnv, UnconditionalMaze

class StepCollector:
    def __init__(self):
        self.steps = []
    
    def update_screen(self, xy_pred, keep_drawing=True, **kwargs):
        # xy_pred is (B, T, 2) or (T, 2)
        # We store CPU numpy copy
        if isinstance(xy_pred, torch.Tensor):
            xy_pred = xy_pred.cpu().numpy()
        self.steps.append(xy_pred.copy())

class DenoisingVizEnv(MazeEnv):
    def __init__(self, policy, args):
        # We need to initialize the parent but we want a larger window.
        # MazeEnv sets window_size in __init__. We must let it run, then modify.
        super().__init__()
        
        self.policy = policy
        self.args = args
        self.mcw_arg = args.maze_cost_weight
        
        # Setup display for 3 panels
        self.single_w, self.single_h = self.gui_size
        self.total_w = self.single_w * 3
        self.total_h = self.single_h
        
        self.window_size = (self.total_w, self.total_h)
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Denoising Visualization: MCW=0 vs MCW={}".format(self.mcw_arg))
        
        # Pre-render maze background to a surface
        self.maze_bg_surface = pygame.Surface(self.gui_size)
        self.draw_maze_background_to_surface(self.maze_bg_surface)
        
        self.font = pygame.font.SysFont(None, 36)
        
        # Setup logging
        self.log_file = open("denoising_log.txt", "w")
        
        # Agent state
        self.agent_history_xy = []
        self.agent_gui_pos = np.array([0, 0])
        self.agent_color = self.RED
        
        # Override batch size to 1 for this visualization
        self.batch_size = 1
        
    def draw_maze_background_to_surface(self, surface):
        maze_img = pygame.surfarray.make_surface(255 - np.swapaxes(np.repeat(self.maze[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8), 0, 1))
        maze_img = pygame.transform.scale(maze_img, self.gui_size)
        surface.blit(maze_img, (0, 0))

    def draw_trajectory_on_surface(self, surface, xy_pred, color, label=None):
        # xy_pred: (B, T, 2)
        time_colors = self.generate_time_color_map(xy_pred.shape[1])
        
        for idx, pred in enumerate(xy_pred):
            for step_idx in range(len(pred) - 1):
                # Use solid color if provided, else time color (blended)
                draw_color = color
                
                # Convert to GUI coordinates
                start_pos = self.xy2gui(pred[step_idx])
                end_pos = self.xy2gui(pred[step_idx + 1])
                
                pygame.draw.circle(surface, draw_color, start_pos, 4)
                # Optional: draw lines for better visibility
                pygame.draw.line(surface, draw_color, start_pos, end_pos, 2)
                
        # Draw Agent
        pygame.draw.circle(surface, self.agent_color, (int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])), 10)
        
        if label:
            text_surf = self.font.render(label, True, (0, 0, 0))
            surface.blit(text_surf, (10, 10))

    def run_comparison(self, start_gui_pos):
        self.update_agent_pos(start_gui_pos)
        
        print(f"Running inference from {start_gui_pos} (GUI) / {self.gui2xy(start_gui_pos)} (XY)")
        
        # 1. Run MCW = 0
        print("Collecting MCW=0 trajectory...")
        self.set_policy_mcw(0.0)
        collector0 = StepCollector()
        self.infer_target(visualizer=collector0)
        
        # 2. Run MCW = Arg
        print(f"Collecting MCW={self.mcw_arg} trajectory...")
        self.set_policy_mcw(self.mcw_arg)
        collector1 = StepCollector()
        self.infer_target(visualizer=collector1)
        
        # 3. Replay and Log
        self.replay_and_log(collector0.steps, collector1.steps)
        
    def set_policy_mcw(self, weight):
        if hasattr(self.policy, 'diffusion'):
             device = get_device_from_parameters(self.policy)
             maze_tensor = torch.from_numpy(self.maze.astype(float)).float().to(device)
             self.policy.diffusion.set_maze(maze_tensor, cost_weight=weight)
             
    def infer_target(self, visualizer=None):
        # Helper to run inference using current agent state
        agent_hist_xy = self.agent_history_xy[-1]
        agent_hist_xy = np.array(agent_hist_xy).reshape(1, 2)
        
        n_obs_steps = self.policy.config.n_obs_steps
        if agent_hist_xy.shape[0] < n_obs_steps:
             # Repeat to fill history
             agent_hist_xy = np.repeat(agent_hist_xy, n_obs_steps, axis=0)
        
        device = get_device_from_parameters(self.policy)
        device_type = "cuda" if device.type == "cuda" else "cpu"
        
        # Prepare observation
        obs = einops.repeat(
                torch.from_numpy(agent_hist_xy).float().to(device), "t d -> b t d", b=self.batch_size
            )
        
        obs_batch = {
            "observation.state": obs,
            "observation.environment_state": obs
        }
        
        # Use fixed seed 0 for consistent comparison between runs
        with torch.autocast(device_type=device_type), seeded_context(0):
            # policy.run_inference calls diffusion.generate_actions which calls conditional_sample
            # conditional_sample calls visualizer.update_screen
            _ = self.policy.run_inference(obs_batch, visualizer=visualizer)

    def replay_and_log(self, steps0, steps1):
        self.log_file.write(f"\n--- New Trajectory Run at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        num_steps = max(len(steps0), len(steps1))
        print(f"Replaying {num_steps} denoising steps...")
        
        for i in range(num_steps):
            s0 = steps0[i] if i < len(steps0) else steps0[-1]
            s1 = steps1[i] if i < len(steps1) else steps1[-1]
            
            # Log
            self.log_step(i, s0, s1)
            
            # Render
            self.screen.fill(self.WHITE)
            
            surf0 = self.maze_bg_surface.copy()
            surf1 = self.maze_bg_surface.copy()
            surf2 = self.maze_bg_surface.copy()
            
            # Draw Panel 1: MCW=0 (Blue)
            self.draw_trajectory_on_surface(surf0, s0, (0, 0, 255), f"MCW=0.0 (Step {i})")
            
            # Draw Panel 2: MCW=Arg (Red)
            self.draw_trajectory_on_surface(surf1, s1, (255, 0, 0), f"MCW={self.mcw_arg} (Step {i})")
            
            # Draw Panel 3: Overlay
            # Draw s0 faint/Blue
            self.draw_trajectory_on_surface(surf2, s0, (100, 100, 255))
            # Draw s1 Red
            self.draw_trajectory_on_surface(surf2, s1, (255, 0, 0), "Overlay")
            
            # Blit to screen
            self.screen.blit(surf0, (0, 0))
            self.screen.blit(surf1, (self.single_w, 0))
            self.screen.blit(surf2, (self.single_w * 2, 0))
            
            # Draw separators
            pygame.draw.line(self.screen, (0,0,0), (self.single_w, 0), (self.single_w, self.total_h), 2)
            pygame.draw.line(self.screen, (0,0,0), (self.single_w * 2, 0), (self.single_w * 2, self.total_h), 2)
            
            pygame.display.flip()
            time.sleep(0.5) # Pause to see step
            
            # Handle quit during replay
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

    def log_step(self, step_idx, traj0, traj1):
        # Taking the first trajectory from the batch for logging
        t0 = traj0[0] 
        t1 = traj1[0]
        
        self.log_file.write(f"\nStep {step_idx}:\n")
        
        self.log_file.write(f"  MCW=0.0:\n")
        self.log_file.write(self.format_traj(t0, indent=4))
        
        self.log_file.write(f"  MCW={self.mcw_arg}:\n")
        self.log_file.write(self.format_traj(t1, indent=4))
        self.log_file.flush()

    def format_traj(self, traj, indent=0):
        # readable text format
        s = ""
        prefix = " " * indent
        for idx, pt in enumerate(traj):
            s += f"{prefix}Pt {idx:2d}: ({pt[0]:.4f}, {pt[1]:.4f})\n"
        return s

    def update_agent_pos(self, new_agent_pos):
        self.agent_gui_pos = np.array(new_agent_pos)
        agent_xy_pos = self.gui2xy(self.agent_gui_pos)
        self.agent_history_xy.append(agent_xy_pos)
        self.agent_history_xy = self.agent_history_xy[-1:]

    def run_interactive(self):
        print("Interactive Mode: Click on the maze to set start position.")
        
        # Initial draw
        self.screen.fill(self.WHITE)
        self.screen.blit(self.maze_bg_surface, (0, 0))
        self.screen.blit(self.maze_bg_surface, (self.single_w, 0))
        self.screen.blit(self.maze_bg_surface, (self.single_w * 2, 0))
        
        # Draw separators
        pygame.draw.line(self.screen, (0,0,0), (self.single_w, 0), (self.single_w, self.total_h), 2)
        pygame.draw.line(self.screen, (0,0,0), (self.single_w * 2, 0), (self.single_w * 2, self.total_h), 2)
        
        txt = self.font.render("Click on LEFT panel to set start pos", True, (0,0,0))
        self.screen.blit(txt, (50, 50))
        pygame.display.flip()
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    # Check if click is in first panel
                    if pos[0] < self.single_w:
                        self.run_comparison(np.array(pos))
                        # After run, clear and wait again
                        print("Ready for next click...")
            self.clock.tick(30)
            
    def run_non_interactive(self, start_x, start_y):
        start_pos = np.array([start_x, start_y])
        self.run_comparison(start_pos)
        # Keep window open? Or quit?
        # Usually test scripts might just finish. 
        # But visualization usually implies waiting to see it.
        # We'll wait for quit.
        print("Finished. Close window to exit.")
        while self.running:
             for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
             self.clock.tick(10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--policy', required=True, type=str, help="Policy name (diffusion/dp)")
    parser.add_argument('-c', "--checkpoint", type=str, default="weights_dp/pretrained_model", help="Path to the checkpoint")
    parser.add_argument('-mcw', '--maze_cost_weight', type=float, required=True, help="Weight for maze wall cost function")
    
    # Mode selection
    parser.add_argument('-i', '--interactive', action='store_true', help="Interactive mode (click to start)")
    
    # Start pos for non-interactive
    parser.add_argument('-x', '--start_x', type=float, default=100.0, help="Start X (GUI coords)")
    parser.add_argument('-y', '--start_y', type=float, default=100.0, help="Start Y (GUI coords)")
    
    # Inference steps
    parser.add_argument('-ni', '--num_inference_steps', type=int, default=10, help="Number of denoising steps (default: 10)")

    args = parser.parse_args()
    
    # Load Policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.policy in ["diffusion", "dp"]:
        policy = DiffusionPolicy.from_pretrained(Path(args.checkpoint), alignment_strategy='post-hoc')
        policy.config.noise_scheduler_type = "DDIM"
        policy.diffusion.num_inference_steps = args.num_inference_steps
        policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
        policy.to(device)
        policy.eval()
    else:
        raise ValueError("Only 'diffusion' or 'dp' policy is supported for this test.")
        
    # Init Env
    env = DenoisingVizEnv(policy, args)
    
    try:
        if args.interactive:
            env.run_interactive()
        else:
            env.run_non_interactive(args.start_x, args.start_y)
    finally:
        env.log_file.close()
        pygame.quit()

if __name__ == "__main__":
    main()

