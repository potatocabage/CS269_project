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

import imageio

class StepCollector:
    def __init__(self):
        self.steps = []
    
    def update_screen(self, xy_pred, keep_drawing=True, label=None, **kwargs):
        # xy_pred is (B, T, 2) or (T, 2)
        # We store CPU numpy copy
        if isinstance(xy_pred, torch.Tensor):
            xy_pred = xy_pred.cpu().numpy()
        self.steps.append({'data': xy_pred.copy(), 'label': label})

class DenoisingVizEnv(MazeEnv):
    def __init__(self, policy, args):
        # We need to initialize the parent but we want a larger window.
        # MazeEnv sets window_size in __init__. We must let it run, then modify.
        super().__init__()
        
        # Allow optional override of GUI panel size via CLI args
        # (keep aspect ratio if only width is provided)
        if getattr(args, 'gui_w', None) is not None:
            gw = int(args.gui_w)
            if getattr(args, 'gui_h', None) is None:
                # preserve aspect ratio
                aspect = float(self.gui_size[1]) / float(self.gui_size[0])
                gh = int(gw * aspect)
            else:
                gh = int(args.gui_h)
            self.gui_size = (gw, gh)

        self.policy = policy
        self.args = args
        self.mcw_arg = args.maze_cost_weight
        
        # Setup display for 3 panels and a legend area below
        self.single_w, self.single_h = self.gui_size
        self.total_w = self.single_w * 3
        # Legend area size (proportional to panel height)
        self.legend_h = int(max(64, self.single_h * 0.22))
        self.legend_margin = int(max(8, self.single_h * 0.04))
        self.total_h = self.single_h + self.legend_h + self.legend_margin

        self.window_size = (self.total_w, self.total_h)
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Denoising Visualization: MCW=0 vs MCW={}".format(self.mcw_arg))
        
        # Pre-render maze background to a surface
        self.maze_bg_surface = pygame.Surface(self.gui_size)
        self.draw_maze_background_to_surface(self.maze_bg_surface)
        
        # Scale fonts relative to GUI panel height so labels and legend fit small windows
        base_h = max(100, self.gui_size[1])
        main_font_size = max(12, int(base_h * 0.08))
        title_font_size = max(12, int(base_h * 0.06))
        legend_font_size = max(10, int(base_h * 0.045))
        label_font_size = max(10, int(base_h * 0.05))

        self.font = pygame.font.SysFont(None, main_font_size)
        self.title_font = pygame.font.SysFont(None, title_font_size)
        self.legend_font = pygame.font.SysFont(None, legend_font_size)
        self.label_font = pygame.font.SysFont(None, label_font_size)
        
        # Setup logging
        self.log_file = open("denoising_log.log", "w")
        
        # Agent state
        self.agent_history_xy = []
        self.agent_gui_pos = np.array([0, 0])
        self.agent_color = self.RED
        self.frames = []
        
        # Override batch size to 1 for this visualization
        self.batch_size = 1
        
    def draw_maze_background_to_surface(self, surface):
        maze_img = pygame.surfarray.make_surface(255 - np.swapaxes(np.repeat(self.maze[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8), 0, 1))
        maze_img = pygame.transform.scale(maze_img, self.gui_size)
        surface.blit(maze_img, (0, 0))

    def draw_trajectory_on_surface(self, surface, xy_pred, color, label=None, width=2):
        # xy_pred: (B, T, 2)
        time_colors = self.generate_time_color_map(xy_pred.shape[1])
        
        for idx, pred in enumerate(xy_pred):
            for step_idx in range(len(pred) - 1):
                # Use solid color if provided, else time color (blended)
                draw_color = color
                
                # Convert to GUI coordinates
                start_pos = self.xy2gui(pred[step_idx])
                end_pos = self.xy2gui(pred[step_idx + 1])
                
                # Draw lines for visibility
                pygame.draw.line(surface, draw_color, start_pos, end_pos, width)
                # Draw small circle at joints
                pygame.draw.circle(surface, draw_color, start_pos, width+1)
                
        # Draw Agent
        pygame.draw.circle(surface, self.agent_color, (int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])), 10)
        
        if label:
            text_surf = self.label_font.render(label, True, (0, 0, 0))
            surface.blit(text_surf, (6, 6))

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
             self.policy.diffusion.set_maze(maze_tensor, cost_weight=weight, apply_on_clean=self.args.clean_guidance)
             
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

    def draw_legend_box(self, surface, x, y):
        # Draw a legend for the trajectory types
        font = self.legend_font
        legend_items = [
            ("Noisy", (200, 200, 200)), # Grey
            ("Clean Est", (0, 255, 0)), # Green
            ("Clean+Grad", (255, 0, 0)), # Red
            ("Baseline", (0, 0, 255)) # Blue (for left panel)
        ]
        # Size legend relative to single panel size
        box_w = int(self.single_w * 0.48)
        box_h = int(self.single_h * 0.22)
        box_w = max(120, box_w)
        box_h = max(80, box_h)

        pygame.draw.rect(surface, (255, 255, 255), (x, y, box_w, box_h))
        pygame.draw.rect(surface, (0, 0, 0), (x, y, box_w, box_h), 1)

        padding_x = int(box_w * 0.06)
        padding_y = int(box_h * 0.12)
        line_y_step = int((box_h - padding_y * 2) / len(legend_items))

        for i, (text, color) in enumerate(legend_items):
            item_y = y + padding_y + i * line_y_step
            pygame.draw.line(surface, color, (x + padding_x, item_y + 6), (x + padding_x + 40, item_y + 6), 3)
            pygame.draw.circle(surface, color, (x + padding_x + 20, item_y + 6), max(3, int(box_h * 0.03)))
            txt_surf = font.render(text, True, (0, 0, 0))
            surface.blit(txt_surf, (x + padding_x + 50, item_y))

    def replay_and_log(self, steps0, steps1):
        self.log_file.write(f"\n--- New Trajectory Run at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        # Group steps by inference step (based on 'noisy' label or order)
        # Policy 0 (unguided) likely has: [noisy, clean, clean_grad(placeholder)] per step
        # Policy 1 (guided) likely has: [noisy, clean, clean_grad] per step
        
        # Helper to flatten if needed or handle grouping
        def group_steps(steps_list):
            grouped = []
            current_group = []
            for s in steps_list:
                # If we see 'noisy', it's the start of a new step (unless it's the very first one)
                if s['label'] == 'noisy' and current_group:
                    grouped.append(current_group)
                    current_group = []
                current_group.append(s)
            if current_group:
                grouped.append(current_group)
            return grouped

        grouped0 = group_steps(steps0)
        grouped1 = group_steps(steps1)
        
        num_steps = max(len(grouped0), len(grouped1))
        print(f"Replaying {num_steps} denoising steps...")
        
        for i in range(num_steps):
            g0 = grouped0[i] if i < len(grouped0) else grouped0[-1]
            g1 = grouped1[i] if i < len(grouped1) else grouped1[-1]
            
            # We want to show the sub-steps "one at a time"
            # Sub-steps are typically: 0:noisy, 1:clean, 2:clean_grad
            
            max_sub = max(len(g0), len(g1))
            
            for sub_idx in range(max_sub):
                # Get data for this sub-step
                # If one runs out of sub-steps, hold the last one
                s0_item = g0[sub_idx] if sub_idx < len(g0) else g0[-1]
                s1_item = g1[sub_idx] if sub_idx < len(g1) else g1[-1]
                
                s0 = s0_item['data']
                s1 = s1_item['data']
                label1 = s1_item['label'] # Use guided policy label for display
                
                # Log (only log once per full step, or maybe just skip logging detailed sub-steps to file)
                if sub_idx == 0:
                    self.log_step(i, s0, s1)
                
                # Render
                self.screen.fill(self.WHITE)
                
                surf0 = self.maze_bg_surface.copy()
                surf1 = self.maze_bg_surface.copy()
                surf2 = self.maze_bg_surface.copy()
                
                # Define standard colors
                COLOR_NOISY = (180, 180, 180) # Grey
                COLOR_CLEAN = (0, 200, 0)     # Green
                COLOR_GRAD = (255, 0, 0)      # Red
                COLOR_BASE = (0, 0, 255)      # Blue (for baseline panel)

                # Draw Panel 1: MCW=0 (Baseline)
                # Unguided usually just has noisy->clean. 
                # We'll show the "current" state in Blue.
                self.draw_trajectory_on_surface(surf0, s0, COLOR_BASE, f"MCW=0.0 (Step {i}.{sub_idx})")
                
                # Draw Panel 2: MCW=Arg (Guided) with specific coloring
                
                if label1 == 'noisy':
                    # Just show noisy state in Grey (or maybe slight Red tint to distinguish from baseline? Let's stick to legend)
                    self.draw_trajectory_on_surface(surf1, s1, COLOR_NOISY, f"MCW={self.mcw_arg} (Step {i}) - Noisy")
                elif label1 == 'clean':
                    # Overlay noisy (Grey, faint) + Clean (Green)
                    noisy_data = g1[0]['data']
                    self.draw_trajectory_on_surface(surf1, noisy_data, COLOR_NOISY, width=1)
                    self.draw_trajectory_on_surface(surf1, s1, COLOR_CLEAN, f"MCW={self.mcw_arg} (Step {i}) - Clean Est")
                elif label1 == 'clean_grad':
                     # Overlay noisy (Grey, faint) + Clean (Green, faint) + Clean+Grad (Red)
                    if len(g1) > 0:
                        noisy_data = g1[0]['data']
                        self.draw_trajectory_on_surface(surf1, noisy_data, COLOR_NOISY, width=1)
                    if len(g1) > 1:
                        clean_data = g1[1]['data']
                        self.draw_trajectory_on_surface(surf1, clean_data, COLOR_CLEAN, width=1) # Faint Green logic handles by thin line? Or need alpha? 
                        # Pygame doesn't do alpha lines easily on existing surface without special handling.
                        # We will trust width=1 and color distinction.
                    
                    self.draw_trajectory_on_surface(surf1, s1, COLOR_GRAD, f"MCW={self.mcw_arg} (Step {i}) - Clean+Grad", width=3)
                else:
                    # Fallback
                    self.draw_trajectory_on_surface(surf1, s1, COLOR_GRAD, f"MCW={self.mcw_arg}")

                
                # Draw Panel 3: Overlay
                # Draw s0 faint/Blue
                self.draw_trajectory_on_surface(surf2, s0, (100, 100, 255))
                # Draw s1 Red (Clean+Grad)
                self.draw_trajectory_on_surface(surf2, s1, COLOR_GRAD, "Overlay")
                
                # Blit to screen (panels occupy the top region)
                self.screen.blit(surf0, (0, 0))
                self.screen.blit(surf1, (self.single_w, 0))
                self.screen.blit(surf2, (self.single_w * 2, 0))

                # Draw separators only across the panel height (not legend area)
                pygame.draw.line(self.screen, (0,0,0), (self.single_w, 0), (self.single_w, self.single_h), 2)
                pygame.draw.line(self.screen, (0,0,0), (self.single_w * 2, 0), (self.single_w * 2, self.single_h), 2)

                # Draw Legend below the panels (centered under middle panel)
                legend_x = self.single_w + int(self.single_w * 0.05)
                legend_y = self.single_h + self.legend_margin
                self.draw_legend_box(self.screen, legend_x, legend_y)

                pygame.display.flip()
                
                if self.args.save_video and not self.args.interactive:
                    # Capture frame
                    frame = pygame.surfarray.array3d(self.screen)
                    frame = frame.transpose([1, 0, 2]) # (w, h, 3) -> (h, w, 3)
                    self.frames.append(frame)
                    
                time.sleep(0.5) # Pause to see sub-step
                
                # Handle quit during replay
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        return
            
            # Additional pause after full step
            time.sleep(0.2)

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

        # Draw separators only across panels
        pygame.draw.line(self.screen, (0,0,0), (self.single_w, 0), (self.single_w, self.single_h), 2)
        pygame.draw.line(self.screen, (0,0,0), (self.single_w * 2, 0), (self.single_w * 2, self.single_h), 2)

        # Instruction text (top area)
        txt = self.title_font.render("Click on LEFT panel to set start pos", True, (0,0,0))
        self.screen.blit(txt, (50, 50))

        # Draw an initial legend below panels so user sees it immediately
        legend_x = self.single_w + int(self.single_w * 0.05)
        legend_y = self.single_h + self.legend_margin
        self.draw_legend_box(self.screen, legend_x, legend_y)
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
            
    def save_video(self, filename):
        if not self.frames:
            print("No frames to save.")
            return
        print(f"Saving video to {filename} ({len(self.frames)} frames)...")
        try:
            imageio.mimsave(filename, self.frames, fps=2) # Slow FPS to match visualization pace
            print("Video saved successfully.")
        except Exception as e:
            print(f"Failed to save video: {e}")

    def run_non_interactive(self, start_x, start_y):
        start_pos = np.array([start_x, start_y])
        self.run_comparison(start_pos)
        
        if self.args.save_video:
             self.save_video(self.args.save_video)
             
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
    parser.add_argument('-cg', '--clean_guidance', action='store_true', help="Apply maze cost gradient on estimated clean sample (DPS style) instead of noisy sample")
    parser.add_argument('-sv', '--save_video', type=str, default=None, help="Filename to save video (mp4) in non-interactive mode")
    # Optional GUI size override (useful to make window smaller/larger)
    parser.add_argument('--gui_w', type=int, default=None, help="Optional GUI panel width in pixels (overrides default)")
    parser.add_argument('--gui_h', type=int, default=None, help="Optional GUI panel height in pixels (overrides default). If omitted, aspect ratio is preserved.")

    args = parser.parse_args()
    
    # Load Policy
    # Prefer CUDA, then Apple's Metal (MPS) if available, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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

