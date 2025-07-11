import torch
import numpy as np
from typing import Dict, Any
import collections

class BoxDeliveryRunner:
    def __init__(self, 
                 env,
                 n_obs_steps: int = 8,
                 n_action_steps: int = 8,
                 max_steps: int = 1000,
                 device: str = 'cuda'):
        self.env = env
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.device = device
        
        # Initialize observation buffer
        self.obs_buffer = collections.deque(maxlen=n_obs_steps)

    def run(self, policy):
        """Run episodes with the given policy"""
        device = self.device
        
        # Reset environment
        obs, info = self.env.reset()
        
        # Get low-dimensional observation
        obs_vert, obs_pos = self.env.generate_observation_low_dim()
        
        # Initialize observation buffer with current observation
        for _ in range(self.n_obs_steps):
            self.obs_buffer.append(obs_pos)
        
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done and step_count < self.max_steps:
            # Create observation dictionary for diffusion policy
            obs_history = np.array(list(self.obs_buffer))  # Shape: (n_obs_steps, obs_dim)
            obs_dict = {
                'obs': torch.from_numpy(obs_history[np.newaxis, ...]).float().to(device)  # Add batch dim
            }
            
            # Get action from policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
            
            # Extract action sequence and take first action
            action_sequence = action_dict['action'].cpu().numpy()
            # action = action_sequence[0, 0]  # First action from first batch
            
            # Convert continuous action to discrete pixel index
            # action = int(np.clip(action, 0, self.env.local_map_pixel_width * self.env.local_map_pixel_width - 1))
            
            # Step environment
            print(action_sequence)
            obs, reward, terminated, truncated, info = self.env.step(action_sequence)
            
            # Update observation buffer
            obs_vert, obs_pos = self.env.generate_observation_low_dim()
            self.obs_buffer.append(obs_pos)
            
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
        
        return {
            'episode_reward': episode_reward,
            'episode_length': step_count,
            'success': info.get('cumulative_boxes', 0) == self.env.num_boxes,
            'boxes_delivered': info.get('cumulative_boxes', 0)
        }