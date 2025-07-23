from diffusion_policy.common.replay_buffer import ReplayBuffer
import numpy as np
import zarr

horizon = 8
obs_key='state_positions'
state_key='goal'
action_key='action'

zarr_path = 'demo_data/box_delivery_expert_demo.zarr'
zarr_path_padded = zarr_path.replace('.zarr', '_padded.zarr')

replay_buffer = ReplayBuffer.copy_from_path(
    zarr_path, keys=[obs_key, state_key, action_key])

replay_buffer_padded = ReplayBuffer.create_from_path(
    zarr_path_padded, mode='a')


# Pad all episodes in the replay buffer to the specified horizon.
for ep_id in range(replay_buffer.n_episodes):
    ep_len = replay_buffer.episode_lengths[ep_id]
    pad_len = max(0, horizon - ep_len)
    if pad_len > 0:
        # Pad obs, action, and state
        padded_episode = {}
        for key in [obs_key, action_key, state_key]:
            ep_data = replay_buffer.get_episode(ep_id)[key]
            pad_data = np.repeat(ep_data[-1:], pad_len, axis=0)
            padded_episode[key] = np.concatenate([ep_data, pad_data], axis=0)

        # Add padded episode
        replay_buffer_padded.add_episode(padded_episode)
    else:
        # No padding needed, just copy the episode
        replay_buffer_padded.add_episode(
            replay_buffer.get_episode(ep_id))