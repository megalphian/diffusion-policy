from typing import Dict
import torch
import numpy as np
import copy

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.normalizer import LinearNormalizer
from diffusion_policy.base_dataset import BaseLowdimDataset

class BoxDeliveryLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='state_positions',
            state_key='goal',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        # TODO: FIND OUT HOW obs_key IS USED
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        box_and_recept = sample[self.obs_key]
        goal = sample[self.state_key]
        # agent_pos = state[:,:2]
        # print("SHAPES INCOMING:")
        # print(box_and_recept.shape, goal.shape)
        obs = np.concatenate([
            box_and_recept.reshape(box_and_recept.shape[0], -1), 
            goal], axis=-1)
        # print("OBS SHAPE:", obs.shape)

        data = {
            'obs': obs, # T, D_o
            'action': sample[self.action_key], # T, D_a
        }
        return data

    def pad_sample(self, sample):
        # extract sequences
        obs = sample[self.obs_key]          # T, D_o
        action = sample[self.action_key]    # T, D_a
        state = sample[self.state_key]      # T, D_s

        T = obs.shape[0]
        pad_len = max(0, self.horizon - T)
        
        if pad_len > 0:
            # repeat last element pad_len times
            pad_obs = np.repeat(obs[-1:], pad_len, axis=0)
            pad_action = np.repeat(action[-1:], pad_len, axis=0)
            pad_state = np.repeat(state[-1:], pad_len, axis=0)

            # concatenate
            obs = np.concatenate([obs, pad_obs], axis=0)
            action = np.concatenate([action, pad_action], axis=0)
            state = np.concatenate([state, pad_state], axis=0)

            sample[self.obs_key] = obs
            sample[self.action_key] = action
            sample[self.state_key] = state

        return sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.pad_sample(self.sampler.sample_sequence(idx))

        # REMOVE: approximate robot position using the first action
        approx_robot_pos = sample[self.action_key][0]
        robot_pos_repeated = np.tile(approx_robot_pos, (sample[self.obs_key].shape[0], 1))
        sample[self.obs_key] = np.concatenate([robot_pos_repeated, sample[self.obs_key]], axis=1)

        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
