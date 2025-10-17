# my_custom_dataset.py
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer

class CoverageDataset(BaseLowdimDataset):
    def __init__(self, zarr_path, horizon=32, pad_before=1, pad_after=7, 
                 obs_key='observations', action_key='actions', 
                 seed=42, val_ratio=0.1):
        super().__init__()
        
        # Load your data
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, action_key])
        
        # Create train/val split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio, seed=seed)
        
        # Setup sequence sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=~val_mask)
        
        # Store keys
        self.obs_key = obs_key
        self.action_key = action_key
        self.train_mask = ~val_mask
        
    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        return {
            'obs': sample[self.obs_key],      # Shape: [horizon, obs_dim]
            'action': sample[self.action_key]  # Shape: [horizon, action_dim]
        }
    
    def get_normalizer(self, mode='limits', **kwargs):
        data = {'obs': self.replay_buffer[self.obs_key],
                'action': self.replay_buffer[self.action_key]}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode)
        return normalizer