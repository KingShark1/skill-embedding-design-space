import torch
from isaacgym.torch_utils import *
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from utils import torch_utils
from learning import ase_network_builder

from learning import amp_agent

class FLOWAgent(amp_agent.AMPAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        return
    
    def init_tensors(self):
        super().init_tensors()

        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['flow_latents'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                dtype=torch.float32, device=self.ppo_device)

        self._flow_latents = torch.zeros((batch_shape[-1], self._latent_dim), dtype=torch.float32,
                                        device=self.ppo_device)

        self.tensor_list += ['flow_latents']

        self._latent_reset_steps = torch.zeros(batch_shape[-1], dtype=torch.int32, device=self.ppo_device)
        num_envs = self.vec_env.env.task.num_envs
        env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.ppo_device)
        self._reset_latent_step_count(env_ids)

        return
