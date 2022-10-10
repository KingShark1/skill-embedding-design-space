from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

from learning import amp_agent

import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0

class FLOWBuilder(network_builder.A2CBuilder):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		return
		
	class Network(network_builder.A2CBuilder):
		pass

	def build(self, name, **kwargs):
		net = FLOWBuilder.Network(self.params, **kwargs)
		return net
