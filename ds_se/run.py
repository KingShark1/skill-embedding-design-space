# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, experiment, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

import numpy as np
import copy
import torch

from learning import amp_agent
from learning import amp_players
from learning import amp_models
from learning import amp_network_builder

from learning import ase_agent
from learning import ase_players
from learning import ase_models
from learning import ase_network_builder

args = None
cfg = None
cfg_train = None

# TODO: Add add Diffusion, Flow, CPC models here with the format underlined here.
def build_alg_runner(algo_observer):
	runner = Runner(algo_observer)
	runner.algo_factory.register_builder('amp', lambda **kwargs : amp_agent.AMPAgent(**kwargs))
	runner.player_factory.register_builder('amp', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
	runner.model_builder.model_factory.register_builder('amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))  
	runner.model_builder.network_factory.register_builder('amp', lambda **kwargs : amp_network_builder.AMPBuilder())
	
	runner.algo_factory.register_builder('ase', lambda **kwargs : ase_agent.ASEAgent(**kwargs))
	runner.player_factory.register_builder('ase', lambda **kwargs : ase_players.ASEPlayer(**kwargs))
	runner.model_builder.model_factory.register_builder('ase', lambda network, **kwargs : ase_models.ModelASEContinuous(network))  
	runner.model_builder.network_factory.register_builder('ase', lambda **kwargs : ase_network_builder.ASEBuilder())
	
	return runner

def main():
	global args
	global cfg
	global cfg_train

	set_np_formatting()
	args = get_args()
	cfg, cfg_train, logdir = load_cfg(args)

	cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

	if args.horovod:
			cfg_train['params']['config']['multi_gpu'] = args.horovod

	if args.horizon_length != -1:
			cfg_train['params']['config']['horizon_length'] = args.horizon_length

	if args.minibatch_size != -1:
			cfg_train['params']['config']['minibatch_size'] = args.minibatch_size
			
	if args.motion_file:
			cfg['env']['motion_file'] = args.motion_file
	
	# Create default directories for weights and statistics
	cfg_train['params']['config']['train_dir'] = args.output_path
	
	vargs = vars(args)

	algo_observer = RLGPUAlgoObserver()

	runner = build_alg_runner(algo_observer)
	runner.load(cfg_train)
	runner.reset()
	runner.run(vargs)

	return

if __name__=="__main__":
	main()