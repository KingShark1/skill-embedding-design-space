from learning import amp_models

class ModelFLOWContinuous(amp_models.ModelAMPContinuous):
    def __init__(self, network):
        super().__init__(network)
        return
    
    def build(self, config):
        net = self.network_builder.build('flow', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelFLOWContinuous.Network(net)
    
    class Network(amp_models.ModelAMPContinuous.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)(self, a2c_network)
            return
        
        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            result = super().forward(input_dict)

            """
            This would get changed, in FLOW models
            """
            if (is_train):
                amp_obs = input_dict['amp_obs']
                enc_pred = self.a2c_network.eval_env(amp_obs)
                result["enc_pred"] = enc_pred
            
            return result