#This file contains the configuration for the SPO algorithm.

class Config:
    def __init__(self):
        #The environment name, with the tuned hyperparameters.
        self.env_name = "LunarLanderContinuous-v3"
        self.seed = 17
        self.total_timesteps = 800_000
        self.steps_per_batch = 4096
        self.update_epochs = 13
        self.num_minibatches = 64
        self.learning_rate = 0.0004526471705959077
        self.gamma = 0.9863912198781083
        self.gae_lambda = 0.9532163174156946
        self.epsilon = 0.2632559378198367
        self.entropy_coeff = 0.003111644391291919
        self.value_loss_coeff = 0.6360807434424953
        self.max_grad_norm = 1.324904070587997
        self.actor_hidden_dims = [256, 256, 256]
        self.critic_hidden_dims = [256, 256, 256]
        self.normalize_advantages = False
        self.eval_interval = 10
        self.save_interval = 20
        self.log_interval = 5
        self.target_reward = 220.0
        self.early_stopping_patience = 20
    
    #This function updates the configuration with the given dictionary.
    def update(self, new_config):
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_dict(self):
        return {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
