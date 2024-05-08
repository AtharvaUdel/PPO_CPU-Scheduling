import torch
from torch.distributions import MultivariateNormal


from network import FeedForwardNN

class PPO:
    def __init__(self, env, obs_enc_dim) -> None:
        # Environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.obs_enc_dim = obs_enc_dim
        self.act_dim = env.action_space.shape[0]

        # ALG STEP 1
        # Actor and critic networks
        self.actor = FeedForwardNN(self.obs_enc_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_enc_dim, self.act_dim)

        # Observations encoder
        self.obs_enc = FeedForwardNN(self.obs_dim, self.obs_enc_dim)

        # Hyperparameters
        self._init_hyperparameters()

        # Multivariate Normal Stats
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def _init_hyperparameters(self):
        # Default hyperparameter values - NEED TO CHANGE
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95 # reward decay

    def learn(self, n_steps):
        n = 0 # number of steps taken
        while n < n_steps: # ALG STEP 2
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
        pass

    def rollout(self):
        # batch data
        batch_obs = []                  # batch observations
        batch_acts = []                 # batch actions
        batch_log_probs = []            # log probs of each action
        batch_rews = []                 # batch rewards
        batch_rtgs = []                 # batch rewards to go
        batch_lens = []                 # episodic lengths in batch

        # Number of timesteps run this batch
        t = 0

        while t < self.timesteps_per_batch: 
            # Rewards this episode
            ep_rews = []

            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps run this batch
                t += 1

                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                # Collect reward, action, and log_prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
            
                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        
        # reshape as tensors 
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # ALG STEP 4 - Compute rewards-to-go
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        # encode the observations and query the actor for mean action
        enc_obs = self.obs_enc(obs)
        mean = self.actor(enc_obs)

        # create multivariate normal distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # sample action from distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # return detached action and log prob
        return action.detach().numpy(), log_prob.detach().numpy()
    
    def compute_rtgs(self, batch_rews):
        # reawards-to-go per episode to return
        batch_rtgs = []

        # iterate through episode backwards
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # running reward
            for rew in reversed(ep_rews):
                discounted_reward = rew + (discounted_reward * self.gamma)
                batch_rtgs.insert(0, discounted_reward)

        # convert to tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

