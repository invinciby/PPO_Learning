from network import FeedForwardNN
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import torch
import numpy as np
from torch import nn 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gymnasium as gym

import time

class PPO:
    def __init__(self, policy_class, env, **hyperparameters):
        assert(type(env.observation_space) == gym.spaces.Box)
        assert(type(env.action_space) == gym.spaces.Box)
        
        # 初始化超参数，包括每个批次的时间步数、每个episode的最大时间步数和折扣因子。
        self._init_hyperparameters(hyperparameters) 
        
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        # 是两个前馈神经网络，分别用于策略网络和值函数网络。
        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        # 定义了动作的高斯分布的协方差矩阵。
        self.cov_var = torch.full(size=(self.act_dim,), fill_value = 0.5)
        self.cov_mat = torch.diag(self.cov_var)
        
        self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}
        
        
    def _init_hyperparameters(self, hyperparameters):
        # 初始化超参数
        self.timesteps_per_batch = 4096
        # 每个批次的步数
        self.max_timesteps_per_episode = 1000
        # 每个episode的最大步数
        self.gamma = 0.99
        # 折扣因子
        self.n_updates_per_iteration = 5
        # 每次迭代的更新次数
        self.clip = 0.5
        # 剪辑因子
        self.lr = 0.005
        
        # 学习率
        self.render = True
        # 是否渲染
        self.render_every_i = 5
        # 每隔多少次渲染一次
        self.save_freq = 5
        # 保存频率
        self.seed = None
        
        # 随机种子
        # 遍历传入的超参数，并赋值给self
        for param,val in hyperparameters.items():
            exec('self.'+param+' = '+str(val))
            
        # 如果随机种子不为空，则设置随机种子
        if self.seed != None:
            assert(type(self.seed) == int)
            
            torch.manual_seed(self.seed) 
            np.random.seed(self.seed)
            print(f"Successfully set seed to {self.seed}.")       
    
    # 收集数据:
    ## 收集观察结果、动作、这些动作的对数概率、rewards, rewards-to-go 以及每个episode的长度。
    def rollout(self):
        # 执行一次rollout，即在一个episode中，通过调用get_action方法获取动作，并执行动作以获取新的观测和奖励。
        batch_obs = [] # 用于存储每个时间步的观测。
        batch_acts  =[] # 用于存储每个时间步的动作。
        batch_log_probs = [] # 用于存储每个时间步的动作的对数概率。
        batch_rews = [] # 用于存储每个时间步的奖励。
        batch_rtgs =[] # 用于存储每个时间步的奖励的折扣累积和。
        batch_lens = [] # 用于存储每个episode的长度。
        
        batch_vals = []
        batch_dones = []
        
        ep_rews = []
        ep_vals = []
        ep_dones = []
        t = 0
        
        while t<self.timesteps_per_batch: # 在每个批次中，执行多个episode。
            ep_rews = [] # 用于存储当前episode中的所有奖励。
            ep_vals = []
            ep_dones = []
            
            obs,_ = self.env.reset() # 重置环境，获取初始观测。
         
                
            done = False
            
            # 在每个episode中，通过调用get_action方法获取动作，并执行动作以获取新的观测和奖励。
            for ep_t in range(self.max_timesteps_per_episode): # 在每个episode中，执行多个时间步。
                if self.render:
                    self.env.render()
                ep_dones.append(done)
                
                t += 1
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                val = self.critic(obs)
                
                # action = self.env.action_space.sample()
                obs, rew, teminated, truncated, _ = self.env.step(action)
                done = teminated or truncated
          
                
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
        
                # if isinstance(obs, tuple):
                #     obs = obs[0]
            
                
                if done:
                    break
                
            batch_lens.append(ep_t+1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
            
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten()
        batch_rtgs = self.compute_rtgs(batch_rews)
        
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
        
    
    
    # learn方法用于训练模型，接受总时间步数作为参数。
    # 在每个时间步，调用rollout方法收集一批数据，然后使用这些数据更新策略网络和值函数网络。
    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0
        i_so_far = 0
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            
            
            
            t_so_far += sum(batch_lens)
            i_so_far +=1
            
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far
            
            if i_so_far % 10 == 0:  # 每10个批次显示一次进度
                print(f"Completed {t_so_far}/{total_timesteps} timesteps ({(t_so_far / total_timesteps) * 100:.2f}%)")
            
            V, _ = self.evaluate(batch_obs, batch_acts)
            
            A_k = batch_rtgs - V.detach()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            
            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratio = torch.exp(curr_log_probs - batch_log_probs)
                
                surr1 = ratio * A_k
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * A_k
                
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()
                
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                
                self.logger['actor_losses'].append(actor_loss.detach())
                
            self._log_summary()
                
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './N_ppo_actor.pth')
                torch.save(self.critic.state_dict(), './N_ppo_critic.pth')
    
    # 根据当前观测获取动作
    def get_action(self, obs):
        # if isinstance(obs, tuple):
        #     obs = obs[0]
        mean = self.actor(obs)
        
        # 使用策略网络的前向传播计算动作的均值，然后从这个均值的高斯分布中采样动作
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        
        # 计算动作的对数概率
        log_prob = dist.log_prob(action)
        
        return action.detach().numpy(), log_prob.detach()
    
    
    
    # 奖励函数：获取奖励
    def compute_rtgs(self, batch_rews):
        batch_rgts =[]
        
        # 对于每个episode的奖励，使用折扣因子计算奖励到目标，并将其插入到列表的开头。
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rgts.insert(0, discounted_reward)
                
        batch_rgts = torch.tensor(batch_rgts, dtype=torch.float)
        return batch_rgts
    
    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        
        
        return V, log_probs
    

    def _log_summary(self):
        """
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
	    """
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

		# Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

		# Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []