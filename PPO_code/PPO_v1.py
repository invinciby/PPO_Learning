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

class RunningStat:
    """用于在线计算均值和标准差"""
    def __init__(self, shape):
        # 初始化计数器n为0
        self.n = 0
        # 初始化均值mean为shape大小的零数组
        self.mean = np.zeros(shape)
        # 初始化方差S为shape大小的零数组
        self.S = np.zeros(shape)
        # 初始化标准差std为shape大小的零数组
        self.std = np.zeros(shape)
        # 初始化方差var为shape大小的零数组
        self.var = np.zeros(shape)

    def update(self, x):
        # 将输入x转换为数组
        x = np.array(x)
        # 计数器n加1
        self.n += 1
        # 如果计数器n为1，则将x赋值给均值mean
        if self.n == 1:
            self.mean = x
        else:
            # 保存旧的均值old_mean
            old_mean = self.mean.copy()
            # 更新均值mean
            self.mean = old_mean + (x - old_mean) / self.n
            # 更新方差S
            self.S = self.S + (x - old_mean) * (x - self.mean)
            # 如果计数器n大于1，则计算方差var，否则计算均值的平方
            self.var = self.S / (self.n - 1) if self.n > 1 else np.square(self.mean)
            # 计算标准差std
            self.std = np.sqrt(self.var)

class PPO:
    def __init__(self, policy_class, env, **hyperparameters):
        """
        初始化策略网络和值函数网络, 并设置超参数, 如学习率、折扣因子、GAE参数、裁剪参数等。
        """
        assert(type(env.observation_space) == gym.spaces.Box)
        assert(type(env.action_space) == gym.spaces.Box)
        
        # 初始化超参数
        self._init_hyperparameters(hyperparameters) 
        
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        # 创建策略网络和值函数网络
        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        # 动态协方差矩阵
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=self.init_cov)
        self.cov_mat = torch.diag(self.cov_var)
        
        # 初始化标准化器
        if self.normalize_observations:
            self.obs_rms = RunningStat(shape=self.obs_dim)
            
        if self.normalize_rewards:
            self.ret_rms = RunningStat(1)
            self.returns = np.zeros(1)
        
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
            'critic_losses': [],    # losses of critic network in current iteration
            'entropy_losses': [],   # entropy losses
            'total_losses': [], 
        }
        
    def _init_hyperparameters(self, hyperparameters):
        # 设置默认超参数
        self.timesteps_per_batch = 4096
        self.max_timesteps_per_episode = 1000
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.n_updates_per_iteration = 10
        self.clip = 0.2
        self.lr = 0.0003
        self.init_cov = 0.5
        self.min_cov = 0.05
        self.cov_decay = 0.999
        self.entropy_coef = 0.01
        self.vf_coef = 0.5
        self.render = True
        self.render_every_i = 10
        self.save_freq = 5
        self.normalize_observations = False  
        self.normalize_rewards = False       
        self.lr_schedule = 'constant'
        self.target_kl = 0.01
        self.early_stopping = True
        self.seed = None
        
        # 更新传入的超参数
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))
            
        # 设置随机种子
        if self.seed is not None:
            assert(type(self.seed) == int)
            torch.manual_seed(self.seed) 
            np.random.seed(self.seed)
            print(f"Successfully set seed to {self.seed}.")      
            
    def normalize_observation(self, obs):
        """标准化观察值"""
        if not self.normalize_observations:
            return obs
        
        if isinstance(obs, np.ndarray):
            self.obs_rms.update(obs)
            return (obs - self.obs_rms.mean) / (self.obs_rms.std + 1e-8)
        else:
            obs_np = obs.detach().numpy() if torch.is_tensor(obs) else obs
            self.obs_rms.update(obs_np)
            return torch.FloatTensor((obs_np - self.obs_rms.mean) / (self.obs_rms.std + 1e-8))
        
    def normalize_reward(self, reward):
        """标准化奖励"""
        if not self.normalize_rewards:
            return reward

        # 确保奖励是标量，不是数组或嵌套结构
        if isinstance(reward, (list, np.ndarray)) and len(np.shape(reward)) > 0:
            reward = float(reward[0]) if len(reward) > 0 else 0.0
        
        self.returns = self.returns * self.gamma + reward
        self.ret_rms.update(np.array([self.returns]))
        return reward / (self.ret_rms.std + 1e-8)
    
    def decay_learning_rate(self, iteration, total_iterations):
        """学习率衰减"""
        total_iterations = max(1, total_iterations)
        
        if self.lr_schedule == 'constant':
            return self.lr
        elif self.lr_schedule == 'linear':
            return self.lr * (1 - iteration / total_iterations)
        elif self.lr_schedule == 'step':
            decay_steps = max(1, total_iterations // 4)
            return self.lr * (0.5 ** (iteration // decay_steps))
        else:
            return self.lr
        
    def update_covariance(self):
        """更新协方差矩阵"""
        self.cov_var = torch.max(
            self.cov_var * self.cov_decay, 
            torch.full_like(self.cov_var, self.min_cov) 
        )
        
        self.cov_mat = torch.diag(self.cov_var)
            
    def rollout(self):
        """收集训练数据"""
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        
        t = 0  # 计数器，跟踪已收集的时间步
        
        while t < self.timesteps_per_batch:
            # 单个episode的数据
            ep_rews = []
            
            # 重置环境
            obs, _ = self.env.reset()
            if self.normalize_observations:
                norm_obs = self.normalize_observation(obs)
            else:
                norm_obs = obs
                
            done = False
            
            for ep_t in range(self.max_timesteps_per_episode):
                # 可选渲染
                if self.render and self.logger['i_so_far'] % self.render_every_i == 0:
                    self.env.render()
                
                t += 1  # 增加总步数计数
                
                # 保存观察
                batch_obs.append(norm_obs)
                
                # 获取动作和其对数概率
                action, log_prob = self.get_action(norm_obs)
                
                # 执行动作
                next_obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 标准化奖励（如果启用）
                if self.normalize_rewards:
                    norm_rew = self.normalize_reward(rew)
                else:
                    norm_rew = rew
                
                # 保存数据
                ep_rews.append(norm_rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                
                # 更新观察
                obs = next_obs
                if self.normalize_observations:
                    norm_obs = self.normalize_observation(obs)
                else:
                    norm_obs = obs
                
                if done:
                    break
                
            # 保存episode长度和奖励
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            
        # 将数据转换为张量
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        
        # 计算rewards-to-go
        batch_rtgs = self.compute_rtgs(batch_rews)
        
        # 更新日志
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    
    def compute_rtgs(self, batch_rews):
        """计算奖励到目标的值"""
        batch_rtgs = []
        
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
                
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    
    def learn(self, total_timesteps):
        """训练模型"""
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        
        t_so_far = 0  # 已执行的总时间步
        i_so_far = 0  # 已执行的迭代次数
        
        # 计算总迭代次数（用于学习率衰减）
        total_iterations = max(1, total_timesteps // self.timesteps_per_batch)
        
        while t_so_far < total_timesteps:
            # 收集一批数据
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            
            # 更新学习率（如果不是常数）
            if self.lr_schedule != 'constant':
                curr_lr = self.decay_learning_rate(i_so_far, total_iterations)
                for param_group in self.actor_optim.param_groups:
                    param_group['lr'] = curr_lr
                for param_group in self.critic_optim.param_groups:
                    param_group['lr'] = curr_lr
            
            # 更新协方差矩阵
            self.update_covariance()
            
            # 更新计数器
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            
            # 更新日志
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far
            
            # 输出进度
            if i_so_far % 10 == 0:
                print(f"Iteration {i_so_far} - Completed {t_so_far}/{total_timesteps} timesteps "
                      f"({(t_so_far / total_timesteps) * 100:.2f}%)")
            
            # 计算值函数估计
            V = self.critic(batch_obs).squeeze()
            
            # 计算优势函数
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            
            # 多次更新策略和值函数
            for _ in range(self.n_updates_per_iteration):
                # 重新评估当前策略
                V = self.critic(batch_obs).squeeze()
                _, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                
                # 计算策略比率
                ratio = torch.exp(curr_log_probs - batch_log_probs)
                
                # 计算PPO目标
                surr1 = ratio * A_k
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * A_k
                
                # 计算actor损失
                actor_loss = (-torch.min(surr1, surr2)).mean()
                
                # 计算critic损失 - 确保形状匹配
                if V.shape != batch_rtgs.shape:
                    # 如果形状不匹配，调整形状
                    if len(V.shape) == 0:  # 如果V是标量
                        V = V.unsqueeze(0)
                    if len(batch_rtgs.shape) == 0:  # 如果batch_rtgs是标量
                        batch_rtgs = batch_rtgs.unsqueeze(0)
                    
                    # 截断或填充到相同长度
                    min_len = min(V.shape[0], batch_rtgs.shape[0])
                    V = V[:min_len]
                    batch_rtgs = batch_rtgs[:min_len]
                
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                
                # 更新actor
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                # 防止梯度爆炸
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optim.step()
                
                # 更新critic
                self.critic_optim.zero_grad()
                critic_loss.backward()
                # 防止梯度爆炸
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optim.step()
                
                # 记录损失
                self.logger['actor_losses'].append(actor_loss.detach())
                self.logger['critic_losses'].append(critic_loss.detach())
            
            # 记录当前批次的训练数据
            self._log_summary()
            
            # 保存模型
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), f'./ppo_actor_{i_so_far}.pth')
                torch.save(self.critic.state_dict(), f'./ppo_critic_{i_so_far}.pth')
    
    def get_action(self, obs):
        """根据观察获取动作"""
        # 确保obs是张量
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        
        # 获取动作均值
        mean = self.actor(obs)
        
        # 创建多元正态分布
        dist = MultivariateNormal(mean, self.cov_mat)
        
        # 从分布中采样动作
        action = dist.sample()
        
        # 计算对数概率
        log_prob = dist.log_prob(action)
        
        return action.detach().numpy(), log_prob.detach().item()
    
    def evaluate(self, batch_obs, batch_acts):
        """评估值函数和对数概率"""
        # 确保输入是正确的张量
        if not isinstance(batch_obs, torch.Tensor):
            batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        if not isinstance(batch_acts, torch.Tensor):
            batch_acts = torch.tensor(batch_acts, dtype=torch.float)
            
        # 计算值函数
        V = self.critic(batch_obs).squeeze()
        
        # 计算对数概率
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        
        return V, log_probs
    
    def _log_summary(self):
        """记录训练摘要"""
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        
        # 安全计算平均值
        batch_lens = self.logger['batch_lens']
        avg_ep_lens = np.mean(batch_lens) if batch_lens else 0
        
        # 安全计算平均奖励
        batch_rews = self.logger['batch_rews']
        # 将奖励列表转换为扁平数组，避免嵌套结构
        ep_rewards = []
        for ep_rew in batch_rews:
            # 确保每个episode奖励是数字而不是复杂结构
            try:
                ep_rewards.append(float(sum(ep_rew)))
            except (TypeError, ValueError):
                # 跳过无法求和的奖励
                continue
        
        avg_ep_rews = np.mean(ep_rewards) if ep_rewards else 0
        
        # 计算平均actor损失
        actor_losses = [loss.float().mean().item() for loss in self.logger['actor_losses']]
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0

        # 格式化输出
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # 打印日志
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Current Covariance: {self.cov_var[0].item():.5f}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # 重置日志数据
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []