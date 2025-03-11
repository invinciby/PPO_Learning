import gymnasium as gym
import sys
import torch

from arguments import get_args
from PPO_v1 import PPO as PPO_v1
from network import FeedForwardNN
from eval_policy import eval_policy


def train(env, hyperparameters, actor_model, critic_model):
    print(f"Training", flush=True)
    
    model = PPO_v1(policy_class=FeedForwardNN, env=env, **hyperparameters)
    
    if actor_model !='' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
        
    elif actor_model != '' or critic_model != '':
        print(f"Error:  要么同时指定演员/评论员模式，要么一个都不指定。 我们不想意外地覆盖任何东西！", flush=True)
        sys.exit(1)
    else:
        print(f"Starting from scratch.", flush=True)
        
    model.learn(total_timesteps=args.total_timesteps)
    
def test(env, actor_model):
    print(f"Testing", flush=True)
    
    if actor_model == '':
        print(f"Error:  必须指定演员模型。", flush=True)
        sys.exit(1)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = FeedForwardNN(obs_dim, act_dim)
    
    policy.load_state_dict(torch.load(actor_model))

    eval_policy(policy=policy, env=env, render=True)
    
def main(args):
    hyperparameters = {
        # 数据采集参数
        'timesteps_per_batch': 2048,          # 较小的批次以减少内存需求
        'max_timesteps_per_episode': 200,     # Pendulum环境每个episode较短
        
        # 折扣和优势估计
        'gamma': 0.99,                        # 折扣因子
        'gae_lambda': 0.95,                   # GAE lambda参数
        
        # 优化参数
        'n_updates_per_iteration': 10,        # 每批次的更新次数
        'lr': 0.0003,                         # 学习率
        'clip': 0.2,                          # PPO裁剪参数
        
        # 探索参数
        'init_cov': 0.8,                      # 初始协方差（更高以鼓励探索）
        'min_cov': 0.1,                       # 最小协方差
        'cov_decay': 0.995,                   # 协方差衰减率（降低使其衰减更慢）
        'entropy_coef': 0.01,                 # 熵正则化系数
        
        # 训练稳定性
        'vf_coef': 0.5,                       # 值函数损失系数
        'target_kl': 0.02,                    # KL散度阈值
        
        # 其他参数
        'normalize_observations': False,      # 先关闭观察标准化
        'normalize_rewards': False,           # 先关闭奖励标准化
 
        'early_stopping': True,               # 启用早停
        'render': True,                       # 是否渲染
        'render_every_i': 5,                  # 每隔多少次迭代渲染一次
        'save_freq': 10                       # 保存频率
    }
    
    env = gym.make(args.env, render_mode='human' if args.mode == 'test' else 'rgb_array')

    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        test(env=env, actor_model=args.actor_model)
        
if __name__ == '__main__':
    args = get_args()
    main(args)