import argparse

def get_args():
    # 创建一个参数解析器
    parser = argparse.ArgumentParser()
    
    # 添加参数，指定模型类型，默认为'test'
    parser.add_argument('--model', dest='mode', type=str, default='test', help='train ,test')
    # 添加参数，指定actor模型路径，默认为'model_save/ppo_actor_90.pth'
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='ppo_actor_V_90.pth')
    # 添加参数，指定critic模型路径，默认为'model_save/ppo_critic_90.pth'
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='ppo_critic_V_90.pth')
    # 添加参数，指定环境类型，默认为'Pendulum-v1'
    parser.add_argument('--env', dest='env', type=str, default='MountainCarContinuous-v0', help='Pendulum-v1, MountainCarContinuous-v0')
    # 添加参数，指定总时间步数，默认为200000
    parser.add_argument('--total_timesteps', dest='total_timesteps', type=int, default=200000, help='total_timesteps')
    
    # 解析参数
    args = parser.parse_args()
    return args