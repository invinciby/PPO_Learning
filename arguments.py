import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', dest='mode', type=str, default='test', help='train ,test')
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='model_save/ppo_actor_90.pth')
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='model_save/ppo_critic_90.pth')
    parser.add_argument('--env', dest='env', type=str, default='Pendulum-v1', help='CartPole-v1, Pendulum-v1')
    parser.add_argument('--total_timesteps', dest='total_timesteps', type=int, default=200000, help='total_timesteps')
    
    args = parser.parse_args()
    return args