def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def rollout(policy, env, render):
    while True:
        obs, _ = env.reset()
        done = False
        
        t = 0
        
        ep_len = 0
        ep_ret = 0
        
        while not done:
            t += 1
            if render:
                env.render()
            
            action = policy(obs).detach().numpy()
            obs, rew, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_ret += rew
            
        ep_len = t
        
        yield ep_len, ep_ret
        
def eval_policy(policy, env, render=False):
    for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render)):
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)