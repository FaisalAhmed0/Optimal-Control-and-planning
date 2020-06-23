# Cross Entropy Method
import numpy as np
import matplotlib.pyplot as plt
import gym
import pybullet_envs
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--env', type=str, required=False)
# parser.add_argument('--iters', type=int, default=10000)
# parser.add_argument('--n_samples', type=int, default=100)
# parser.add_argument('--elites_perc', type=float, default=0.3)

# args = parser.parse_args()

# envs = ['MountainCarContinuous-v0', 'InvertedPendulumBulletEnv-v0']
# env = gym.make(args.env)
# actions_high = env.action_space.high
# actions_low = env.action_space.low
# T = env.spec.max_episode_steps

# iters = args.iters
# n_samples = args.n_samples
# elites_perc = args.elites_perc



class StateReset(gym.Wrapper):
    def __init__(self, env=None):
        super(StateReset, self).__init__(env)

    def step(self, action):
        return self.env.step(action)

    def reset(self, state):
    	self.state = state
    	return np.array(state)



class CEM():
	def __init__(self, env_name, n_samples):
		self.env_name = env_name
		self.n_samples = n_samples 
		self.env = StateReset(gym.make(env_name))

	# Evaluate the objective
	def J(self, state,Actions):
		state = self.env.reset(self.start_state)
		# print(f'start_state {self.start_state}')
		# print(f'state {state}')
		# print(f'state {self.env.state}')
		total_reward = 0
		done = False
		for action in Actions:
			state, reward, done, _ = env.step(action)
			total_reward += reward
			if done:
				break
		return total_reward

	def evaluate_actions(self, sampled_actions, state):
		evaluations = []
		for actions in sampled_actions:
			evaluations.append(self.J(state, actions))
		return evaluations

	def generateActionsFromUniform(self, n_samples,low, high, action_shape):
		A = []
		for i in range(n_samples):
			A.append(np.random.uniform(low=low, high=high, size=(T,action_shape)))
		return A

	def generateActionsFromGaussian(self, n_samples,mean, std,action_shape):
		A = []
		for i in range(n_samples):
			A.append(np.random.normal(mean, std, size=(T,action_shape)))
		return A

	def cem_optimizer(self, state):
		self.start_state = state.copy()
		mean_returns = []
		for i in range(iters):
			if i == 0:
				sampled_actions = self.generateActionsFromUniform(n_samples, actions_low, actions_high, env.action_space.shape[0])
			else:
				sampled_actions = self.generateActionsFromGaussian(n_samples, elites_mean, elites_cov,  env.action_space.shape[0])
				# sampled_actions = np.clip(sampled_actions, actions_low, actions_high)
			evaluations =  self.evaluate_actions(sampled_actions, self.start_state)
			sampled_actions_dict = {f'{k}': sampled_action for k, sampled_action in zip(evaluations, sampled_actions)}
			sampled_actions_dict_sorted = {k:v for k,v in sorted(sampled_actions_dict.items(), key=lambda item: float(item[0]), reverse=True)}
			elites = np.array([sample for sample in sampled_actions_dict_sorted.values()][:int(elites_perc*n_samples)])
			elites_mean = np.mean(elites, axis=0)
			elites_cov = np.std(elites, axis=0)
			# print(f'elites_mean.shape {elites_mean.shape}')
			mean_returns.append(np.mean(evaluations))
			# if (i%100==0):
			# 	print(f'step: {i} mean return: {np.mean(evaluations)} ')
		return elites[0]

# def test_policy(actions, env, state,render=False, record=True):
#     total_reward = 0.0
#     if record:
#         env = gym.wrappers.Monitor(env, "recording", force=True)
#     done = False
#     env = StateReset(env)
#     state = env.reset(state)
#     # print('after cem', env.state)
#     # env.reset()
#     for action in actions:
#         if render:
#             env.render()
#         state, reward, done, _ = env.step(action)
#         total_reward += reward
#         if done:
#         	break
#     env.close()
#     return total_reward


# if __name__=='__main__':
# 	state = env.reset().copy()
# 	# print('before cem', env.state)
# 	cem = CEM(envs[0], n_samples)
# 	actions = cem.cem_optimizer(state)
# 	total_reward = test_policy(actions, env, state, render=True, record=False)
# 	print(f"Total reward: {total_reward}")


    

	





