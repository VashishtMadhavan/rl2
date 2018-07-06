import gym
import argparse
import numpy as np
import os
from utils import *
from policy import LSTMPolicy
from a2c import A2C

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_env', type=str, default="MediumBandit-v0", help='env for meta-training')
	parser.add_argument('--test_env', type=str, default="EasyBandit-v0", help='env for meta-testing')
	parser.add_argument('--train_eps', type=int, default=int(2e4), help='training episodes')
	parser.add_argument('--test_eps', type=int, default=300, help='test episodes')
	parser.add_argument('--seed', type=int, default=1, help='experiment seed')

	# Training Hyperparameters
	parser.add_argument('--hidden', type=int, default=48, help='hidden layer dimensions')
	parser.add_argument('--gamma', type=float, default=0.8, help='discount factor')
	args = parser.parse_args()

	env = gym.make(args.train_env)
	env.seed(args.seed)

	eval_env = gym.make(args.test_env)
	eval_env.seed(args.seed)

	algo = A2C(env=env,
		session=get_session(),
	    policy_cls=LSTMPolicy,
	    hidden_dim=args.hidden,
	    action_dim=env.action_space.n,
		scope='a2c')

	save_iter = args.train_eps // 20
	average_returns = []
	average_regret = []
	average_subopt = []

	for ep in range(args.train_eps):
		obs = env.reset()
		done = False
		ep_X, ep_R, ep_A, ep_V, ep_D = [], [], [], [], []
		track_R = 0; track_regret = np.max(env.unwrapped.probs) * env.unwrapped.n
		best_action = np.argmax(env.unwrapped.probs); num_suboptimal = 0
		action_hist = np.zeros(env.action_space.n)
		algo.reset()

		while not done:
			action, value = algo.get_actions(obs[None])
			new_obs, rew, done, info = env.step(action)
			track_R += rew
			num_suboptimal += int(action != best_action)
			action_hist[action] += 1

			ep_X.append(obs)
			ep_A.append(action)
			ep_V.append(value)
			ep_R.append(rew)
			ep_D.append(done)

			obs = new_obs
		_, last_value = algo.get_actions(obs[None])
		ep_X = np.asarray(ep_X, dtype=np.float32)
		ep_R = np.asarray(ep_R, dtype=np.float32)
		ep_A = np.asarray(ep_A, dtype=np.int32)
		ep_V = np.squeeze(np.asarray(ep_V, dtype=np.float32))
		ep_D = np.asarray(ep_D, dtype=np.float32)

		if ep_D[-1] == 0:
			disc_rew = discount_with_dones(ep_R.to_list() + [np.squeeze(last_value)], ep_D.to_list() + [0], args.gamma)[:-1]
		else:
			disc_rew = discount_with_dones(ep_R.tolist(), ep_D.tolist(), args.gamma)
		ep_adv = disc_rew - ep_V
		track_regret -= track_R

		train_info = algo.train(ep_X=ep_X, ep_A=ep_A, ep_R=ep_R, ep_adv=ep_adv)
		average_returns.append(track_R)
		average_regret.append(track_regret)
		average_subopt.append(num_suboptimal)

		if ep % save_iter == 0 and ep != 0:
			print("Episode: {}".format(ep))
			print("ActionHist: {}".format(action_hist))
			print("Probs: {}".format(env.unwrapped.probs))
			print("MeanReward: {}".format(np.mean(average_returns[-50:])))
			print("MeanRegret: {}".format(np.mean(average_regret[-50:])))
			print("NumSuboptimal: {}".format(np.mean(average_subopt[-50:])))
			print()

	print()
	test_regrets = []; test_rewards = []
	for test_ep in range(args.test_eps):
		obs = eval_env.reset()
		algo.reset()
		done = False
		track_regret = np.max(eval_env.unwrapped.probs) * eval_env.unwrapped.n
		track_R = 0

		while not done:
			action, value = algo.get_actions(obs[None])
			new_obs, rew, done, info = eval_env.step(action)
			obs = new_obs
			track_R += rew

		test_regrets.append(track_regret - track_R)
		test_rewards.append(track_R)
	print('Mean Test Cumulative Regret: {}'.format(np.mean(test_regrets)))
	print('Mean Test Reward: {}'.format(np.mean(test_rewards)))

if __name__=='__main__':
	main()
