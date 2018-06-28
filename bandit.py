import gym
import argparse
import numpy as np
import os
from utils import *
from policy import LSTMPolicy
from a2c import A2C

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--k', type=int, default=5, help='num arms')
	parser.add_argument('--n', type=int, default=10, help='num trials')
	parser.add_argument('--train_eps', type=int, default=int(2e4), help='training episodes')
	parser.add_argument('--seed', type=int, default=1, help='experiment seed')

	# Training Hyperparameters
	parser.add_argument('--hidden', type=int, default=48, help='hidden layer dimensions')
	parser.add_argument('--gamma', type=float, default=0.8, help='discount factor')
	args = parser.parse_args()

	env_id = "Bandit_k{}_n{}-v0".format(args.k, args.n)
	env = gym.make(env_id)

	algo = A2C(env=env,
		session=get_session(),
	    policy_cls=LSTMPolicy,
	    hidden_dim=args.hidden,
	    action_dim=env.action_space.n,
		scope='a2c')

	save_iter = args.train_eps // 10

	for ep in range(args.train_eps):
		obs = env.reset()
		done = False
		ep_X, ep_R, ep_A, ep_V, ep_D = [], [], [], [], []
		track_R = 0

		while not done:
			action, value = algo.get_actions(obs[None])
			new_obs, rew, done, info = env.step(action)
			track_R += rew

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

		train_info = algo.train(ep_X=ep_X, ep_A=ep_A, ep_R=ep_R, ep_adv=ep_adv)
		algo.reset()

		print("Episode: ", ep)
		print("Episode Reward: ", track_R)
		for k in sorted(train_info.keys()):
			print("{}: {}".format(k, train_info[k]))

if __name__=='__main__':
	main()