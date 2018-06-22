import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils import *

class A2C:
	def __init__(self, 
		env,
		session,
		scope,
		policy_cls,
		policy_kwargs,
		encode_state=False,
		grad_clip=10):

		self.input_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.n
		self.hidden_dim = hidden_dim
		self.vf_ceof = 0.5
		self.ent_coef = 0.01
		self.step_size = 7e-5
		self.session = session

		if encode_state:
			pass
			# TODO: add something here for encoding
		else:
			self.encoder = None

		self.X = tf.placeholder(tf.float32, [None, input_dim])
		self.ADV = tf.placeholder(tf.float32, [None])
		self.A = tf.placeholder(tf.int32,   [None])
		self.R = tf.placeholder(tf.float32, [None])
		
		self.policy = policy_cls(scope=scope, inputs=self.X, **policy_kwargs)

		dist = tf.distributions.Categorical(logits=self.policy.pi)
		neglogpi = tf.nn.sparse_softmax_cross_entropy(logits=self.policy.pi, labels=self.A)
		self.pg_loss = tf.reduce_mean(self.ADV * neglogpi)
		self.vf_loss = tf.reduce_mean(tf.square(tf.squeeze(self.policy.V) - self.R) / 2.)			
		self.entropy = tf.reduce_mean(dist.entropy())
		self.loss = self.pg_loss - self.ent_coef * self.entropy + self.vf_ceof * self.vf_loss
		self.act = dist.sample()

		opt = tf.train.RMSPropOptimizer(learning_rate=self.step_size, decay=0.99, epsilon=1e-5)
		self.train_op = minimize_and_clip(opt, self.loss, var_list=self.policy.variables, clip_val=grad_clip)

		initialize(self.session)

	def get_actions(self, X):
		if not self.policy.recurrent:
			actions, values = self.session.run([self.act, self.policy.V], feed_dict={self.X: X})
		else:
			actions, values, c_out, h_out = self.session.run([self.act, self.policy.V, self.policy.c_out, self.h_out],
				feed_dict={self.X: X,
				self.policy.c_in: self.policy.prev_c,
				self.policy.h_in: self.policy.prev_h})
			self.policy.prev_c = c_out 
			self.policy.prev_h = h_out
		return actions, values

	def reset(self):
		self.policy.reset()
		
	def save_policy(self):
		# TODO: set filename
		save_filename = None
		self.policy.save(self.session, save_filename)

	def train(self, ep_X, ep_A, ep_R, ep_adv):
		train_dict = {self.X: ep_X, self.ADV: ep_adv, self.A: ep_A, self.R: ep_R}
		if not self.policy.recurrent:
			train_dict = {self.policy.c_in: self.policy.c_init, self.policy_h_in: self.policy.h_init}		
		pLoss, vLoss, ent, _ = self.session.run([self.pg_loss, self.vf_loss, self.entropy, self.train_op],
			feed_dict=train_dict)
		info = {}
		info['policy_loss'] = pLoss
		info['value_loss'] = vLoss
		info['policy_entropy'] = ent
		return info