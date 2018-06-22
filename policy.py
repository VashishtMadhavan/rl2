import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils import *

class LSTMPolicy:
	def __init__(scope,
		inputs,
		action_dim,
		hidden_dim=256,
		activation=tf.nn.relu):

		self.scope = scope
		self.hidden_dim = hidden_dim
		self.activation_fn = activation
		self.action_dim = action_dim

		with tf.variable_scope(self.scope):
			lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True, activation=self.activation_fn)
			self.c_init = np.zeros((1, lstm.state_size.c), np.float32)
			self.h_init = np.zeros((1, lstm.state_size.h), np.float32)

			self.c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
			self.h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])

			lstm_out, lstm_state = tf.nn.dynamic_rnn(lstm, 
				inputs=tf.expand_dims(inputs, [0]),
				initial_state= tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in),
				sequence_length=tf.shape(inputs)[:1],
				time_major=False
			)
			lstm_c, lstm_h = lstm_state
			self.c_out = lstm_c[:1, :]
			self.h_out = lstm_h[:1, :]
			lstm_out_flat = tf.reshape(lstm_out, [-1, self.hidden_dim])

			self._pi = layers.fully_connected(lstm_out_flat, num_outputs=self.action_dim, activation_fn=None)
			self._v = layers.fully_connected(lstm_out_flat, num_outputs=1, activation_fn=None)
			self._variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

		self.prev_c = self.c_init.copy()
		self.prev_h = self.h_init.copy()

	@property
	def recurrent(self):
		return True

	@property
	def V(self):
		return self._v

	@property
	def pi(self):
		return self._pi

	@property
	def variables(self):
		return self._variables

	def reset(self):
		self.prev_c = self.c_init.copy()
		self.prev_h = self.h_init.copy()

	def save(self, session, filename):
		pass

