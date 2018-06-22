from vec_env.vec_frame_stack import VecFrameStack
from vec_env.subproc_vec_env import SubprocVecEnv

import os
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import random
import h5py


"""

Gen Purpose Utility Functions

"""

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0)
    tf_config.gpu_options.allow_growth=True
    session = tf.Session(config=tf_config)
    return session


ALREADY_INITIALIZED = set()
def initialize(session):
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    session.run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)

def create_vec_env(env_id, seed, num_threads):
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env
        return _thunk
    env = SubprocVecEnv([make_env(i + start_index) for i in range(num_threads)])
    return env