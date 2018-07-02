import argparse
import math
from collections import namedtuple
from itertools import count
import random
from operator import attrgetter
import copy

import gym
import numpy as np
from gym import wrappers

import torch
from ddpg import DDPG
from naf import NAF
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from replay_memory import ReplayMemory, Transition
import pickle



def parse_arguments():
    global parser
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--algo', default='DDPG',
                        help='algorithm to use: DDPG | NAF')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='name of the environment to run')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                        help='discount factor for model (default: 0.001)')
    parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                        help='initial noise scale (default: 0.3)')
    parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                        help='final noise scale (default: 0.3)')
    parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                        help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=4, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size (default: 128)')
    parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--num_episodes', type=int, default=100, metavar='N',
                        help='number of episodes (default: 1000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                        help='number of episodes (default: 128)')
    parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                        help='model updates per simulator step (default: 5)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')


if __name__=="__main__":
    parse_arguments()
    args = parser.parse_args()
    args.env_name = "Springmass-v0"
    print("Running environment" + str(args.env_name))

    env = NormalizedActions(gym.make(args.env_name))
    env = wrappers.Monitor(env, '/tmp/{}-experiment'.format(args.env_name), force=True)
    env.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = DDPG(args.gamma, args.tau, args.hidden_size, env.observation_space.shape[0], env.action_space)

    # Load saved policy
    agent.actor.load_state_dict((torch.load('params_monopod.pt')))

    for j in range(3):
        state = torch.Tensor([env.reset()])
        # test_episode_ERL_reward = 0.0
        for t in range(args.num_steps):
            # forward pass through the actor network
            action = agent.select_action(state, exploration=None)
            next_state, reward, done, _ = env.step(action.numpy()[0])
            # test_episode_ERL_reward += reward
            env.render()

            next_state = torch.Tensor([next_state])
            state = next_state

            if done:
                env.reset()
