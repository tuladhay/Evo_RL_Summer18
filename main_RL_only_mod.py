import argparse
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
    parser = argparse.ArgumentParser(description='PyTorch DDPG')
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
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 128)')
    parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--num_episodes', type=int, default=2000, metavar='N',
                        help='number of episodes (default: 1000)')
    parser.add_argument('--hidden_size', type=int, default=32, metavar='N',
                        help='number of episodes (default: 128)')
    parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                        help='model updates per simulator step (default: 5)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')

if __name__ == "__main__":
    parse_arguments()
    args = parser.parse_args()
    args.env_name = "Springmass-v0"
    print("Running environment" + str(args.env_name))

    env = NormalizedActions(gym.make(args.env_name))
    env = wrappers.Monitor(env, '/tmp/{}-experiment'.format(args.env_name), force=True)
    env.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    '''
    DEFINE THE ACTOR RL AGENT
    '''
    if args.algo == "NAF":
        agent = NAF(args.gamma, args.tau, args.hidden_size,
                    env.observation_space.shape[0], env.action_space)
        print("Initialized NAF")
    else:
        agent = DDPG(args.gamma, args.tau, args.hidden_size,
                     env.observation_space.shape[0], env.action_space)
        print("Initialized DDPG actor")

    '''
    DEFINE REPLAY BUFFER AND NOISE
    '''
    memory = ReplayMemory(args.replay_size)
    ounoise = OUNoise(env.action_space.shape[0])

    # TODO: MOVE THE TRAINING CODE BELOW TO ITS RESPECTIVE FUNCTIONS
    rewards = []  # during training
    rewards_test_DDPG = []

    print("Number of hidden units = " + str(args.hidden_size))
    print("Batch size = " + str(args.batch_size))
    print("Number of episodes : " + str(args.num_episodes))
    for i_episode in range(args.num_episodes):
        '''
        #############
        The DDPG part
        #############
        '''
        state = torch.Tensor([env.reset()])  # algo line 6
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                          i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()
        episode_reward = 0
        eps = int(args.num_steps/10)

        for t in range(eps):  # line 7
            # forward pass through the actor network
            action = agent.select_action(state, ounoise)  # line 8
            done = False
            for i in range(10):
                next_state, reward, done, _ = env.step(action.numpy()[0])  # line 9
                episode_reward += reward

                action = torch.Tensor(action)
                mask = torch.Tensor([not done])
                next_state = torch.Tensor([next_state])
                reward = torch.Tensor([reward])

                if done:
                    # print("Done")
                    break

                if i_episode % 1 == 0:
                    env.render()

                memory.push(state, action, mask, next_state, reward)  # line 10

                state = next_state

            if len(memory) > args.batch_size * 5:
                for _ in range(args.updates_per_step):
                    transitions = memory.sample(args.batch_size)  # line 11
                    batch = Transition(*zip(*transitions))

                    agent.update_parameters(batch)
            if done:
                break

        rewards.append(episode_reward)

        '''
            ##################
            Run DDPG policy
            ##################
        '''
        for j in range(3):
            state = torch.Tensor([env.reset()])
            test_episode_DDPG_reward = 0.0
            for t in range(args.num_steps):
                # forward pass through the actor network
                action = agent.select_action(state, exploration=None)
                next_state, reward, done, _ = env.step(action.numpy()[0])
                test_episode_DDPG_reward += reward

                next_state = torch.Tensor([next_state])
                state = next_state

                # print("Test run, Action: " + str(action))
                if done:
                    break
                # env.render()

        test_episode_DDPG_reward = np.mean(test_episode_DDPG_reward)
        rewards_test_DDPG.append(test_episode_DDPG_reward)
        print("DDPG Test Reward = " + str(test_episode_DDPG_reward))

        ''' Print the training performance'''
        print("Training: Episode: {}, noise: {}, reward: {}, average reward: {}".format(i_episode, ounoise.scale,
                                                                                        rewards[-1],
                                                                                        np.mean(rewards[-10:])))
        print()
        print()

    env.close()

    pickling_on = open("Springmass_RL_rewards.p", "wb")
    pickle.dump(rewards_test_DDPG, pickling_on)
    pickling_on.close()

    # Save model
    torch.save(agent.actor.state_dict(), 'params_springmass_ddpg.pt')
