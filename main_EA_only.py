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


class Evo:
    def __init__(self, num_evo_actors, evo_episodes=1):
        '''
        :param num_evo_actors: This is the number of genes/actors you want to have in the population
        :param evo_episodes: This is the number of evaluation episodes for each gene. See Algo1: 7, and Table 1
        population: initalizes 10 genes/actors
        num_elites: number of genes/actor that are selected, and do not undergo mutation (unless they are
                    selected again in the tournament selection
        tournament_genes: number of randomly selected genes to take the max(fitness) from,
                        and then put it back into the population

        noise_mean: mean for the gaussian noise for mutation
        noise_stddev: standard deviation for the gaussian noise for mutation
        '''
        self.num_actors = num_evo_actors
        self.population = [DDPG(args.gamma, args.tau, args.hidden_size,
                                env.observation_space.shape[0], env.action_space) for _ in range(10)]
        print("Initializing Evolutionary Actors")
        self.evo_episodes = evo_episodes
        self.elite_percentage = 0.1
        self.num_elites = int(self.elite_percentage * self.num_actors)
        self.tournament_genes = 3  # TODO: make it a percentage

        self.noise_mean = 0.0
        self.noise_stddev = 0.1

        self.save_fitness = []
        self.best_policy = copy.deepcopy(self.population[0])  # for saving policy purposes

    def initialize_fitness(self):
        '''
        Adds and attribute "fitness" to the genes/actors in the list of population,
        and sets the fitness of all genes/actor in the population to 0
        '''
        for gene in self.population:
            gene.fitness = 0.0
        print("Initialized gene fitness")

    def evaluate_pop(self):
        for gene in self.population:
            fitness = []
            for ep in range(self.evo_episodes):
                evo_state = torch.Tensor([env.reset()])
                evo_episode_reward = 0
                for t in range(args.num_steps):
                    evo_action = gene.select_action(evo_state)
                    evo_next_state, evo_reward, evo_done, _ = env.step(evo_action.numpy()[0])
                    evo_episode_reward += evo_reward

                    evo_action = torch.Tensor(evo_action)
                    evo_mask = torch.Tensor([not evo_done])
                    evo_next_state = torch.Tensor([evo_next_state])
                    evo_reward = torch.Tensor([evo_reward])

                    memory.push(evo_state, evo_action, evo_mask, evo_next_state, evo_reward)
                    evo_state = copy.copy(evo_next_state)

                    if ep % 20 == 0:
                        env.render()

                    if evo_done:
                        # print("Done")
                        break
                    # <end of time-steps>
                fitness.append(evo_episode_reward)
                # <end of episodes>
            fitness = sum(fitness) / self.evo_episodes  # Algo2: 12
            gene.fitness = copy.copy(fitness)

    def rank_pop_selection_mutation(self):
        '''
        This function takes the current evaluated population (of k , then ranks them according to their fitness,
        then selects a number of elites (e), and then selects a set S of (k-e) using tournament selection.
        It then calls the mutation function to add mutation to the set S of genes.
        In the end this will replace the current population with a new one.
        '''
        ranked_pop = copy.deepcopy(sorted(self.population, key=lambda x: x.fitness, reverse=True))  # Algo1: 9
        elites = ranked_pop[:self.num_elites]
        self.best_policy = elites[0]  # for saving policy purposes
        set_s = []

        for i in range(len(ranked_pop) - len(elites)):
            tournament_genes = [random.choice(ranked_pop) for _ in range(self.tournament_genes)]
            tournament_winner = max(tournament_genes, key=attrgetter('fitness'))
            set_s.append(copy.deepcopy(tournament_winner))

        mutated_set_S = copy.deepcopy(self.mutation(set_s))
        self.population = []
        # Addition of lists
        self.population = copy.deepcopy(elites + mutated_set_S)
        # print("Best fitness = " + str(elites[0].fitness))

        self.save_fitness.append(elites[0].fitness)

    def mutation(self, set_s):
        """
        :param set_s: This is the set of (k-e) genes that are going to be mutated by adding noise
        :return: Returns the mutated set of (k-e) genes

        Adds noise to the weights and biases of each layer of the network
        But why is a noise (out of 1) being added? Since we cant really say how big or small the parameters should be.
        """
        for i in range(len(set_s)):
            ''' Noise to Linear 1 weights and biases'''
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.linear1.weight))
            noise = torch.FloatTensor(noise)
            # gene.actor.linear1.weight.data = gene.actor.linear1.weight.data + noise
            noise = torch.mul(set_s[i].actor.linear1.weight.data, noise)
            set_s[i].actor.linear1.weight.data = copy.deepcopy(set_s[i].actor.linear1.weight.data + noise)

            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.linear1.bias))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.linear1.bias.data, noise)
            set_s[i].actor.linear1.bias.data = copy.deepcopy(set_s[i].actor.linear1.bias.data + noise)

            '''Noise to Linear 2 weights and biases'''
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.linear2.weight))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.linear2.weight.data, noise)
            set_s[i].actor.linear2.weight.data = copy.deepcopy(set_s[i].actor.linear2.weight.data + noise)

            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.linear2.bias))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.linear2.bias.data, noise)
            set_s[i].actor.linear2.bias.data = copy.deepcopy(set_s[i].actor.linear2.bias.data + noise)

            ''' Noise to mu layer weights and biases'''
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.mu.weight))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.mu.weight.data, noise)
            set_s[i].actor.mu.weight.data = copy.deepcopy(set_s[i].actor.mu.weight.data + noise)

            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.mu.bias))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.mu.bias.data, noise)
            set_s[i].actor.mu.bias.data = copy.deepcopy(set_s[i].actor.mu.bias.data + noise)

            ''' LayerNorm 1'''
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.layerN1.weight))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.layerN1.weight.data, noise)
            set_s[i].actor.layerN1.weight.data = copy.deepcopy(set_s[i].actor.layerN1.weight.data + noise)

            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.layerN1.bias))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.layerN1.bias.data, noise)
            set_s[i].actor.layerN1.bias.data = copy.deepcopy(set_s[i].actor.layerN1.bias.data + noise)

            ''' LayerNorm 2'''
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.layerN2.weight))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.layerN2.weight.data, noise)
            set_s[i].actor.layerN2.weight.data = copy.deepcopy(set_s[i].actor.layerN2.weight.data + noise)

            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.layerN2.bias))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.layerN2.bias.data, noise)
            set_s[i].actor.layerN2.bias.data = copy.deepcopy(set_s[i].actor.layerN2.bias.data + noise)

            ''' LayerNorm MU'''
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.layerNmu.weight))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.layerNmu.weight.data, noise)
            set_s[i].actor.layerNmu.weight.data = copy.deepcopy(set_s[i].actor.layerNmu.weight.data + noise)

            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.layerNmu.bias))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.layerNmu.bias.data, noise)
            set_s[i].actor.layerNmu.bias.data = copy.deepcopy(set_s[i].actor.layerNmu.bias.data + noise)

        return set_s


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
    # if args.algo == "NAF":
    #     agent = NAF(args.gamma, args.tau, args.hidden_size,
    #                 env.observation_space.shape[0], env.action_space)
    #     print("Initialized NAF")
    # else:
    #     agent = DDPG(args.gamma, args.tau, args.hidden_size,
    #                  env.observation_space.shape[0], env.action_space)
    #     print("Initialized DDPG actor")

    '''
    DEFINE REPLAY BUFFER AND NOISE
    '''
    memory = ReplayMemory(args.replay_size)
    ounoise = OUNoise(env.action_space.shape[0])

    '''
    #############################
    Initialize the Evolution Part
    #############################
    '''
    evo = Evo(10)
    evo.initialize_fitness()

    # TODO: MOVE THE TRAINING CODE BELOW TO ITS RESPECTIVE FUNCTIONS
    rewards = []  # during training
    rewards_test_ERL = []  # during testing ERL policy
    #rewards_test_DDPG = []

    print("Number of hidden units = " + str(args.hidden_size))
    print("Batch size = " + str(args.batch_size))
    print("Number of episodes : " + str(args.num_episodes))
    for i_episode in range(args.num_episodes):
        '''
        Here, num_episodes correspond to the generations in Algo 1.
        In every generation, the population is evaluated, ranked, mutated, and re-instered into population
        '''
        evo.evaluate_pop()
        evo.rank_pop_selection_mutation()

        print("Episode: "+str(i_episode) + "    Evolutionary Fitness = " + str(evo.best_policy.fitness))

        '''
        #############
        The DDPG part
        #############
        '''
        # state = torch.Tensor([env.reset()])  # algo line 6
        # ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
        #                                                                   i_episode) / args.exploration_end + args.final_noise_scale
        # ounoise.reset()
        # episode_reward = 0
        #
        # for t in range(args.num_steps):  # line 7
        #     # forward pass through the actor network
        #     action = agent.select_action(state, ounoise)  # line 8
        #     next_state, reward, done, _ = env.step(action.numpy()[0])  # line 9
        #     episode_reward += reward
        #
        #     action = torch.Tensor(action)
        #     mask = torch.Tensor([not done])
        #     next_state = torch.Tensor([next_state])
        #     reward = torch.Tensor([reward])
        #
        #     # if i_episode % 10 == 0:
        #     #     env.render()
        #
        #     memory.push(state, action, mask, next_state, reward)  # line 10
        #
        #     state = next_state
        #
        #     if len(memory) > args.batch_size * 5:
        #         for _ in range(args.updates_per_step):
        #             transitions = memory.sample(args.batch_size)  # line 11
        #             batch = Transition(*zip(*transitions))
        #
        #             agent.update_parameters(batch)
        #
        #     if done:
        #         break
        # rewards.append(episode_reward)
        #
        # '''
        # ###############
        # Synchronization
        # ###############
        # '''
        # if i_episode % 10 == 0:
        #     weakest_in_pop_index = evo.population.index(min(evo.population, key=attrgetter('fitness')))
        #     evo.population[weakest_in_pop_index] = copy.deepcopy(agent)
        #     print("Synchronized")

        '''
        ##################
        Run test episodes
        ##################
            >> First pick the agent with the best fitness in the population
            >> Then run 5 episodes of that on the environment, and average the reward

        '''
        test_actor_index = evo.population.index(max(evo.population, key=attrgetter('fitness')))
        test_actor_ERL = copy.deepcopy(evo.population[test_actor_index])
        test_episode_ERL_reward = 0.0

        #test_episode_DDPG_reward = 0.0

        '''
            ##############
            Run ERL policy
            ##############
        '''
        for j in range(3):
            state = torch.Tensor([env.reset()])
            test_episode_ERL_reward = 0.0
            for t in range(args.num_steps):
                # forward pass through the actor network
                action = test_actor_ERL.select_action(state, exploration=None)
                next_state, reward, done, _ = env.step(action.numpy()[0])
                test_episode_ERL_reward += reward

                next_state = torch.Tensor([next_state])
                state = next_state

                if done:
                    break
        test_episode_ERL_reward = np.mean(test_episode_ERL_reward)
        rewards_test_ERL.append(test_episode_ERL_reward)
        print("              EA Test Reward = " + str(test_episode_ERL_reward))

        '''
            ##################
            Run DDPG policy
            ##################
        '''
        # for j in range(3):
        #     state = torch.Tensor([env.reset()])
        #     test_episode_DDPG_reward = 0.0
        #     for t in range(args.num_steps):
        #         # forward pass through the actor network
        #         action = agent.select_action(state, exploration=None)
        #         next_state, reward, done, _ = env.step(action.numpy()[0])
        #         test_episode_DDPG_reward += reward
        #
        #         next_state = torch.Tensor([next_state])
        #         state = next_state
        #
        #         if done:
        #             break
        #test_episode_DDPG_reward = np.mean(test_episode_DDPG_reward)
        #rewards_test_DDPG.append(test_episode_DDPG_reward)
        #print("DDPG Test Reward = " + str(test_episode_DDPG_reward))

        ''' Print the training performance'''
#        print("Training: Episode: {}, noise: {}, reward: {}, average reward: {}".format(i_episode, ounoise.scale,
#                                                                                        rewards[-1],
#                                                                                        np.mean(rewards[-10:])))
        print()
        print()

    env.close()
    pickling_on = open("SwimmerV2_EVO_fitness_training_LN_final.p",
                       "wb")  # basically Shaw said this is not what you want to look at, instead see ERL testing rewards
    pickle.dump(evo.save_fitness, pickling_on)
    pickling_on.close()

    pickling_on = open("Swimmer_EVO_rewards_testing_LN_final.p", "wb")
    pickle.dump(rewards_test_ERL, pickling_on)
    pickling_on.close()

    # Save model
    torch.save(evo.best_policy.actor.state_dict(), 'params_monopod.pt')

    # torch.save(the_model.state_dict(), PATH)
