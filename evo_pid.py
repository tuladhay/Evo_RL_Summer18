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
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from replay_memory import ReplayMemory, Transition
import pickle
from pid import PD
import random


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
    parser.add_argument('--num_steps', type=int, default=10000, metavar='N',
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
    def __init__(self, num_evo_actors, target, delta):
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
        # PD initialization takes(self, target, delta)
        self.population = [PD(target, delta) for _ in range(self.num_actors)]
        print("Initializing Evolutionary PD Controllers")
        self.evo_episodes = 1
        self.elite_percentage = 0.1
        self.tournament_percentage = 0.25
        self.num_elites = int(self.elite_percentage * self.num_actors)
        self.tournament_genes = int(self.tournament_percentage * self.num_actors)

        self.noise_mean = 0.0
        self.noise_stddev = 0.1

        self.save_fitness = []
        self.best_policy = copy.deepcopy(self.population[0])  # for saving policy purposes

    def initialize_gains(self):
        for gene in self.population:
            gene.kp = random.uniform(0, 1000)
            gene.kd = random.uniform(0, 200)
        print("Initialized gene gains")

    def evaluate_pop(self):
        for gene in self.population:
            fitness = []
            for ep in range(self.evo_episodes):
                # evo_state = torch.Tensor([env.reset()])
                evo_state = torch.Tensor([env.reset()])
                evo_episode_reward = 0
                for t in range(args.num_steps):
                    evo_state = evo_state.data.numpy()[0]
                    position = evo_state.data[0]
                    evo_action = gene.compute_pd_output(position)
                    evo_next_state, evo_reward, evo_done, _ = env.step(evo_action)
                    evo_episode_reward += evo_reward

                    # evo_action = torch.Tensor(evo_action)
                    # evo_mask = torch.Tensor([not evo_done])
                    # evo_mask = torch.Tensor([not evo_done])
                    evo_next_state = torch.Tensor([evo_next_state])
                    # evo_reward = torch.Tensor([evo_reward])

                    # memory.push(evo_state, evo_action, evo_mask, evo_next_state, evo_reward)
                    evo_state = copy.copy(evo_next_state)
                    #
                    # if ep % 20 == 0:
                    #     env.render()
                    env.render()

                    if evo_done:
                        print("Done")
                        break
                    # evo_state = evo_state.data.numpy()[0]
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

        mutated_set_s = copy.deepcopy(self.mutation(set_s))
        self.population = []
        # Addition of lists
        self.population = copy.deepcopy(elites + mutated_set_s)
        self.save_fitness.append(elites[0].fitness)

    def mutation(self, set_s):
        """
        :param set_s: This is the set of (k-e) genes that are going to be mutated by adding noise
        :return: Returns the mutated set of (k-e) genes

        Adds noise to the weights and biases of each layer of the network
        """
        for i in range(len(set_s)):
            ''' Noise to Kp'''
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev)
            noise = set_s[i].kp*noise
            set_s[i].kp = copy.deepcopy(set_s[i].kp + noise)

            ''' Noise to Kd'''
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev)
            noise = set_s[i].kd*noise
            set_s[i].kd = copy.deepcopy(set_s[i].kd + noise)

        return set_s


if __name__ == "__main__":
    parse_arguments()
    args = parser.parse_args()
    args.env_name = "Springmass-v0"
    print("Running environment" + str(args.env_name))

    env = gym.make(args.env_name)
    # env = wrappers.Monitor(env, '/tmp/{}-experiment'.format(args.env_name), force=True)
    env.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Control
    pd_target = 8.0
    delta = 0.005

    '''
    #############################
    Initialize the Evolution Part
    #############################
    '''
    evo = Evo(10, pd_target, delta)    # initializes population with the target and delta
    # make sure the pd_target is also manually copied in the reward function
    evo.initialize_gains()

    # TODO: MOVE THE TRAINING CODE BELOW TO ITS RESPECTIVE FUNCTIONS
    rewards = []  # during training
    rewards_testing = []  # This appends to a list so that the progress across episodes can be plotted

    print("Number of episodes : " + str(args.num_episodes))
    for i_episode in range(args.num_episodes):
        '''
        Here, num_episodes correspond to the generations in Algo 1.
        In every generation, the population is evaluated, ranked, mutated, and re-instered into population
        '''
        evo.evaluate_pop()
        evo.rank_pop_selection_mutation()

        print("Episode: "+str(i_episode) + "    Training Fitness = " + str(evo.best_policy.fitness))

    env.close()

    # pickling_on = open("Springmass_evo_pd_v1_fitness.p", "wb")
    # pickle.dump(rewards_testing, pickling_on)
    # pickling_on.close()

    print("The best performing PD values were: ")
    print("Kp = " + str(evo.best_policy.kp) + "\nKd = " + str(evo.best_policy.kd))
