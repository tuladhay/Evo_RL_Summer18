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
import matplotlib.pyplot as plt


def parse_arguments():
    global parser
    parser = argparse.ArgumentParser(description='PID tuning using evolutionary algorithm')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='name of the environment to run')
    parser.add_argument('--seed', type=int, default=4, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--num_episodes', type=int, default=100, metavar='N',
                        help='number of episodes (default: 1000)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--population_size', type=int, default=200, metavar='N',
                        help='population size (default: 100)')


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
        if self.num_elites < 1:
            self.num_elites = 1
        self.tournament_genes = int(self.tournament_percentage * self.num_actors)

        self.noise_mean = 0.0
        self.noise_stddev = 0.2

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
                    position = evo_state
                    evo_action = gene.compute_pd_output(position)
                    evo_next_state, evo_reward, evo_done, _ = env.step(evo_action)
                    evo_episode_reward += evo_reward

                    evo_next_state = torch.Tensor([evo_next_state])
                    evo_state = copy.copy(evo_next_state)
                    #
                    # if ep % 20 == 0:
                    #     env.render()

                    # env.render()

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

    evo = Evo(args.population_size, pd_target, delta)    # initializes population with the target and delta
    # make sure the pd_target is also manually copied in the reward function
    evo.initialize_gains()
    print("Population size = " + str(args.population_size))

    rewards = []  # recorded during training rollout
    rewards_testing = []  # This appends to a list so that the progress across episodes can be plotted

    print("Number of episodes : " + str(args.num_episodes))
    for i_episode in range(args.num_episodes):
        '''
        Here, num_episodes correspond to the generations in Algo 1.
        In every generation, the population is evaluated, ranked, mutated, and re-inserted into population
        '''
        evo.evaluate_pop()
        evo.rank_pop_selection_mutation()

        print("Episode: "+str(i_episode) + "    Fitness = " + str(evo.best_policy.fitness))

    env.close()

    # pickling_on = open("Springmass_evo_pd_v1_fitness.p", "wb")
    # pickle.dump(rewards_testing, pickling_on)
    # pickling_on.close()

    print("The best performing PD values were: ")
    print("Kp = " + str(evo.best_policy.kp) + "\nKd = " + str(evo.best_policy.kd))

    """
    ################
    TEST RUN
    ################
    """
    state = torch.Tensor([env.reset()])
    episode_reward = 0
    position_array = []  # for visualization
    target_array = []
    print("Num steps = " + str(args.num_steps))
    for t in range(1000):
        state = state.data.numpy()[0]
        position = state
        position_array.append(position)
        target_array.append(pd_target)

        action = evo.best_policy.compute_pd_output(position)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        next_state = torch.Tensor([next_state])
        state = copy.copy(next_state)

        # env.render()

    env.close()

    plt.plot(position_array)
    plt.plot(target_array)
    plt.title("PID response for Kp: " + str(evo.best_policy.kp) + " Kd: " + str(evo.best_policy.kd))
    plt.show()

    plt.plot(evo.save_fitness)
    plt.title("Best gene fitness vs epochs")
    plt.show()


