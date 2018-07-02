import argparse
import random
from operator import attrgetter

import gym
import numpy as np
import torch
from ddpg import DDPG
from test_xor import XOR

import copy


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
    parser.add_argument('--num_episodes', type=int, default=15000, metavar='N',
                        help='number of episodes (default: 1000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                        help='number of episodes (default: 128)')
    parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                        help='model updates per simulator step (default: 5)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')

class Evo():
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
                           2, np.array([1])) for _ in range(10)]
        print("Initializing Evolutionary Actors")
        self.evo_episodes = evo_episodes
        self.elite_percentage = 0.1
        self.num_elites = int(self.elite_percentage*self.num_actors)
        self.tournament_genes = 3   # TODO: make it a percentage

        self.noise_mean = 0.0
        self.noise_stddev = 0.1

        self.save_fitness = []
        self.best_policy = copy.deepcopy(self.population[0])    # for saving policy purposes

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
            for _ in range(100):
                state = torch.Tensor(problem.current_state)
                episode_reward = 0
                for t in range(args.num_steps):
                    action = gene.select_action(state)
                    one = 1.0
                    zero = 0.0
                    if action > 0.5:
                        action = 1
                    else:
                        action = 0
                    next_state, reward, done = problem.step(action)
                    episode_reward += reward

                    state = next_state

                    if done:
                        break
                    # <end of time-steps>
                fitness.append(episode_reward)
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
        self.best_policy = elites[0]    # for saving policy purposes
        set_s = []

        for i in range(len(ranked_pop)-len(elites)):
            tournament_genes = [random.choice(ranked_pop) for _ in range(self.tournament_genes)]
            tournament_winner = max(tournament_genes, key=attrgetter('fitness'))
            set_s.append(copy.deepcopy(tournament_winner))

        # print("Before mutation")
        # print(set_s[0].actor.mu.bias[:5])

        mutated_set_S = copy.deepcopy(self.mutation(set_s))

        # print("After mutation")
        # print(mutated_set_S[0].actor.mu.bias[:5])
        # print()

        self.population = []
        # Addition of lists
        self.population = copy.deepcopy(elites + mutated_set_S)
        print("Best fitness = " + str(elites[0].fitness))

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

        return set_s




if __name__ == "__main__":
    parse_arguments()
    args = parser.parse_args()

    '''
    DEFINE THE ACTOR RL AGENT
    '''
    agent = DDPG(args.gamma, args.tau, 5, 2, np.array([1]))    # hidden=5, obs_space=2, action_space=1
    print("Initialized DDPG actor")

    '''
    DEFINE REPLAY BUFFER AND NOISE
    '''

    '''
    Initialize the Evolution Part
    '''
    problem = XOR()
    problem.reset()

    evo = Evo(10)
    evo.initialize_fitness()

    # TODO: MOVE THE TRAINING CODE BELOW TO ITS RESPECTIVE FUNCTIONS
    rewards = []

    for i_episode in range(1000):
        '''
        Here, num_episodes correspond to the generations in Algo 1.
        In every generation, the population is evaluated, ranked
        '''
        evo.evaluate_pop()
        evo.rank_pop_selection_mutation()

        print("Episode: " + str(i_episode))



