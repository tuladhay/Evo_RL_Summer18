import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

def MSELoss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Actor(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ''' Lets see if LayerNorm will improve performance'''
        self.layerN1 = nn.LayerNorm(num_inputs)
        self.layerN2 = nn.LayerNorm(hidden_size)
        self.layerNmu = nn.LayerNorm(hidden_size)

        self.linear1 = nn.Linear(num_inputs, hidden_size)   # has 2 parameters: weights, biases

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.mul_(10)
        self.linear2.bias.data.mul_(10)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(10)
        self.mu.bias.data.mul_(10)

        print("num_actions = " + str(num_outputs))
        print("num_inputs = " + str(num_inputs))


    def forward(self, inputs):
        x = inputs
        x = self.layerN1(x)
        x = F.tanh(self.linear1(x))
        x = self.layerN2(x)
        x = F.tanh(self.linear2(x))

        mu = F.tanh(self.mu(x))
        return mu

    
class Critic(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]
        self.bn0 = nn.BatchNorm1d(num_inputs)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)

        '''
        Modifications to the code so that it uses the same layer units as the original paper
        Uncomment the code below and change the hidden_size in the nn.linear declarations
        '''
        # print("Critic hidden nodes modified to match the original paper")
        # critic_linear1_hidden_size = 200
        # critic_laction_hidden_size = 200
        # critic_linear2_hidden_size = 400
        # print("Critic linear1 hidden size = 200")
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)

        self.linear_action = nn.Linear(num_outputs, hidden_size)
        self.bn_a = nn.BatchNorm1d(hidden_size)
        self.bn_a.weight.data.fill_(1)
        self.bn_a.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        # Read experimental details section from Evolutionary Reinforcement Learning (ERL) paper
        x = inputs
        x = self.bn0(x)
        x = F.tanh(self.linear1(x))
        a = F.tanh(self.linear_action(actions))    # Actions were not included until the second hidden layer of Q
        x = torch.cat((x, a), 1)
        x = F.tanh(self.linear2(x))

        V = self.V(x)
        return V


class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=5e-5)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=5e-4)

        self.gamma = gamma
        self.tau = tau
        self.fitness = 0.0

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, exploration=None):
        self.actor.eval()    # Set the network in evaluation mode
        with torch.no_grad():
            mu = self.actor((Variable(state)))
            # since Actor class has only one function, I think it goes through the "forward"
            # Now it has passed through the actor>forward
        self.actor.train()    # Set the network in training model
        mu = mu.data
        if exploration is not None:
            mu += torch.Tensor(exploration.noise())

        return mu.clamp(-1, 1)


    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        with torch.no_grad():
            next_state_batch = Variable(torch.cat(batch.next_state))
        
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = torch.unsqueeze(reward_batch, 1)
        expected_state_action_batch = reward_batch + (self.gamma * next_state_action_values)	# line 12

        self.critic_optim.zero_grad()  # zero_grad, Clears the gradients of all optimized

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = MSELoss(state_action_batch, expected_state_action_batch)	# line 13
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()	# zero out the gradients

        policy_loss = -self.critic((state_batch),self.actor((state_batch)))	# line 14

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)	# line 15
        soft_update(self.critic_target, self.critic, self.tau)
