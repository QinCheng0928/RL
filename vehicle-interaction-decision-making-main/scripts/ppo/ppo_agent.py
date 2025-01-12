import torch
import torch.nn as nn
import torch.optim as optim
from ppo.ppo_network import Actor, Critic

class PPOAgent:
    def __init__(self, state_dim, action_dim, num_agents, lr, gamma, K_epochs, eps_clip):
        self.actor = Actor(state_dim, action_dim, num_agents)
        self.critic = Critic(state_dim, num_agents,action_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.MseLoss = nn.MSELoss()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.memory = Memory()

    def select_action(self, states):
        with torch.no_grad():
            action_probs = self.actor(states)
            action_list = torch.distributions.Categorical(action_probs)
            action = action_list.sample().item()
            
        return action

    def learn(self):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list to tensor
        old_states = torch.stack(self.memory.states).detach()
        old_actions = torch.stack(self.memory.actions).detach()
        old_logprobs = torch.stack(self.memory.logprobs).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            action_probs = self.actor(old_states)
            dist = torch.distributions.Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = self.critic(old_states)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.gather(1, old_actions.unsqueeze(1)).squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.gather(1, old_actions.unsqueeze(1)).squeeze(), rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
