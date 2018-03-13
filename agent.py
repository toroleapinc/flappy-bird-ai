"""DQN Agent."""
import math
import numpy as np
import torch
import torch.optim as optim
from dqn import DQN, DuelingDQN
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class DQNAgent:
    def __init__(self, input_size=4, n_actions=2, hidden=256, gamma=0.99,
                 lr=1e-4, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=80000,
                 batch_size=32, target_update=1000, use_dueling=True, use_per=True):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        Model = DuelingDQN if use_dueling else DQN
        self.policy_net = Model(input_size, hidden, n_actions).to(self.device)
        self.target_net = Model(input_size, hidden, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer() if use_per else ReplayBuffer()

    @property
    def epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-self.steps / self.epsilon_decay)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q = self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            return q.argmax(dim=1).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        if isinstance(self.memory, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)

        td_error = (current_q - target_q).abs()
        loss = (weights * (current_q - target_q) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        if isinstance(self.memory, PrioritizedReplayBuffer):
            self.memory.update_priorities(indices, td_error.detach().cpu().numpy())
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()
