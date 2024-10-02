import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random 
from collections import deque
import matplotlib.pyplot as plt

class networkDefine(nn.Module):
    def __init__(self, state_size, action_size):
        super(networkDefine, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.memory.append(experience)

    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        return np.vstack(states), np.array(actions), np.array(rewards), np.vstack(next_states), np.array(dones)

    def __len__(self):
        return len(self.memory)
class dqnA:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99    
        self.epsilon = 1.0   
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update = 10 
        self.memory = ReplayBuffer(buffer_size=10000, batch_size=self.batch_size)

        self.qnetwork_local = networkDefine(state_size, action_size)
        self.qnetwork_target = networkDefine(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(np.arange(self.action_size))
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return np.argmax(action_values.numpy())

    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(torch.FloatTensor(next_states)).detach().max(1)[0].unsqueeze(1)
        Q_targets = torch.FloatTensor(rewards).unsqueeze(1) + (self.gamma * Q_targets_next * (1 - torch.FloatTensor(dones).unsqueeze(1)))

        Q_expected = self.qnetwork_local(torch.FloatTensor(states)).gather(1, torch.LongTensor(actions).unsqueeze(1))

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

def train_dqn(env, agent, n_episodes=1000, max_t=200):
    scores = []
    scores_window = deque(maxlen=100)

    for e in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            #env.render()
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        
        if e % 10 == 0:
            print(f"Episode {e}\tAverage Score: {np.mean(scores_window)}")

        if e % agent.target_update == 0:
            agent.update_target_network()

        if np.mean(scores_window) >= 195.0:
            print(f"Environment solved in {e} episodes!")
            break

    return scores

def plot_scores(scores):
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Progreso del entrenamiento')
    plt.show()

def evaluate_agent(env, agent, n_episodes=10):
    for i in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            env.render(mode="human")  
            action = agent.act(state)
            state, reward, done, _, _ = env.step(action)

    env.close()

env = gym.make('CartPole-v1', render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = dqnA(state_size, action_size)

scores = train_dqn(env, agent)
plot_scores(scores)
evaluate_agent(env, agent)
