import configparser
import os
import pickle
import random
import sys
from collections import deque

import gym
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self, state_space, actions):
        super(ANN, self).__init__()
        # inputs state space of Lunar Lander
        # outputs actions of Lunar Lander
        self.hidden_nodes = int(config.get('NN-Config', 'hidden_nodes', fallback=64))
        self.in_layer = nn.Linear(state_space, self.hidden_nodes)
        self.hidden_layer_1 = nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.out_layer = nn.Linear(self.hidden_nodes, actions)

    def forward(self, state):
        data = F.relu(self.in_layer(state))
        data = F.relu(self.hidden_layer_1(data))
        return self.out_layer(data)


class DQNLearningAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.device_type = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.main_nn = ANN(self.state_space, self.action_space).to(self.device_type)
        self.learning_rate = float(config.get('DQN-Agent', 'learning_rate', fallback=1e-3))
        self.discount_factor = float(config.get('DQN-Agent', 'discount_factor', fallback=0.99))
        self.target_nn = ANN(self.state_space, self.action_space).to(self.device_type)
        self.optimizer = torch.optim.Adam(self.main_nn.parameters(), lr=self.learning_rate)
        self.replay_memory = ExperienceMemory(self.device_type)
        self.soft_update_factor = float(config.get('DQN-Agent', 'soft_update_factor', fallback=1e-3))

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_memory.add_experience(state, action, reward, next_state, done)

    def sample_and_learn(self):
        if len(self.replay_memory) > self.replay_memory.sample_size:
            self.learn_from_experiences(self.replay_memory.sample_experiences(), self.discount_factor)

    def epsilon_greedy_action(self, current_state, epsilon=0.):
        current_state = torch.from_numpy(current_state).float().unsqueeze(0).to(self.device_type)
        self.main_nn.eval()
        with torch.no_grad():  # stop gradient computation
            qs = self.main_nn(current_state)  # fetch all q values for the state
        self.main_nn.train()
        if random.random() < epsilon:
            return random.choice(np.arange(self.action_space))
        else:
            return np.argmax(qs.cpu().data.numpy())

    def learn_from_experiences(self, experiences, discount_factor):
        states, actions, rewards, state_primes, done = experiences
        q_hat = self.target_nn(state_primes).detach().max(1)[0].unsqueeze(1)
        target_qs = rewards + (discount_factor * q_hat * (1 - done))
        expected_qs = self.main_nn(states).gather(1, actions)
        self.optimizer.zero_grad()
        loss_mse = F.mse_loss(target_qs, expected_qs)
        loss_mse.backward()
        self.optimizer.step()
        # soft update target network from main network
        for target_param, local_param in zip(self.target_nn.parameters(), self.main_nn.parameters()):
            target_param.data.copy_(
                self.soft_update_factor * local_param.data + (1.0 - self.soft_update_factor) * target_param.data)


class ExperienceMemory(object):
    def __init__(self, device_type):
        self.memory_capacity = int(config.get('Experience-Memory', 'memory-capacity', fallback=1e5))
        self.sample_size = int(config.get('Experience-Memory', 'sample_size', fallback=64))
        self.device_type = device_type
        self.experience_buffer = deque(maxlen=self.memory_capacity)

    def add_experience(self, s, a, r, s_prime, done):
        self.experience_buffer.append((s, a, r, s_prime, done))

    def __len__(self):
        return len(self.experience_buffer)

    def sample_experiences(self):
        random_experiences = random.sample(self.experience_buffer, k=self.sample_size)
        e_states, e_actions, e_rewards, e_next_states, e_done = [], [], [], [], []
        for experience in random_experiences:
            e_states.append(experience[0])
            e_actions.append(experience[1])
            e_rewards.append(experience[2])
            e_next_states.append(experience[3])
            e_done.append(experience[4])
        # convert numpy to tensors
        e_states = torch.from_numpy(np.array(e_states)).float().to(self.device_type)
        e_actions = torch.from_numpy(np.reshape(e_actions, (self.sample_size, -1))).long().to(self.device_type)
        e_rewards = torch.from_numpy(np.reshape(e_rewards, (self.sample_size, -1))).float().to(self.device_type)
        e_next_states = torch.from_numpy(np.array(e_next_states)).float().to(self.device_type)
        e_done = torch.from_numpy(np.reshape(e_done, (self.sample_size, -1)).astype(np.uint8)).float().to(
            self.device_type)
        return e_states, e_actions, e_rewards, e_next_states, e_done


def train_dqn_agent(save_model=False):
    env = gym.make('LunarLander-v2')
    num_state_features = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('No. of state features: {}'.format(num_state_features))
    print('No. of actions: {}'.format(num_actions))
    agent = DQNLearningAgent(num_state_features, num_actions)
    epsilon_start = float(config.get('DQN-Agent', 'epsilon_start', fallback=1.0))
    epsilon_end = float(config.get('DQN-Agent', 'epsilon_end', fallback=.01))
    epsilon_decay = float(config.get('DQN-Agent', 'epsilon_decay', fallback=.99))
    update_weights_frequency = int(config.get('DQN-Agent', 'update_weights_frequency', fallback=2))

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # queue to hold last 100 scores
    current_epsilon = epsilon_start
    current_frame = 0
    for episode in range(1, n_episodes + 1):
        current_state = env.reset()
        score = 0
        for t in range(max_steps_per_episode):
            a_t = agent.epsilon_greedy_action(current_state, current_epsilon)
            s_prime_t, r_t, done, _ = env.step(a_t)
            agent.store_transition(current_state, a_t, r_t, s_prime_t, done)
            current_state = s_prime_t
            score += r_t
            current_frame = (current_frame + 1) % update_weights_frequency
            if current_frame == 0:
                agent.sample_and_learn()
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        current_epsilon = max(epsilon_end, epsilon_decay * current_epsilon)  # decrease current_epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {:d} - {:d}\tAverage Score: {:.2f}'.format(episode - 100, episode, np.mean(scores_window)))
        if episode % 300 == 0 and save_model:
            torch.save(agent.main_nn.state_dict(), 'model/snapshot_episode_' + str(episode) + '.pth')

    # save the trained model
    if save_model:
        torch.save(agent.main_nn.state_dict(), 'model/snapshot_episode_final.pth')
        pickle.dump(scores, open("results/scores_train.p", "wb"))
        plot_training_graphs()
    env.close()
    return scores


def plot_training_graphs():
    learning_rate = float(config.get('DQN-Agent', 'learning_rate', fallback=1e-3))
    discount_factor = float(config.get('DQN-Agent', 'discount_factor', fallback=0.99))
    epsilon_decay = float(config.get('DQN-Agent', 'epsilon_decay', fallback=0.99))
    sample_size = int(config.get('Experience-Memory', 'sample_size', fallback=64))
    replay_memory_capacity = config.get('Experience-Memory', 'memory_capacity', fallback=64)
    lr_patch = mpatches.Patch(label=r"$\alpha$=" + str(learning_rate), color='green')
    gamma_patch = mpatches.Patch(label=r"$\gamma$=" + str(discount_factor), color='green')
    epsilon_decay_patch = mpatches.Patch(label='epsilon_decay=' + str(epsilon_decay), color='green')
    sample_size_patch = mpatches.Patch(label='sample_size=' + str(sample_size), color='green')
    replay_memory_capacity_patch = mpatches.Patch(label='replay memory capacity=' + str(replay_memory_capacity),
                                                  color='green')
    train_scores = pickle.load(open("results/scores_train.p", "rb"))
    scores_df = pd.DataFrame(train_scores)
    scores_df['rolling_mean'] = scores_df[scores_df.columns[0]].rolling(100).mean()
    scores_df['Episodes'] = np.arange(scores_df.shape[0])
    ax = plt.gca()
    ax.set_title('Training Rewards')
    plt.ylabel("Rewards")
    colors = ['tab:blue', 'red']
    scores_df.plot(color=colors, kind='line', x='Episodes', linewidth=2, ax=ax)
    plt.legend(handles=[lr_patch, gamma_patch, epsilon_decay_patch, replay_memory_capacity_patch, sample_size_patch],
               loc='lower right')
    plt.savefig('{}/training_scores.png'.format('graphs'), dpi=300)
    plt.show()


def test_trained_agent():
    # test the trained agent
    env = gym.make('LunarLander-v2')
    num_state_features = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('No. of state features: {}'.format(num_state_features))
    print('No. of actions: {}'.format(num_actions))
    agent = DQNLearningAgent(num_state_features, num_actions)
    agent.main_nn.load_state_dict(torch.load('model/snapshot_episode_final.pth'))
    scores_window = deque(maxlen=100)
    scores = []
    for episode in range(1, 100 + 1):
        score = 0
        current_state = env.reset()
        done = False
        while not done:
            a_t = agent.epsilon_greedy_action(current_state)
            s_prime_t, r_t, done, _ = env.step(a_t)
            score += r_t
            current_state = s_prime_t
            env.render()
        scores.append(score)
        scores_window.append(score)
    print('\rEpisode {:d} - {:d}\tAverage Score: {:.2f}'.format(0, 100, np.mean(scores_window)))
    pickle.dump(scores, open("results/scores_test.p", "wb"))
    env.close()
    plot_testing_graphs()


def plot_testing_graphs():
    train_scores = pickle.load(open("results/scores_test.p", "rb"))
    mean_score = mpatches.Patch(label="Mean Score=" + "{:.2f}".format(np.mean(train_scores)), color='green')
    scores_df = pd.DataFrame(train_scores)
    scores_df['rolling_mean'] = scores_df[scores_df.columns[0]].mean()
    scores_df['Episodes'] = np.arange(scores_df.shape[0])
    ax = plt.gca()
    ax.set_title('Testing Rewards')
    plt.ylabel("Rewards")
    colors = ['tab:blue', 'red']
    scores_df.plot(color=colors, kind='line', x='Episodes', linewidth=2, ax=ax)
    plt.legend(handles=[mean_score], loc='upper right')
    plt.savefig('{}/testing_scores.png'.format('graphs'), dpi=300)
    plt.show()


def train_different_lr():
    lrs = [1e-5 * (10 ** i) for i in range(5)]
    scores_lr = {}
    for lr in lrs:
        print("learning rate: {:.6f}\n".format(lr), end="")
        config.read('hyper-parameters.ini')
        config['DQN-Agent']['learning_rate'] = str(lr)
        scores_lr[lr] = train_dqn_agent()
    pickle.dump(scores_lr, open("results/scores_lr.p", "wb"))
    plot_scores_2_learning_rate()


def plot_scores_2_learning_rate():
    lrs = [1e-5 * (10 ** i) for i in range(5)]
    scores_lr = pickle.load(open("results/scores_lr.p", "rb"))
    scores_lr_pd = None
    for lr in lrs:
        df = pd.DataFrame(pd.Series(list(scores_lr[lr])).rolling(100).mean(),
                          columns=[r"$\alpha$=" + '{:.0e}'.format(lr)])
        scores_lr_pd = pd.concat([scores_lr_pd, df], axis=1)
    scores_lr_pd = scores_lr_pd.fillna(method='ffill')
    scores_lr_pd['Episodes'] = np.arange(scores_lr_pd.shape[0])
    ax = plt.gca()
    ax.set_title('100 Episodes Moving Average')
    plt.ylabel("Rewards")
    scores_lr_pd.plot(kind='line', x='Episodes', y=scores_lr_pd.columns[0], ax=ax)
    scores_lr_pd.plot(kind='line', x='Episodes', y=scores_lr_pd.columns[1], ax=ax)
    scores_lr_pd.plot(kind='line', x='Episodes', y=scores_lr_pd.columns[2], ax=ax)
    scores_lr_pd.plot(kind='line', x='Episodes', y=scores_lr_pd.columns[3], ax=ax)
    scores_lr_pd.plot(kind='line', x='Episodes', y=scores_lr_pd.columns[4], ax=ax)
    plt.savefig('{}/hp_learning_rate.png'.format('graphs'), dpi=300)
    plt.show()


def train_different_discount_rate():
    scores_gamma = {}
    for gamma in np.linspace(0.5, 0.99, num=5):
        print("discount rate: {:.3f}\n".format(gamma), end="")
        config.read('hyper-parameters.ini')
        config['DQN-Agent']['discount_factor'] = str(gamma)
        scores_gamma[gamma] = train_dqn_agent()
    pickle.dump(scores_gamma, open("results/scores_gamma.p", "wb"))
    plot_scores_2_discount_rate()


def plot_scores_2_discount_rate():
    scores_discount_factor = pickle.load(open("results/scores_gamma.p", "rb"))
    scores_discount_factor_pd = None
    for discount in np.linspace(0.5, 0.99, num=5):
        df = pd.DataFrame(pd.Series(list(scores_discount_factor[discount])).rolling(100).mean(),
                          columns=[r"$\gamma$=" + '{:.3f}'.format(discount)])
        scores_discount_factor_pd = pd.concat([scores_discount_factor_pd, df], axis=1)
    scores_discount_factor_pd = scores_discount_factor_pd.fillna(method='ffill')
    scores_discount_factor_pd['Episodes'] = np.arange(scores_discount_factor_pd.shape[0])
    ax = plt.gca()
    ax.set_title('100 Episodes Moving Average')
    plt.ylabel("Rewards")
    scores_discount_factor_pd.plot(kind='line', x='Episodes', y=scores_discount_factor_pd.columns[0], ax=ax)
    scores_discount_factor_pd.plot(kind='line', x='Episodes', y=scores_discount_factor_pd.columns[1], ax=ax)
    scores_discount_factor_pd.plot(kind='line', x='Episodes', y=scores_discount_factor_pd.columns[2], ax=ax)
    scores_discount_factor_pd.plot(kind='line', x='Episodes', y=scores_discount_factor_pd.columns[3], ax=ax)
    scores_discount_factor_pd.plot(kind='line', x='Episodes', y=scores_discount_factor_pd.columns[4], ax=ax)
    plt.savefig('{}/hp_discount_rate.png'.format('graphs'), dpi=300)
    plt.show()


def train_different_epsilon_decays():
    scores_epsilon_decays = {}
    epsilon_decays = [0.7, 0.8, 0.9, 0.99, 0.999]
    for epsilon_decay in epsilon_decays:
        print("epsilon decay: {}\n".format(epsilon_decay), end="")
        config.read('hyper-parameters.ini')
        config['DQN-Agent']['epsilon_decay'] = str(epsilon_decay)
        scores_epsilon_decays[epsilon_decay] = train_dqn_agent()
    pickle.dump(scores_epsilon_decays, open("results/scores_epsilon_decays.p", "wb"))
    plot_scores_2_epsilon_decays_graph()


def plot_scores_2_epsilon_decays_graph():
    epsilon_decays = [0.7, 0.8, 0.9, 0.99, 0.999]
    scores_epsilon_decays = pickle.load(open("results/scores_epsilon_decays.p", "rb"))
    scores_epsilon_decays_pd = None
    for epsilon_decay in epsilon_decays:
        df = pd.DataFrame(pd.Series(list(scores_epsilon_decays[epsilon_decay])).rolling(100).mean(),
                          columns=["Epsilon Decay=" + str(epsilon_decay)])
        scores_epsilon_decays_pd = pd.concat([scores_epsilon_decays_pd, df], axis=1)
    scores_epsilon_decays_pd = scores_epsilon_decays_pd.fillna(method='ffill')
    scores_epsilon_decays_pd['Episodes'] = np.arange(scores_epsilon_decays_pd.shape[0])
    ax = plt.gca()
    ax.set_title('100 Episodes Moving Average')
    plt.ylabel("Rewards")
    scores_epsilon_decays_pd.plot(kind='line', x='Episodes', y=scores_epsilon_decays_pd.columns[0], ax=ax)
    scores_epsilon_decays_pd.plot(kind='line', x='Episodes', y=scores_epsilon_decays_pd.columns[1], ax=ax)
    scores_epsilon_decays_pd.plot(kind='line', x='Episodes', y=scores_epsilon_decays_pd.columns[2], ax=ax)
    scores_epsilon_decays_pd.plot(kind='line', x='Episodes', y=scores_epsilon_decays_pd.columns[3], ax=ax)
    scores_epsilon_decays_pd.plot(kind='line', x='Episodes', y=scores_epsilon_decays_pd.columns[4], ax=ax)
    plt.savefig('{}/hp_epsilon_decay.png'.format('graphs'), dpi=300)
    plt.show()


if __name__ == '__main__':
    artifact_dirs = ['model', 'results', 'graphs']
    for directory in artifact_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    config = configparser.ConfigParser()
    config.read('hyper-parameters.ini')
    n_episodes = 1500
    max_steps_per_episode = 1000
    # script options
    # 1 - Train the agent
    # 2 - Test the trained agent
    # 3 - Train with different learning rates
    # 4 - Train with different discount rates
    # 5 - Train with different epsilon decays
    if len(sys.argv) > 1:
        option = sys.argv[1]
        if option == '1':
            train_dqn_agent(True)
        elif option == '2':
            test_trained_agent()
        elif option == '3':
            train_different_lr()
        elif option == '4':
            train_different_discount_rate()
        elif option == '5':
            train_different_epsilon_decays()
    else:
        # nothing is passed just train the agent
        train_dqn_agent(True)
