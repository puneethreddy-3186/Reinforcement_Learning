import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
from scipy.linalg import block_diag

solvers.options['show_progress'] = False


class Player:
    def __init__(self, row_pos, col_pos, player_name=None):
        self.player_name = player_name
        self.row_pos = row_pos
        self.col_pos = col_pos

    def update_state(self, row_pos, col_pos):
        self.row_pos = row_pos
        self.col_pos = col_pos

    def get_grid_position(self):
        return np.array([self.row_pos, self.col_pos])


# Soccer Environment
class GridWorld:
    def __init__(self):
        self.player_list = None
        self.goal_columns = None
        self.action_dict = None
        self.ball_possession = None
        # ['N'=0,'S'=1,'E'=2,'W'=3,'ST'=4]
        self.action_dict = {0: [-1, 0], 1: [1, 0], 2: [0, 1], 3: [0, -1], 4: [0, 0]}
        self.action_space = len(self.action_dict)
        # 8 grid positions each for player 1 and 2, 2 for ball possessions
        self.state_space = (8, 8, 2)

    def reset(self):
        # Initial player positions as per Figure 4 of the paper
        self.player_list = [Player(0, 2, 'A'), Player(0, 1, 'B')]
        self.ball_possession = 1
        self.goal_columns = {self.player_list[0].player_name: 0, self.player_list[1].player_name: 3}
        return self.state()

    def state(self):
        return [self.player_list[0].row_pos * 4 + self.player_list[0].col_pos,
                self.player_list[1].row_pos * 4 + self.player_list[1].col_pos, self.ball_possession]

    def move_player(self, new_positions, actions, index_1, index_2):
        new_positions[index_1] = self.player_list[index_1].get_grid_position() + self.action_dict[actions[index_1]]
        if (new_positions[index_1] == self.player_list[index_2].get_grid_position()).all():
            # if 1st mover possess ball, the ball is lost to 2nd mover
            if self.ball_possession == index_1:
                self.ball_possession = index_2

        # no collision, update player's  pos
        elif GridWorld.is_valid_position(new_positions[index_1]):
            self.player_list[index_1].update_state(new_positions[index_1][0], new_positions[index_1][1])
            player1 = self.player_list[index_1]
            player2 = self.player_list[index_2]
            if player1.col_pos == self.goal_columns[player1.player_name] and self.ball_possession == index_1:
                rewards = ([1, -1][index_1]) * np.array([100, -100])
                done = 1
                return self.state(), rewards, done
            elif player1.col_pos == self.goal_columns[player2.player_name] and self.ball_possession == index_1:
                rewards = ([1, -1][index_1]) * np.array([-100, 100])
                done = 1
                return self.state(), rewards, done
        return None, None, None

    @staticmethod
    def is_valid_position(player_grid_pos):
        return player_grid_pos[0] in range(0, 2) and player_grid_pos[1] in range(0, 4)

    def step(self, actions):
        # randomly choose which player to move first
        f_move_index = np.random.choice([0, 1], 1)[0]
        s_move_index = 1 - f_move_index
        new_positions = [self.player_list[0].get_grid_position(), self.player_list[1].get_grid_position()]
        unknown_actions = [e for e in actions if e not in self.action_dict]
        if len(unknown_actions) > 0:
            return self.state(), np.array([0, 0]), 0
        else:
            # move 1st player
            next_state, rewards, done = self.move_player(new_positions, actions, f_move_index, s_move_index)
            if next_state is None:
                # move 2nd player
                next_state, rewards, done = self.move_player(new_positions, actions, s_move_index, f_move_index)
            if next_state is None:
                next_state, rewards, done = self.state(), np.array([0, 0]), 0
        return next_state, rewards, done


class SolverQLearning:
    def __init__(self):
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999993
        self.alpha = 1.0
        self.alpha_min = 0.001
        self.alpha_decay = 0.999993

    @staticmethod
    def epsilon_greedy_action(Q, state, epsilon, action_list):
        if np.random.random() < epsilon:
            return np.random.choice(action_list, 1)[0]
        return np.random.choice(
            np.flatnonzero(Q[state[0]][state[1]][state[2]] == Q[state[0]][state[1]][state[2]].max()))

    def solve(self):
        print("Q-learner Algorithm")
        q_diff_values = []
        env = GridWorld()
        dim_q = np.concatenate((env.state_space, [env.action_space]))
        q_1 = np.zeros(dim_q)
        q_2 = np.zeros(dim_q)
        count = 0
        start_time = time.time()
        action_list = list(env.action_dict.keys())
        while count < n_episodes:
            state = env.reset()
            while True:
                if count % 1000 == 0:
                    msg = '\rEpisode {}\t Time: {:.2f}\t Epsilon: {:.3f}\t Alpha: {:.3f}'.format(
                        count,
                        time.time() - start_time,
                        self.epsilon,
                        self.alpha)
                    print(msg, end="")

                # Q value of player A at state S take action South before update
                before = q_1[2][1][1][1]
                # eps-greedy to generate action
                actions = [self.epsilon_greedy_action(q_1, state, self.epsilon, action_list),
                           self.epsilon_greedy_action(q_2, state, self.epsilon, action_list)]
                # get next state, reward and game termination flag
                next_state, rewards, done = env.step(actions)
                count += 1
                # update Q values
                q_1_current_state = q_1[state[0]][state[1]][state[2]][actions[0]]
                q_1_next_state_all_actions = q_1[next_state[0]][next_state[1]][next_state[2]]
                q_1[state[0]][state[1]][state[2]][actions[0]] = (1 - self.alpha) * q_1_current_state + self.alpha * (
                        rewards[0] + self.gamma * np.max(q_1_next_state_all_actions) * (1 - done))
                q_2_current_state = q_2[state[0]][state[1]][state[2]][actions[1]]
                q_2_next_state_all_actions = q_2[next_state[0]][next_state[1]][next_state[2]]
                q_2[state[0]][state[1]][state[2]][actions[1]] = (1 - self.alpha) * q_2_current_state + self.alpha * (
                        rewards[1] + self.gamma * np.max(q_2_next_state_all_actions) * (1 - done))

                after = q_1[2][1][1][1]
                q_diff_values.append(abs(after - before))
                if done:
                    break

                state = next_state
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)

                self.alpha *= self.alpha_decay
                self.alpha = max(self.alpha_min, self.alpha)
        pickle.dump(q_diff_values, open("results/Q-learner.p", "wb"))
        error_plot(np.array(q_diff_values), 'Q-learner')
        return q_diff_values


class SolverFriendQ:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999993
        self.alpha = 1.0
        self.alpha_min = 0.001
        self.alpha_decay = 0.999993

    @staticmethod
    def epsilon_greedy_action(Q, state, epsilon, action_list):
        if np.random.random() < epsilon:
            return np.random.choice(action_list, 1)[0]
        max_idx = np.where(Q[state[0]][state[1]][state[2]] == np.max(Q[state[0]][state[1]][state[2]]))
        return max_idx[1][np.random.choice(range(len(max_idx[0])), 1)[0]]

    def solve(self):
        print("Friend-Q Algorithm")
        np.random.seed(23456)
        q_diff_values = []
        env = GridWorld()
        dim_q = np.concatenate((env.state_space, [env.action_space, env.action_space]))
        q_1 = np.ones(dim_q)
        q_2 = np.ones(dim_q)
        count = 0
        start_time = time.time()
        action_list = list(env.action_dict.keys())
        while count < n_episodes:
            state = env.reset()
            while True:
                if count % 1000 == 0:
                    msg = '\rEpisode {}\t Time: {:.2f}\t Epsilon: {:.3f} \t Alpha: {:.3f}'.format(
                        count,
                        time.time() - start_time,
                        self.epsilon,
                        self.alpha)
                    print(msg, end="")

                # Q value of player B sticking and player A's action towards South before update
                before = q_1[2][1][1][4][1]
                # eps-greedy to generate action
                actions = [self.epsilon_greedy_action(q_1, state, self.epsilon, action_list),
                           self.epsilon_greedy_action(q_2, state, self.epsilon, action_list)]
                # get next state, reward and game termination flag
                next_state, rewards, done = env.step(actions)
                count += 1
                # update Q values
                q_1_current_state = q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]]
                q_1_next_state_all_actions = q_1[next_state[0]][next_state[1]][next_state[2]]
                q1_target = rewards[0] + ((1 - done) * self.gamma * np.max(q_1_next_state_all_actions))
                q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (1 - self.alpha) * q_1_current_state \
                                                                            + self.alpha * q1_target
                q_2_current_state = q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]]
                q_2_next_state_all_actions = q_2[next_state[0]][next_state[1]][next_state[2]]
                q2_target = rewards[1] + ((1 - done) * self.gamma * np.max(q_2_next_state_all_actions))
                q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (1 - self.alpha) * q_2_current_state \
                                                                            + self.alpha * q2_target

                # Q value of player B sticking and player A's action towards South after update
                after = q_1[2][1][1][4][1]
                q_diff_values.append(abs(after - before))
                if done:
                    break

                state = next_state
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)
                self.alpha *= self.alpha_decay
                self.alpha = max(self.alpha_min, self.alpha)

        pickle.dump(q_diff_values, open("results/Friend-Q.p", "wb"))
        error_plot(q_diff_values, 'Friend-Q')
        return q_diff_values


class SolverFoeQ:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999993
        self.alpha = 1.0
        self.alpha_min = 0.001
        self.alpha_decay = 0.999993

    @staticmethod
    def epsilon_greedy_action(pi, state, epsilon, action_list):
        if np.random.random() < epsilon:
            return np.random.choice(action_list, 1)[0]
        else:
            return np.random.choice(action_list, 1, p=pi[state[0]][state[1]][state[2]])[0]

    @staticmethod
    def max_min_solver(q, state, action_space):
        # cvxopt minimizes matrix c
        c = [-1] + [0 for i in range(action_space)]
        c = matrix(np.array(c, dtype="float"))

        # constraints G*x <= h
        G = np.array(q[state[0]][state[1]][state[2]], dtype="float") * -1
        G = np.vstack([G, -np.eye(action_space)])
        utility_col = [1 for i in range(action_space)] + [0 for i in range(action_space)]
        G = matrix(np.insert(G, 0, utility_col, axis=1))
        h = ([0 for i in range(2 * action_space)])
        h = matrix(np.array(h, dtype="float"))

        # contraints Ax = b
        A = [0.0] + [1.0 for i in range(action_space)]
        A = matrix([[i] for i in A])
        b = matrix(1.0)
        try:
            sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver='glpk', options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})
            if sol['x'] is not None:
                prob = np.abs(sol['x'][1:]).reshape((5,)) / sum(np.abs(sol['x'][1:]))
                val = np.array(sol['x'][0])
            else:
                prob = None
                val = None
        except:
            prob = None
            val = None
        return prob, val

    def solve(self):
        print("Foe-Q Algorithm")
        q_diff_values = []
        env = GridWorld()
        dim_q = np.concatenate((env.state_space, [env.action_space, env.action_space]))
        q_1 = np.ones(dim_q)
        q_2 = np.ones(dim_q)
        dim_pi = np.concatenate((env.state_space, [env.action_space]))
        pi_1 = np.ones(dim_pi) / env.action_space
        pi_2 = np.ones(dim_pi) / env.action_space
        dim_v = env.state_space
        v_1 = np.ones(dim_v)
        v_2 = np.ones(dim_v)
        count = 0
        start_time = time.time()
        action_list = list(env.action_dict.keys())
        while count < n_episodes:
            state = env.reset()
            while True:
                if count % 1000 == 0:
                    msg = '\rEpisode {}\t Time: {:.2f}\t Epsilon: {:.3f}\t Alpha: {:.3f}'.format(
                        count,
                        time.time() - start_time,
                        self.epsilon,
                        self.alpha)
                    print(msg, end="")

                # Q value of player B sticking and player A's action towards South before update
                before = q_1[2][1][1][4][1]
                # eps-greedy to generate action
                actions = [self.epsilon_greedy_action(pi_1, state, self.epsilon, action_list),
                           self.epsilon_greedy_action(pi_2, state, self.epsilon, action_list)]
                # get next state, reward and game termination flag
                next_state, rewards, done = env.step(actions)
                count += 1
                # update Q values
                q_1_current_state = q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]]
                q1_target = (rewards[0] + self.gamma * v_1[next_state[0]][next_state[1]][next_state[2]])
                q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (1 - self.alpha) * q_1_current_state \
                                                                            + self.alpha * q1_target

                prob, val = self.max_min_solver(q_1, state, env.action_space)
                if prob is not None:
                    pi_1[state[0]][state[1]][state[2]] = prob
                    v_1[state[0]][state[1]][state[2]] = val

                q_2_current_state = q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]]
                q2_target = (rewards[0] + self.gamma * v_2[next_state[0]][next_state[1]][next_state[2]])
                q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (1 - self.alpha) * q_2_current_state \
                                                                            + self.alpha * q2_target

                prob, val = self.max_min_solver(q_2, state, env.action_space)
                if prob is not None:
                    pi_2[state[0]][state[1]][state[2]] = prob
                    v_2[state[0]][state[1]][state[2]] = val

                # Q value of player B sticking and player A's action towards South after update
                after = q_1[2][1][1][4][1]
                q_diff_values.append(abs(after - before))
                if done:
                    break

                state = next_state
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)
                self.alpha *= self.alpha_decay
                self.alpha = max(self.alpha_min, self.alpha)
        pickle.dump(q_diff_values, open("results/Foe-Q.p", "wb"))
        error_plot(q_diff_values, 'Foe-Q')
        return q_diff_values


class SolverCEQ:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999993
        self.alpha = 1.0
        self.alpha_min = 0.001
        self.alpha_decay = 0.999993

    @staticmethod
    def epsilon_greedy_action(pi, state, epsilon):
        if np.random.random() < epsilon:
            index = np.random.choice(np.arange(25), 1)
            return np.array([index // 5, index % 5]).reshape(2)

        else:
            index = np.random.choice(np.arange(25), 1, p=pi[state[0]][state[1]][state[2]].reshape(25))
            return np.array([index // 5, index % 5]).reshape(2)

    @staticmethod
    def build_ce_constraints(q_1, q_2, state, row_indices, col_indices):
        # row player constraints
        q_1_actions = q_1[state[0]][state[1]][state[2]]
        s = block_diag(q_1_actions - q_1_actions[0, :], q_1_actions - q_1_actions[1, :],
                       q_1_actions - q_1_actions[2, :], q_1_actions - q_1_actions[3, :],
                       q_1_actions - q_1_actions[4, :])
        p_1_constraints = s[row_indices, :]

        # column player constraints
        q_2_actions = q_2[state[0]][state[1]][state[2]]
        s = block_diag(q_2_actions - q_2_actions[0, :], q_2_actions - q_2_actions[1, :],
                       q_2_actions - q_2_actions[2, :],
                       q_2_actions - q_2_actions[3, :], q_2_actions - q_2_actions[4, :])
        p_2_constraints = s[row_indices, :][:, col_indices]
        return np.append(p_1_constraints, p_2_constraints, axis=0)

    @staticmethod
    def ce_solver(q_1, q_2, state, row_indices, col_indices):
        c = matrix((q_1[state[0]][state[1]][state[2]] + q_2[state[0]][state[1]][state[2]].T).reshape(25))
        # construct rationality constraints
        G = matrix(
            np.append(SolverCEQ.build_ce_constraints(q_1, q_2, state, row_indices, col_indices), -np.eye(25), axis=0))
        h = matrix(np.zeros(G.size[0]) * 0.0)
        # construct probability constraints
        A = matrix(np.ones((1, G.size[1])))
        b = matrix(1.0)

        # error-handling mechanism
        try:
            sol = solvers.lp(c=c, G=G, h=h, A=A, b=b)
            if sol['x'] is not None:
                prob = np.abs(np.array(sol['x']).reshape((5, 5))) / sum(np.abs(sol['x']))
                val_1 = np.sum(prob * q_1[state[0]][state[1]][state[2]])
                val_2 = np.sum(prob * q_2[state[0]][state[1]][state[2]].T)
            else:
                prob = None
                val_1 = None
                val_2 = None
        except:
            prob = None
            val_1 = None
            val_2 = None

        return prob, val_1, val_2

    def solve(self):
        print("CE-Q Algorithm")
        np.random.seed(23456)
        q_diff_values = []
        env = GridWorld()
        dim_q = np.concatenate((env.state_space, [env.action_space, env.action_space]))
        q_1 = np.ones(dim_q)
        q_2 = np.ones(dim_q)
        dim_pi = np.concatenate((env.state_space, [env.action_space, env.action_space]))
        joint_pi = np.ones(dim_pi) / (env.action_space * env.action_space)
        dim_v = env.state_space
        v_1 = np.ones(dim_v)
        v_2 = np.ones(dim_v)
        count = 0
        start_time = time.time()
        ce_row_indices = [i for i in range(25) if i not in [0, 6, 12, 18, 24]]
        ce_col_indices = [x + 5 * y for x in range(5) for y in range(5)]
        while count < n_episodes:
            done = 0
            j = 0
            state = env.reset()
            while not done and j <= 100:
                if count % 1000 == 0:
                    msg = '\rEpisode {}\t Time: {:.2f} \t Percentage: {:.2f}%\t Epsilon: {:.3f} \t Alpha: {:.3f}'.format(
                        count,
                        time.time() - start_time,
                        count * 100 / n_episodes,
                        self.epsilon,
                        self.alpha)
                    print(msg, end="")

                count, j = count + 1, j + 1
                # Q value of player B sticking and player A's action towards South before update
                before = q_1[2][1][1][4][1]
                # eps-greedy to generate action
                self.epsilon = self.epsilon_decay ** count
                actions = self.epsilon_greedy_action(joint_pi, state, self.epsilon)
                self.alpha = self.alpha_decay ** count
                # get next state, reward and game termination flag
                next_state, rewards, done = env.step(actions)
                # update Q values
                q_1_current_state = q_1[state[0]][state[1]][state[2]][actions[0]][actions[1]]
                q1_target = (rewards[0] + self.gamma * v_1[next_state[0]][next_state[1]][next_state[2]])
                q_1[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (1 - self.alpha) * q_1_current_state \
                                                                            + self.alpha * q1_target

                q_2_current_state = q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]]
                q2_target = (rewards[1] + self.gamma * v_1[next_state[0]][next_state[1]][next_state[2]].T)
                q_2[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (1 - self.alpha) * q_2_current_state \
                                                                            + self.alpha * q2_target
                prob, val_1, val_2 = self.ce_solver(q_1, q_2, state, ce_row_indices, ce_col_indices)
                if prob is not None:
                    joint_pi[state[0]][state[1]][state[2]] = prob
                    v_1[state[0]][state[1]][state[2]] = val_1
                    v_2[state[0]][state[1]][state[2]] = val_2

                state = next_state
                # Q value of player B sticking and player A's action towards South after update
                after = q_1[2][1][1][4][1]
                q_diff_values.append(abs(after - before))

        pickle.dump(q_diff_values, open("results/CE-Q.p", "wb"))
        error_plot(q_diff_values, 'CE-Q')
        return q_diff_values


def error_plot(errors, title):
    plt.rcParams['agg.path.chunksize'] = 1000
    plt.plot(errors, linestyle='-', linewidth=0.6)
    plt.title(title)
    plt.ylim(0, 0.5)
    plt.xlabel('Simulation Iterations')
    plt.ylabel('Q-value Difference')
    plt.ticklabel_format(style='sci', axis='x',
                         scilimits=(0, 0), useMathText=True)
    plt.savefig('{}/{}.png'.format('graphs', title), dpi=300)
    plt.show()


if __name__ == '__main__':
    artifact_dirs = ['results', 'graphs']
    for directory in artifact_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

    n_episodes = int(1e6)
    # script options
    # 1 - Solver Q-Learning
    # 2 - Solver Friend-Q
    # 3 - Solver Foe-Q
    # 4 - Solver CE-Q
    if len(sys.argv) > 1:
        option = sys.argv[1]
        if option == '1':
            SolverQLearning().solve()
        elif option == '2':
            SolverFriendQ().solve()
        elif option == '3':
            SolverFoeQ().solve()
        elif option == '4':
            SolverCEQ().solve()
    else:
        SolverQLearning().solve()
