import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def generate_state_vector(state, non_terminal_states):
    return np.array([1 if i == state else 0 for i in non_terminal_states])


def calculate_reward(sequence):
    return 1 if 6 in sequence else 0


def generate_sequence_vectors(sequence, all_states):
    return np.array([generate_state_vector(sequence[i], all_states[1:-1]) for i in range(len(sequence) - 1)]).T


def generate_training_sequences(start_state, all_states, no_sequences):
    sequences = list()
    non_terminal_states = all_states[1:-1]
    for i in range(no_sequences):
        current_state = start_state
        episode = list()
        episode.append(current_state)
        while current_state in non_terminal_states:
            if np.random.choice(['left', 'right']) == 'left':
                current_state = all_states[all_states.index(current_state) - 1]
            else:
                current_state = all_states[all_states.index(current_state) + 1]
            if current_state in non_terminal_states:
                episode.append(current_state)
        episode.append(current_state)
        sequences.append(episode)
    return sequences


def generate_training_data(start_state, no_sequences, no_training_data, all_states):
    training_data = list()
    for i in range(no_training_data):
        training_data.append(generate_training_sequences(start_state, all_states, no_sequences))
    return training_data


def td_lambda_update(_lambda, alpha, sequence_vector, weights, sequence_reward):
    # weights of all non terminal states.
    # et holds the eligibility traces for all the previously visited non terminal states.
    et = np.array([[0, 0, 0, 0, 0]]).T
    delta_w = np.array([[0, 0, 0, 0, 0]]).T
    for t in range(sequence_vector.shape[1]):
        et = _lambda * et + sequence_vector[:, [t]]
        if t == sequence_vector.shape[1] - 1:
            delta_w = delta_w + alpha * (sequence_reward - np.dot(weights.T, sequence_vector[:, [t]])) * et
        else:
            delta_w = delta_w + alpha * (np.dot(weights.T, sequence_vector[:, [t + 1]])
                                         - np.dot(weights.T, sequence_vector[:, [t]])) * et
    return delta_w


def generate_true_weights(all_states):
    identity = np.identity(5)
    tp_non_terminal_states = list()
    for i in range(len(all_states[1:-1])):
        tp_non_terminal_states.append(
            [0.5 if i == j - 1 or i == j + 1 else 0 for j in range(len(all_states[1:-1]))])
    tp_terminal_states = list()
    terminal_states = [all_states[0], all_states[len(all_states) - 1]]
    for i in terminal_states:
        tp_terminal_states.append(
            [0.5 if i + 1 == j or i - 1 == j else 0 for j in (all_states[1:-1])])
    q = np.array(tp_non_terminal_states)
    h_sum = np.sum(np.multiply(np.array(tp_terminal_states), np.array([[0], [1]])), axis=0)
    h = h_sum.reshape(-1, 1)  # transition probabilities from non terminal- terminal states * z(rewards A=0 and G=1)
    true_weights = np.dot(np.linalg.inv(identity - q), h)
    return true_weights


def replicate_figure_3(no_sequences, no_training_data):
    # all alphabetical states (A-G) are indexed as running numbers staring from 0 to 6
    all_states = [i for i in range(7)]
    start_state = 3
    learning_parameters = np.array([0.005])
    lambdas = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    true_weights = generate_true_weights(all_states)
    training_data = generate_training_data(start_state, no_sequences, no_training_data,
                                           all_states)
    results = []
    for _lambda in lambdas:
        for lp in learning_parameters:
            root_mean_squares = list()
            for training_set in training_data:
                weights = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]]).T
                while True:
                    previous_weights = np.copy(weights)
                    delta_w_sum = np.array([[0, 0, 0, 0, 0]]).T
                    for sequence in training_set:
                        sequence_vector = generate_sequence_vectors(sequence, all_states)
                        sequence_reward = calculate_reward(sequence)
                        delta_w_sum = delta_w_sum + td_lambda_update(_lambda, lp,
                                                                     sequence_vector, weights, sequence_reward)
                    weights += delta_w_sum

                    if np.sum(np.absolute(previous_weights - weights)) < 1e-3:
                        break

                error = (true_weights - np.array(weights))
                root_mean_squares.append(np.sqrt(np.average(np.power(error, 2))))

            result = [_lambda, lp, np.mean(root_mean_squares)]
            results.append(result)
    show_figure_3(results)


def show_figure_3(results):
    lambda_to_min_rms = lambda_2_min_rms(results)
    df = pd.DataFrame(lambda_to_min_rms)
    df.columns = ["lmbda", "rms-error"]
    df.head()
    plt.figure(num=None, figsize=(12, 6), dpi=100)
    plt.margins(.05)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Error (RMS)")
    plt.title("Figure 3")
    plt.text(0.9, .18, "Widrow-Hoff", ha="center", va="center", rotation=0, size=15)
    plt.xticks(df["lmbda"])
    plt.plot('lmbda', 'rms-error', data=df, marker='o', color='blue', linewidth=2)
    plt.show()


def lambda_2_min_rms(results):
    lambda_dict = dict()
    for result in results:
        if result[0] not in lambda_dict.keys():
            lambda_dict[result[0]] = list()
        lambda_dict[result[0]].append(result[2])
    lambda_to_min_rms = list()
    for entry in lambda_dict.keys():
        lambda_to_min_rms.append([entry, sorted(lambda_dict[entry])[0]])
    return lambda_to_min_rms


def replicate_figure_4_and_5(no_sequences, no_training_data, experimental):
    # states A, B, C, D, E, F, G are indexed as 0, 1, 2, 3, 4, 5, 6
    all_states = [i for i in range(7)]
    start_state = 3
    learning_parameters = [i * .05 for i in range(0, 13)]
    lambdas = [i * .1 for i in range(0, 11)]
    true_weights = generate_true_weights(all_states)
    training_data = generate_training_data(start_state, no_sequences, no_training_data,
                                           all_states)
    results = []
    for _lambda in lambdas:
        for lp in learning_parameters:
            root_mean_squares = []
            for training_set in training_data:
                weights = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]]).T
                for sequence in training_set:
                    sequence_vector = generate_sequence_vectors(sequence, all_states)
                    sequence_reward = calculate_reward(sequence)
                    weights = weights + td_lambda_update(_lambda, lp,
                                                         sequence_vector, weights, sequence_reward)

                error = (true_weights - np.array(weights))
                root_mean_squares.append(np.sqrt(np.average(np.power(error, 2))))

            result = [_lambda, lp, np.mean(root_mean_squares)]
            results.append(result)

    show_figure_4(results)
    if experimental:
        show_figure_5_exp(results)
    else:
        show_figure_5(results)


def show_figure_4(results):
    data = pd.DataFrame(results)
    data.columns = ['lmbda', 'alpha', 'rms-error']
    data.head()
    lambda_0_data_df = create_data_2_plot([r for r in results if 0.0 <= r[0] < 0.1 and r[2] < 0.99])
    lambda_dot3_data_df = create_data_2_plot([r for r in results if 0.3 <= r[0] < 0.4 and r[2] < 0.99])
    lambda_dot8_data_df = create_data_2_plot([r for r in results if 0.8 <= r[0] < 0.9 and r[2] < 0.99])
    lambda_1_data_df = create_data_2_plot([r for r in results if 1.0 <= r[0] and r[2] < 0.99])
    plt.figure(num=None, figsize=(12, 6), dpi=100)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Error (RMS)")
    plt.title("Figure 4")
    plt.plot('alpha', 'rms-error', data=lambda_0_data_df, marker='o', color='blue', linewidth=2, label=r"$\lambda$=0")
    plt.plot('alpha', 'rms-error', data=lambda_dot3_data_df, marker='o', color='olive', linewidth=2,
             label=r"$\lambda$=0.3")
    plt.plot('alpha', 'rms-error', data=lambda_dot8_data_df, marker='o', color='red', linewidth=2,
             label=r"$\lambda$=0.8")
    plt.plot('alpha', 'rms-error', data=lambda_1_data_df, marker='o', color='black', linewidth=2,
             label=r"$\lambda$=1 (Widrow-Hoff)")
    plt.legend(loc="upper left")
    plt.show()


def create_data_2_plot(lambda_data):
    lambda_data_df = pd.DataFrame(lambda_data)
    lambda_data_df.columns = ['lmbda', 'alpha', 'rms-error']
    lambda_data_df.drop('lmbda', 1, inplace=True, )
    return lambda_data_df


def show_figure_5_exp(results):
    lambda_to_min_rms = lambda_2_min_rms(results)
    data = pd.DataFrame(lambda_to_min_rms)
    data.columns = ['lmbda', 'rms-error']
    data.head()
    plt.figure(num=None, figsize=(12, 6), dpi=100)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Error (RMS) \n using best " + r"$\alpha$")
    plt.title("Figure 5")
    plt.text(0.98, .21, "Widrow-Hoff", ha="center", va="center", rotation=0, size=15)
    plt.plot('lmbda', 'rms-error', data=data, marker='o', color='blue', linewidth=2)
    plt.show()


def show_figure_5(results):
    lambda_to_min_rms = lambda_2_min_rms(results)
    data = pd.DataFrame(lambda_to_min_rms)
    data.columns = ['lmbda', 'rms-error']
    data.head()
    plt.figure(num=None, figsize=(12, 6), dpi=100)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Error (RMS) \n using best " + r"$\alpha$")
    plt.title("Figure 5")
    plt.text(0.9, .18, "Widrow-Hoff", ha="center", va="center", rotation=0, size=15)
    plt.plot('lmbda', 'rms-error', data=data, marker='o', color='blue', linewidth=2)
    plt.show()


if __name__ == '__main__':
    sequence_size = 10
    training_data_sample_size = 100
    replicate_figure_3(sequence_size, training_data_sample_size)
    replicate_figure_4_and_5(sequence_size, training_data_sample_size, False)
    # additional experiment with reduced sequence size
    if len(sys.argv) > 1 and sys.argv[1]:
        sequence_size = 5
        replicate_figure_4_and_5(sequence_size, training_data_sample_size, True)
