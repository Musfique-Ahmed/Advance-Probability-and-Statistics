import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(25)

P = np.array([
    [0.1, 0.6, 0.3],
    [0.4, 0.4, 0.2],
    [0.3, 0.3, 0.4]
], dtype=float)

n_states = P.shape[0] # Number of states = number of rows in P

# def simulate_markov_chain(transition_matrix, initial_state, num_steps):
#     """
#     Simulates a Markov chain given a transition matrix and an initial state.

#     Parameters:
#     transition_matrix (np.ndarray): A square matrix where element (i, j) represents the probability of transitioning from state i to state j.
#     initial_state (int): The starting state index.
#     num_steps (int): The number of steps to simulate.

#     Returns:
#     list: A list of states visited during the simulation.
#     """
#     states = [initial_state]
#     current_state = initial_state

#     for _ in range(num_steps):
#         current_state = rng.choice(
#             len(transition_matrix),
#             p=transition_matrix[current_state]
#         )
#         states.append(current_state)

#     return states



