import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(25)

P = np.array([
    [0.1, 0.6, 0.3],
    [0.4, 0.4, 0.2],
    [0.3, 0.3, 0.4]
], dtype=float)

n_states = P.shape[0] # Number of states = number of rows in P
N_steps = 20000 # Number of steps to simulate

state = 0 # Initial state
counts = np.zeros(n_states, dtype=int) # Count visits to each state, [0, 0, 0]
pie_sim = np.zeros((N_steps, n_states), dtype=float) # Store empirical distribution over time
# [1 0 0] at step 0, 
# [0.5 0.5 0] at step 1, 
# [0.33333333 0.33333333 0.33333333] at step 2, 
# ...

for step in range(N_steps):
    counts[state] += 1 # Increment count for current state
    pie_sim[step] = counts / (step + 1) # Update empirical distribution 
    # if i devide counts by steps we get the limiting probability distribution for that step
    # step + 1 because step starts from 0

    # Transition to next state based on current state's probabilities
    state = rng.choice(n_states, p=P[state]) # choice(3, p=[0.1, 0.6, 0.3]) if state=0 (for numpy choice)
    # choice(3) returns 0, 1, or 2 with equal probability (1/3 each) if no p= is given
    # choice([0, 1, 2]) returns 0, 1, or 2 with equal probability (1/3 each) if no p= is given
    # choice(3, p=[0.1, 0.6, 0.3]) returns 0 with prob 0.1, 1 with prob 0.6, 2 with prob 0.3 if state=0 (for numpy choice)
    # choice([0, 1, 2], p=[0.1, 0.6, 0.3]) returns 0 with prob 0.1, 1 with prob 0.6, 2 with prob 0.3 if state=0 (for numpy choice)
    # choice(3, p=[0.1, 0.6, 0.3]) if state=0 (for vanilla python choice) [vanilla python choice doesn't have p= option][Python's random.choices has p= option]
    # choice([0, 1, 2], p=[0.1, 0.6, 0.3]) if state=0 (for vanilla python choice) [vanilla python choice doesn't have p= option][Python's numpy random.choices has p= option]



# Theoritical value of limiting probability distribution
# Solve the equation pie = pie * P

# Find the stationary distribution by solving the equation pie = pie * P
# This can be done by finding the eigenvector corresponding to the eigenvalue 1
# or by using the fact that in the limit, pie * P = pie

eigvals, eigvecs = np.linalg.eig(P.T) # Transpose P to get right eigenvectors
# eigvals: array of eigenvalues [ 1. +0.j -0.3+0.j  0.2+0.j]
# eigvecs: 2D array where each column is an eigenvector corresponding to an eigenvalue
# eigvecs[:, 0] is the eigenvector corresponding to eigenvalue 1

i = np.argmin(np.abs(eigvals - 1.0)) # Index of eigenvalue closest to 1 (should be exactly 1)
# np.abs(eigvals - 1.0) gives array of absolute differences from 1
# np.argmin(...) gives the index of the minimum value in that array
# (1.001, -10, 20) - 1 = (0.001, 11, 21) -> argmin = 0 (ARGMIN gives index of minimum value in array)
pie_theory = np.real(eigvecs[:, i]) # Take the real part of the eigenvector (in case of numerical noise)
# pie_theory is the stationary distribution (not normalized yet)
# np.real(...) takes the real part of complex numbers (in case of numerical noise)
# np.real(:, i) takes the i-th column of eigvecs (the eigenvector corresponding to eigenvalue 1)

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



