# Queueing & Markov Simulation Plots
# - DTMC: empirical distribution -> stationary vector
# - M/M/1: Pn vs (1-rho)rho^n, running mean of Wq vs theory, histogram of Wq
# Requirements: numpy, matplotlib (no seaborn)

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# 1) Discrete-time Markov chain
# -----------------------------
rng = np.random.default_rng(42)

# Ergodic 3-state chain
P = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
    [0.1, 0.3, 0.6]
], dtype=float)

n_states = P.shape[0]
N_steps = 50_000

state = 0
counts = np.zeros(n_states, dtype=float)
running_pi = np.zeros((N_steps, n_states), dtype=float)

for t in range(N_steps):
    state = rng.choice(n_states, p=P[state])
    counts[state] += 1.0
    running_pi[t] = counts / (t + 1)

# Theoretical stationary distribution (eigenvector for eigenvalue 1)
eigvals, eigvecs = np.linalg.eig(P.T)
i = np.argmin(np.abs(eigvals - 1.0))
pi_theory = np.real(eigvecs[:, i])
pi_theory = pi_theory / np.sum(pi_theory)

# Plot: empirical vs theoretical stationary
plt.figure(figsize=(7, 4.2))
for s in range(n_states):
    plt.plot(running_pi[:, s], label=f"Empirical π[{s}]")
for s in range(n_states):
    plt.hlines(pi_theory[s], xmin=0, xmax=N_steps, linestyles='dashed',
               label=f"Theory π[{s}]={pi_theory[s]:.3f}")
plt.title("DTMC: Convergence of Empirical Distribution to Stationary π")
plt.xlabel("Steps")
plt.ylabel("Probability")
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# -----------------------------
# 2) M/M/1 queue (event simulation, FIFO)
# -----------------------------
rng = np.random.default_rng(123)

lam = 2.0  # arrival rate λ
mu  = 3.0  # service rate μ
rho = lam / mu
assert rho < 1.0, "System must be stable (ρ < 1)."

N_customers = 30_000  # number of customers who START service (for Wq estimate)

t = 0.0
next_arrival = rng.exponential(1.0 / lam)
next_departure = float('inf')         # server idle initially
waiting_queue = []                    # arrival times waiting in FIFO line

waiting_times = []                    # Wq recorded at service start
running_mean_Wq = []                  # running average of Wq
time_in_n = defaultdict(float)        # time spent with n customers in system

def n_in_system():
    # number waiting + 1 if server busy
    return len(waiting_queue) + (0 if np.isinf(next_departure) else 1)

while len(waiting_times) < N_customers:
    # Select next event
    if next_arrival <= next_departure:
        t_next = next_arrival
        event = 'arrival'
    else:
        t_next = next_departure
        event = 'departure'

    # Accumulate time in current state n
    dt = t_next - t
    time_in_n[n_in_system()] += dt
    t = t_next

    if event == 'arrival':
        # If idle, start service immediately; else join queue
        if np.isinf(next_departure):
            waiting_times.append(0.0)
            running_mean_Wq.append(float(np.mean(waiting_times)))
            service_time = rng.exponential(1.0 / mu)
            next_departure = t + service_time
        else:
            waiting_queue.append(t)

        next_arrival = t + rng.exponential(1.0 / lam)

    else:  # departure
        if waiting_queue:
            a = waiting_queue.pop(0)
            w = t - a  # waited until server became free
            waiting_times.append(w)
            running_mean_Wq.append(float(np.mean(waiting_times)))
            service_time = rng.exponential(1.0 / mu)
            next_departure = t + service_time
        else:
            next_departure = float('inf')

# Empirical Pn via time-average
total_time = sum(time_in_n.values())
n_max = 12
Pn_emp = np.array([time_in_n[n] / total_time for n in range(n_max + 1)])

# Theoretical Pn and Wq
Pn_theory = np.array([(1 - rho) * (rho ** n) for n in range(n_max + 1)])
Wq_theory = rho / (mu - lam)

# Plot A: Pn (empirical vs theory)
x = np.arange(n_max + 1)

plt.figure(figsize=(7, 4.2))
plt.bar(x - 0.2, Pn_emp, width=0.4, label="Empirical P(n)")
plt.plot(x + 0.2, Pn_theory, marker='o', linestyle='--', label="Theoretical P(n)")
plt.title(f"M/M/1: Time-average State Probabilities vs Theory (ρ={rho:.2f})")
plt.xlabel("n (customers in system)")
plt.ylabel("Probability")
plt.xticks(x)
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# Plot B: running mean of Wq vs theory
plt.figure(figsize=(7, 4.2))
plt.plot(running_mean_Wq, label="Running mean Wq (simulation)")
plt.hlines(Wq_theory, xmin=0, xmax=len(running_mean_Wq), linestyles='dashed',
           label=f"Theoretical Wq={Wq_theory:.3f}")
plt.title("M/M/1: Convergence of Average Waiting Time in Queue (Wq)")
plt.xlabel("Customers who started service")
plt.ylabel("Average waiting time in queue")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# Plot C: histogram of waiting times with theoretical mean
plt.figure(figsize=(7, 4.2))
plt.hist(waiting_times, bins=60, density=True, alpha=0.7)
plt.axvline(Wq_theory, linestyle='--', label=f"Theoretical mean Wq={Wq_theory:.3f}")
plt.title("M/M/1: Distribution of Waiting Times in Queue")
plt.xlabel("Waiting time in queue")
plt.ylabel("Density")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# Quick numeric sanity-checks
avg_Wq_sim = float(np.mean(waiting_times))
Lq_sim = lam * avg_Wq_sim
Lq_theory = (rho ** 2) / (1 - rho)
print(f"rho = {rho:.3f}")
print(f"Average Wq (sim) = {avg_Wq_sim:.4f},  Wq (theory) = {Wq_theory:.4f}")
print(f"Lq (sim via Little) = {Lq_sim:.4f},  Lq (theory) = {Lq_theory:.4f}")
