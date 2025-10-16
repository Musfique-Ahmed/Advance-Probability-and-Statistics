import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

rng = np.random.default_rng(26)

λ = 2.0  # Arrival rate
μ = 3.0  # Service rate

ρ = λ / μ

n_customers = 30000 # number of customers who start service in the system (for M/M/1 this is equivalent to number of departures) (for Wq estimation we need number of arrivals)


t = 0.0

next_arrival = rng.exponential(1/λ) # Time of next arrival
next_departure = float('inf') # No departure scheduled initially (server is idle initially so next departure is infinity)
waiting_queue = [] # Queue of arrival times of customers waiting for service
waiting_times = [] # List of waiting times of all customers
running_mean_Wq = [] # Running mean of waiting times in queue
time_in_n = defaultdict(float) # Total time spent in each state (number of customers in system)


def num_in_system():
    """Return the current number of customers in the system (in service + in queue)."""
    return len(waiting_queue) + (1 if next_departure < float('inf') else 0)