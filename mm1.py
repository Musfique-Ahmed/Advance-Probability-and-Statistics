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


while len(waiting_times) < n_customers:
    if next_arrival < next_departure:
        # Next event is an arrival
        t_next = next_arrival
        event = 'arrival'
    else:
        # Next event is a departure
        t_next = next_departure
        event = 'departure'
    
    δt = t_next - t # Time until next event
    time_in_n[num_in_system()] += δt # Update time spent in current state
    t = t_next # Advance time to next event


    if event == 'arrival':
        if np.isinf(next_departure):
            waiting_times.append(0.0) # No waiting time if server is idle
            running_mean_Wq.append(float(np.mean(waiting_times))) # Update running mean
            service_time = rng.exponential(1.0 / μ) # Sample service time
            next_departure = t + service_time # Schedule departure
        else:
            waiting_queue.append(t) # Add arrival time to queue
        next_arrival = t + rng.exponential(1.0 / λ) # Schedule next arrival

    else: # event == 'departure'
        if waiting_queue:
            arrival_time = waiting_queue.pop(0) # Get arrival time of next customer in queue
            waiting_time = t - arrival_time # Calculate waiting time
            waiting_times.append(waiting_time) # Record waiting time
            running_mean_Wq.append(float(np.mean(waiting_times))) # Update running mean
            service_time = rng.exponential(1.0 / μ) # Sample service time
            next_departure = t + service_time # Schedule next departure
        else:
            next_departure = float('inf') # No customers left, server becomes idle


# Theoretical values
Wq_theory = ρ / (μ * (1 - ρ)) # Average waiting time in queue
