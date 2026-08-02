import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    values = np.array(values, dtype=float)
    transitions = np.array(transitions, dtype=float)
    rewards = np.array(rewards, dtype=float)

    q_values = rewards + gamma * np.sum(transitions * values[np.newaxis, np.newaxis, :], axis=2)

    return np.max(q_values, axis=1).tolist()