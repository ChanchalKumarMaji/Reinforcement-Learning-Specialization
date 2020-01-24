import numpy as np
import matplotlib.pyplot as plt


def compute_value_function(num_states, action_range):
    """
    Computes the value function for the 1000 state random walk as described in Sutton and Barto (2017).
    :return: The value function for states 1 to 1000. Index 0 is not used in this array (i.e. should remain 0).
    """
    state_prob = 0.5 / float(action_range)
    gamma = 1
    theta = 0.000001

    V = np.zeros(num_states+1)

    delta = np.infty
    i = 0
    while delta > theta:
        i += 1
        delta = 0.0
        for s in range(1, num_states+1):
            v = V[s]
            value_sum = 0.0
            for transition in range(1, action_range+1):
                right = s + transition
                right_reward = 0
                if right > num_states:
                    right_reward = 1
                    right = 0

                left = s - transition
                left_reward = 0
                if left < 1:
                    left_reward = -1
                    left = 0

                value_sum += state_prob * ((right_reward + gamma * V[right]) + (left_reward + gamma * V[left]))

            V[s] = value_sum
            delta = max(delta, np.abs(v-V[s]))
    
    np.save("data/true_V", V[1:])
    return V[1:]

if __name__ == '__main__':
    num_states = 500
    action_range = 100
    V = compute_value_function(num_states, action_range)
    
    np.save("true_V", V)
    plt.plot(range(1, num_states+1), V[1:])
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.show()
