import numpy as np
import matplotlib.pyplot as plt
import pickle

plt_legend_dict = {"expected_sarsa_agent": "Expected SARSA with neural network",
                   "random_agent": "Random"}
path_dict = {"expected_sarsa_agent": "results/",
             "random_agent": "./"}

plt_label_dict = {"expected_sarsa_agent": "Sum of\nreward\nduring\nepisode"}

def smooth(data, k):
    num_episodes = data.shape[1]
    num_runs = data.shape[0]

    smoothed_data = np.zeros((num_runs, num_episodes))

    for i in range(num_episodes):
        if i < k:
            smoothed_data[:, i] = np.mean(data[:, :i+1], axis = 1)   
        else:
            smoothed_data[:, i] = np.mean(data[:, i-k:i+1], axis = 1)    
        

    return smoothed_data

# Function to plot result
def plot_result(data_name_array):
    plt_agent_sweeps = []
    
    fig, ax = plt.subplots(figsize=(8,6))

    
    for data_name in data_name_array:
        
        # load data
        filename = 'sum_reward_{}'.format(data_name).replace('.','')
        sum_reward_data = np.load('{}/{}.npy'.format(path_dict[data_name], filename))

        # smooth data
        smoothed_sum_reward = smooth(data = sum_reward_data, k = 100)
        
        mean_smoothed_sum_reward = np.mean(smoothed_sum_reward, axis = 0)

        plot_x_range = np.arange(0, mean_smoothed_sum_reward.shape[0])
        graph_current_agent_sum_reward, = ax.plot(plot_x_range, mean_smoothed_sum_reward[:], label=plt_legend_dict[data_name])
        plt_agent_sweeps.append(graph_current_agent_sum_reward)
    
    ax.legend(handles=plt_agent_sweeps, fontsize = 13)
    ax.set_title("Learning Curve", fontsize = 15)
    ax.set_xlabel('Episodes', fontsize = 14)
    ax.set_ylabel(plt_label_dict[data_name_array[0]], rotation=0, labelpad=40, fontsize = 14)
    ax.set_ylim([-300, 300])

    plt.tight_layout()
    plt.show()     
