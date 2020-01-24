import numpy as np
import matplotlib.pyplot as plt

plt1_legend_dict = {"td_agent": "approximate values learned by\n TD with neural network", 
                    "td_agent_5000_episodes": "approximate values learned by\n TD with neural network",
                    "td_agent_tilecoding": "approximate values learned by\n TD with tile-coding"}


plt2_legend_dict = {"td_agent": "TD with neural network", 
                    "td_agent_5000_episodes": "TD with neural network",
                    "td_agent_tilecoding": "TD with tile-coding"}


plt2_label_dict = {"td_agent": "RMSVE\n averaged\n over\n 20 runs", 
                   "td_agent_5000_episodes": "RMSVE\n averaged\n over\n 20 runs",
                   "td_agent_tilecoding": "RMSVE\n averaged\n over\n 20 runs"}


# Function to plot result
def plot_result(data_name_array):
    
    true_V = np.load('data/true_V.npy')

    plt1_agent_sweeps = []
    plt2_agent_sweeps = []
    
    # two plots: learned state-value and learning curve (RMSVE)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    
    for data_name in data_name_array:

        # plot1
        filename = 'V_{}'.format(data_name).replace('.','')
        current_agent_V = np.load('results/{}.npy'.format(filename))
        current_agent_V = current_agent_V[-1, :]


        plt1_x_legend = range(1,len(current_agent_V[:]) + 1)
        graph_current_agent_V, = ax[0].plot(plt1_x_legend, current_agent_V[:], label=plt1_legend_dict[data_name])
        plt1_agent_sweeps.append(graph_current_agent_V)
        
        # plot2
        filename = 'RMSVE_{}'.format(data_name).replace('.','')
        RMSVE_data = np.load('results/{}.npz'.format(filename))
        current_agent_RMSVE = np.mean(RMSVE_data["rmsve"], axis = 0)

        plt2_x_legend = np.arange(0, RMSVE_data["num_episodes"]+1, RMSVE_data["eval_freq"])
        graph_current_agent_RMSVE, = ax[1].plot(plt2_x_legend, current_agent_RMSVE[:], label=plt2_legend_dict[data_name])
        plt2_agent_sweeps.append(graph_current_agent_RMSVE)
                
          
    # plot1: 
    # add True V
    plt1_x_legend = range(1,len(true_V[:]) + 1)
    graph_true_V, = ax[0].plot(plt1_x_legend, true_V[:], label="$v_{\pi}$")
    
    ax[0].legend(handles=[*plt1_agent_sweeps, graph_true_V], fontsize = 13)
    
    ax[0].set_title("State Value", fontsize = 15)
    ax[0].set_xlabel('State', fontsize = 14)
    ax[0].set_ylabel('Value\n scale', rotation=0, labelpad=15, fontsize = 14)

    plt1_xticks = [1, 100, 200, 300, 400, 500]
    plt1_yticks = [-1.0, 0.0, 1.0]
    ax[0].set_xticks(plt1_xticks)
    ax[0].set_xticklabels(plt1_xticks, fontsize=13)
    ax[0].set_yticks(plt1_yticks)
    ax[0].set_yticklabels(plt1_yticks, fontsize=13)
    
    
    # plot2:
    ax[1].legend(handles=plt2_agent_sweeps, fontsize = 13)
    
    ax[1].set_title("Learning Curve", fontsize = 15)
    ax[1].set_xlabel('Episodes', fontsize = 14)
    ax[1].set_ylabel(plt2_label_dict[data_name_array[0]], rotation=0, labelpad=40, fontsize = 14)

    plt2_yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ax[1].tick_params(axis="x", labelsize=13)
    ax[1].set_yticks(plt2_yticks)
    ax[1].set_yticklabels(plt2_yticks, fontsize = 13)

    plt.tight_layout()
    plt.show()      