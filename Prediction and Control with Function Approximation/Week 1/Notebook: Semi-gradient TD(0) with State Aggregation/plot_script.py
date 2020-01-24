import numpy as np
import matplotlib.pyplot as plt

# Function to plot result
def plot_result(agent_parameters, directory):
    
    true_V = np.load('data/true_V.npy')

    for num_g in agent_parameters["num_groups"]:
        plt1_agent_sweeps = []
        plt2_agent_sweeps = []
        
        # two plots: learned state-value and learning curve (RMSVE)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
            
        for step_size in agent_parameters["step_size"]:
            
            # plot1
            filename = 'V_TD_agent_agg_states_{}_step_size_{}'.format(num_g, step_size).replace('.','')
            current_agent_V = np.load('{}/{}.npy'.format(directory, filename))

            plt1_x_legend = range(1,len(current_agent_V[:]) + 1)
            graph_current_agent_V, = ax[0].plot(plt1_x_legend, current_agent_V[:], label="approximate values: state aggregation: {}, step-size: {}".format(num_g, step_size))
            plt1_agent_sweeps.append(graph_current_agent_V)
            
            # plot2
            filename = 'RMSVE_TD_agent_agg_states_{}_step_size_{}'.format(num_g, step_size).replace('.','')
            current_agent_RMSVE = np.load('{}/{}.npy'.format(directory, filename))

            plt2_x_legend = range(1,len(current_agent_RMSVE[:]) + 1)
            graph_current_agent_RMSVE, = ax[1].plot(plt2_x_legend, current_agent_RMSVE[:], label="approximate values: state aggregation: {}, step-size: {}".format(num_g, step_size))
            plt2_agent_sweeps.append(graph_current_agent_RMSVE)
            
          
        # plot1: 
        # add True V
        plt1_x_legend = range(1,len(true_V[:]) + 1)
        graph_true_V, = ax[0].plot(plt1_x_legend, true_V[:], label="$v_\pi$")
        
        ax[0].legend(handles=[*plt1_agent_sweeps, graph_true_V])
        
        ax[0].set_title("Learned State Value after 2000 episodes")
        ax[0].set_xlabel('State')
        ax[0].set_ylabel('Value\n scale', rotation=0, labelpad=15)

        plt1_xticks = [1, 100, 200, 300, 400, 500]#, 600, 700, 800, 900, 1000]
        plt1_yticks = [-1.0, 0.0, 1.0]
        ax[0].set_xticks(plt1_xticks)
        ax[0].set_xticklabels(plt1_xticks)
        ax[0].set_yticks(plt1_yticks)
        ax[0].set_yticklabels(plt1_yticks)
        
        
        # plot2:
        ax[1].legend(handles=plt2_agent_sweeps)
        
        ax[1].set_title("Learning Curve")
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('RMSVE\n averaged over 50 runs', rotation=0, labelpad=40)

        plt2_xticks = range(0, 210, 20) # [0, 10, 20, 30, 40, 50, 60, 70, 80]
        plt2_xticklabels = range(0, 2100, 200) # [0, 100, 200, 300, 400, 500, 600, 700, 800]
        plt2_yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        ax[1].set_xticks(plt2_xticks)
        ax[1].set_xticklabels(plt2_xticklabels)
        ax[1].set_yticks(plt2_yticks)
        ax[1].set_yticklabels(plt2_yticks)
        
        plt.tight_layout()
        plt.suptitle("{}-State Aggregation".format(num_g),fontsize=16, fontweight='bold', y=1.03)
        plt.show()      
