import matplotlib
import numpy as np
import matplotlib.pyplot as plt

class Manager:
    def __init__(self, env_info, agent_info, true_values_file=None, experiment_name=None):
        self.experiment_name = experiment_name
        
        self.grid_h, self.grid_w = env_info["grid_height"], env_info["grid_width"]
        self.cmap = matplotlib.cm.viridis
        self.cmap.set_bad('black', 1.)
            
        self.values_table = None
        self.policy = agent_info["policy"]
        
        self.true_values_file = true_values_file
    
    def compute_values_table(self, values):
        self.values_table = np.empty((self.grid_h, self.grid_w))
        self.values_table.fill(np.nan)
        for state in range(len(values)):
            self.values_table[np.unravel_index(state, (self.grid_h, self.grid_w))] = values[state]
                    
    def compute_RMSVE(self):
        return (np.sqrt(np.nanmean((self.values_table - self.true_values) ** 2)))
    
    def visualize(self, values, episode_num):
        if not hasattr(self, "fig"):
            self.fig = plt.figure(figsize=(10, 20))
            plt.ion()
            
            if self.true_values_file is not None:
                self.cmap_VE = matplotlib.cm.Reds
                self.cmap_VE.set_bad('black', 1.)
                self.ax = self.fig.add_subplot(311)
                self.RMSVE_LOG = []
                self.true_values = np.load(self.true_values_file)
            else:
                self.true_values = None

        self.fig.clear()
        if self.true_values is not None:
            plt.subplot(311)
        self.compute_values_table(values)
        plt.xticks([])
        plt.yticks([])
        im = plt.imshow(self.values_table, cmap=self.cmap, interpolation='nearest', origin='upper')
        
        for state in range(self.policy.shape[0]):
            for action in range(self.policy.shape[1]):
                y, x = np.unravel_index(state, (self.grid_h, self.grid_w))
                pi = self.policy[state][action]
                if pi == 0:
                    continue
                if action == 0:
                    plt.arrow(x, y, 0,  -0.5 * pi, fill=False, length_includes_head=True, head_width=0.1, 
                              alpha=0.5)
                if action == 1: 
                    plt.arrow(x, y, -0.5 * pi, 0, fill=False, length_includes_head=True, head_width=0.1, 
                              alpha=0.5)
                if action == 2:
                    plt.arrow(x, y, 0, 0.5 * pi, fill=False, length_includes_head=True, head_width=0.1, 
                              alpha=0.5)
                if action == 3:
                    plt.arrow(x, y, 0.5 * pi, 0, fill=False, length_includes_head=True, head_width=0.1, 
                              alpha=0.5)
        
        plt.title((("" or self.experiment_name) + "\n") + "Predicted Values, Episode: %d" % episode_num)
        plt.colorbar(im, orientation='horizontal')
        
        if self.true_values is not None:
            plt.subplot(312)
            plt.xticks([])
            plt.yticks([])
            im = plt.imshow((self.values_table - self.true_values) ** 2, origin='upper', cmap=self.cmap_VE)
            plt.title('Squared Value Error: $(v_{\pi}(S) - \hat{v}(S))^2$')
            plt.colorbar(im, orientation='horizontal')
            self.RMSVE_LOG.append((episode_num, self.compute_RMSVE()))
            
            plt.subplot(313)
            plt.plot([x[0] for x in self.RMSVE_LOG], [x[1] for x in self.RMSVE_LOG])
            plt.xlabel("Episode")
            plt.ylabel("RMSVE", rotation=0, labelpad=20)
            plt.title("Root Mean Squared Value Error")
        self.fig.canvas.draw()
    
    def run_tests(self, values, RMSVE_threshold):
        assert self.true_values is not None, "This function can only be called once the true values are given during " +\
               "runtime."
        self.compute_values_table(values)
        mask = ~(np.isnan(self.values_table) | np.isnan(self.true_values))
        if self.compute_RMSVE() < RMSVE_threshold and np.allclose(self.true_values[mask], self.values_table[mask]):
            pass
        else:
            assert False
            
    def __del__(self):
        pass #plt.close()