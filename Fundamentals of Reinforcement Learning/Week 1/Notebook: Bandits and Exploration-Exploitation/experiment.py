#!/usr/bin/env python

from __future__ import print_function
import drifter_distractor_env, switched_drifter_distractor_env
import random_agent, weight_change_agent
from rl_glue import RLGlue
import numpy as np

def main():
    env = drifter_distractor_env.Environment
    env = switched_drifter_distractor_env.Environment

    agents = [random_agent.Agent, weight_change_agent.Agent]
    agent_types = ["absolute_error", "squared_error", "weight_change"]

    for agent_type in agent_types:
        agent = agents[1]

        agent_info = {"num_actions": 4,
                      "action_selection": "softmax",
                      "agent_type": agent_type}
        env_info= {}

        num_runs = 1
        num_steps = 100000

        actions = [0 for _ in range(4)]

        errors = []

        for run in range(num_runs):
            rl_glue = RLGlue(env, agent)
            rl_glue.rl_init(agent_info, env_info)
            rl_glue.rl_start()

            for step in range(num_steps):
                reward, state, action, is_terminal = rl_glue.rl_step()
                actions[action] += 1

        # np.save("data/squared_error", rl_glue.agent.track_actions)
        np.save("data/{}".format(agent_type), rl_glue.agent.track_actions)
        # print(rl_glue.environment.arm_1)
        # print(rl_glue.environment.arm_2)
        # print(rl_glue.environment.arm_3)
        # print(rl_glue.environment.arm_4)
        print(actions)


if __name__ == "__main__":
    main()