# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from ipdb import set_trace

def cost_per_step(env, pt, prev_pt, costs, goal, q_val):
    
    # compute_reward(self, achieved_goal, goal, info)


    ### assume that the reward function is known
    new_ag = pt[:,3:6]         
    d = np.linalg.norm(new_ag - goal, axis=-1)
    # step_rews = -(d > 0.2).astype(np.float32)
    step_rews = -(d > 0.05).astype(np.float32)

    costs -= step_rews + 0.01*q_val[:,0]
    return costs

def calculate_costs(env, resulting_states_list, resulting_Q_list, goal):
    """Rank various predicted trajectories (by cost)

    Args:
        resulting_states_list :
            predicted trajectories
            [ensemble_size, horizon+1, N, statesize]
        actions :
            the actions that were "executed" in order to achieve the predicted trajectories
            [N, h, acsize]
        reward_func :
            calculates the rewards associated with each state transition in the predicted trajectories
        evaluating :
            determines whether or not to use model-disagreement when selecting which action to execute
            bool
        take_exploratory_actions :
            determines whether or not to use model-disagreement when selecting which action to execute
            bool

    Returns:
        cost_for_ranking : cost associated with each candidate action sequence [N,]
    """

    ensemble_size = len(resulting_states_list)
    ###########################################################
    ## some reshaping of the predicted trajectories to rate
    ###########################################################

    N = len(resulting_states_list[0][0])

    #resulting_states_list is [ensSize, H+1, N, statesize]
    resulting_states = []
    for timestep in range(len(resulting_states_list[0])): # loops over H+1
        all_per_timestep = []
        for entry in resulting_states_list: # loops over ensSize
            all_per_timestep.append(entry[timestep])
        all_per_timestep = np.concatenate(all_per_timestep)  #[ensSize*N, statesize]
        resulting_states.append(all_per_timestep)
    #resulting_states is now [H+1, ensSize*N, statesize]

    #resulting_Q_list is [ensSize, H+1, N, Qsize]
    resulting_Q = []
    for timestep in range(len(resulting_Q_list[0])): # loops over H+1
        all_per_timestep = []
        for entry in resulting_Q_list: # loops over ensSize
            all_per_timestep.append(entry[timestep])
        all_per_timestep = np.concatenate(all_per_timestep)  #[ensSize*N, Qsize]
        resulting_Q.append(all_per_timestep)
    #resulting_Q is now [H+1, ensSize*N, Qsize]


    ###########################################################
    ## calculate costs associated with each predicted trajectory
    ######## treat each traj from each ensemble as just separate trajs
    ###########################################################

    #init vars for calculating costs
    costs = np.zeros((N * len(resulting_states_list),))
    prev_pt = resulting_states[0]

    # set_trace()
    #accumulate cost over each timestep
    for pt_number in range(len(resulting_states_list[0]) - 1):

        #array of "current datapoint" [(ensemble_size*N) x state]
        pt = resulting_states[pt_number + 1]
        q_val = resulting_Q[pt_number]
        costs = cost_per_step(env, pt, prev_pt, costs, goal, q_val)
        #update
        prev_pt = np.copy(pt)

    #consolidate costs (ensemble_size*N) --> (N)
    new_costs = []
    for i in range(N):
        # 1-a0 1-a1 1-a2 ... 2-a0 2-a1 2-a2 ... 3-a0 3-a1 3-a2...
        new_costs.append(costs[i::N])  #start, stop, step

    #mean and std cost (across ensemble) [N,]
    mean_cost = np.mean(new_costs, 1)
    std_cost = np.std(new_costs, 1)


    return mean_cost, std_cost
