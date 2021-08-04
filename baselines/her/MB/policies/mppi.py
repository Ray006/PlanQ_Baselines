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
import copy
import matplotlib.pyplot as plt
from ipdb import set_trace
# my imports
from baselines.her.MB.samplers import trajectory_sampler
from baselines.her.MB.utils.helper_funcs import do_groundtruth_rollout
from baselines.her.MB.utils.helper_funcs import turn_acs_into_acsK
from baselines.her.MB.utils.calculate_costs import calculate_costs


class MPPI(object):

    def __init__(self, env, dyn_models, ac_dim, params):

        ###########
        ## params
        ###########
        self.K = params.K
        self.horizon = params.horizon
        self.N = params.num_control_samples
        self.dyn_models = dyn_models
        self.reward_func = None

        #############
        ## init mppi vars
        #############
        self.ac_dim = ac_dim
        self.env=env

        self.use_exponential = params.use_exponential
        self.noise_type = params.noise_type
        self.beta = params.beta
        self.alpha = params.alpha   ## noise factor

        # self.mppi_only = False
        self.mppi_only = params.mppi_only
        self.mppi_mean = np.zeros((self.horizon, self.ac_dim))  #start mean at 0
        self.sample_velocity = params.rand_policy_sample_velocities
        self.sigma = params.mppi_mag_noise * np.ones(self.ac_dim)
        self.mppi_beta = params.mppi_beta
        self.mppi_kappa = params.mppi_kappa
        
        self.max_u = 1



    ###################################################################
    ###################################################################
    #### update action mean using weighted average of the actions (by their resulting scores)
    ###################################################################
    ###################################################################

    ###### modified from pddm
    def mppi_update(self, scores, all_samples):

        if self.mppi_only:

            # print('use mppi planner only instead of ddpg policy')
            #########################
            ## how each sim's score compares to the best score
            ##########################
            S = np.exp(self.mppi_kappa * (scores - np.max(scores)))  # [N,]
            denom = np.sum(S) + 1e-10

            ##########################
            ## weight all actions of the sequence by that sequence's resulting reward
            ##########################
            S_shaped = np.expand_dims(np.expand_dims(S, 1), 2)  #[N,1,1]
            weighted_actions = (all_samples * S_shaped)  #[N x H x acDim]
            self.mppi_mean = np.sum(weighted_actions, 0) / denom

            ##########################
            ## return 1st element of the mean, which corresps to curr timestep
            ##########################
            return self.mppi_mean[0]

        else:
            # self.gamma = 10000
            
            #########################
            ## how each sim's score compares to the best score
            ##########################
            S = np.exp(self.beta * (scores - np.max(scores)))  # [N,]
            denom = np.sum(S) + 1e-10

            # set_trace()
            ##########################
            ## weight all actions of the sequence by that sequence's resulting reward
            ##########################
            S_shaped = np.expand_dims(S, 1)  #[N,1]
            weighted_actions = (all_samples * S_shaped)  #[N x acDim]
            selected_action = np.sum(weighted_actions, 0) / denom

            selected_action = np.tile(selected_action,(1,1))
            ##########################
            ## return 1st element of the mean, which corresps to curr timestep
            ##########################
            return selected_action


    def get_action(self, curr_state, goal, act_ddpg, evaluating, take_exploratory_actions, noise_factor_discount):

        # set_trace()

        if self.mppi_only:
            past_action = self.mppi_mean[0].copy()
            self.mppi_mean[:-1] = self.mppi_mean[1:]

            np.random.seed()  # to get different action samples for each rollout

            ##############################
            #### get action sequences ####
            ##############################
            #sample noise from normal dist, scaled by sigma
            if(self.sample_velocity):
                eps_higherRange = np.random.normal(
                    loc=0, scale=1.0, size=(self.N, self.horizon,
                                            self.ac_dim)) * self.sigma
                lowerRange = 0.3*self.sigma
                num_lowerRange = int(0.1*self.N)
                eps_lowerRange = np.random.normal(
                    loc=0, scale=1.0, size=(num_lowerRange, self.horizon,
                                            self.ac_dim)) * lowerRange
                eps_higherRange[-num_lowerRange:] = eps_lowerRange
                eps_mppi=eps_higherRange.copy()
            else:
                eps_mppi = np.random.normal(
                    loc=0, scale=1.0, size=(self.N, self.horizon,
                                            self.ac_dim)) * self.sigma

            # actions = mean + noise... then smooth the actions temporally
            all_samples = eps_mppi.copy()
            for i in range(self.horizon):

                if(i==0):
                    all_samples[:, i, :] = self.mppi_beta*(self.mppi_mean[i, :] + eps_mppi[:, i, :]) + (1-self.mppi_beta)*past_action
                else:
                    all_samples[:, i, :] = self.mppi_beta*(self.mppi_mean[i, :] + eps_mppi[:, i, :]) + (1-self.mppi_beta)*all_samples[:, i-1, :]

            # The resulting candidate action sequences:
            # all_samples : [N, horizon, ac_dim]
            all_samples = np.clip(all_samples, -1, 1)

            all_acs = all_samples  ### by ray


            # noise_factor = 0.2
            # eps = np.random.normal(loc=0, scale=1.0, size=(self.N, self.ac_dim)) * noise_factor
            # act_ddpg_tile = np.tile(act_ddpg, (self.N, 1))
            # first_acts = act_ddpg_tile + eps

            # set_trace()

            resulting_states_list, resulting_Q_list = self.dyn_models.do_forward_sim_for_mppi_only(curr_state, goal, all_acs)
            resulting_states_list = np.swapaxes(resulting_states_list, 0,1)  #[ensSize, horizon+1, N, statesize]
            resulting_Q_list = np.swapaxes(resulting_Q_list, 0,1)  #[ensSize, horizon+1, N, statesize]

            ############################
            ### evaluate the predicted trajectories
            ############################

            # calculate costs [N,]
            #### here, use costs only!!!
            costs, mean_costs, std_costs = calculate_costs(self.env, resulting_states_list, resulting_Q_list, goal, evaluating, take_exploratory_actions)

            # uses all paths to update action mean (for horizon steps)
            # Note: mppi_update needs rewards, so pass in -costs
            selected_action = self.mppi_update(-costs, all_samples)

            selected_action = np.tile(selected_action,(1,1))

            # set_trace()

            return selected_action


        else:

            np.random.seed()  # to get different action samples for each rollout
            
            # from ipdb import set_trace
            # set_trace()

            # if noise_factor_discount==0:
            #     return act_ddpg

            if self.noise_type == 'gaussian':
                ##### gaussian noise
                # eps = np.random.normal(loc=0, scale=1.0, size=(self.N, self.ac_dim)) * noise_factor
                eps = np.random.normal(loc=0, scale=1.0, size=(self.N, self.ac_dim)) * self.alpha * noise_factor_discount
            if self.noise_type == 'uniform':
                ##### gaussian noise
                eps = np.random.uniform(low=-self.max_u, high=self.max_u, size=(self.N, self.ac_dim)) * self.alpha * noise_factor_discount

            act_ddpg_tile = np.tile(act_ddpg, (self.N, 1))
            first_acts = act_ddpg_tile + eps

            first_acts = np.clip(first_acts, -self.max_u, self.max_u)  #### actions are \in [-1,1]
            # from ipdb import set_trace
            # set_trace()

            #################################################
            ### Get result of executing those candidate action sequences
            #################################################
            resulting_states_list, resulting_Q_list = self.dyn_models.do_forward_sim(curr_state, goal, first_acts)
            resulting_states_list = np.swapaxes(resulting_states_list, 0,1)  #[ensSize, horizon+1, N, statesize]
            resulting_Q_list = np.swapaxes(resulting_Q_list, 0,1)  #[ensSize, horizon+1, N, statesize]

            ############################
            ### evaluate the predicted trajectories
            ############################

            # calculate costs [N,]
            #### here, use mean_costs only!!!
            costs, mean_costs, std_costs = calculate_costs(self.env, resulting_states_list, resulting_Q_list, goal, evaluating, take_exploratory_actions)


            # from ipdb import set_trace
            # set_trace()

            if self.use_exponential:
                # ######### use exponential function to weight actions
                selected_action = self.mppi_update(-costs, first_acts)
            
            else:
                #### don't use exponential func to weight actions.
                if (costs == costs.min()).all():
                    selected_action = act_ddpg
                    # self.act_NN+=1
                else:
                    # self.act_plan+=1
                    # set_trace()
                    idx = np.where(costs==costs.min())[0]
                    # selected_action = first_acts[idx][0]
                    selected_action = first_acts[idx].mean(axis=0)
                    
                    selected_action = np.tile(selected_action,(1,1))

            return selected_action
