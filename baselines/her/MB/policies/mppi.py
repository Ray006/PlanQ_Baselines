
import numpy as np
import copy
import matplotlib.pyplot as plt
from ipdb import set_trace
# my imports
from baselines.her.MB.samplers import trajectory_sampler
from baselines.her.MB.utils.helper_funcs import do_groundtruth_rollout
from baselines.her.MB.utils.helper_funcs import turn_acs_into_acsK

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
            S = np.exp(self.mppi_kappa * (scores - np.max(scores)))  # [N,]
            denom = np.sum(S) + 1e-10

            S_shaped = np.expand_dims(np.expand_dims(S, 1), 2)  #[N,1,1]
            weighted_actions = (all_samples * S_shaped)  #[N x H x acDim]
            self.mppi_mean = np.sum(weighted_actions, 0) / denom

            return self.mppi_mean[0]

        else:
            # self.gamma = 10000
            
            S = np.exp(self.beta * (scores - np.max(scores)))  # [N,]
            denom = np.sum(S) + 1e-10

            # set_trace()
            S_shaped = np.expand_dims(S, 1)  #[N,1]
            weighted_actions = (all_samples * S_shaped)  #[N x acDim]
            selected_action = np.sum(weighted_actions, 0) / denom

            selected_action = np.tile(selected_action,(1,1))

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
            costs, mean_costs, std_costs = self.calculate_costs(self.env, resulting_states_list, resulting_Q_list, goal, evaluating, take_exploratory_actions)

            # uses all paths to update action mean (for horizon steps)
            # Note: mppi_update needs rewards, so pass in -costs
            selected_action = self.mppi_update(-costs, all_samples)

            selected_action = np.tile(selected_action,(1,1))

            # set_trace()
            return selected_action

        else:
            np.random.seed()  # to get different action samples for each rollout
        
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


            resulting_states_list, resulting_Q_list = self.dyn_models.do_forward_sim(curr_state, goal, first_acts)

            costs, mean_costs, std_costs = self.calculate_costs(self.env, resulting_states_list, resulting_Q_list, goal, evaluating, take_exploratory_actions)

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



    def reward_fun(self, env, obs, next_obs, goal):
        available_envs={'FetchReach-v1':next_obs[:,:,0:3], 'FetchPush-v1':next_obs[:,:,3:6],'FetchSlide-v1':next_obs[:,:,3:6],'FetchPickAndPlace-v1':next_obs[:,:,3:6],  #3:6
        # 'HandReach-v0':next_obs[:,:,-15:], #-15:
        'HandManipulateBlockRotateZ-v0':next_obs[:,:,-7:],'HandManipulateEggRotate-v0':next_obs[:,:,-7:],'HandManipulatePenRotate-v0':next_obs[:,:,-7:]}  #-7:

        assert env.spec.id in available_envs.keys(),  'Oops! The environment tested is not available!'

        achieved_goal = available_envs[env.spec.id]
        # assume that the reward function is known.
        all_r = env.envs[0].compute_reward(achieved_goal, goal, 'NoNeed')

        return all_r


    def calculate_costs(self, env, resulting_states_list, resulting_Q_list, goal, evaluating, take_exploratory_actions):
    
        H, ensemble_size, N, s_dim= np.array(resulting_states_list).shape
        H=H-1

        resulting_states=np.reshape(resulting_states_list, (H+1, ensemble_size*N, -1))
        resulting_Q=np.reshape(resulting_Q_list, (H, ensemble_size*N, -1))

        #init vars for calculating costs
        costs = np.zeros((N * ensemble_size,))

        goal = np.tile(goal,(H, ensemble_size*N,1))
        all_r = self.reward_fun(env, resulting_states[:-1], resulting_states[1:], goal)

        costs1 = np.zeros((N * ensemble_size,))
        gamma = 0.98
        for t in range(H):
            
            q_val = resulting_Q[t]
            step_rews = all_r[t] 

            # costs -= pow(gamma,t) * step_rews  ### vanilla PDDM
            costs1 -= (H-t-1) * pow(gamma,t) * step_rews + pow(gamma,t) * q_val[:,0]   ### ours 

        
        # set_trace()
        scores_reshape = costs.reshape(ensemble_size, N)
        new_costs = np.swapaxes(scores_reshape, 0,1)    

        #mean and std cost (across ensemble) [N,]
        mean_cost = np.mean(new_costs, 1)
        std_cost = np.std(new_costs, 1)


        #rank by rewards
        if evaluating:
            cost_for_ranking = mean_cost
        #sometimes rank by model disagreement, and sometimes rank by rewards
        else:
            if take_exploratory_actions:
                cost_for_ranking = mean_cost - 4 * std_cost
                # print("   ****** taking exploratory actions for this rollout")
            else:
                cost_for_ranking = mean_cost


        # return mean_cost, std_cost
        return cost_for_ranking, mean_cost, std_cost