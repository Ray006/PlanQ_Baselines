
import numpy as np
from ipdb import set_trace


def reward_fun(env, obs, next_obs, goal):
    available_envs={'FetchReach-v1':next_obs[:,:,0:3], 'FetchPush-v1':next_obs[:,:,3:6],'FetchSlide-v1':next_obs[:,:,3:6],'FetchPickAndPlace-v1':next_obs[:,:,3:6],  #3:6
    # 'HandReach-v0':next_obs[:,:,-15:], #-15:
    'HandManipulateBlockRotateZ-v0':next_obs[:,:,-7:],'HandManipulateEggRotate-v0':next_obs[:,:,-7:],'HandManipulatePenRotate-v0':next_obs[:,:,-7:]}  #-7:

    assert env.spec.id in available_envs.keys(),  'Oops! The environment tested is not available!'

    achieved_goal = available_envs[env.spec.id]
    # assume that the reward function is known.
    all_r = env.envs[0].compute_reward(achieved_goal, goal, 'NoNeed')

    return all_r


def calculate_costs(env, resulting_states_list, resulting_Q_list, goal, evaluating, take_exploratory_actions):
   

    H, ensemble_size, N, s_dim= np.array(resulting_states_list).shape
    H=H-1

    resulting_states=np.reshape(resulting_states_list, (H+1, ensemble_size*N, -1))
    resulting_Q=np.reshape(resulting_Q_list, (H, ensemble_size*N, -1))

    #init vars for calculating costs
    costs = np.zeros((N * ensemble_size,))



    goal = np.tile(goal,(H, ensemble_size*N,1))
    all_r = reward_fun(env, resulting_states[:-1], resulting_states[1:], goal)

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
