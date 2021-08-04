
import numpy as np
from ipdb import set_trace


def cost_per_step(t, H, env, pt, prev_pt, costs, goal, q_val):
    
    gamma = 0.98

    available_envs={'FetchReach-v1':pt[:,0:3], 'FetchPush-v1':pt[:,3:6],'FetchSlide-v1':pt[:,3:6],'FetchPickAndPlace-v1':pt[:,3:6],  #3:6
    # 'HandReach-v0':pt[:,-15:], #-15:
    'HandManipulateBlockRotateZ-v0':pt[:,-7:],'HandManipulateEggRotate-v0':pt[:,-7:],'HandManipulatePenRotate-v0':pt[:,-7:]}  #-7:

    assert env.spec.id in available_envs.keys(),  'Oops! The environment tested is not available!'

    achieved_goal = available_envs[env.spec.id]
    goal = np.tile(goal,(achieved_goal.shape[0],1))

    # assume that the reward function is known.
    step_rews = env.envs[0].compute_reward(achieved_goal, goal, 'NoNeed')

    # costs -= pow(gamma,t) * step_rews  ### vanilla PDDM

    costs -= (H-t-1) * pow(gamma,t) * step_rews + pow(gamma,t) * q_val[:,0]   ### ours 

    return costs
##############################
##############################

# def calculate_costs(env, resulting_states_list, resulting_Q_list, goal, evaluating, take_exploratory_actions):
def calculate_costs(env, resulting_states_list, resulting_Q_list, goal, evaluating, take_exploratory_actions):
   

    # from ipdb import set_trace
    # set_trace()

    H, ensemble_size, N, s_dim= np.array(resulting_states_list).shape
    H=H-1

    resulting_states=np.reshape(resulting_states_list, (H+1, ensemble_size*N, -1))
    resulting_Q=np.reshape(resulting_Q_list, (H, ensemble_size*N, -1))

    #init vars for calculating costs
    costs = np.zeros((N * len(resulting_states_list),))
    prev_pt = resulting_states[0]


    ##############################
    ##########  v3  ##############
    ##############################
    # set_trace()
    H = len(resulting_states_list[0]) - 1  ### -1 due to the terminal state   
    for pt_number in range(H):
        #array of "current datapoint" [(ensemble_size*N) x state]
        pt = resulting_states[pt_number + 1]
        q_val = resulting_Q[pt_number]
        costs = cost_per_step(pt_number, H, env, pt, prev_pt, costs, goal, q_val)
        #update
        prev_pt = np.copy(pt)
    ##############################
    ##############################

    #consolidate costs (ensemble_size*N) --> (N)
    new_costs = []
    for i in range(N):
        # 1-a0 1-a1 1-a2 ... 2-a0 2-a1 2-a2 ... 3-a0 3-a1 3-a2...
        new_costs.append(costs[i::N])  #start, stop, step

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
