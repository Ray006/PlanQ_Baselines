
# ############  to search module in current path ###########################
import os
import sys
addr_ = os.getcwd()
sys.path.append(addr_)
# ############  to search module in current path ###########################
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import numpy.random as npr
from random import shuffle
import tensorflow as tf
import pickle
import argparse
import traceback
from ipdb import set_trace
# set_trace()

#my imports
from baselines.her.MB.policies.policy_random import Policy_Random
from baselines.her.MB.utils.helper_funcs import *
from baselines.her.MB.regressors.dynamics_model import Dyn_Model
from baselines.her.MB.policies.mpc_rollout import MPCRollout
from baselines.her.MB.utils.loader import Loader
from baselines.her.MB.utils.saver import Saver
from baselines.her.MB.utils.data_processor import DataProcessor
from baselines.her.MB.utils.data_structures import *
from baselines.her.MB.utils.convert_to_parser_args import convert_to_parser_args
from baselines.her.MB.utils import config_reader

from baselines.her.MB.policies.mppi import MPPI

SCRIPT_DIR = os.path.dirname(__file__)


#### H10N500a0.3b1e6
class MB_class:
    def __init__(self, env, buffer_size, dims, policy):
        self.buffer_size = buffer_size
        self.rollouts = []
        # self.num_data = 0

        para_dict={
                    'seed': [0],
                    'num_fc_layers': [2],
                    'depth_fc_layers': [400],
                    'ensemble_size': [5],
                    'K': [1],
                    'batchsize': [512],
                    'lr': [0.001],
                    'nEpoch': [30],

                    ##########################
                    ##### controller
                    ##########################
                    'horizon': [5],
                    'num_control_samples': [500],
                    'alpha': [0.3],        ### noise factor

                    'use_exponential': [True],
                    'noise_type': ['gaussian'],
                    'beta': [1000000],        ### exponentially weighted factor, like the one mppi-kappa

                    #################### PlanQ-S or PlanQ-P  (begin) ########################################################
                    'mppi_only': [True],        ### sample actions from planner, PlanQ-S == PDDM + PlanQ
                    # 'mppi_only': [False],        ### sample actions from policy, PlanQ-P == DDPG/HER + PlanQ
                    #################### PlanQ-S or PlanQ-P  (begin) ########################################################

                    'rand_policy_sample_velocities': [False],
                    'mppi_kappa': [10],     ### for mppi planner
                    'mppi_beta': [0.6],
                    'mppi_mag_noise': [0.8],
                  }
        
        para_dict['s_dim'] = [dims['o']]
        para_dict['g_dim'] = [dims['g']]
        para_dict['a_dim'] = [dims['u']]

        self.para = para_dict
        #convert job dictionary to different format
        args_list = config_dict_to_flags(para_dict)
        self.args = convert_to_parser_args(args_list)


        ### set seeds
        npr.seed(self.args.seed)
        tf.set_random_seed(self.args.seed)
        ### data types
        self.args.tf_datatype = tf.float32
        self.args.np_datatype = np.float32
        ### supervised learning noise, added to the training dataset
        self.args.noiseToSignal = 0.01
        #initialize data processor
        self.data_processor = DataProcessor(self.args)
        
        ### init model
        s_dim, a_dim = dims['o'], dims['u']
        inputSize = s_dim + a_dim
        outputSize = s_dim
        acSize = a_dim

        self.dyn_models = Dyn_Model(inputSize, outputSize, acSize, policy, params=self.args)
        self.planner = MPPI(env, self.dyn_models, a_dim, params=self.args)
        self.model_was_learned = False

        self.env = env


    def store_rollout(self, episode):

        rollout = Rollout(episode['o'][0], episode['u'][0])
        self.rollouts.append(rollout)
        
        num_rollouts = len(self.rollouts)
        lenth_each_rollout = self.rollouts[0].actions.shape[0]

        self.num_data = num_rollouts * lenth_each_rollout
        
        if self.num_data > self.buffer_size: 
            self.rollouts.pop(0)
            # print('d')


    def get_data_dim(self):
        assert len(self.rollouts)>0
        state_dim = self.rollouts[0].states.shape[-1]
        action_dim = self.rollouts[0].actions.shape[-1]
        return state_dim, action_dim

    def get_rollout(self):
               
        rollouts_train = []
        rollouts_val = []

        num_mpc_rollouts = len(self.rollouts)
        shuffle(self.rollouts)

        for i,rollout in enumerate(self.rollouts):
            if i<int(num_mpc_rollouts * 0.9): rollouts_train.append(rollout)
            else: rollouts_val.append(rollout)

        if rollouts_val == []:
            rollout=rollouts_train[0]
            rollouts_train.pop[0]
            rollouts_val.append(rollout)

        return rollouts_train, rollouts_val

    def run_job(self):  ## v3, outside session 
        self.model_was_learned = True
        ### get data from the buffer
        rollouts_trainOnPol, rollouts_valOnPol = self.get_rollout()
        # set_trace()
        #convert (rollouts --> dataset)
        dataset_trainOnPol = self.data_processor.convertRolloutsToDatasets(rollouts_trainOnPol)
        dataset_valOnPol = self.data_processor.convertRolloutsToDatasets(rollouts_valOnPol)
        ### update model mean/std
        inputSize, outputSize, acSize = check_dims(dataset_trainOnPol) # just for printing
        self.data_processor.update_stats(self.dyn_models, dataset_trainOnPol) # mean/std of all data
        #preprocess datasets to mean0/std1 + clip actions
        preprocessed_data_trainOnPol = self.data_processor.preprocess_data(dataset_trainOnPol)
        preprocessed_data_valOnPol = self.data_processor.preprocess_data(dataset_valOnPol)
        #convert datasets (x,y,z) --> training sets (inp, outp)
        inputs_onPol, outputs_onPol = self.data_processor.xyz_to_inpOutp(preprocessed_data_trainOnPol)
        inputs_val_onPol, outputs_val_onPol = self.data_processor.xyz_to_inpOutp(preprocessed_data_valOnPol)

        ## train model
        training_loss, training_lists_to_save = self.dyn_models.train(
            inputs_onPol,
            outputs_onPol,
            self.args.nEpoch,
            inputs_val_onPol=inputs_val_onPol,
            outputs_val_onPol=outputs_val_onPol)

        return 0