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


class MB_class:
    def __init__(self, env, buffer_size, dims, policy):
        self.buffer_size = buffer_size
        self.rollouts = []
        # self.num_data = 0

        para_dict={
                    'use_gpu': [1],
                    # 'use_gpu': [0],
                    'gpu_frac': [0.5],
                    #########################
                    ##### run options
                    #########################
                    'job_name': ['ant'],
                    'seed': [0],
                    #########################
                    ##### experiment options
                    #########################
                    ## noise
                    'make_aggregated_dataset_noisy': [True],
                    'make_training_dataset_noisy': [True],
                    'rollouts_noise_actions': [False],
                    'rollouts_document_noised_actions': [False],

                    ##########################
                    ##### dynamics model
                    ##########################
                    ## arch
                    'num_fc_layers': [2],
                    'depth_fc_layers': [400],
                    # 'ensemble_size': [3],
                    'ensemble_size': [5],
                    'K': [1],
                    ## model training
                    'batchsize': [512],
                    'lr': [0.001],
                    'nEpoch': [30],
                    # 'nEpoch': [40],
                    # 'nEpoch_init': [30],
                    ##########################
                    ##### controller
                    ##########################
                    ## MPC
                    # 'horizon': [5],
                    'horizon': [10],
                    # 'horizon': [15],
                    'num_control_samples': [500],
                    'controller_type': ['mppi'],

                    ## exponential
                    'use_exponential': [True],
                    # 'use_exponential': [False],
                    'alpha': [0.5],        ### noise factor
                    'noise_type': ['gaussian'],
                    # 'noise_type': ['uniform'],
                    'beta': [1000000],        ### exponentially weighted factor, like the one mppi-kappa
                    # 'beta': [20],        ### exponentially weighted factor, like the one mppi-kappa



                    # 'mppi_only': [True],        ### use mppi planner or not
                    'mppi_only': [False],        ### use mppi planner or not

                    # 'rand_policy_sample_velocities': [True],
                    'rand_policy_sample_velocities': [False],
                    'mppi_kappa': [10],     ### for mppi planner
                    # 'mppi_kappa': [50],     ### hand for mppi planner
                    'mppi_beta': [0.6],
                    'mppi_mag_noise': [0.8],

                  }

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
        # #initialize saver
        # saver = Saver(save_dir, sess)
        # saver_data = DataPerIter()

        # self.sess = tf.Session(config=get_gpu_config(self.args.use_gpu, self.args.gpu_frac))
        
        ### init model
        s_dim, a_dim = dims['o'], dims['u']
        inputSize = s_dim + a_dim
        outputSize = s_dim
        acSize = a_dim

        self.dyn_models = Dyn_Model(inputSize, outputSize, acSize, policy, params=self.args)
        self.planner = MPPI(env, self.dyn_models, a_dim, params=self.args)
        self.model_was_learned = False


    def store_rollout(self, episode):

        rollout = Rollout(episode['o'][0], episode['u'][0])
        self.rollouts.append(rollout)
        
        num_rollouts = len(self.rollouts)
        lenth_each_rollout = self.rollouts[0].actions.shape[0]
        num_data = num_rollouts * lenth_each_rollout
        
        if num_data > self.buffer_size: 
            # set_trace()
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
            if i<int(num_mpc_rollouts * 0.9):
                rollouts_train.append(rollout)
            else:
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

        # set_trace()
        # tf.reset_default_graph()  #### ????   
        # with tf.Session(config=get_gpu_config(self.args.use_gpu, self.args.gpu_frac)) as sess:

        # #re-initialize all vars (randomly) if training from scratch
        # self.sess.run(tf.global_variables_initializer())

        ## train model
        training_loss, training_lists_to_save = self.dyn_models.train(
            inputs_onPol,
            outputs_onPol,
            self.args.nEpoch,
            inputs_val_onPol=inputs_val_onPol,
            outputs_val_onPol=outputs_val_onPol)

            # #########################################################
            # ### save everything about this iter of model training
            # #########################################################
            # trainingLoss_perIter.append(training_loss)

            # saver_data.training_losses = trainingLoss_perIter
            # saver_data.training_lists_to_save = training_lists_to_save

            # saver_data.train_rollouts_onPol = rollouts_trainOnPol
            # saver_data.val_rollouts_onPol = rollouts_valOnPol

            # ### save all info from this training iteration
            # saver.save_model()
            # saver.save_training_info(saver_data)

        return 0





# def main():

#     #####################
#     # training args
#     #####################

#     parser = argparse.ArgumentParser(
#         # Show default value in the help doc.
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

#     parser.add_argument(
#         '-c',
#         '--config',
#         nargs='*',
#         help=('Path to the job data config file. This is specified relative '
#             'to working directory'))

#     parser.add_argument(
#         '-o',
#         '--output_dir',
#         default='output',
#         help=
#         ('Directory to output trained policies, logs, and plots. A subdirectory '
#          'is created for each job. This is speficified relative to  '
#          'working directory'))

#     parser.add_argument('--use_gpu', action="store_true")
#     parser.add_argument('-frac', '--gpu_frac', type=float, default=0.9)
#     general_args = parser.parse_args()

#     #####################
#     # job configs
#     #####################

#     # Get the job config files
#     jobs = config_reader.process_config_files(general_args.config)
#     assert jobs, 'No jobs found from config.'

#     # Create the output directory if not present.
#     output_dir = general_args.output_dir
#     if not os.path.isdir(output_dir):
#         os.makedirs(output_dir)
#     output_dir = os.path.abspath(output_dir)

#     # Run separate experiment for each variant in the config
#     for index, job in enumerate(jobs):

#         #add an index to jobname, if there is more than 1 job
#         if len(jobs)>1:
#             job['job_name'] = '{}_{}'.format(job['job_name'], index)

#         #convert job dictionary to different format
#         args_list = config_dict_to_flags(job)
#         args = convert_to_parser_args(args_list)

#         #copy some general_args into args
#         args.use_gpu = general_args.use_gpu
#         args.gpu_frac = general_args.gpu_frac

#         #directory name for this experiment
#         job['output_dir'] = os.path.join(output_dir, job['job_name'])

#         ################
#         ### run job
#         ################

#         try:
#             run_job(args, job['output_dir'])
#         except (KeyboardInterrupt, SystemExit):
#             print('Terminating...')
#             sys.exit(0)
#         except Exception as e:
#             print('ERROR: Exception occured while running a job....')
#             traceback.print_exc()

# if __name__ == '__main__':
#     main()
