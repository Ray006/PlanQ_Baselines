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

import sys
import argparse

def convert_to_parser_args(args_source=sys.argv[1:]):

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_minimal', action="store_true")
    parser.add_argument('--make_training_dataset_noisy', action="store_true")

    parser.add_argument('--s_dim', type=int, default=2)
    parser.add_argument('--g_dim', type=int, default=2)
    parser.add_argument('--a_dim', type=int, default=2)

    parser.add_argument('--rand_policy_sample_velocities', action="store_true")
    # arch
    parser.add_argument('--num_fc_layers', type=int, default=2)
    parser.add_argument('--depth_fc_layers', type=int, default=64)
    parser.add_argument('--ensemble_size', type=int, default=1) #ensemble size
    parser.add_argument('--K', type=int, default=1) #number of past states for input to model

    # model training
    parser.add_argument('--batchsize', type=int, default=500) #batchsize for each gradient step
    parser.add_argument('--lr', type=float, default=0.001) #learning rate
    parser.add_argument('--nEpoch', type=int, default=40) #epochs of training

    #######################
    ### controller
    #######################

    # MPC
    parser.add_argument('--horizon', type=int, default=7) #planning horizon
    parser.add_argument('--num_control_samples', type=int, default=700) #number of candidate ac sequences
    
    parser.add_argument('--use_exponential', action="store_true")#ray
    parser.add_argument('--noise_type', type=str, default='gaussian')#ray
    parser.add_argument('--alpha', type=float, default=0.3) #ray
    parser.add_argument('--beta', type=float, default=1000) #ray

    parser.add_argument('--mppi_only', action="store_true")#ray
    # mppi
    parser.add_argument('--mppi_kappa', type=float, default=1.0) #reward weighting
    parser.add_argument('--mppi_mag_noise', type=float, default=0.9) #magnitude of sampled noise
    parser.add_argument('--mppi_beta', type=float, default=0.8) #smoothing

    args = parser.parse_args(args_source)
    return args
