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
from functools import partial
from baselines.her.MB.utils.data_structures import *
from baselines.her.MB.utils.helper_funcs import add_noise


class DataProcessor:

    def __init__(self,params):

        #vars
        self.params = params
        self.normalization_data = MeanStd()
        self.array_datatype = partial(np.array, dtype=params.np_datatype)

    def get_normalization_data(self):
        return self.normalization_data

    def set_normalization_data(self, model, normalization_data):
        self.normalization_data = normalization_data
        self.update_model_normalization(model)

    def update_stats(self, model, dataset_trainOnPol):

        temp_x = dataset_trainOnPol.dataX
        temp_y = dataset_trainOnPol.dataY
        temp_z = dataset_trainOnPol.dataZ


        #clip actions, before calculating mean/std
        temp_y = np.clip(temp_y, -1, 1)

        # recalculate mean/std based on the full (rand+onPol) updated dataset
        self.normalization_data.mean_x = np.mean(
            np.concatenate(temp_x, 0), axis=0)
        self.normalization_data.std_x = np.std(
            np.concatenate(temp_x, 0), axis=0)

        self.normalization_data.mean_y = np.mean(
            np.concatenate(temp_y, 0), axis=0)
        self.normalization_data.std_y = np.std(
            np.concatenate(temp_y, 0), axis=0)

        self.normalization_data.mean_z = np.mean(temp_z, axis=0)
        self.normalization_data.std_z = np.std(temp_z, axis=0)

        #update the mean/std values of model
        self.update_model_normalization(model)

    def preprocess_data(self, dataset):

        #make data mean0/std1
        if dataset.dataX.shape[0]>0:
            x_preprocessed = (dataset.dataX - self.normalization_data.mean_x
                             ) / self.normalization_data.std_x
            y_preprocessed = (dataset.dataY - self.normalization_data.mean_y
                             ) / self.normalization_data.std_y
            z_preprocessed = (dataset.dataZ - self.normalization_data.mean_z
                             ) / self.normalization_data.std_z

            #clip actions to (-1,1)
            y_preprocessed = np.clip(y_preprocessed, -1, 1)

            return Dataset(x_preprocessed, y_preprocessed, z_preprocessed)
        else:
            return Dataset()

    def xyz_to_inpOutp(self, dataset_preprocessed):

        if dataset_preprocessed.dataX.shape[0]>0:
            inputs = np.concatenate(
                (dataset_preprocessed.dataX, dataset_preprocessed.dataY), axis=2)
            outputs = np.copy(dataset_preprocessed.dataZ)
            return inputs, outputs
        else:
            return np.array([]), np.array([])

    def update_model_normalization(self, model):

        if type(model)==list:
            for model_indiv in model:
                model_indiv.normalization_data = copy.deepcopy(
                    self.normalization_data)
        else:
            model.normalization_data = copy.deepcopy(self.normalization_data)

    def convertRolloutsToDatasets(self, rollouts):

        if len(rollouts)==0:
            return Dataset()

        all_states_K = []
        all_actions_K = []
        all_differences_K = []
        all_differences_single = []
        K = self.params.K

        for rollout in rollouts:

            states = rollout.states
            actions = rollout.actions

            #the past K timesteps
            for step in range(K, len(states)):
                all_states_K.append(states[step - K:step])
                all_actions_K.append(actions[step - K:step])
                all_differences_single.append(states[step] - states[step - 1])

        #training labels:
        dataZ = np.array(all_differences_single)

        #turn the list of rollouts into large arrays of data
        dataX = self.array_datatype(np.array(all_states_K))
        dataY = self.array_datatype(np.array(all_actions_K))
        dataZ = self.array_datatype(np.array(dataZ))

        #add some supervised learning noise (to help model training)
        if self.params.make_training_dataset_noisy:
            dataX = add_noise(dataX, self.params.noiseToSignal)
            dataZ = add_noise(dataZ, self.params.noiseToSignal)

        return Dataset(dataX, dataY, dataZ)
