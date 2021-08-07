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
import numpy.random as npr
import tensorflow as tf
import time
import math

#my imports
from baselines.her.MB.regressors.feedforward_network import feedforward_network
from ipdb import set_trace


class Dyn_Model:
    """
    This class implements: init, train, get_loss, do_forward_sim
    """

    def __init__(self,
                 inputSize,
                 outputSize,
                 acSize,
                 policy,
                 params,
                 normalization_data=None):

        # init vars
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.acSize = acSize
        self.normalization_data = normalization_data
        self.sess = policy.sess
        self.get_ddpg_act = policy.get_actions
        self.getQval = policy.get_Q_value_for_mb_only

        self.main_network_reuse = policy.main_network_reuse 

        # params
        self.params = params
        self.ensemble_size = self.params.ensemble_size
        self.print_minimal = self.params.print_minimal
        self.batchsize = self.params.batchsize
        self.K = self.params.K
        self.tf_datatype = self.params.tf_datatype

        self.mppi_only = params.mppi_only
        self.H = self.params.horizon
        self.N = self.params.num_control_samples


        self.scope = 'dynamics_model'
        
        with tf.variable_scope(self.scope):

            ## create placeholders
            self.create_placeholders()
            ## clip actions
            # of acs outside of range -1 to 1
            first, second = tf.split(self.inputs_, [(inputSize - self.acSize), self.acSize], 3)
            second = tf.clip_by_value(second, -1, 1)
            self.inputs_clipped = tf.concat([first, second], axis=3)

            ## define forward pass
            self.define_forward_pass()

        self.graph_do_forward_sim()

        mb_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        tf.variables_initializer(mb_var).run()

        # self.sess.run(tf.global_variables_initializer())

        # print(tf.global_variables())


    def create_placeholders(self):
        self.inputs_ = tf.placeholder( self.tf_datatype, shape=[self.ensemble_size, None, self.K, self.inputSize], name='nn_inputs')
        self.labels_ = tf.placeholder( self.tf_datatype, shape=[None, self.outputSize], name='nn_labels')


    def define_forward_pass(self):

        #optimizer
        self.opt = tf.train.AdamOptimizer(self.params.lr)

        self.curr_nn_outputs = []
        self.mses = []
        self.train_steps = []

        for i in range(self.ensemble_size):

            # forward pass through this network
            this_output = feedforward_network(
                self.inputs_clipped[i], self.inputSize, self.outputSize,
                self.params.num_fc_layers, self.params.depth_fc_layers, self.tf_datatype, scope=i)
            self.curr_nn_outputs.append(this_output)

            # loss of this network's predictions
            this_mse = tf.reduce_mean(
                tf.square(self.labels_ - this_output))
            self.mses.append(this_mse)

            # this network's weights
            # this_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=str(i))
            this_theta = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + str(i))

            # train step for this network
            gv = [(g, v) for g, v in self.opt.compute_gradients(
                this_mse, this_theta) if g is not None]
            self.train_steps.append(self.opt.apply_gradients(gv))

        self.predicted_outputs = self.curr_nn_outputs


    def train(self,
              data_inputs_onPol,
              data_outputs_onPol,
              nEpoch,
              inputs_val_onPol=None,
              outputs_val_onPol=None):

        #init vars
        np.random.seed()
        start = time.time()
        training_loss_list = []
        val_loss_list_rand = []
        val_loss_list_onPol = []
        val_loss_list_xaxis = []
        rand_loss_list = []
        onPol_loss_list = []


        data_inputs = data_inputs_onPol.copy()
        data_outputs = data_outputs_onPol.copy()

        #dims
        nData_onPol = data_inputs_onPol.shape[0]
        nData = nData_onPol

        
        mb_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        tf.variables_initializer(mb_var).run()

        #training loop
        for i in range(nEpoch):

            #reset tracking variables to 0
            sum_training_loss = 0
            num_training_batches = 0

            ##############################
            ####### training loss
            ##############################

            #randomly order indices (equivalent to shuffling)
            range_of_indices = np.arange(data_inputs.shape[0])
            all_indices = npr.choice(range_of_indices, size=(data_inputs.shape[0],), replace=False)

            for batch in range(int(math.floor(nData / self.batchsize))):

                #walk through the shuffled new data
                data_inputs_batch = data_inputs[
                    all_indices[batch * self.batchsize:(batch + 1) *
                                self.batchsize]]  #[bs x K x dim]
                data_outputs_batch = data_outputs[all_indices[
                    batch * self.batchsize:(batch + 1) * self.
                    batchsize]]  #[bs x dim]

                #one iteration of feedforward training
                this_dataX = np.tile(data_inputs_batch,
                                     (self.ensemble_size, 1, 1, 1))
                _, losses, outputs, true_output = self.sess.run(
                    [
                        self.train_steps, self.mses, self.curr_nn_outputs,
                        self.labels_
                    ],
                    feed_dict={
                        self.inputs_: this_dataX,
                        self.labels_: data_outputs_batch
                    })
                loss = np.mean(losses)

                training_loss_list.append(loss)
                sum_training_loss += loss
                num_training_batches += 1

            mean_training_loss = sum_training_loss / num_training_batches

            if ((i % 10 == 0) or (i == (nEpoch - 1))):
                ##############################
                ####### validation loss on onPol
                ##############################

                #loss on on-pol validation set
                val_loss_onPol = self.get_loss(inputs_val_onPol,
                                                outputs_val_onPol)
                val_loss_list_onPol.append(val_loss_onPol)

                ##############################
                ####### training loss on onPol
                ##############################

                if (nData_onPol > 0):
                    loss_onPol = self.get_loss(
                        data_inputs_onPol,
                        data_outputs_onPol,
                        fraction_of_data=0.5,
                        shuffle_data=True)
                    onPol_loss_list.append(loss_onPol)

            if not self.print_minimal:
                if ((i % 10) == 0 or (i == (nEpoch - 1))):
                    print("\n=== Epoch {} ===".format(i))
                    print("    train loss: ", mean_training_loss)
                    print("    val onPol: ", val_loss_onPol)

        if not self.print_minimal:
            print("Training duration: {:0.2f} s".format(time.time() - start))

        lists_to_save = dict(
            training_loss_list = training_loss_list,
            val_loss_list_rand = val_loss_list_rand,
            val_loss_list_onPol = val_loss_list_onPol,
            val_loss_list_xaxis = val_loss_list_xaxis,
            rand_loss_list = rand_loss_list,
            onPol_loss_list = onPol_loss_list,)

        #done
        return mean_training_loss, lists_to_save


    def get_loss(self,
                 inputs,
                 outputs,
                 fraction_of_data=1.0,
                 shuffle_data=False):

        """ get prediction error of the model on the inputs """

        #init vars
        nData = inputs.shape[0]
        avg_loss = 0
        iters_in_batch = 0

        if shuffle_data:
            range_of_indices = np.arange(inputs.shape[0])
            indices = npr.choice(
                range_of_indices, size=(inputs.shape[0],), replace=False)

        for batch in range(int(math.floor(nData / self.batchsize) * fraction_of_data)):

            # Batch the training data
            if shuffle_data:
                dataX_batch = inputs[indices[batch * self.batchsize:
                                             (batch + 1) * self.batchsize]]
                dataZ_batch = outputs[indices[batch * self.batchsize:
                                              (batch + 1) * self.batchsize]]
            else:
                dataX_batch = inputs[batch * self.batchsize:(batch + 1) *
                                     self.batchsize]
                dataZ_batch = outputs[batch * self.batchsize:(batch + 1) *
                                      self.batchsize]

            #one iteration of feedforward training
            this_dataX = np.tile(dataX_batch, (self.ensemble_size, 1, 1, 1))
            z_predictions_multiple, losses = self.sess.run(
                [self.curr_nn_outputs, self.mses],
                feed_dict={
                    self.inputs_: this_dataX,
                    self.labels_: dataZ_batch
                })
            loss = np.mean(losses)

            avg_loss += loss
            iters_in_batch += 1

        if iters_in_batch==0:
            return 0
        else:
            return (avg_loss / iters_in_batch)




    def clip_actions(self, input_data):
        ## clip actions to range -1 to 1
        first, second = tf.split(input_data, [(self.inputSize - self.acSize), self.acSize], -1)
        second = tf.clip_by_value(second, -1, 1)
        inputs_clipped = tf.concat([first, second], axis=-1)   
        return inputs_clipped  

    # #by ray
    def transition_planQ(self, obs, act):

        states_preprocessed  = tf.divide((obs  - self.ph_mean_curr_s), self.ph_std_curr_s)
        actions_preprocessed = tf.divide((act  - self.ph_mean_curr_a), self.ph_std_curr_a)
        inputs_list = tf.concat((states_preprocessed, actions_preprocessed), axis=-1)
        inputs_clipped_planQ = self.clip_actions(inputs_list)   #### in cpu version, there is no clip here.

        nn_outputs_planQ = []
        with tf.variable_scope(self.scope, reuse=True):        
            for i in range(self.ensemble_size):
                # set_trace()
                this_output_planQ = feedforward_network( inputs_clipped_planQ[i], self.inputSize, self.outputSize,
                                                self.params.num_fc_layers, self.params.depth_fc_layers, self.tf_datatype, scope=i)
                nn_outputs_planQ.append(this_output_planQ)
        state_differences = tf.multiply( nn_outputs_planQ, self.ph_std_diff_s ) + self.ph_mean_diff_s
        next_states_NK = obs + state_differences

        return next_states_NK

    def graph_do_forward_sim(self):

        '''  current state and action placeholders '''
        self.inputs_curr_s = tf.placeholder( self.tf_datatype, shape=[1, self.params.s_dim], name='curr_s')
        self.inputs_goal = tf.placeholder( self.tf_datatype, shape=[1, 3], name='goal')
        if self.mppi_only:
            self.inputs_curr_a = tf.placeholder( self.tf_datatype, shape=[self.N, self.H, self.params.a_dim], name='curr_a')
        else:
            self.inputs_curr_a = tf.placeholder( self.tf_datatype, shape=[self.N, self.params.a_dim], name='curr_a')

        '''  mean and std placeholders '''
        self.ph_mean_curr_s = tf.placeholder( self.tf_datatype, shape=[self.params.s_dim], name='m_curr_s')
        self.ph_mean_diff_s = tf.placeholder( self.tf_datatype, shape=[self.params.s_dim], name='m_diff_s')
        self.ph_mean_curr_a = tf.placeholder( self.tf_datatype, shape=[self.params.a_dim], name='m_curr_a')
        self.ph_std_curr_s = tf.placeholder( self.tf_datatype, shape=[self.params.s_dim], name='std_curr_s')
        self.ph_std_diff_s = tf.placeholder( self.tf_datatype, shape=[self.params.s_dim], name='std_diff_s')
        self.ph_std_curr_a = tf.placeholder( self.tf_datatype, shape=[self.params.a_dim], name='std_curr_a')

        '''  all placeholders '''
        ph_input = [self.inputs_curr_s, self.inputs_goal, self.inputs_curr_a]  
        ph_norm = [self.ph_mean_curr_s, self.ph_mean_curr_a, self.ph_mean_diff_s, self.ph_std_curr_s, self.ph_std_curr_a, self.ph_std_diff_s]  
        self.all_ph_graph = ph_input + ph_norm

        # set_trace()
        ''' tile state and action -->[ensemble_size, N, a_dim]'''
        obs_N = tf.tile( tf.reshape(self.inputs_curr_s, [1,1,-1]), (self.ensemble_size, self.N, 1))
        self.goal_N = tf.tile( tf.reshape(self.inputs_goal, [1,1,-1]), (self.ensemble_size, self.N, 1))
        
        if self.mppi_only:
            self.actions = tf.transpose(self.inputs_curr_a,(1,0,2))
            self.actions = tf.concat([self.actions, self.actions[0:1]],0)  #### the while_loop "needs" the one more action
            act_N = tf.tile( self.actions[0:1], (self.ensemble_size, 1, 1))
        else:
            act_N = tf.tile( tf.reshape(self.inputs_curr_a, [1,self.N,-1]), (self.ensemble_size, 1, 1))

        ''' do H steps rollouts '''
        obs_ta =  tf.TensorArray(size=self.H, dynamic_size=False, dtype=tf.float32)
        act_ta =  tf.TensorArray(size=self.H, dynamic_size=False, dtype=tf.float32)
        def rollout_loop_body(t, xxx_todo_changeme):
            (obs, act, obs_ta, act_ta) = xxx_todo_changeme
            
            next_obs = self.transition_planQ(obs, act)
            if self.mppi_only:
                next_act = tf.tile( self.actions[t+1:t+2], (self.ensemble_size, 1, 1))
                next_act = tf.reshape(next_act,[self.ensemble_size, self.N, self.params.a_dim])
            else:
                next_act = self.main_network_reuse(next_obs, self.goal_N)

            obs_ta = obs_ta.write(t, obs)
            act_ta = act_ta.write(t, act)
            return t+1, (next_obs, next_act, obs_ta, act_ta)

        _, (final_obs, final_act, obs_ta, act_ta) = tf.while_loop(
            lambda t, _: t < self.H,
            rollout_loop_body,
            [0, (obs_N, act_N, obs_ta, act_ta)]
        )
        ### compile the TensorArrays into useful tensors
        obss = obs_ta.stack()
        final_obs = tf.reshape(final_obs, [1, self.ensemble_size, self.N, self.params.s_dim])
        all_obss = tf.concat([obss, final_obs],0)
        all_acts = act_ta.stack()
        
        ''' reshape the state list and action list for calculating reward, q-value, and done '''
        all_obs = tf.reshape(all_obss, (self.H+1, self.ensemble_size * self.N, self.params.s_dim)) ## [H+1, ensSize, N, statesize] -> [H+1, ensSize * N, statesize]
        all_act = tf.reshape(all_acts, (self.H,   self.ensemble_size * self.N, self.params.a_dim)) ## [H, ensSize, N, asize] -> [H, ensSize * N, asize]

        all_goal= tf.tile( tf.reshape(self.inputs_goal, [1,1,-1]), (self.H, self.ensemble_size*self.N, 1))

        self.s_list = all_obs
        self.curr_s_list = all_obs[:-1]
        self.next_s_list = all_obs[1: ]
        self.curr_a_list = all_act

        self.Q_values = self.main_network_reuse(self.curr_s_list, all_goal, self.curr_a_list)

    def do_forward_sim(self, states, goal, actions):

        ''' input '''
        data_input = [states, goal, actions]
        data_norm = [self.normalization_data.mean_x, self.normalization_data.mean_y, self.normalization_data.mean_z, self.normalization_data.std_x, self.normalization_data.std_y, self.normalization_data.std_z]
        all_input_data = data_input + data_norm

        ''' output '''
        dict_data = dict(list(zip(self.all_ph_graph, all_input_data)))
        output = [self.s_list, self.Q_values]

        ''' run '''
        s_list, q_list = self.sess.run(output, feed_dict=dict_data)

        # set_trace()
        return s_list, q_list
