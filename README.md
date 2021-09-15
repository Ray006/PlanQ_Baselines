<!-- <img src="data/logo.jpg" width=25% align="right" /> [![Build status](https://travis-ci.org/openai/baselines.svg?branch=master)](https://travis-ci.org/openai/baselines) -->

This repo is built on top of OpenAI Baselines(https://github.com/openai/baselines). To run the PlanQ algorithm, install Baselines first according to OpenAI guide.  

## Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```
- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases, you may use
    ```bash 
    pip install tensorflow-gpu==1.14 # if you have a CUDA-compatible gpu and proper drivers
    ```
- Install baselines package
    ```bash
    pip install -e .
    ```
- Install MuJoCo
    ```bash
    mujoco=1.5 (http://www.mujoco.org)
    ```

## Run
<!-- CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 6-10 python -m baselines.run --alg=her --env=dclaw_turn-v0 --num_timesteps=4e6 --n_cycles=100 --log_path=../../data/Ray_data/mp/test -->
Before you run PlanQ, Make sure you can run OpenAI Baselines successfully.

```bash
cd planQ_baselines
```

##### FetchPush-v1
```
python -m baselines.run --alg=her --env=FetchPush-v1 --num_timesteps=1e6 --n_cycles=100 --log_path=../../data/push
```
##### FetchPickAndPlace-v1
```
python -m baselines.run --alg=her --env=FetchPickAndPlace-v1 --num_timesteps=1.5e6 --n_cycles=100 --log_path=../../data/PickAndPlace
```

##### dclaw_turn-v0
```
python -m baselines.run --alg=her --env=dclaw_turn-v0 --num_timesteps=2e5 --n_cycles=10 --log_path=../../data/dclaw_turn
```
##### HandManipulateBlockRotateZ-v0
```
python -m baselines.run --alg=her --env=HandManipulateBlockRotateZ-v0 --num_timesteps=3e6 --n_cycles=100 --log_path=../../data/block
```

## Which Algo to run?
we can run different algorithms (DDPG, HER, PDDM, PlanQ-S, PlanQ-P) with different settings.

##### 1. To select algorithm DDPG, make sure:

 -   'replay_strategy': 'none',   
 -   'use_planner': False,        
        ```bash
        PATH: planQ_baselines/baselines/her/experiment/config.py
        ```

##### 2. To select algorithm PlanQ-P(PDDM+PlanQ), make sure:

 -   'replay_strategy': 'none',   
 -   'use_planner': True,
        ```bash
        PATH: planQ_baselines/baselines/her/experiment/config.py
        ```
    
 -   'mppi_only': [False],
        ```bash
        PATH: planQ_baselines/baselines/her/MB/model_based.py
        ```
    
 -   costs -= (self.H-t-1) * pow(gamma,t) * step_rews + pow(gamma,t) * q_val[:,0] ## use Q-values
        ```bash
        PATH: planQ_baselines/baselines/her/MB/policies/mppi.py
        ```

##### 3. To select algorithm DDPG+HER, make sure:

 -   'replay_strategy': 'future',   
 -   'use_planner': False,        
        ```bash
        PATH: planQ_baselines/baselines/her/experiment/config.py
        ```
##### 4. To select algorithm PlanQ-P(PDDM+HER+PlanQ), make sure:

 -   'replay_strategy': 'future',   
 -   'use_planner': True,
        ```bash
        PATH: planQ_baselines/baselines/her/experiment/config.py
        ```
    
 -   'mppi_only': [False],
        ```bash
        PATH: planQ_baselines/baselines/her/MB/model_based.py
        ```
    
 -   costs -= (self.H-t-1) * pow(gamma,t) * step_rews + pow(gamma,t) * q_val[:,0] ## use Q-values
        ```bash
        PATH: planQ_baselines/baselines/her/MB/policies/mppi.py
        ```

##### 5. To select algorithm PDDM, make sure:

 -   'use_planner': True,
        ```bash
        PATH: planQ_baselines/baselines/her/experiment/config.py
        ```
 -   'mppi_only': [True],
        ```bash
        PATH: planQ_baselines/baselines/her/MB/model_based.py
        ```
 -   costs -= pow(gamma,t) * step_rews  ## use the immediate rewards only
        ```bash
        PATH: planQ_baselines/baselines/her/MB/policies/mppi.py
        ```

##### 6. To select algorithm PlanQ-S(PDDM+PlanQ), make sure:

 -   'use_planner': True,
        ```bash
        PATH: planQ_baselines/baselines/her/experiment/config.py
        ```
 -   'mppi_only': [True],
        ```bash
        PATH: planQ_baselines/baselines/her/MB/model_based.py
        ```
 -   costs -= (self.H-t-1) * pow(gamma,t) * step_rews + pow(gamma,t) * q_val[:,0] ## use Q-values
        ```bash
        PATH: planQ_baselines/baselines/her/MB/policies/mppi.py
        ```




## Modification
Where have we modified to the Baselines?

1. Added an environment folder 'dclaw_for_planQ' in path:
```bash
planQ_baselines/dclaw_for_planQ
```

2. Added an dynamics model trainning folder 'MB' in path:

```bash
planQ_baselines/baselines/her/MB
```

