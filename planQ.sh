#!/bin/bash

read -p "input experiment name, env, GPU, first cpu, number of cpus (e.g. test push 0 0 2): " experiment_name env gpu first_cpu num_cpu_per_exp

case $env in 
reach)
	env=FetchReach-v1
	num_timesteps=1e6
	;;
pap)
	env=FetchPickAndPlace-v1
	num_timesteps=1.5e6
	;;
push)
	env=FetchPush-v1
	num_timesteps=1e6
	;;
dclaw)
	env=dclaw_turn-v0
	num_timesteps=2e5
	;;
block)
	env=HandManipulateBlockRotateZ-v0
	num_timesteps=3e6
	;;
baxter)
	env=BaxterPickAndPlace-v1
	num_timesteps=2e6
	;;
*)
	echo -e "\e[1;31mPlease input valid env.\e[0m"
	exit 1
	;;
esac

cd && cd planQ_baselines
# cd && cd planQ_baselines_real_reach

if [ $? -eq 1 ]; then
	echo "can't find the direction"
	exit 1
fi

screen -dmS $experiment_name

for i in {1..4}
do
	screen -dr $experiment_name -p 0 -X stuff "screen -t win_$i \n"
done

sleep 1s

from_cpu=$[$first_cpu]
to_cpu=$[$first_cpu+$num_cpu_per_exp-1]
	
for i in {0..4}
do

	# content="conda activate gym_baxter"
	content="conda activate baselines"
	screen -dr $experiment_name -p $i -X stuff "$content \n"
	sleep 0.1s

	# content="CUDA_VISIBLE_DEVICES=$gpu taskset --cpu-list $from_cpu-$to_cpu python -m baselines.run --alg=her --env=$env --num_timesteps=$num_timesteps --n_cycles=100 --log_path=/home/ray/data_planQ/real_reach/$experiment_name"	 
	content="CUDA_VISIBLE_DEVICES=$gpu taskset --cpu-list $from_cpu-$to_cpu python -m baselines.run --alg=her --env=$env --num_timesteps=$num_timesteps --n_cycles=100 --log_path=../../data/Ray_data/mp/$experiment_name"	 

	# content="CUDA_VISIBLE_DEVICES=$gpu taskset --cpu-list $from_cpu-$to_cpu python -m baselines.run --alg=her --env=$env --num_timesteps=$num_timesteps --n_cycles=100 --log_path=/home/ray/data_planQ/block/$experiment_name"	 
	# content="CUDA_VISIBLE_DEVICES=$gpu taskset --cpu-list $from_cpu-$to_cpu python -m baselines.run --alg=her --env=$env --num_timesteps=$num_timesteps --n_cycles=10 --log_path=/home/ray/data_planQ/dclaw/$experiment_name"	 
	# screen -dr $experiment_name -p $i -X stuff "$content \n"
	screen -dr $experiment_name -p $i -X stuff "$content"
	screen -dr $experiment_name -p $i -X stuff "\n"
	
	sleep 2s
	
	from_cpu=$[$from_cpu+$num_cpu_per_exp]
	to_cpu=$[$to_cpu+$num_cpu_per_exp]
done
