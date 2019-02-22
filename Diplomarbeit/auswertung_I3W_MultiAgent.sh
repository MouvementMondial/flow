#!/bin/bash 

source activate flow_2

for i in {1..365}
do
    python /home/thorsten/flow_2/flow/flow/visualize/visualizer_rllib_log_MultiI3W.py /home/thorsten/ray_results/IntersectionExample/PPO_MultiAgentIntersectionEnv-v0_0_2019-02-21_11-07-01v5vhj7jt $i --num-rollouts 20 --render_mode no_render
done
