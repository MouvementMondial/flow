#!/bin/bash 

source activate flow_2

for i in {1..1000}
do
    python /home/thorsten/flow_2/flow/flow/visualize/visualizer_rllib_log_MultiI3W.py /home/thorsten/ray_results/IntersectionExample/PPO_MultiAgentIntersectionEnv-v0_0_2019-02-22_00-55-50s2hylffo $i --num-rollouts 10 --render_mode no_render
done
