#!/bin/bash 

source activate flow_2

for i in {-10..1}
do
    for j in {-10..11}
    do
        python /home/thorsten/flow_2/flow/flow/visualize/visualizer_rllib_log_MultiI3W_teamspirit.py /home/thorsten/ray_results/IntersectionExample/PPO_MultiAgentIntersectionEnv_sharedPolicy_TeamSpirit-v0_0_2019-04-08_18-41-29iqy7yfqi 600 --teamspirit_0 $i --teamspirit_1 $j --num-rollouts 100 --render_mode no_render
    done
done
