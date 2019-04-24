"""
Environment used to train a stop-and-go dissipating controller.

This is the environment that was used in:

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and
Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol.
abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465
"""

import numpy as np
from gym.spaces.box import Box
from flow.multiagent_envs.multiagent_env import MultiEnv

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # bounds on the ranges of ring road lengths the autonomous vehicle is
    # trained on
    'ring_length': [220, 270],
}


class MultiWaveAttenuationPOEnv(MultiEnv):
    """Multiagent shared model version of WaveAttenuationPOEnv

    Intended to work with Lord Of The Rings Scenario.
    Note that this environment current
    only works when there is one autonomous vehicle
    on each ring.

    Required from env_params: See parent class

    States
        See parent class
    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class

    """

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=0, high=1, shape=(3,), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        add_params = self.scenario.net_params.additional_params
        num_rings = add_params['num_rings']
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(int(self.scenario.vehicles.num_rl_vehicles / num_rings), ),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        obs = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

            # normalizers
            max_speed = 15.
            max_length = self.env_params.additional_params['ring_length'][1]

            observation = np.array([
                self.k.vehicle.get_speed(rl_id) / max_speed,
                (self.k.vehicle.get_speed(lead_id) -
                 self.k.vehicle.get_speed(rl_id))
                / max_speed,
                self.k.vehicle.get_headway(rl_id) / max_length
            ])
            obs.update({rl_id: observation})

        return obs

    def _apply_rl_actions(self, rl_actions):
        """Split the accelerations by ring"""
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            accel = list(rl_actions.values())
            self.k.vehicle.apply_acceleration(rl_ids, accel)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rew = {}
        for rl_id in rl_actions.keys():
            edge_id = rl_id.split('_')[1]
            edges = self.gen_edges(edge_id)
            vehs_on_edge = self.k.vehicle.get_ids_by_edge(edges)
            vel = np.array([
                self.k.vehicle.get_speed(veh_id)
                for veh_id in vehs_on_edge
            ])
            if any(vel < -100) or kwargs['fail']:
                return 0.

            target_vel = self.env_params.additional_params['target_velocity']
            max_cost = np.array([target_vel] * len(vehs_on_edge))
            max_cost = np.linalg.norm(max_cost)

            cost = vel - target_vel
            cost = np.linalg.norm(cost)

            rew[rl_id] = max(max_cost - cost, 0) / max_cost
        return rew

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
            self.k.vehicle.set_observed(lead_id)

    def gen_edges(self, i):
        """Return the edges corresponding to the rl id"""
        return ['top_{}'.format(i), 'left_{}'.format(i),
                'right_{}'.format(i), 'bottom_{}'.format(i)]
                
class MultiWaveAttenuationPOEnv_dipl(MultiEnv):
    """Multiagent shared model version of WaveAttenuationPOEnv

    Intended to work with Lord Of The Rings Scenario.
    Note that this environment current
    only works when there is one autonomous vehicle
    on each ring.

    Required from env_params: See parent class

    States
        See parent class
    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class

    """

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=0,
            high=1,
            shape=(4,),
            dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-3,
            high=3,
            shape=(1, ),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        obs = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

            # normalizers
            max_speed = 15.0
            max_length = 206.2

            pos_rl_id = self.k.vehicle.get_x_by_id(rl_id)
            distance_kp = max_length - pos_rl_id

            observation = np.array([
                self.k.vehicle.get_speed(rl_id) / max_speed,
                (self.k.vehicle.get_speed(lead_id) -
                 self.k.vehicle.get_speed(rl_id))
                / max_speed,
                self.k.vehicle.get_headway(rl_id) / max_length,
                distance_kp
            ])
            obs.update({rl_id: observation})

        return obs

    def _apply_rl_actions(self, rl_actions):
        """Split the accelerations by ring"""
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            accel = list(rl_actions.values())
            self.k.vehicle.apply_acceleration(rl_ids, accel)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rew = {}
        
        for rl_id in rl_actions.keys():            
            target_vel = self.env_params.additional_params['target_velocity']
            rew[rl_id] = self.k.vehicle.get_speed(rl_id) / target_vel

        return rew

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
            self.k.vehicle.set_observed(lead_id)

    def gen_edges(self, i):
        """Return the edges corresponding to the rl id"""
        return ['top_{}'.format(i), 'left_{}'.format(i),
                'right_{}'.format(i), 'bottom_{}'.format(i)]

    def step(self, rl_actions):
        """Advance the environment by one step.

        Assigns actions to autonomous and human-driven agents (i.e. vehicles,
        traffic lights, etc...). Actions that are not assigned are left to the
        control of the simulator. The actions are then used to advance the
        simulator by the number of time steps requested per environment step.

        Results from the simulations are processed through various classes,
        such as the Vehicle and TrafficLight kernels, to produce standardized
        methods for identifying specific network state features. Finally,
        results from the simulator are used to generate appropriate
        observations.

        Parameters
        ----------
        rl_actions : array_like
            an list of actions provided by the rl algorithm

        Returns
        -------
        observation : dict of array_like
            agent's observation of the current environment
        reward : dict of floats
            amount of reward associated with the previous state/action pair
        done : dict of bool
            indicates whether the episode has ended
        info : dict
            contains other diagnostic information from the previous action
        """
        idsBegin = self.k.vehicle.get_ids()
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # perform acceleration actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    accel_contr = self.k.vehicle.get_acc_controller(veh_id)
                    action = accel_contr.get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform lane change actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.k.vehicle.get_controlled_lc_ids():
                    target_lane = self.k.vehicle.get_lane_changing_controller(
                        veh_id).get_action(self)
                    direction.append(target_lane)
                self.k.vehicle.apply_lane_change(
                    self.k.vehicle.get_controlled_lc_ids(),
                    direction=direction)

            # perform (optionally) routing actions for all vehicle in the
            # network, including rl and sumo-controlled vehicles
            routing_ids = []
            routing_actions = []

            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(veh_id)
                    routing_actions.append(route_contr.choose_route(self))
            self.k.vehicle.choose_routes(routing_ids, routing_actions)    
            
            self.apply_rl_actions(rl_actions)

            self.additional_command()
            
            nrOfVeh_t = len(self.k.vehicle.get_ids())
            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            nrOfVeh_t_plus_1 = len(self.k.vehicle.get_ids())
            crash=False
            if nrOfVeh_t_plus_1 < nrOfVeh_t:
                crash = True
                print("Crash anzahl")
                break
            
            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            if crash:
                print("crash collision")
                break

        if len(self.k.vehicle.get_ids()) != 4:
                print("############8 != 4")

        states = self.get_state()
        done = {key: key in self.k.vehicle.get_arrived_ids()
                for key in states.keys()}
        if crash:
            done['__all__'] = True
            states = {}
            reward = {}
            observation = np.zeros(4)
            for idBegin in idsBegin:
                states[idBegin] = observation
                reward[idBegin] = 0
            infos = {key: {} for key in states.keys()}
            return states,reward,done,infos

        else:
            done['__all__'] = False
        infos = {key: {} for key in states.keys()}

        clipped_actions = self.clip_actions(rl_actions)
        reward = self.compute_reward(clipped_actions, fail=crash)
        if crash:
            print(states)
            print(reward)
        if len(states) != len(reward):
            print("#########   LENGTH!!")
            print(states)
            print(done)
            print(reward)
        return states, reward, done, infos

