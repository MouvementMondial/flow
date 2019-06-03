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

class MultiTenaciousDEnv(MultiEnv):
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

    # helper left or right
    def get_left_or_right(self, rl_id):
        edge = self.k.vehicle.get_edge(rl_id)
        if edge=="right_upper_E" or edge=="right_upper" or edge=="right_lower" or edge=="right_lower_E":
            return "right"
        elif edge=="left_upper_E" or edge=="left_upper" or edge=="left_lower" or edge=="left_lower_E":
            return "left"
        else:
            return "dia"
             
    def get_state(self):
        """See class definition."""
        obs = {}
        #for rl_id in self.k.vehicle.get_rl_ids():
            #lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

            # normalizers
            #max_speed = 15.0
            #max_length = 206.2

            ## 01
            #s01_speed = self.k.vehicle.get_speed(rl_id) / max_speed
            
            # 02
            #s02_dSpeed = (self.k.vehicle.get_speed(lead_id)-self.k.vehicle.get_speed(rl_id))/max_speed
            
            # 03
            #s03_headway = self.k.vehicle.get_headway(rl_id) / max_length
            
            # 04
            #pos_rl_id = self.k.vehicle.get_x_by_id(rl_id)
            #s04_distanceKP = (max_length-pos_rl_id)/max_length

            # 05 06
            #max_pos_1st = 73.2
            #max_pos_2nd = 73.2
            #speed_1st = 0.0
            #speed_2nd = 0.0
            #side = self.get_left_or_right(rl_id)
            #for rl_id2 in self.k.vehicle.get_rl_ids():
            #    if side == "dia":
            #        break
            #    side2 = self.get_left_or_right(rl_id2)
            #    if side2 != side:
            #        if side2 != "dia":
            #            pos2 = self.k.vehicle.get_x_by_id(rl_id2)
            #            if pos2 > max_pos_1st:
            #                max_pos_2nd = max_pos_1st
            #                max_pos_1st = pos2
            #                speed_2nd = speed_1st
            #                speed_1st = self.k.vehicle.get_speed(rl_id2) / max_speed
            #            elif pos2 > max_pos_2nd:
            #                smax_pos_2nd = pos2
            #                speed_2nd = self.k.vehicle.get_speed(rl_id2) / max_speed
            #max_pos_1st = max_pos_1st
            #max_pos_2nd = max_pos_2nd
            #s05_distanceKP_1st = (max_length-max_pos_1st)/max_length
            #s06_distanceKP_2nd = (max_length-max_pos_2nd)/max_length

            #speed_1 = 0
            #speed_2 = 0
            #if rl_id == "rl_0":
            #    speed_1 = self.k.vehicle.get_speed("rl_0") / max_speed
            #    speed_2 = self.k.vehicle.get_speed("rl_1") / max_speed
            #    dist_kp = self.k.vehicle.get_x
            #else:
            #    speed_1 = self.k.vehicle.get_speed("rl_1") / max_speed
            #    speed_2 = self.k.vehicle.get_speed("rl_0") / max_speed

            # s07
            #s07_distanceKPdiff = s04_distanceKP-s05_distanceKP_1st
 
            # create state 
            #observation = np.array([
                #s01_speed,
                #s02_dSpeed,
            #     speed_1,
            #     speed_2,
            #     s04_distanceKP,
                #s05_distanceKP_1st,
                #s06_distanceKP_2nd,
            #    s07_distanceKPdiff
            #])
            #if rl_id == "rl_0":
            #    print(observation)
            #obs.update({rl_id: observation})

        # normalizers
        max_speed = 15.0
        max_length = 206.2
        dist_KP = 67.5

        speed_0 = [ self.k.vehicle.get_speed("rl_0") / max_speed ]
        speed_1 = [ self.k.vehicle.get_speed("rl_1") / max_speed ]
        
        pos_0   = self.k.vehicle.get_x_by_id("rl_0")
        pos_1   = self.k.vehicle.get_x_by_id("rl_1")
        distance_kp_0 =   [ (dist_KP-pos_0)    *4 / max_length ]
        distance_kp_1 =   [ (dist_KP-pos_1)    *4 / max_length ]
        distance_0_to_1 = [ (pos_1-pos_0) *4 / max_length ]
        distance_1_to_0 = [ (pos_0-pos_1) *4 / max_length ]

        obs_0 = np.array(speed_0 + speed_1 + distance_kp_0 + distance_0_to_1)
        obs_1 = np.array(speed_1 + speed_0 + distance_kp_1 + distance_1_to_0)

        obs.update({"rl_0":obs_0})
        obs.update({"rl_1":obs_1})

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
            rew[rl_id] = self.k.vehicle.get_speed(rl_id) *0.1

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
            crashAnzahl=False
            crash=False
            if nrOfVeh_t_plus_1 < nrOfVeh_t:
                crashAnza = True
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

        #if len(self.k.vehicle.get_ids()) != 4:
        #        print("############8 != 4")

        states = self.get_state()
        done = {key: key in self.k.vehicle.get_arrived_ids()
                for key in states.keys()}

        infos = {key: {} for key in states.keys()}

        clipped_actions = self.clip_actions(rl_actions)
        reward = self.compute_reward(clipped_actions, fail=crash)

        if crashAnzahl or len(states) != len(reward):
            done['__all__'] = True
            states = {}
            reward = {}
            observation = np.zeros(4)
            for idBegin in idsBegin:
                states[idBegin] = observation
                reward[idBegin] = 0
            infos = {key: {} for key in states.keys()}
            return states,reward,done,infos

        if crash:
            done['__all__'] = True
            states = {}
            reward = {}
            observation = np.zeros(4)
            for idBegin in idsBegin:
                states[idBegin] = observation
                reward[idBegin] = -100
            infos = {key: {} for key in states.keys()}
            return states,reward,done,infos

        finishDistance = False
        for veh_id in self.k.vehicle.get_ids():
            edge = self.k.vehicle.get_edge(veh_id)
            if edge == "right_upper" or edge == "left_upper":
                finishDistance = True
                break
        
        if finishDistance:
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
        return states, reward, done, infos

