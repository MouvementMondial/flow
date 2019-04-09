import numpy as np
import numpy as np
from gym.spaces.box import Box
from flow.multiagent_envs.multiagent_env import MultiEnv
from flow.core.params import InitialConfig
import os

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 10,
    
    # teamspirit
    "ap_teamspirit_0": -1,
    "ap_teamspirit_1": -1,
    "ap_teamspirit_shuffle": False

}

class MultiAgentIntersectionEnv(MultiEnv):
    
    @property
    def observation_space(self):
        return Box(
            low=0,
            high=1,
            shape=(4, ),
            dtype=np.float32)
            
    @property
    def action_space(self):
        return Box(
            low=-3,
            high=3,
            shape=(1, ),
            dtype=np.float32)
		    
    def get_state(self):
		
		# reward: v_selbst, p_selbst, v_anderer, p_anderer
        #speed_0 = [self.k.vehicle.get_speed("rl_0") / self.k.scenario.max_speed()]
        #speed_1 = [self.k.vehicle.get_speed("rl_1") / self.k.scenario.max_speed()]
        #pos_0   = [self.k.vehicle.get_x_by_id("rl_0") / self.k.scenario.length()]
        #pos_1   = [self.k.vehicle.get_x_by_id("rl_1") / self.k.scenario.length()]
        
        #obs_0 = np.array(speed_0 + pos_0 + speed_1 + pos_1)
        #obs_1 = np.array(speed_1 + pos_1 + speed_0 + pos_0)

        # reward: v_selbst, v_anderer, abstand_zu_fzg, abstand_zu_kp
        speed_0 = [ self.k.vehicle.get_speed("rl_0") / self.k.scenario.max_speed() ]
        speed_1 = [ self.k.vehicle.get_speed("rl_1") / self.k.scenario.max_speed() ]
        
        pos_0   = self.k.vehicle.get_x_by_id("rl_0")
        pos_1   = self.k.vehicle.get_x_by_id("rl_1")
        distance_kp_0 =   [ (40-pos_0)    *4 / self.k.scenario.length() ]
        distance_kp_1 =   [ (40-pos_1)    *4 / self.k.scenario.length() ]
        distance_0_to_1 = [ (pos_1-pos_0) *4 / self.k.scenario.length() ]
        distance_1_to_0 = [ (pos_0-pos_1) *4 / self.k.scenario.length() ]

        obs_0 = np.array(speed_0 + speed_1 + distance_kp_0 + distance_0_to_1)
        obs_1 = np.array(speed_1 + speed_0 + distance_kp_1 + distance_1_to_0)

        obs = {}
        obs.update({"rl_0":obs_0})
        obs.update({"rl_1":obs_1})

        # reward: v1,v2,p1,p2
        #for rl_id in self.k.vehicle.get_rl_ids():
        #    speed = [self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed()
        #         for veh_id in self.k.vehicle.get_ids()]
        #    pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.scenario.length()
        #         for veh_id in self.k.vehicle.get_ids()]
        #    observation = np.array(speed + pos)
        #    obs.update({rl_id:observation})
        return obs
		
    def _apply_rl_actions(self, rl_actions):
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            accel = list(rl_actions.values())
            self.k.vehicle.apply_acceleration(rl_ids, accel)

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}
        #if kwargs['fail']:
        #    return 0

        rew = {}
        rew["rl_0"] = 0
        rew["rl_1"] = 0

        # reward 1: forward progress
        rew = {}
        rew_rl_0 = self.k.vehicle.get_speed("rl_0")*0.1
        rew_rl_1 = self.k.vehicle.get_speed("rl_1")*0.1
        rew["rl_0"] = rew_rl_0
        rew["rl_1"] = rew_rl_1

        # reward 2: first wins 
        #rew = {}
        #pos_0 = self.k.vehicle.get_x_by_id("rl_0")
        #pos_1 = self.k.vehicle.get_x_by_id("rl_1")
        #if pos_0 > pos_1:
        #    rew["rl_0"] = 1
        #    rew["rl_1"] = 0
        #else:
        #    rew["rl_0"] = 0
        #    rew["rl_1"] = 1
        
        #reward 2: velocity
        #rew = {}
        #rew_rl_0 = self.k.vehicle.get_speed("rl_0")*0.1 - 2
        #rew_rl_1 = self.k.vehicle.get_speed("rl_1")*0.1 - 2
        #rew["rl_0"] = rew_rl_0
        #rew["rl_1"] = rew_rl_1

        return rew

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
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # safe old pos (teleportation / pos set to 0 if collision, timeout(?) or a car leaves the scenario
            old_pos_0 = self.k.vehicle.get_x_by_id("rl_0")
            old_pos_1 = self.k.vehicle.get_x_by_id("rl_1") 

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
            
            # save screenshots
            save_screenshots = True
            if save_screenshots:	
                if not os.path.exists("./screenshots"):
                    os.mkdir("./screenshots")
                self.k.kernel_api.gui.screenshot("View #0","./screenshots/sumo_"+str(self.step_counter)+".png")

            self.k.vehicle.update(reset=False)

            nrOfVeh_t_plus_1 = len(self.k.vehicle.get_ids())
            crash_nr=False
            crash=False
            if nrOfVeh_t_plus_1 < nrOfVeh_t:
                crash_nr = True
                print("Crash anzahl")
                break
                
            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)
          
            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            if crash:
                print("crash collision")
                break
            
            # update the colors of vehicles
            if self.sim_params.render:
                #self.k.vehicle.update_vehicle_colors()
                self.k.vehicle.set_color('rl_0',(0,0,255))
                self.k.vehicle.set_color('rl_1',(255,0,0))
   
        # test if the agent should terminate due to a crash      
        states = self.get_state()
        done = {key: key in self.k.vehicle.get_arrived_ids()
                for key in states.keys()}
        if crash or crash_nr:
            done['__all__'] = True
            states = {}
            observation = np.zeros(4)
            states["rl_0"] = observation
            states["rl_1"] = observation
            infos = {key: {} for key in states.keys()}
       
            #reward when episode finished
            reward = {}
            rew_0 = 0
            rew_1 = 0
            
            #if old_pos_0 > old_pos_1:
            #    rew_0 += 300
            #    rew_1 += 0
            #else:
            #    rew_0 += 0
            #    rew_1 += 300

            # reward if crash, no reward if one vehicle leaves the scenario
            if crash:	
                rew_0 += -100
                rew_1 += -100
            else:
                rew_0 += 0
                rew_1 += 0
            
            reward["rl_0"] = rew_0
            reward["rl_1"] = rew_1
            
            clipped_actions = self.clip_actions(rl_actions)
            return states, reward, done, infos
            
        else:
            done['__all__'] = False

        infos = {key: {} for key in states.keys()}

        clipped_actions = self.clip_actions(rl_actions)
        reward = self.compute_reward(clipped_actions, fail=crash)

        return states, reward, done, infos

class MultiAgentTeamSpiritIntersectionEnv(MultiEnv):
    
    @property
    def observation_space(self):
        return Box(
            low=0,
            high=1,
            shape=(4, ),
            dtype=np.float32)
            
    @property
    def action_space(self):
        return Box(
            low=-3,
            high=3,
            shape=(1, ),
            dtype=np.float32)
		    
    def get_state(self):
		
		#TODO: jeweils aufteilen in eigene Beobachtung und die des anderen
        obs = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            speed = [self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed()
                 for veh_id in self.k.vehicle.get_ids()]
            pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.scenario.length()
                 for veh_id in self.k.vehicle.get_ids()]
            observation = np.array(speed + pos)
            obs.update({rl_id:observation})
        return obs
		
    def _apply_rl_actions(self, rl_actions):
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            accel = list(rl_actions.values())
            self.k.vehicle.apply_acceleration(rl_ids, accel)

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}
        #if kwargs['fail']:
        #    return 0
        # reward: forward progress
        rew = {}
        rew_rl_0 = self.k.vehicle.get_speed("rl_0")*0.1
        rew_rl_1 = self.k.vehicle.get_speed("rl_1")*0.1
        rew["rl_0"] = 1.0*rew_rl_0+0.0*rew_rl_1
        rew["rl_1"] = 0.2*rew_rl_0+0.8*rew_rl_1
        return rew

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
            
            # save screenshots
            save_screenshots = False
            if save_screenshots:	
                if not os.path.exists("./screenshots"):
                    os.mkdir("./screenshots")
                self.k.kernel_api.gui.screenshot("View #0","./screenshots/"+str(self.step_counter)+".png")

            self.k.vehicle.update(reset=False)

            nrOfVeh_t_plus_1 = len(self.k.vehicle.get_ids())
            crash_nr=False
            crash=False
            if nrOfVeh_t_plus_1 < nrOfVeh_t:
                crash_nr = True
                print("Crash anzahl")
                break
                
            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)
          
            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            if crash:
                print("crash collision")
                break
            
            # update the colors of vehicles
            if self.sim_params.render:
                #self.k.vehicle.update_vehicle_colors()
                self.k.vehicle.set_color('rl_0',(0,0,255))
                self.k.vehicle.set_color('rl_1',(255,0,0))
   
        # test if the agent should terminate due to a crash      
        states = self.get_state()
        done = {key: key in self.k.vehicle.get_arrived_ids()
                for key in states.keys()}
        if crash or crash_nr:
            done['__all__'] = True
            states = {}
            observation = np.zeros(4)
            states["rl_0"] = observation
            states["rl_1"] = observation
            infos = {key: {} for key in states.keys()}
            reward = {}
            if crash:	
                reward["rl_0"] = -100
                reward["rl_1"] = -100
            else:
                reward["rl_0"] = 0
                reward["rl_1"] = 0
            clipped_actions = self.clip_actions(rl_actions)
            return states, reward, done, infos
            
        else:
            done['__all__'] = False

        infos = {key: {} for key in states.keys()}

        clipped_actions = self.clip_actions(rl_actions)
        reward = self.compute_reward(clipped_actions, fail=crash)

        return states, reward, done, infos
        
class MultiAgentIntersectionEnv_baseline_1(MultiEnv):
    
    #FCFS with amax
    @property
    def observation_space(self):
        return Box(
            low=0,
            high=1,
            shape=(4, ),
            dtype=np.float32)
            
    @property
    def action_space(self):
        return Box(
            low=-3,
            high=3,
            shape=(1, ),
            dtype=np.float32)
		    
    def get_state(self):
		
		# state: v_selbst, p_selbst, v_anderer, p_anderer
        #speed_0 = [self.k.vehicle.get_speed("rl_0") / self.k.scenario.max_speed()]
        #speed_1 = [self.k.vehicle.get_speed("rl_1") / self.k.scenario.max_speed()]
        #pos_0   = [self.k.vehicle.get_x_by_id("rl_0") / self.k.scenario.length()]
        #pos_1   = [self.k.vehicle.get_x_by_id("rl_1") / self.k.scenario.length()]
        
        #obs_0 = np.array(speed_0 + pos_0 + speed_1 + pos_1)
        #obs_1 = np.array(speed_1 + pos_1 + speed_0 + pos_0)

        # state: v_selbst, v_anderer, abstand_zu_fzg, abstand_zu_kp
        speed_0 = [ self.k.vehicle.get_speed("rl_0") / self.k.scenario.max_speed() ]
        speed_1 = [ self.k.vehicle.get_speed("rl_1") / self.k.scenario.max_speed() ]
        
        pos_0   = self.k.vehicle.get_x_by_id("rl_0")
        pos_1   = self.k.vehicle.get_x_by_id("rl_1")
        distance_kp_0 =   [ (40-pos_0)    *4 / self.k.scenario.length() ]
        distance_kp_1 =   [ (40-pos_1)    *4 / self.k.scenario.length() ]
        distance_0_to_1 = [ (pos_1-pos_0) *4 / self.k.scenario.length() ]
        distance_1_to_0 = [ (pos_0-pos_1) *4 / self.k.scenario.length() ]

        obs_0 = np.array(speed_0 + speed_1 + distance_kp_0 + distance_0_to_1)
        obs_1 = np.array(speed_1 + speed_0 + distance_kp_1 + distance_1_to_0)

        obs = {}
        obs.update({"rl_0":obs_0})
        obs.update({"rl_1":obs_1})

        # state: v1,v2,p1,p2
        #for rl_id in self.k.vehicle.get_rl_ids():
        #    speed = [self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed()
        #         for veh_id in self.k.vehicle.get_ids()]
        #    pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.scenario.length()
        #         for veh_id in self.k.vehicle.get_ids()]
        #    observation = np.array(speed + pos)
        #    obs.update({rl_id:observation})
        return obs
		
    def _apply_rl_actions(self, rl_actions):
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            #accel = list(rl_actions.values())
            baseline_1_accel = [3,3]
            self.k.vehicle.apply_acceleration(rl_ids, baseline_1_accel)

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}
        #if kwargs['fail']:
        #    return 0

        rew = {}
        rew["rl_0"] = 0
        rew["rl_1"] = 0

        # reward 1: forward progress
        #rew = {}
        #rew_rl_0 = self.k.vehicle.get_speed("rl_0")*0.1
        #rew_rl_1 = self.k.vehicle.get_speed("rl_1")*0.1
        #rew["rl_0"] = rew_rl_0
        #rew["rl_1"] = rew_rl_1

        # reward 2: first wins 
        #rew = {}
        #pos_0 = self.k.vehicle.get_x_by_id("rl_0")
        #pos_1 = self.k.vehicle.get_x_by_id("rl_1")
        #if pos_0 > pos_1:
        #    rew["rl_0"] = 1
        #    rew["rl_1"] = 0
        #else:
        #    rew["rl_0"] = 0
        #    rew["rl_1"] = 1
        
        #reward 2: velocity
        rew = {}
        rew_rl_0 = self.k.vehicle.get_speed("rl_0")*0.1
        rew_rl_1 = self.k.vehicle.get_speed("rl_1")*0.1
        rew["rl_0"] = rew_rl_0
        rew["rl_1"] = rew_rl_1

        return rew

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
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # safe old pos (teleportation / pos set to 0 if collision, timeout(?) or a car leaves the scenario
            old_pos_0 = self.k.vehicle.get_x_by_id("rl_0")
            old_pos_1 = self.k.vehicle.get_x_by_id("rl_1") 

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
            
            # save screenshots
            save_screenshots = False
            if save_screenshots:	
                if not os.path.exists("./screenshots"):
                    os.mkdir("./screenshots")
                self.k.kernel_api.gui.screenshot("View #0","./screenshots/"+str(self.step_counter)+".png")

            self.k.vehicle.update(reset=False)

            nrOfVeh_t_plus_1 = len(self.k.vehicle.get_ids())
            crash_nr=False
            crash=False
            if nrOfVeh_t_plus_1 < nrOfVeh_t:
                crash_nr = True
                print("Crash anzahl")
                break
                
            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)
          
            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            if crash:
                print("crash collision")
                break
            
            # update the colors of vehicles
            if self.sim_params.render:
                #self.k.vehicle.update_vehicle_colors()
                self.k.vehicle.set_color('rl_0',(0,0,255))
                self.k.vehicle.set_color('rl_1',(255,0,0))
   
        # test if the agent should terminate due to a crash      
        states = self.get_state()
        done = {key: key in self.k.vehicle.get_arrived_ids()
                for key in states.keys()}
        if crash or crash_nr:
            done['__all__'] = True
            states = {}
            observation = np.zeros(4)
            states["rl_0"] = observation
            states["rl_1"] = observation
            infos = {key: {} for key in states.keys()}
       
            #reward when episode finished
            reward = {}
            rew_0 = 0
            rew_1 = 0
            
            #if old_pos_0 > old_pos_1:
            #    rew_0 += 300
            #    rew_1 += 0
            #else:
            #    rew_0 += 0
            #    rew_1 += 300

            # reward if crash, no reward if one vehicle leaves the scenario
            if crash:	
                rew_0 += -100
                rew_1 += -100
            else:
                rew_0 += 0
                rew_1 += 0
            
            reward["rl_0"] = rew_0
            reward["rl_1"] = rew_1
            
            clipped_actions = self.clip_actions(rl_actions)
            return states, reward, done, infos
            
        else:
            done['__all__'] = False

        infos = {key: {} for key in states.keys()}

        clipped_actions = self.clip_actions(rl_actions)
        reward = self.compute_reward(clipped_actions, fail=crash)

        return states, reward, done, infos

class MultiAgentIntersectionEnv_baseline_2(MultiEnv):
    
    # amax and amax/2
    @property
    def observation_space(self):
        return Box(
            low=0,
            high=1,
            shape=(4, ),
            dtype=np.float32)
            
    @property
    def action_space(self):
        return Box(
            low=-3,
            high=3,
            shape=(1, ),
            dtype=np.float32)
		    
    def get_state(self):
		
		# state: v_selbst, p_selbst, v_anderer, p_anderer
        #speed_0 = [self.k.vehicle.get_speed("rl_0") / self.k.scenario.max_speed()]
        #speed_1 = [self.k.vehicle.get_speed("rl_1") / self.k.scenario.max_speed()]
        #pos_0   = [self.k.vehicle.get_x_by_id("rl_0") / self.k.scenario.length()]
        #pos_1   = [self.k.vehicle.get_x_by_id("rl_1") / self.k.scenario.length()]
        
        #obs_0 = np.array(speed_0 + pos_0 + speed_1 + pos_1)
        #obs_1 = np.array(speed_1 + pos_1 + speed_0 + pos_0)

        # state: v_selbst, v_anderer, abstand_zu_fzg, abstand_zu_kp
        speed_0 = [ self.k.vehicle.get_speed("rl_0") / self.k.scenario.max_speed() ]
        speed_1 = [ self.k.vehicle.get_speed("rl_1") / self.k.scenario.max_speed() ]
        
        pos_0   = self.k.vehicle.get_x_by_id("rl_0")
        pos_1   = self.k.vehicle.get_x_by_id("rl_1")
        distance_kp_0 =   [ (40-pos_0)    *4 / self.k.scenario.length() ]
        distance_kp_1 =   [ (40-pos_1)    *4 / self.k.scenario.length() ]
        distance_0_to_1 = [ (pos_1-pos_0) *4 / self.k.scenario.length() ]
        distance_1_to_0 = [ (pos_0-pos_1) *4 / self.k.scenario.length() ]

        obs_0 = np.array(speed_0 + speed_1 + distance_kp_0 + distance_0_to_1)
        obs_1 = np.array(speed_1 + speed_0 + distance_kp_1 + distance_1_to_0)

        obs = {}
        obs.update({"rl_0":obs_0})
        obs.update({"rl_1":obs_1})

        # state: v1,v2,p1,p2
        #for rl_id in self.k.vehicle.get_rl_ids():
        #    speed = [self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed()
        #         for veh_id in self.k.vehicle.get_ids()]
        #    pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.scenario.length()
        #         for veh_id in self.k.vehicle.get_ids()]
        #    observation = np.array(speed + pos)
        #    obs.update({rl_id:observation})
        return obs
		
    def _apply_rl_actions(self, rl_actions):
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            #accel = list(rl_actions.values())
            baseline_2_accel = [2,3]
            self.k.vehicle.apply_acceleration(rl_ids, baseline_2_accel)

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}
        #if kwargs['fail']:
        #    return 0

        rew = {}
        rew["rl_0"] = 0
        rew["rl_1"] = 0

        # reward 1: forward progress
        #rew = {}
        #rew_rl_0 = self.k.vehicle.get_speed("rl_0")*0.1
        #rew_rl_1 = self.k.vehicle.get_speed("rl_1")*0.1
        #rew["rl_0"] = rew_rl_0
        #rew["rl_1"] = rew_rl_1

        # reward 2: first wins 
        #rew = {}
        #pos_0 = self.k.vehicle.get_x_by_id("rl_0")
        #pos_1 = self.k.vehicle.get_x_by_id("rl_1")
        #if pos_0 > pos_1:
        #    rew["rl_0"] = 1
        #    rew["rl_1"] = 0
        #else:
        #    rew["rl_0"] = 0
        #    rew["rl_1"] = 1
        
        #reward 2: velocity
        rew = {}
        rew_rl_0 = self.k.vehicle.get_speed("rl_0")*0.1
        rew_rl_1 = self.k.vehicle.get_speed("rl_1")*0.1
        rew["rl_0"] = rew_rl_0
        rew["rl_1"] = rew_rl_1

        return rew

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
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # safe old pos (teleportation / pos set to 0 if collision, timeout(?) or a car leaves the scenario
            old_pos_0 = self.k.vehicle.get_x_by_id("rl_0")
            old_pos_1 = self.k.vehicle.get_x_by_id("rl_1") 

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
            
            # save screenshots
            save_screenshots = False
            if save_screenshots:	
                if not os.path.exists("./screenshots"):
                    os.mkdir("./screenshots")
                self.k.kernel_api.gui.screenshot("View #0","./screenshots/"+str(self.step_counter)+".png")

            self.k.vehicle.update(reset=False)

            nrOfVeh_t_plus_1 = len(self.k.vehicle.get_ids())
            crash_nr=False
            crash=False
            if nrOfVeh_t_plus_1 < nrOfVeh_t:
                crash_nr = True
                print("Crash anzahl")
                break
                
            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)
          
            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            if crash:
                print("crash collision")
                break
            
            # update the colors of vehicles
            if self.sim_params.render:
                #self.k.vehicle.update_vehicle_colors()
                self.k.vehicle.set_color('rl_0',(0,0,255))
                self.k.vehicle.set_color('rl_1',(255,0,0))
   
        # test if the agent should terminate due to a crash      
        states = self.get_state()
        done = {key: key in self.k.vehicle.get_arrived_ids()
                for key in states.keys()}
        if crash or crash_nr:
            done['__all__'] = True
            states = {}
            observation = np.zeros(4)
            states["rl_0"] = observation
            states["rl_1"] = observation
            infos = {key: {} for key in states.keys()}
       
            #reward when episode finished
            reward = {}
            rew_0 = 0
            rew_1 = 0
            
            #if old_pos_0 > old_pos_1:
            #    rew_0 += 300
            #    rew_1 += 0
            #else:
            #    rew_0 += 0
            #    rew_1 += 300

            # reward if crash, no reward if one vehicle leaves the scenario
            if crash:	
                rew_0 += -100
                rew_1 += -100
            else:
                rew_0 += 0
                rew_1 += 0
            
            reward["rl_0"] = rew_0
            reward["rl_1"] = rew_1
            
            clipped_actions = self.clip_actions(rl_actions)
            return states, reward, done, infos
            
        else:
            done['__all__'] = False

        infos = {key: {} for key in states.keys()}

        clipped_actions = self.clip_actions(rl_actions)
        reward = self.compute_reward(clipped_actions, fail=crash)

        return states, reward, done, infos

class MultiAgentIntersectionEnv_baseline_3(MultiEnv):
    
    #FCFS with amax and amax/2
    @property
    def observation_space(self):
        return Box(
            low=0,
            high=1,
            shape=(4, ),
            dtype=np.float32)
            
    @property
    def action_space(self):
        return Box(
            low=-3,
            high=3,
            shape=(1, ),
            dtype=np.float32)
		    
    def get_state(self):
		
		# state: v_selbst, p_selbst, v_anderer, p_anderer
        #speed_0 = [self.k.vehicle.get_speed("rl_0") / self.k.scenario.max_speed()]
        #speed_1 = [self.k.vehicle.get_speed("rl_1") / self.k.scenario.max_speed()]
        #pos_0   = [self.k.vehicle.get_x_by_id("rl_0") / self.k.scenario.length()]
        #pos_1   = [self.k.vehicle.get_x_by_id("rl_1") / self.k.scenario.length()]
        
        #obs_0 = np.array(speed_0 + pos_0 + speed_1 + pos_1)
        #obs_1 = np.array(speed_1 + pos_1 + speed_0 + pos_0)

        # state: v_selbst, v_anderer, abstand_zu_fzg, abstand_zu_kp
        speed_0 = [ self.k.vehicle.get_speed("rl_0") / self.k.scenario.max_speed() ]
        speed_1 = [ self.k.vehicle.get_speed("rl_1") / self.k.scenario.max_speed() ]
        
        pos_0   = self.k.vehicle.get_x_by_id("rl_0")
        pos_1   = self.k.vehicle.get_x_by_id("rl_1")
        distance_kp_0 =   [ (40-pos_0)    *4 / self.k.scenario.length() ]
        distance_kp_1 =   [ (40-pos_1)    *4 / self.k.scenario.length() ]
        distance_0_to_1 = [ (pos_1-pos_0) *4 / self.k.scenario.length() ]
        distance_1_to_0 = [ (pos_0-pos_1) *4 / self.k.scenario.length() ]

        obs_0 = np.array(speed_0 + speed_1 + distance_kp_0 + distance_0_to_1)
        obs_1 = np.array(speed_1 + speed_0 + distance_kp_1 + distance_1_to_0)

        obs = {}
        obs.update({"rl_0":obs_0})
        obs.update({"rl_1":obs_1})

        # state: v1,v2,p1,p2
        #for rl_id in self.k.vehicle.get_rl_ids():
        #    speed = [self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed()
        #         for veh_id in self.k.vehicle.get_ids()]
        #    pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.scenario.length()
        #         for veh_id in self.k.vehicle.get_ids()]
        #    observation = np.array(speed + pos)
        #    obs.update({rl_id:observation})
        return obs
		
    def _apply_rl_actions(self, rl_actions):
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            #accel = list(rl_actions.values())
            
            pos_0   = self.k.vehicle.get_x_by_id("rl_0")
            pos_1   = self.k.vehicle.get_x_by_id("rl_1")
            
            #print(rl_ids)
            #print([pos_0,pos_1])
            
            
            if pos_0 < pos_1:
                if rl_ids[0] == "rl_0":
                     baseline_3_accel = [2,3]
                else:
                      baseline_3_accel = [3,2]
            else:
                if rl_ids[0] == "rl_0":
                     baseline_3_accel = [3,2]
                else:
                      baseline_3_accel = [2,3]
            self.k.vehicle.apply_acceleration(rl_ids, baseline_3_accel)

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}
        #if kwargs['fail']:
        #    return 0

        rew = {}
        rew["rl_0"] = 0
        rew["rl_1"] = 0

        # reward 1: forward progress
        #rew = {}
        #rew_rl_0 = self.k.vehicle.get_speed("rl_0")*0.1
        #rew_rl_1 = self.k.vehicle.get_speed("rl_1")*0.1
        #rew["rl_0"] = rew_rl_0
        #rew["rl_1"] = rew_rl_1

        # reward 2: first wins 
        #rew = {}
        #pos_0 = self.k.vehicle.get_x_by_id("rl_0")
        #pos_1 = self.k.vehicle.get_x_by_id("rl_1")
        #if pos_0 > pos_1:
        #    rew["rl_0"] = 1
        #    rew["rl_1"] = 0
        #else:
        #    rew["rl_0"] = 0
        #    rew["rl_1"] = 1
        
        #reward 2: velocity
        rew = {}
        rew_rl_0 = self.k.vehicle.get_speed("rl_0")*0.1
        rew_rl_1 = self.k.vehicle.get_speed("rl_1")*0.1
        rew["rl_0"] = rew_rl_0
        rew["rl_1"] = rew_rl_1

        return rew

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
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # safe old pos (teleportation / pos set to 0 if collision, timeout(?) or a car leaves the scenario
            old_pos_0 = self.k.vehicle.get_x_by_id("rl_0")
            old_pos_1 = self.k.vehicle.get_x_by_id("rl_1") 

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
            
            # save screenshots
            save_screenshots = False
            if save_screenshots:	
                if not os.path.exists("./screenshots"):
                    os.mkdir("./screenshots")
                self.k.kernel_api.gui.screenshot("View #0","./screenshots/"+str(self.step_counter)+".png")

            self.k.vehicle.update(reset=False)

            nrOfVeh_t_plus_1 = len(self.k.vehicle.get_ids())
            crash_nr=False
            crash=False
            if nrOfVeh_t_plus_1 < nrOfVeh_t:
                crash_nr = True
                print("Crash anzahl")
                break
                
            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)
          
            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            if crash:
                print("crash collision")
                break
            
            # update the colors of vehicles
            if self.sim_params.render:
                #self.k.vehicle.update_vehicle_colors()
                self.k.vehicle.set_color('rl_0',(0,0,255))
                self.k.vehicle.set_color('rl_1',(255,0,0))
   
        # test if the agent should terminate due to a crash      
        states = self.get_state()
        done = {key: key in self.k.vehicle.get_arrived_ids()
                for key in states.keys()}
        if crash or crash_nr:
            done['__all__'] = True
            states = {}
            observation = np.zeros(4)
            states["rl_0"] = observation
            states["rl_1"] = observation
            infos = {key: {} for key in states.keys()}
       
            #reward when episode finished
            reward = {}
            rew_0 = 0
            rew_1 = 0
            
            #if old_pos_0 > old_pos_1:
            #    rew_0 += 300
            #    rew_1 += 0
            #else:
            #    rew_0 += 0
            #    rew_1 += 300

            # reward if crash, no reward if one vehicle leaves the scenario
            if crash:	
                rew_0 += -100
                rew_1 += -100
            else:
                rew_0 += 0
                rew_1 += 0
            
            reward["rl_0"] = rew_0
            reward["rl_1"] = rew_1
            
            clipped_actions = self.clip_actions(rl_actions)
            return states, reward, done, infos
            
        else:
            done['__all__'] = False

        infos = {key: {} for key in states.keys()}

        clipped_actions = self.clip_actions(rl_actions)
        reward = self.compute_reward(clipped_actions, fail=crash)

        return states, reward, done, infos

class MultiAgentIntersectionEnv_sharedPolicy_TeamSpirit(MultiEnv):
        
    @property
    def observation_space(self):
        return Box(
            low=0,
            high=1,
            shape=(5, ),
            dtype=np.float32)
            
    @property
    def action_space(self):
        return Box(
            low=-3,
            high=3,
            shape=(1, ),
            dtype=np.float32)
		    
    def get_state(self):
		
		# reward: v_selbst, p_selbst, v_anderer, p_anderer
        #speed_0 = [self.k.vehicle.get_speed("rl_0") / self.k.scenario.max_speed()]
        #speed_1 = [self.k.vehicle.get_speed("rl_1") / self.k.scenario.max_speed()]
        #pos_0   = [self.k.vehicle.get_x_by_id("rl_0") / self.k.scenario.length()]
        #pos_1   = [self.k.vehicle.get_x_by_id("rl_1") / self.k.scenario.length()]
        
        #obs_0 = np.array(speed_0 + pos_0 + speed_1 + pos_1)
        #obs_1 = np.array(speed_1 + pos_1 + speed_0 + pos_0)

        # reward: v_selbst, v_anderer, abstand_zu_fzg, abstand_zu_kp
        speed_0 = [ self.k.vehicle.get_speed("rl_0") / self.k.scenario.max_speed() ]
        speed_1 = [ self.k.vehicle.get_speed("rl_1") / self.k.scenario.max_speed() ]
        
        pos_0   = self.k.vehicle.get_x_by_id("rl_0")
        pos_1   = self.k.vehicle.get_x_by_id("rl_1")
        distance_kp_0 =   [ (40-pos_0)    *4 / self.k.scenario.length() ]
        distance_kp_1 =   [ (40-pos_1)    *4 / self.k.scenario.length() ]
        distance_0_to_1 = [ (pos_1-pos_0) *4 / self.k.scenario.length() ]
        distance_1_to_0 = [ (pos_0-pos_1) *4 / self.k.scenario.length() ]

        team_0 = 0
        team_1 = 0
        if self.env_params.additional_params['ap_teamspirit_shuffle']:
            team_0 = self.teamspirit_0
            team_1 = self.teamspirit_1
        else:
            team_0 = self.env_params.additional_params['ap_teamspirit_0']
            team_1 = self.env_params.additional_params['ap_teamspirit_1']

        #print(team_0)
        #print(team_1) 

        obs_0 = np.array(speed_0 + speed_1 + distance_kp_0 + distance_0_to_1 + [team_0])
        obs_1 = np.array(speed_1 + speed_0 + distance_kp_1 + distance_1_to_0 + [team_1])

        obs = {}
        obs.update({"rl_0":obs_0})
        obs.update({"rl_1":obs_1})

        # reward: v1,v2,p1,p2
        #for rl_id in self.k.vehicle.get_rl_ids():
        #    speed = [self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed()
        #         for veh_id in self.k.vehicle.get_ids()]
        #    pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.scenario.length()
        #         for veh_id in self.k.vehicle.get_ids()]
        #    observation = np.array(speed + pos)
        #    obs.update({rl_id:observation})
        return obs
		
    def _apply_rl_actions(self, rl_actions):
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            accel = list(rl_actions.values())
            self.k.vehicle.apply_acceleration(rl_ids, accel)

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}
        #if kwargs['fail']:
        #    return 0

        rew = {}
        rew["rl_0"] = 0
        rew["rl_1"] = 0 

        # reward 1: forward progress
        team_0 = 0
        team_1 = 0
        
        #print(self.env_params.additional_params['teamspirit_shuffle'])
        #print(self.env_params.additional_params['teamspirit_0'])
        #print(self.env_params.additional_params['teamspirit_1'])
        
        if self.env_params.additional_params['ap_teamspirit_shuffle']:
            team_0 = self.teamspirit_0
            team_1 = self.teamspirit_1
        else:
            team_0 = self.env_params.additional_params['ap_teamspirit_0']
            team_1 = self.env_params.additional_params['ap_teamspirit_1']
        
        #print(team_0)
        #print(team_1)    
        
        # teamspirit Faktor von -1_1 zurück auf 0_1 ziehen    
        team_0 = (team_0+1)/2.0
        team_1 = (team_1+1)/2.0
        rew = {}
        rew_rl_0 = self.k.vehicle.get_speed("rl_0")*0.1
        rew_rl_1 = self.k.vehicle.get_speed("rl_1")*0.1
        
        # fix und 2*flexibel
        #rew["rl_0"] = rew_rl_0 + rew_rl_0 * (1-team_0) * 2 + rew_rl_1 * team_0 * 2 
        #rew["rl_1"] = rew_rl_1 + rew_rl_1 * (1-team_1) * 2 + rew_rl_0 * team_1 * 2
        
        # nur flexibel
        #rew["rl_0"] = rew_rl_0 * (1-team_0) + rew_rl_1 * team_0 
        #rew["rl_1"] = rew_rl_1 * (1-team_1) + rew_rl_0 * team_1
        
        # fix und kleiner Anteil vom flexiblen (besser mit 2-fach Testen?)
        #rew["rl_0"] = rew_rl_0 + rew_rl_1 * team_0 * 2
        #rew["rl_1"] = rew_rl_1 + rew_rl_0 * team_1 * 2
        
        rew["rl_0"] = rew_rl_0
        rew["rl_1"] = rew_rl_1 - 2 * rew_rl_0 
       

        # reward 2: first wins 
        #rew = {}
        #pos_0 = self.k.vehicle.get_x_by_id("rl_0")
        #pos_1 = self.k.vehicle.get_x_by_id("rl_1")
        #if pos_0 > pos_1:
        #    rew["rl_0"] = 1
        #    rew["rl_1"] = 0
        #else:
        #    rew["rl_0"] = 0
        #    rew["rl_1"] = 1
        
        #reward 2: velocity
        #rew = {}
        #rew_rl_0 = self.k.vehicle.get_speed("rl_0")*0.1 - 2
        #rew_rl_1 = self.k.vehicle.get_speed("rl_1")*0.1 - 2
        #rew["rl_0"] = rew_rl_0
        #rew["rl_1"] = rew_rl_1

        return rew

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
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # safe old pos (teleportation / pos set to 0 if collision, timeout(?) or a car leaves the scenario
            old_pos_0 = self.k.vehicle.get_x_by_id("rl_0")
            old_pos_1 = self.k.vehicle.get_x_by_id("rl_1") 

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
            
            # save screenshots
            save_screenshots = False
            if save_screenshots:	
                if not os.path.exists("./screenshots"):
                    os.mkdir("./screenshots")
                self.k.kernel_api.gui.screenshot("View #0","./screenshots/sumo_"+str(self.step_counter)+".png")

            self.k.vehicle.update(reset=False)

            nrOfVeh_t_plus_1 = len(self.k.vehicle.get_ids())
            crash_nr=False
            crash=False
            if nrOfVeh_t_plus_1 < nrOfVeh_t:
                crash_nr = True
                print("Crash anzahl")
                break
                
            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)
          
            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            if crash:
                print("crash collision")
                break
            
            # update the colors of vehicles
            if self.sim_params.render:
                #self.k.vehicle.update_vehicle_colors()
                self.k.vehicle.set_color('rl_0',(0,0,255))
                self.k.vehicle.set_color('rl_1',(255,0,0))
   
        # test if the agent should terminate due to a crash      
        states = self.get_state()
        done = {key: key in self.k.vehicle.get_arrived_ids()
                for key in states.keys()}
        if crash or crash_nr:
            done['__all__'] = True
            states = {}
            observation = np.zeros(5)
            states["rl_0"] = observation
            states["rl_1"] = observation
            infos = {key: {} for key in states.keys()}
       
            #reward when episode finished
            reward = {}
            rew_0 = 0
            rew_1 = 0
            
            #if old_pos_0 > old_pos_1:
            #    rew_0 += 300
            #    rew_1 += 0
            #else:
            #    rew_0 += 0
            #    rew_1 += 300

            # reward if crash, no reward if one vehicle leaves the scenario
            if crash:	
                rew_0 += -100
                rew_1 += -100
            else:
                rew_0 += 0
                rew_1 += 0
            
            reward["rl_0"] = rew_0
            reward["rl_1"] = rew_1
            
            clipped_actions = self.clip_actions(rl_actions)
            return states, reward, done, infos
            
        else:
            done['__all__'] = False

        infos = {key: {} for key in states.keys()}

        clipped_actions = self.clip_actions(rl_actions)
        reward = self.compute_reward(clipped_actions, fail=crash)

        return states, reward, done, infos

