"""Test environment used to run simulations in the absence of autonomy."""
from flow.core.params import InitialConfig
from flow.envs.base_env import Env
from gym.spaces.box import Box
from flow.core import rewards
import numpy as np
import os

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 10,
}

class IntersectionEnv(Env):
    """Fully observed intersection environment.

	Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s

    States
        The state consists of the velocities and absolute position of all
        vehicles in the network. This assumes a constant number of vehicles.

    Actions
        Actions are a list of acceleration for each rl vehicles, bounded by the
        maximum accelerations and decelerations specified in EnvParams.

    Rewards
        The reward is zero at every step.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """
    
    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, scenario, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            #shape=(self.k.vehicle.num_rl_vehicles, ),
            shape=(2, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=0,
            high=1,
            #shape=(2 * self.k.vehicle.num_vehicles, ),
            shape=(4, ),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        self.k.vehicle.apply_acceleration(
            self.k.vehicle.get_rl_ids(), rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if kwargs.get('fail'):
            return 0   
        return rewards.rl_forward_progress(self,gain=0.1)

    def get_state(self):
        """See class definition."""
        speed = [self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed()
                 for veh_id in self.k.vehicle.get_ids()]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.scenario.length()
               for veh_id in self.k.vehicle.get_ids()]

        return np.array(speed + pos)

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)
                
    def step(self, rl_actions):
        """Advance the environment by one step.

        Assigns actions to autonomous and human-driven agents (i.e. vehicles,
        traffic lights, etc...). Actions that are not assigned are left to the
        control of the simulator. The actions are then used to advance the
        simulator by the number of time steps requested per environment step.

        Results from the simulations are processed through various classes,
        such as the Vehicles and TrafficLights classes, to produce standardized
        methods for identifying specific network state features. Finally,
        results from the simulator are used to generate appropriate
        observations.

        Parameters
        ----------
        rl_actions: numpy ndarray
            an list of actions provided by the rl algorithm

        Returns
        -------
        observation: numpy ndarray
            agent's observation of the current environment
        reward: float
            amount of reward associated with the previous state/action pair
        done: bool
            indicates whether the episode has ended
        info: dict
            contains other diagnostic information from the previous action
        """
        
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # perform acceleration actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    action = self.k.vehicle.get_acc_controller(
                        veh_id).get_action(self)
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

            # perform (optionally) routing actions for all vehicles in the
            # network, including RL and SUMO-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) \
                        is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(
                        veh_id)
                    routing_actions.append(route_contr.choose_route(self))

            self.k.vehicle.choose_routes(routing_ids, routing_actions)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            nrOfVeh_t = len(self.k.vehicle.get_ids())
            self.k.simulation.simulation_step()
            
            # save screenshots
            save_screenshots = True
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
            
            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

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
            
            # render a frame
            self.render()

        # test if the agent should terminate due to a crash
        if crash or crash_nr:
            done = True
        else:
            done = False

        # compute the info for each agent
        infos = {}
        
        if done:
            if crash:
                reward = -100
            else:
                reward = 0
            next_observation = np.zeros(4)
            return next_observation, reward, done, infos

        states = self.get_state()

        # collect information of the state of the network based on the
        # environment class used
        self.state = np.asarray(states).T

        # collect observation new state associated with action
        next_observation = np.copy(states)
        old_observation = np.copy(states)

        # test if the agent should terminate due to a crash
        done = crash

        # compute the info for each agent
        infos = {}

        # compute the reward
        rl_clipped = self.clip_actions(rl_actions)
        reward = self.compute_reward(rl_clipped, fail=crash)

        return next_observation, reward, done, infos
