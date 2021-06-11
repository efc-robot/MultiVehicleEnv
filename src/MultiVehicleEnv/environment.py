from typing import Any, Callable, Dict, List,Union

import gym
from gym import spaces
import numpy as np
from .basic import World
from .GUI import GUI


T_action = Union[List[int],List[List[int]]]
# environment for all vehicles in the multi-vehicle world
# currently code assumes that no vehicle will be created/destroyed at runtime!
class MultiVehicleEnv(gym.Env):
    def __init__(self, world:World,
                 reset_callback:Callable=None,
                 reward_callback:Callable=None,
                 observation_callback:Callable=None,
                 info_callback:Callable=None,
                 done_callback:Callable=None,
                 shared_reward:bool = False):

        self.world = world
        self.vehicle_list = self.world.vehicle_list
        # set required vectorized gym env property
        self.vehicle_number = len(self.world.vehicle_list)
        self.shared_reward = shared_reward

        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.total_time:float = 0.0

        # action spaces
        self.action_space:List[spaces.Discrete] = []
        for vehicle in self.vehicle_list:
            self.action_space.append(spaces.Discrete(len(vehicle.discrete_table)))

        # observation space
        self.observation_space = []
        for vehicle in self.vehicle_list:
            if self.observation_callback is None:
                obs_dim = 0
            else:
                obs_dim = len(self.observation_callback(vehicle, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
        self.GUI = None

    # get info used for benchmarking
    def _get_info(self, vehicle):
        if self.info_callback is None:
            return {}
        return self.info_callback(vehicle, self.world)

    # get observation for a particular vehicle
    def _get_obs(self, vehicle):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(vehicle, self.world)

    # get dones for a particular vehicle
    # unused right now -- vehicle are allowed to go beyond the viewing screen
    def _get_done(self, vehicle):
        if self.done_callback is None:
            return False
        return self.done_callback(vehicle, self.world)

    # get reward for a particular vehicle
    def _get_reward(self, vehicle):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(vehicle, self.world)

    def step(self, action_n:T_action):
        obs_n:List[np.ndarray] = []
        reward_n:List[float] = []
        done_n:List[bool] = []
        info_n:Dict[str,Any] = {'n': []}
        self.vehicle_list = self.world.vehicle_list
        # set action for each vehicle
        for i, vehicle in enumerate(self.vehicle_list):
            if isinstance(action_n[i],int):
                action_i = action_n[i]
            else:
                action_i = list(action_n[i]).index(1)
            [ctrl_vel_b,ctrl_phi] = vehicle.discrete_table[action_i]
            vehicle.state.ctrl_vel_b = ctrl_vel_b
            vehicle.state.ctrl_phi = ctrl_phi
        # advance world state
        self.world.step()
        # record observation for each vehicle
        for vehicle in self.vehicle_list:
            obs_n.append(self._get_obs(vehicle))
            reward_n.append([self._get_reward(vehicle)])
            done_n.append(self._get_done(vehicle))

            info_n['n'].append(self._get_info(vehicle))
        
        if  'max_step_number' in self.world.data_slot.keys():
            step_done = self.world.data_slot['max_step_number']<=self.world.data_slot['total_step_number']
            info_n['step_done'] = step_done

        # all vehicles get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.vehicle_number
        return obs_n, reward_n, done_n, info_n

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        self.world.total_time = 0.0
        # record observations for each vehicle
        obs_n = []
        self.vehicle_list = self.world.vehicle_list
        for vehicle_list in self.vehicle_list:
            obs_n.append(self._get_obs(vehicle_list))
        return obs_n
    
    def render(self,mode = 'human'):
        if self.GUI is None:
            self.GUI = GUI(port_type='direct',gui_port=self,fps = 24)
            self.GUI.init_viewer()
            self.GUI.init_object()
        self.GUI._render()
        direction = []
        for a in self.world.vehicle_list:
            direction.append(a.data_slot['direction_obs'])
        pass
    
    def ros_step(self,total_time):
        self.world.ros_step(total_time)