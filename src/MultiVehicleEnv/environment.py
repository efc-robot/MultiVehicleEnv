from typing import Any, BinaryIO, Callable, Dict, List,Union
import time
import gym
from gym import spaces
import numpy as np
from .basic import World
from .GUI import GUI
import copy
import pickle

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
                 updata_callback:Callable=None,
                 GUI_port:Union[str,None] = '/dev/shm/gui_port',
                 shared_reward:bool = False):

        self.world = world
        # set required vectorized gym env property
        self.shared_reward = shared_reward

        self.GUI_port = GUI_port
        self.GUI_file:Union[BinaryIO,None]=None

        
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.updata_callback = updata_callback
        # record total real-world time past by
        self.total_time:float = 0.0

        # action spaces
        self.action_space:List[spaces.Discrete] = []
        for vehicle in self.world.vehicle_list:
            if vehicle.discrete_table is None:
                self.action_space.append(spaces.Box(low=-1.0,high=1.0,shape=(2,)))
            else:
                self.action_space.append(spaces.Discrete(len(vehicle.discrete_table)))
        # observation space
        self.observation_space = []
        for vehicle in self.world.vehicle_list:
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
        return self.reward_callback(vehicle, self.world, self.old_world)

    def step(self, action_n:T_action):
        obs_n:List[np.ndarray] = []
        reward_n:List[float] = []
        done_n:List[bool] = []
        info_n:Dict[str,Any] = {'n': []}
        # set action for each vehicle
        for i, vehicle in enumerate(self.world.vehicle_list):
            if vehicle.discrete_table is None:
                ctrl_vel_b = action_n[i][0]
                ctrl_phi = action_n[i][1]
            else:
                if isinstance(action_n[i],int):
                    action_i = action_n[i]
                else:
                    action_i = list(action_n[i]).index(1)
                [ctrl_vel_b,ctrl_phi] = vehicle.discrete_table[action_i]
            vehicle.state.ctrl_vel_b = ctrl_vel_b
            vehicle.state.ctrl_phi = ctrl_phi
        # advance world state
        self.old_world = copy.deepcopy(self.world)
        
        
        for idx in range(self.world.sim_step):
            self.total_time += self.world.sim_t
            self.world._update_one_sim_step()
            self.world._check_collision()
            if self.GUI_port is not None:
                # if use GUI, slow down the simulation speed
                time.sleep(self.world.sim_t)
                self.dumpGUI()


        # record observation for each vehicle
        for vehicle in self.world.vehicle_list:
            reward_n.append(self._get_reward(vehicle))
        for vehicle in self.world.vehicle_list:            
            done_n.append(self._get_done(vehicle))
        for vehicle in self.world.vehicle_list:
            info_n['n'].append(self._get_info(vehicle))
        if self.updata_callback is not None:
            self.updata_callback(self.world)
        for vehicle in self.world.vehicle_list:
            obs_n.append(self._get_obs(vehicle))
        
        
        if  'max_step_number' in self.world.data_slot.keys():
            step_done = self.world.data_slot['max_step_number']<=self.world.data_slot['total_step_number']
            info_n['TimeLimit.truncated'] = step_done

        # all vehicles get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * len(self.world.vehicle_list)
        return obs_n, reward_n, done_n, info_n

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        self.total_time = 0.0
        # record observations for each vehicle
        obs_n = []
        for vehicle_list in self.world.vehicle_list:
            obs_n.append(self._get_obs(vehicle_list))
        return obs_n
    
    def render(self, mode = 'human'):
        if self.GUI is None:
            self.GUI = GUI(port_type='direct', gui_port=self, fps = 24)
            self.GUI.init_viewer()
            self.GUI.init_object()
        self.GUI._render()
    
    def ros_step(self,total_time):
        self.total_time = total_time
        self.world._check_collision()
        self.dumpGUI()
    
    def dumpGUI(self, port_type = 'file'):
        GUI_data = {'field_range':self.world.field_range,
                    'total_time':self.total_time,
                    'vehicle_list':self.world.vehicle_list,
                    'landmark_list':self.world.landmark_list,
                    'obstacle_list':self.world.obstacle_list,
                    'info':self.world.data_slot}
        if port_type == 'direct':
            return copy.deepcopy(GUI_data)
        if port_type == 'file':
            if self.GUI_port is not None and self.GUI_file is None:
                try:
                    self.GUI_file = open(self.GUI_port, "w+b")
                except IOError:
                    print('open GUI_file %s failed'%self.GUI_port)
            if self.GUI_port is not None:
                self.GUI_file.seek(0)
                pickle.dump(GUI_data,self.GUI_file)
                self.GUI_file.flush()