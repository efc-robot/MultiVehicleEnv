from typing import *
import numpy as np
import pickle
import time
import copy

class VehicleState(object):
    def __init__(self):
        # center point position in x,y axis
        self.coordinate:List[float]= [0.0,0.0]
        # direction of car axis
        self.theta:float = 0.0
        # linear velocity of back point
        self.vel_b:float = 0.0
        # deflection angle of front wheel
        self.phi:float = 0.0
        # Control signal of linear velocity of back point
        self.ctrl_vel_b:float = 0.0
        # Control signal of deflection angle of front wheel
        self.ctrl_phi:float = 0.0
        # cannot move if movable is False. Default is True and be setted as False when crashed.
        self.movable:bool = True
        # Default is False and be setted as True when crashed into other collideable object.
        self.crashed:bool = False

class Vehicle(object):
    def __init__(self):
        # safe radius of the vehicle
        self.r_safe:float = 0.24
        # length of the vehicle
        self.L_car:float = 0.30
        # width of the vehicle
        self.W_car:float = 0.20
        # distance between front and back wheel
        self.L_axis:float = 0.20
        # coefficient of back whell velocity control
        self.K_vel:float = 0.18266
        # coefficient of front wheel deflection control
        self.K_phi:float = 0.298
        # the acceleration of the back whell velocity
        self.dv_dt:float = 2.0
        # the angular acceleration of the back whell velocity
        self.dphi_dt:float = 3.0
        # the color of the vehicle
        self.color:list[Union[list[float] , float]] = [[0.0,0.0,0.0],0.0]
        # the discrete action table of the vehicle, action_code -> (ctrl_vel,ctrl_phi) 
        self.discrete_table:Dict[int,Tuple[float,float]] = {0:( 0.0, 0.0),
                               1:( 1.0, 0.0), 2:( 1.0, 1.0), 3:( 1.0, -1.0),
                               4:(-1.0, 0.0), 5:(-1.0, 1.0), 6:(-1.0, -1.0)}
        self.data_slot:Dict[str,Any]= {}
        # the state of the vehicle
        self.state:VehicleState = VehicleState()

class EntityState(object):
    def __init__(self):
        # center point position in x,y axis
        self.coordinate:List[float] = [0.0,0.0]

class Entity(object):
    def __init__(self):
        # the redius of the entity
        self.radius:float = 1.0
        # true if the entity can crash into other collideable object
        self.collideable:bool = False
        # the color of the entoty
        self.color:list[Union[list[float] , float]] = [[0.0,0.0,0.0],0.0]
        # the state of the entity
        self.state:EntityState = EntityState()

# multi-vehicle world
class World(object):
    def __init__(self):
        # list of vehicles and entities (can change at execution-time!)
        self.vehicle_list:List[Vehicle] = []
        self.landmark_list:List[Entity] = []
        self.obstacle_list:List[Entity] = []
        # range of the main field
        self.field_range:List[float] = [-1.0,-1.0,1.0,1.0]
        # GUI port for GUI display, by a file in shared memory
        self.GUI_port:str = '/dev/shm/gui_port'
        # File handle for GUI
        self.GUI_file:Union[BinaryIO,None] = None
        # Sim state for flow control not used yet
        self.sim_state:str = 'init'

        # real-world time duration for one MDP step
        self.step_t:float = 1.0

        # split the step_t into sim_step pieces state simulation
        self.sim_step:int = 1000
        # record total real-world time past by
        self.total_time:float = 0.0

        # the data slot for additional data defined in scenario
        self.data_slot:Dict[str,Any] = {}
    
    # real-world time duration for one state simulation step
    @property
    def sim_t(self)->float:
        return self.step_t/self.sim_step
    
    # coordinate of the main field center
    @property
    def field_center(self)->Tuple[float,float]:
        center_x = (self.field_range[0] + self.field_range[2])/2
        center_y = (self.field_range[1] + self.field_range[3])/2 
        return (center_x, center_y)

    # width and height of the main field center
    @property
    def field_half_size(self)->Tuple[float,float]:
        width = (self.field_range[2] - self.field_range[0])/2
        height = (self.field_range[3] - self.field_range[1])/2
        return (width, height)

    # return all entities in the world. May not be used because Vehicle and Entit have no common parent class
    @property
    def entities(self)->List[Union[Vehicle,Entity]]:
        result_list:List[Union[Vehicle,Entity]] = []
        for vehicle in self.vehicle_list:
            result_list.append(vehicle)
        for landmark in self.landmark_list:
            result_list.append(landmark)
        for obstacle in self.obstacle_list:
            result_list.append(obstacle)
        return result_list

    # return all vehicles controllable by external policies
    @property
    def policy_vehicles(self):
        raise NotImplementedError()

    # return all vehicles controlled by world scripts
    @property
    def scripted_vehicles(self):
        raise NotImplementedError()
    
    # update the physical state for one sim_step.
    def _update_one_sim_step(self):
        for vehicle in self.vehicle_list:
            state = vehicle.state
            # if the vehicle is not movable, skip update its physical state
            if state.movable:
                new_state_data = _update_one_sim_step_warp(state, vehicle, self.sim_t)
                state.coordinate[0], state.coordinate[1], state.theta = new_state_data

    # check collision state for each vehicle
    def _check_collision(self):
        for idx_a, vehicle_a in enumerate(self.vehicle_list):
            if vehicle_a.state.crashed :
                continue

            # check if the agent_a crashed into other agent_b
            for idx_b, vehicle_b in enumerate(self.vehicle_list):
                if idx_a == idx_b:
                    continue
                dist = ((vehicle_a.state.coordinate[0]-vehicle_b.state.coordinate[0])**2
                      +(vehicle_a.state.coordinate[1]-vehicle_b.state.coordinate[1])**2)**0.5
                if dist < vehicle_a.r_safe + vehicle_b.r_safe:
                    vehicle_a.state.collision = True
                    vehicle_a.state.movable = False
                    break

            # check if the agent_a crashed into a obstacle
            for obstacle in self.obstacle_list:
                dist = ((vehicle_a.state.coordinate[0]-obstacle.state.coordinate[0])**2
                      +(vehicle_a.state.coordinate[1]-obstacle.state.coordinate[1])**2)**0.5
                if dist < vehicle_a.r_safe + obstacle.radius:
                    vehicle_a.state.collision = True
                    vehicle_a.state.movable = False
                    break
    
    # pickle the GUI data and dump to the sheard memory file
    def dumpGUI(self, port_type = 'file'):
        GUI_data = {'field_range':self.field_range,
                    'total_time':self.total_time,
                    'vehicle_list':self.vehicle_list,
                    'landmark_list':self.landmark_list,
                    'obstacle_list':self.obstacle_list,
                    'info':self.data_slot}
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

    # update state of the world
    def step(self):
        if self.GUI_port is not None:
            for idx in range(self.sim_step):
                self.total_time += self.sim_t
                self._update_one_sim_step()
                self._check_collision()
                # if use GUI, slow down the simulation speed
                time.sleep(self.sim_t)
                self.dumpGUI()
        else:
            for idx in range(self.sim_step):
                self.total_time += self.sim_t
                self._update_one_sim_step()
                self._check_collision()

    def ros_step(self,total_time):
        self.total_time = total_time
        self._check_collision()
        self.dumpGUI()
                
# warp one sim step into a function with pure math calculation
def _update_one_sim_step_warp(state:VehicleState, vehicle:Vehicle, dt:float):
    def linear_update(x:float, dx_dt:float, dt:float, target:float)->float:
        if x < target:
            return min(x + dx_dt*dt, target)
        elif x > target:
            return max(x - dx_dt*dt, target)
        return x
    target_vel_b = state.ctrl_vel_b * vehicle.K_vel
    state.vel_b = linear_update(state.vel_b, vehicle.dv_dt, dt, target_vel_b) 
    target_phi = state.ctrl_phi * vehicle.K_phi
    state.phi = linear_update(state.phi, vehicle.dphi_dt, dt, target_phi) 
    
    update_data = _update_one_sim_step_njit(state.phi,
                                       state.vel_b,
                                       state.theta,
                                       vehicle.L_axis,
                                       state.coordinate[0],
                                       state.coordinate[1],
                                       dt)
    return update_data

# core math calculation part for speed up by numba
# if want to speed up, please uncomment the followed code
#from numba import njit
#@njit
def _update_one_sim_step_njit(_phi:float, _vb:float, _theta:float, _L:float, _x:float, _y:float, dt:float)->Tuple[float,float,float]:
    sth = np.sin(_theta)
    cth = np.cos(_theta)
    _xb = _x - cth*_L/2.0
    _yb = _y - sth*_L/2.0
    tphi = np.tan(_phi)
    _omega = _vb/_L*tphi
    _delta_theta = _omega * dt
    if abs(_phi)>0.00001:
        _rb = _L/tphi
        _delta_tao = _rb*(1-np.cos(_delta_theta))
        _delta_yeta = _rb*np.sin(_delta_theta)
    else:
        _delta_tao = _vb*dt*(_delta_theta/2.0)
        _delta_yeta = _vb*dt*(1-_delta_theta**2/6.0)
    _xb += _delta_yeta*cth - _delta_tao*sth
    _yb += _delta_yeta*sth + _delta_tao*cth
    _theta += _delta_theta
    _theta = (_theta/3.1415926)%2*3.1415926

    nx = _xb + np.cos(_theta)*_L/2.0
    ny = _yb + np.sin(_theta)*_L/2.0
    ntheta = _theta
    return nx,ny,ntheta