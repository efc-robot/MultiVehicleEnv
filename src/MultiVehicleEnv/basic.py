from typing import *
import numpy as np
from .geometry import laser_circle_dist


class Sensor(object):
    def __init__(self):
        pass


class Lidar(Sensor):
    def __init__(self):
        self.angle_min = -np.pi
        self.angle_max = np.pi
        self.N_laser = 720
        self.range_min = 0.15
        self.range_max = 12.0


class VehicleState(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
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
        self.vehicle_id:str = 'none'
        # safe radius of the vehicle
        self.r_safe:float = 0.17
        # length of the vehicle
        self.L_car:float = 0.250
        # width of the vehicle
        self.W_car:float = 0.185
        # distance between front and back wheel
        self.L_axis:float = 0.20
        # coefficient of back whell velocity control
        self.K_vel:float = 0.361
        # coefficient of front wheel deflection control
        self.K_phi:float = 0.561
        # the acceleration of the back whell velocity
        self.dv_dt:float = 2.17
        # the angular acceleration of the back whell velocity
        self.dphi_dt:float = 2.10
        # the color of the vehicle
        self.color:list[Union[list[float] , float]] = [[0.0,0.0,0.0],0.0]
        # the discrete action table of the vehicle, action_code -> (ctrl_vel,ctrl_phi) 
        self.discrete_table:Dict[int,Tuple[float,float]] = {0:( 0.0, 0.0),
                               1:( 1.0, 0.0), 2:( 1.0, 1.0), 3:( 1.0, -1.0),
                               4:(-1.0, 0.0), 5:(-1.0, 1.0), 6:(-1.0, -1.0)}
        self.data_slot:Dict[str,Any]= {}

        self.lidar:Lidar = None
        # the state of the vehicle
        self.state:VehicleState = VehicleState()


class EntityState(object):
    def __init__(self):
        # center point position in x,y axis
        self.coordinate:List[float] = [0.0,0.0]
        # direction of entity axis
        self.theta:float = 0.0


class Obstacle(object):
    def __init__(self):
        # the redius of the entity
        self.radius:float = 1.0
        # true if the entity can crash into other collideable object
        self.collideable:bool = False
        # the color of the entoty
        self.color:list[Union[list[float] , float]] = [[0.0,0.0,0.0],0.0]
        # the state of the entity
        self.state:EntityState = EntityState()

class Landmark(object):
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
    def __init__(self, kinetic_mode = None, lidar_mode = None):
        # list of vehicles and entities (can change at execution-time!)
        self.vehicle_list:List[Vehicle] = []
        self.landmark_list:List[Landmark] = []
        self.obstacle_list:List[Obstacle] = []
        self.vehicle_id_list = []
        self.data_interface = {}

        self.kinetic_mode = kinetic_mode
        self.lidar_mode = lidar_mode

        if self.kinetic_mode == 'linear':
            self.linear_matrix = None
        
        if self.lidar_mode == 'table':
            #the data for liar scaning by look-up table
            self.lidar_table={}
            self.lidar_NUM_R=101    
            self.lidar_r_vec_vec=np.array([])
            self.lidar_size_vec =np.array([])

        # range of the main field
        self.field_range:List[float] = [-1.0,-1.0,1.0,1.0]

        # real-world time duration for one MDP step
        self.step_t:float = 1.0

        # split the step_t into sim_step pieces state simulation
        self.sim_step:int = 1000

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
    def entities(self)->List[Union[Vehicle,Obstacle,Landmark]]:
        result_list:List[Union[Vehicle,Obstacle,Landmark]] = []
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
    
    def assign_data_step(self):
        for vehicle in self.vehicle_list:
            state = vehicle.state
            data_interface = self.data_interface[vehicle.vehicle_id]
            state.coordinate[0] = data_interface['x']
            state.coordinate[1] = data_interface['y']
            state.theta = data_interface['theta']

    # update the physical state for one sim_step.
    def _update_one_sim_step(self):
        for vehicle in self.vehicle_list:
            state = vehicle.state
            data_interface = self.data_interface[vehicle.vehicle_id]
            data_interface['x'] = state.coordinate[0]
            data_interface['y'] = state.coordinate[1]
            data_interface['theta'] = state.theta
            
            # if the vehicle is not movable, skip update its physical state
            if state.movable:
                if self.kinetic_mode is None:
                    new_state_data = _update_one_sim_step_warp(state, vehicle, self.sim_t)
                else:
                    new_state_data = _update_one_sim_step_linear_warp(state, vehicle, self.sim_t, self.linear_matrix)
                data_interface['x'], data_interface['y'], data_interface['theta'] = new_state_data
    
    def _update_laser_sim_step(self):
        for vehicle in self.vehicle_list:
            if vehicle.lidar is None:
                continue
            state = vehicle.state
            lidar_c = np.array(state.coordinate)
            lidar_angle = state.theta
            lidar = vehicle.lidar
            obstacle_list = self.obstacle_list
            data_interface = self.data_interface[vehicle.vehicle_id]
            if self.lidar_mode is None:
                lidar_data = lidar_scan(lidar_c, lidar_angle, lidar, obstacle_list)
            else:
                lidar_data = lidar_scan_table(lidar_c, 
                                              lidar_angle, 
                                              lidar, 
                                              obstacle_list,
                                              self.lidar_table,
                                              self.lidar_size_vec,
                                              self.lidar_r_vec_vec)
            data_interface['lidar'] = lidar_data[0]

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
                    vehicle_a.state.crashed = True
                    vehicle_a.state.movable = False
                    break

            # check if the agent_a crashed into a obstacle
            for obstacle in self.obstacle_list:
                dist = ((vehicle_a.state.coordinate[0]-obstacle.state.coordinate[0])**2
                      +(vehicle_a.state.coordinate[1]-obstacle.state.coordinate[1])**2)**0.5
                if dist < vehicle_a.r_safe + obstacle.radius:
                    vehicle_a.state.crashed = True
                    vehicle_a.state.movable = False
                    break
                
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



def _update_one_sim_step_linear_warp(state:VehicleState, vehicle:Vehicle, dt:float, A):

    
    delta_x = (A[1][0] * state.vel_b+ A[3][0] * state.ctrl_vel_b)
    delta_y = (A[2][1] * state.phi+ A[4][1] * state.ctrl_phi)
    delta_theta =(A[6][2] * state.vel_b+ A[10][2] * state.ctrl_vel_b) * state.phi 
    delta_vb = (A[3][3]  * state.ctrl_vel_b + A[1][3]  * state.vel_b)
    delta_phi = (A[4][4] * state.ctrl_phi + A[2][4] * state.phi)

    #delta_x, delta_y, delta_theta, delta_vb, delta_phi = [0,0,0,0,0]
    
    state.vel_b += delta_vb
    state.phi += delta_phi
    theta = state.theta
    ct = np.cos(theta)
    st = np.sin(theta)
    x = state.coordinate[0]
    y = state.coordinate[1]
    nx = x + ct * delta_x - st * delta_y
    ny = y + st * delta_x + ct * delta_y

    ntheta = theta + delta_theta

    update_data = nx,ny,ntheta
    
    return update_data


def lidar_scan(lidar_c, lidar_angle, lidar: Lidar, obstacle_list: List[Obstacle]):
    ranges = np.ones(lidar.N_laser)*lidar.range_max
    intensities = np.zeros(lidar.N_laser) 
    angle_increment = (lidar.angle_max - lidar.angle_min) / lidar.N_laser
    for lidar_idx in range(lidar.N_laser):
        angle = lidar_angle + angle_increment * lidar_idx + lidar.angle_min
        final_dist = lidar.range_max
        final_intensities = 0.0
        for obstacle in obstacle_list:
            dist, inten = laser_circle_dist(lidar_c, angle, lidar.range_max, obstacle.state.coordinate, obstacle.radius)
            if dist < final_dist:
                final_dist = dist
                final_intensities = inten
        ranges[lidar_idx] = final_dist
        intensities[lidar_idx] = final_intensities
    return ranges, intensities



def lidar_scan_table(lidar_c, lidar_angle, lidar: Lidar, obstacle_list: List[Obstacle],table:dict,size_vec,r_vec_vec ): 
    
    lidar1=lidar
    lidar_Angle=lidar_angle
    lidar_Centroid=lidar_c
    obs_vec=obstacle_list
    PI=3.1415926535
    if len(obs_vec) == 0:
        final_ranges = np.ones(lidar1.N_laser)*lidar1.range_max
        final_intensities = np.zeros(lidar1.N_laser)
        return final_ranges,final_intensities

    coordinate_vec=np.zeros((2,len(obs_vec)))
    for i in range(len(obs_vec)):
        coordinate_vec[:,i]=obs_vec[i].state.coordinate-lidar_Centroid

    rel_coordinate_vec=np.zeros((2,len(obs_vec)))
    #转换成相对坐标
    rel_coordinate_vec[[0],:]=coordinate_vec[[0],:]*np.cos(lidar_Angle)+coordinate_vec[[1],:]*np.sin(lidar_Angle)
    rel_coordinate_vec[[1],:]=-coordinate_vec[[0],:]*np.sin(lidar_Angle)+coordinate_vec[[1],:]*np.cos(lidar_Angle)

    


    #输入坐标(obs_x,obs_y) （朝向theta省略）
    dis_vec=np.sqrt(rel_coordinate_vec[0,:]**2+rel_coordinate_vec[1,:]**2)
    theta_vec=np.arctan2(rel_coordinate_vec[1,:],rel_coordinate_vec[0,:])/PI*180 #-pi~pi  -180~180
    #精度为360度/N_laser  得到偏移量(注意：偏移量是往右循环移位，所以应该对应负角度)
    #之前没有降采样，这里也没有插值，而是直接取整来偏移  

    offset_by_theta = np.round(theta_vec*(lidar1.N_laser/360))*(-1)
    offset_by_theta=(offset_by_theta%lidar1.N_laser).astype('int')
    #查找表
    #一系列不同距离的球   暂时都在x轴上
    outputs_ranges=np.ones([lidar1.N_laser, 1])*lidar1.range_max*np.ones((1,len(dis_vec)))
    outputs_intensities=np.zeros([lidar1.N_laser, 1])*np.ones((1,len(obs_vec)))

    
    
    for i in range(len(dis_vec)):
        #根据半径
        size_index=np.abs(size_vec - obs_vec[i].radius ).argmin() 
        ranges_vec=table['r'][size_index]
        intensities_vec=table['i'][size_index]

        index=np.abs(r_vec_vec[size_index] - dis_vec[i]).argmin()     
        #根据距离查找ranges
        output_range=ranges_vec[:,[index]]
        output_range=np.vstack((output_range[offset_by_theta[i]:], output_range[:offset_by_theta[i]]))
        outputs_ranges[:,[i]]=output_range
        #查找intense
        output_intensities=intensities_vec[:,[index]]
        output_intensities=np.vstack((output_intensities[offset_by_theta[i]:], output_intensities[:offset_by_theta[i]]))
        outputs_intensities[:,[i]]=output_intensities
    #ranges"叠加"
    final_ranges=np.ones(lidar1.N_laser)*lidar1.range_max
    index_min=np.argmin(outputs_ranges,axis=1)
    for i in range(len(final_ranges)):
        final_ranges[i]=outputs_ranges[i,index_min[i]]
    #intense叠加
    final_intensities=np.zeros(lidar1.N_laser)
    index_max=np.argmax(outputs_intensities,axis=1)
    for i in range(len(final_intensities)):
        final_intensities[i]=outputs_intensities[i,index_max[i]]



    return final_ranges,final_intensities
