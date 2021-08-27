import numpy as np
from MultiVehicleEnv.basic import World, Vehicle, Entity
from MultiVehicleEnv.scenario import BaseScenario
from MultiVehicleEnv.utils import coord_data_dist, naive_inference, hsv2rgb

import math
from numpy.linalg import norm

def default_vehicle():
    agent = Vehicle()
    agent.r_safe     = 0.17
    agent.L_car      = 0.25
    agent.W_car      = 0.18
    agent.L_axis     = 0.2
    agent.K_vel      = 0.707
    agent.K_phi      = 0.596
    agent.dv_dt      = 10.0
    agent.dphi_dt    = 10.0
    agent.color      = [[1.0,0.0,0.0],1.0]
    return agent


def f_Relative(groundtruth, P, N):
    '''
    Inputs:
        groundtruth
        P: predicted topo
        N: number of nodes
    Outputs:
        P_final: topo rotated and transited
        Gamma: rotation mat
        t: transit argument
    '''
    I_Na = np.eye(N)
    # SVD
    M_h = P.T
    L = I_Na - np.ones([N, N]) * 1/N
    M = groundtruth.T
    S = np.matmul( np.matmul(M_h.T, L), M )
    U1, Lambda1, V1 = np.linalg.svd(S)
    
    #TODO: why should we use.T
    U1 = U1.T
    V1 = V1.T
    
    # calculate gamma mat and T mat
    gamma = np.matmul(V1, U1.T)
    t = np.matmul((M.T - np.matmul(gamma, M_h.T)), np.ones([N,1])) * 1 / N

    # V mat
    v_x = np.kron(np.ones([N,1]), np.expand_dims(np.array([1,0]),axis=-1))
    v_y = np.kron(np.ones([N,1]), np.expand_dims(np.array([0,1]),axis=-1))

    # rotation correction and transit correction
    P_array = P
    P_final = np.expand_dims(np.matmul(np.kron(I_Na, gamma), P_array.flatten(order='F')), axis=-1) + np.matmul(np.concatenate([v_x, v_y], axis=1), t)

    return P_final, gamma, t



def error_rel_g(p1, p2, n):
    p2_rel,_,_ = f_Relative(p1, p2, n)
    error2 = p2_rel - np.expand_dims(p1.reshape(-1, order='F'), axis=-1)
    error_rel = np.sqrt(1/n * np.sum(error2 ** 2))

    return error_rel

class Scenario(BaseScenario):
    def make_world(self,args):
        # init a world instance
        self.args = args
        world = World()

        #for simulate real world
        world.step_t = args.step_t
        world.sim_step = args.sim_step
        world.field_range = [-2.0,-2.0,2.0,2.0]

        num_agent_ray = 60


        # set world.GUI_port as port dir when usegui is true
        if args.usegui:
            world.GUI_port = args.guiport
        else:
            world.GUI_port = None

        # add 4 agents
        agent_number = args.num_agents

        # ideal formation topo side length
        world.ideal_side_len = args.ideal_side_len

        # calculate the ideal formation topo
        world.ideal_topo_point = [[], []]
        for i in range(agent_number):
            world.ideal_topo_point[0].append(world.ideal_side_len / np.sqrt(2) * np.cos(i/agent_number*2*np.pi))
            world.ideal_topo_point[1].append(world.ideal_side_len / np.sqrt(2) * np.sin(i/agent_number*2*np.pi))

        world.vehicle_list = []
        for idx in range(agent_number):
            vehicle_id = 'AKM_'+str(idx+1)
            agent = default_vehicle()
            agent.vehicle_id = vehicle_id
            agent.ray        = np.zeros((num_agent_ray, 2))
            agent.min_ray    = np.array([0, len(agent.ray)//2])
            agent.color = [hsv2rgb((idx/agent_number)*360,1.0,1.0), 0.8]
            world.vehicle_list.append(agent)
            world.vehicle_id_list.append(vehicle_id)
            world.data_interface[vehicle_id] = {}
            world.data_interface[vehicle_id]['x'] = 0.0
            world.data_interface[vehicle_id]['y'] = 0.0
            world.data_interface[vehicle_id]['theta'] = 0.0

            

        # add landmark_list
        landmark_number = args.num_landmarks
        world.landmark_list = []
        for idx in range(landmark_number):
            entity = Entity()
            entity.radius = 0.4
            entity.collideable = False
            entity.color  = [[0.0,1.0,0.0],0.1]
            world.landmark_list.append(entity)

        # add obstacle_list
        obstacle_number = args.num_obstacles
        world.obstacle_list = []
        for idx in range(obstacle_number):
            entity = Entity()
            entity.radius = 0.2
            entity.collideable = True
            entity.color  = [[0.0,0.0,0.0],1.0]
            world.obstacle_list.append(entity)

        world.data_slot['max_step_number'] = 20
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world:World):
        world.data_slot['total_step_number'] = 0
        # set random initial states
        for agent in world.vehicle_list:
            agent.state.theta = np.random.uniform(0,1/2*3.14159)
            #agent.state.theta = 1/4 * 3.14159 
            agent.state.vel_b = 0
            agent.state.phi = 0
            agent.state.ctrl_vel_b = 0
            agent.state.ctrl_phi = 0
            agent.state.movable = True
            agent.state.crashed = False
        
        # place all landmark,obstacle and vehicles in the field with out conflict
        conflict = True
        while conflict:
            conflict = False
            all_circle = []
            for landmark in world.landmark_list:
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    #norm_pos = np.random.uniform(-0.5,+0.5)
                    norm_pos = 8
                    norm_pos = norm_pos + (0.5 if norm_pos>0 else -0.5)
                    landmark.state.coordinate[idx] = norm_pos * scale + trans
                all_circle.append((landmark.state.coordinate[0],landmark.state.coordinate[1],landmark.radius))
    
            for obstacle in world.obstacle_list:
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-1,+1)
                    obstacle.state.coordinate[idx] = norm_pos * scale*0.5 + trans # Why?
                all_circle.append((obstacle.state.coordinate[0],obstacle.state.coordinate[1],obstacle.radius))


            for i, agent in enumerate(world.vehicle_list):
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-1,1)
                    #norm_pos = world.ideal_topo_point[idx][i]
                    agent.state.coordinate[idx] = norm_pos * scale + trans
                    #agent.state.coordinate[idx] = norm_pos
                all_circle.append((agent.state.coordinate[0],agent.state.coordinate[1],agent.r_safe))
            
            for idx_a in range(len(all_circle)):
                for idx_b in range(idx_a+1,len(all_circle)):
                    x_a = all_circle[idx_a][0]
                    y_a = all_circle[idx_a][1]
                    r_a = all_circle[idx_a][2]
                    x_b = all_circle[idx_b][0]
                    y_b = all_circle[idx_b][1]
                    r_b = all_circle[idx_b][2]
                    dis = ((x_a - x_b)**2 + (y_a - y_b)**2)**0.5
                    if dis < r_a + r_b:
                        conflict = True
                        break
                if conflict:
                    break

        for agent in world.vehicle_list:
            agent_data = world.data_interface[agent.vehicle_id]
            target_x = world.landmark_list[0].state.coordinate[0]
            target_y = world.landmark_list[0].state.coordinate[1]
            target_data = {'x':target_x, 'y':target_y}
            agent.dis2goal = coord_data_dist(agent_data, target_data)
            agent.dis2goal_prev = agent.dis2goal
        
        for landmark in world.landmark_list:
            landmark.color[1] = 0.1
        # set real landmark and make it color solid
        world.data_slot['real_landmark'] = np.random.randint(len(world.landmark_list))
        real_landmark = world.landmark_list[world.data_slot['real_landmark']]
        real_landmark.color[1] = 1.0
        
        # encode 4 directions into [0,1,2,3]
        def encode_direction(direction):
            if direction[0] > 0 and direction[1] > 0:
                return 0
            if direction[0] < 0 and direction[1] > 0:
                return 1
            if direction[0] < 0 and direction[1] < 0:
                return 2
            if direction[0] > 0 and direction[1] < 0:
                return 3
        # decode 4 direction code [0,1,2,3] into onehot vector
        world.data_slot['direction_decoder'] = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        # give each agent a direction code as a human hint
        for agent in world.vehicle_list:
            direction = [real_landmark.state.coordinate[0] - agent.state.coordinate[0],
                         real_landmark.state.coordinate[1] - agent.state.coordinate[1]]
            agent.data_slot['direction_obs'] = world.data_slot['direction_decoder'][encode_direction(direction)]




                        
    def reward(self, agent:Vehicle, world:World, old_world:World = None):

        return 0

    def get_relAngle(self, center, target):
        ang = math.atan2(target.state.coordinate[1] - center['y'], target.state.coordinate[0] - center['x']) - center['theta']
        if abs(ang) > np.pi:
            ang -= np.sign(ang) * 2 * np.pi
        return ang

    def formation_reward(self, agent, world):
        agent_data = world.data_interface[agent.vehicle_id]
        pos_rel = [[],[]] 
        for any_agent in world.vehicle_list:
            any_agent_data = world.data_interface[any_agent.vehicle_id]
            pos_rel[0].append(any_agent_data['x'] - agent_data['x'])
            pos_rel[1].append(any_agent_data['y'] - agent_data['y'])
        topo_err = error_rel_g(np.array(world.ideal_topo_point), np.array(pos_rel), len(world.vehicle_list))

        return -topo_err

    def observation(self,agent:Vehicle, world:World):
        
        agent_data = world.data_interface[agent.vehicle_id]
        print(agent_data)
        agent.dis2goal_prev = agent.dis2goal

        target_x = world.landmark_list[0].state.coordinate[0]
        target_y = world.landmark_list[0].state.coordinate[1]
        target_data = {'x':target_x, 'y':target_y}
        agent.dis2goal = coord_data_dist(agent_data, target_data)
        agent.ang2goal = self.get_relAngle(agent_data, world.landmark_list[0])
        # get positions of all obstacle_list in this agent's reference frame



        agt_dis = []
        agt_ang = []
        target_dis = [np.array([agent.dis2goal])] 
        target_ang = [np.array([self.get_relAngle(agent_data, world.landmark_list[0])])]
        #formation_err = [np.array([self.formation_reward(agent, world)])]
        formation_err = [np.array([0])]

        for entity in world.vehicle_list:
            if entity == agent: 
                continue
            dis2agt = np.array([norm(np.array(entity.state.coordinate) - np.array(agent.state.coordinate))])
            ang = self.get_relAngle(agent_data, entity)
            agt_dis.append(dis2agt)
            agt_ang.append(np.array([ang]))

        for _ in range(2):
            agt_dis.append(np.array([0.0]))
            agt_ang.append(np.array([0.0]))
        print(agt_dis , agt_ang , target_ang , target_dis , formation_err)
        return np.concatenate(agt_dis + agt_ang + target_ang + target_dis + formation_err)


    def info(self, agent:Vehicle, world:World):
        agent_info:dict = {}
        return agent_info