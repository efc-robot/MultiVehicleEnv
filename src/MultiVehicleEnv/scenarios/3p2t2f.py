import numpy as np
from MultiVehicleEnv.basic import World, Vehicle, Entity
from MultiVehicleEnv.scenario import BaseScenario
from MultiVehicleEnv.utils import coord_dist, naive_inference


class Scenario(BaseScenario):
    def make_world(self,args):
        # init a world instance
        world = World()
        
        # direction reward score coefficient
        self.direction_alpha:float = args.direction_alpha
        
        # direction reward type, train:init in reset, keyboard:set by keyboard, disable:no direction reward
        self.add_direction_encoder:str = args.add_direction_encoder
        assert self.add_direction_encoder in ['train','keyboard','disable']

        #for simulate real world
        world.step_t = args.step_t
        world.sim_step = args.sim_step
        world.field_range = [-2.4,-2.4,2.4,2.4]

        # define the task parameters
        # one agent can know where is the lanmark
        world.data_slot['view_threshold'] = 1.0
        # store the keyboard direction
        world.data_slot['key_direction'] = 0

        # set world.GUI_port as port dir when usegui is true
        if args.usegui:
            world.GUI_port = args.gui_port
        else:
            world.GUI_port = None

        # add 3 agents
        agent_number = 3
        world.vehicle_list = []
        for idx in range(agent_number):
            agent = Vehicle()
            agent.r_safe     = 0.17
            agent.L_car      = 0.25
            agent.W_car      = 0.18
            agent.L_axis     = 0.2
            agent.K_vel      = 0.18266
            agent.K_phi      = 0.298
            agent.dv_dt      = 2.0
            agent.dphi_dt    = 3.0
            agent.color      = [[1.0,idx/3.0,idx/3.0],1.0]
            world.vehicle_list.append(agent)
            

        # add landmark_list
        landmark_number = 2
        world.landmark_list = []
        for idx in range(landmark_number):
            entity = Entity()
            entity.radius = 0.2
            entity.collideable = False
            entity.color  = [[0.0,1.0,0.0],0.5]
            world.landmark_list.append(entity)

        # add obstacle_list
        obstacle_number = 2
        world.obstacle_list = []
        for idx in range(obstacle_number):
            entity = Entity()
            entity.radius = 0.2
            entity.collideable = True
            entity.color  = [[0.0,0.0,0.0],1.0]
            world.obstacle_list.append(entity)

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world:World):

        # set random initial states
        for agent in world.vehicle_list:
            agent.state.theta = np.random.uniform(0,2*3.14159)
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
                    norm_pos = np.random.uniform(-0.5,+0.5)
                    norm_pos = norm_pos + (0.5 if norm_pos>0 else -0.5)
                    landmark.state.coordinate[idx] = norm_pos * scale + trans
                all_circle.append((landmark.state.coordinate[0],landmark.state.coordinate[1],landmark.radius))
    
            for obstacle in world.obstacle_list:
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-1,+1)
                    obstacle.state.coordinate[idx] = norm_pos * scale*0.5 + trans
                all_circle.append((obstacle.state.coordinate[0],obstacle.state.coordinate[1],obstacle.radius))
    
            for agent in world.vehicle_list:
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-1,+1)
                    agent.state.coordinate[idx] = norm_pos * scale + trans
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
                        
    def reward(self, agent:Vehicle, world:World):
                # Adversaries are rewarded for collisions with agents
        rew:float = 0.0

        # direction reward
        prefer_action = naive_inference(agent.state.coordinate[0],
                                        agent.state.coordinate[1],
                                        agent.state.theta)
        same_direction = np.sign(agent.state.ctrl_vel_b) == prefer_action[0] and np.sign(agent.state.ctrl_phi) == prefer_action[1]
        if same_direction:
            rew += self.direction_alpha * 1.0


        # reach reward
        Allreach = True
        real_landmark = world.landmark_list[world.data_slot['real_landmark']]
        for agent_a in world.vehicle_list:
            dist = coord_dist(agent_a.state.coordinate, real_landmark.state.coordinate)
            if dist > agent_a.r_safe +real_landmark.radius:
                Allreach = False
        if Allreach:
            rew += 1.0
            
        # collision reward
        if agent.state.crashed:
            rew -= 1.0

        return rew

    def observation(self,agent:Vehicle, world:World):
        def get_pos(obj , main_agent:Vehicle = None):
            main_shift_x = 0.0 if main_agent  is None else main_agent.state.coordinate[0]
            main_shift_y = 0.0 if main_agent  is None else main_agent.state.coordinate[1]
            x = obj.state.coordinate[0] - main_shift_x 
            y = obj.state.coordinate[1] - main_shift_y

            if isinstance(obj,Vehicle):
                ctheta = np.cos(obj.state.theta)
                stheta = np.sin(obj.state.theta)
                return [x, y, ctheta, stheta]
            elif isinstance(obj,Entity):
                return [x, y]
            else:
                raise TypeError


        # get positions of all obstacle_list in this agent's reference frame
        obstacle_pos = []
        for obstacle in world.obstacle_list:
            epos = np.array([obstacle.state.coordinate[0] - agent.state.coordinate[0],
                             obstacle.state.coordinate[1] - agent.state.coordinate[1]])
            obstacle_pos.append(epos)

        agent_pos = [get_pos(agent)]

        # check in view
        in_view = 0.0
        landmark_pos = []
        for landmark in world.landmark_list:
            dist = coord_dist(agent.state.coordinate, landmark.state.coordinate)
            if dist < world.data_slot['view_threshold']:
                in_view = 1.0
                landmark_pos.append(get_pos(landmark, agent))
            else:
                landmark_pos.append([0,0])

        # communication of all other agents
        other_pos = []
      
        for other in world.vehicle_list:
            if other is agent: continue
            other_pos.append(get_pos(other,agent))
        
        
        if  self.add_direction_encoder == 'train':
            return np.concatenate([agent.data_slot['direction_obs']]+agent_pos + landmark_pos + obstacle_pos + other_pos + [[in_view]] )
        elif self.add_direction_encoder == 'keyboard':
            key_direction = world.data_slot['key_direction']
            one_hot_direction = world.data_slot['direction_decoder'][key_direction]
            return np.concatenate([[one_hot_direction] +  agent_pos + landmark_pos + obstacle_pos + other_pos + [in_view]])
        elif self.add_direction_encoder == 'disable':
            return np.concatenate(agent_pos + landmark_pos + obstacle_pos + other_pos)



    def info(self, agent:Vehicle, world:World):
        agent_info:dict = {}
        return agent_info