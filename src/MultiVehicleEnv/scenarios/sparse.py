import numpy as np
from MultiVehicleEnv.basic import World, Vehicle, Entity
from MultiVehicleEnv.scenario import BaseScenario
from MultiVehicleEnv.utils import coord_dist, naive_inference


class Scenario(BaseScenario):
    def make_world(self,args):
        # init a world instance
        world = World()

        #for simulate real world
        world.step_t = args.step_t
        world.sim_step = args.sim_step
        world.field_range = [-2.0,-2.0,2.0,2.0]

        # set world.GUI_port as port dir when usegui is true
        if args.usegui:
            world.GUI_port = args.gui_port
        else:
            world.GUI_port = None

        # add 3 agents
        agent_number = 1
        world.vehicle_list = []
        for idx in range(agent_number):
            agent = Vehicle()
            agent.r_safe     = 0.17
            agent.L_car      = 0.25
            agent.W_car      = 0.18
            agent.L_axis     = 0.2
            agent.K_vel      = 0.707
            agent.K_phi      = 0.298
            agent.dv_dt      = 2.0
            agent.dphi_dt    = 3.0
            agent.color      = [[1.0,idx/3.0,idx/3.0],1.0]
            agent.discrete_table = None
            world.vehicle_list.append(agent)
            

        # add landmark_list
        landmark_number = 1
        world.landmark_list = []
        for idx in range(landmark_number):
            entity = Entity()
            entity.radius = 0.4
            entity.collideable = False
            entity.color  = [[0.0,1.0,0.0],0.1]
            world.landmark_list.append(entity)

        # add obstacle_list
        obstacle_number = 0
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
        for landmark in world.landmark_list:
            landmark.color[1] = 0.1
        # set real landmark and make it color solid
        world.data_slot['real_landmark'] = np.random.randint(len(world.landmark_list))
        real_landmark = world.landmark_list[world.data_slot['real_landmark']]
        real_landmark.color[1] = 1.0
                        
    def reward(self, agent:Vehicle, world:World):
        # Adversaries are rewarded for collisions with agents
        rew:float = 0.0
        real_landmark = world.landmark_list[world.data_slot['real_landmark']]



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
        
        world.data_slot['total_step_number'] += 1
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
            landmark_pos.append(get_pos(landmark, agent))


        # communication of all other agents
        other_pos = []
      
        for other in world.vehicle_list:
            if other is agent: continue
            other_pos.append(get_pos(other,agent))
        return np.concatenate(agent_pos + landmark_pos + obstacle_pos + other_pos)



    def info(self, agent:Vehicle, world:World):
        agent_info:dict = {}
        return agent_info