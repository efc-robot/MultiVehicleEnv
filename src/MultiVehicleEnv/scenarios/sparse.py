import numpy as np
from MultiVehicleEnv.basic import World, Vehicle, Entity
from MultiVehicleEnv.scenario import BaseScenario
from MultiVehicleEnv.utils import coord_dist, naive_inference

def reset_state(agent:Vehicle):
    agent.state.vel_b = 0
    agent.state.phi = 0
    agent.state.ctrl_vel_b = 0
    agent.state.ctrl_phi = 0
    agent.state.movable = True
    agent.state.crashed = False
    agent.state.theta
    agent.state.coordinate[0] = 0
    agent.state.coordinate[1] = 0

def check_confilict(world:World):
    coord_list = []
    radius_list = []
    for landmark in world.landmark_list:
        coord_list.append(landmark.state.coordinate)
        radius_list.append(landmark.radius)
    for obstacle in world.obstacle_list:
        coord_list.append(obstacle.state.coordinate)
        radius_list.append(obstacle.radius)
    for agent in world.vehicle_list:
        coord_list.append(agent.state.coordinate)
        radius_list.append(agent.r_safe)
    
    for idx_a in range(len(coord_list)):
        for idx_b in range(idx_a+1,len(coord_list)):
            dis = coord_dist(coord_list[idx_a], coord_list[idx_b])
            if dis < radius_list[idx_a] + radius_list[idx_b]:
                return True
    return False

class Scenario(BaseScenario):
    def make_world(self,args):
        self.reward_coef = args.reward_coef 
        self.control_coef = args.control_coef
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

        world.data_slot['max_step_number'] = 40
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world:World):
        world.data_slot['total_step_number'] = 0
        # set random initial states
        for agent in world.vehicle_list:
            reset_state(agent)
            agent.state.theta = np.random.uniform(0,2*3.14159)
            
        
        # place all landmark,obstacle and vehicles in the field with out conflict
        safe_count = 1000
        while True:
            if safe_count == 0:
                print('can not place objects with no conflict')
            safe_count -= 1
            for landmark in world.landmark_list:
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-1,+1)
                    landmark.state.coordinate[idx] = norm_pos * scale + trans

            for obstacle in world.obstacle_list:
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-1,+1)
                    obstacle.state.coordinate[idx] = norm_pos * scale + trans
    
            for agent in world.vehicle_list:
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-1,+1)
                    agent.state.coordinate[idx] = norm_pos * scale + trans
            
            conflict = check_confilict(world)
            if not conflict:
                break
            


    def done(self, agent:Vehicle, world:World):
        return False
    
    def reward(self, agent:Vehicle, world:World, old_world:World = None):
        rew:float = 0.0

        # reach reward
        Allreach = True

        for agent_old, agent_new, landmark in zip(old_world.vehicle_list, world.vehicle_list, world.landmark_list):
            dist_old = coord_dist(agent_old.state.coordinate, landmark.state.coordinate)
            dist_new = coord_dist(agent_new.state.coordinate, landmark.state.coordinate)
            rew += self.reward_coef*(dist_old-dist_new)
            if dist_new < landmark.radius:
                rew += 10.0
            
        for agent_a in world.vehicle_list:
            rew += self.control_coef * agent_a.state.ctrl_vel_b**2
            
        # collision reward
        if agent.state.crashed:
            rew -= 10.0
        
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
        landmark_pos = []
        for landmark in world.landmark_list:
            landmark_pos.append(get_pos(landmark))


        # communication of all other agents
        other_pos = []
      
        for other in world.vehicle_list:
            if other is agent: continue
            other_pos.append(get_pos(other,agent))
        return np.concatenate(agent_pos + landmark_pos + obstacle_pos + other_pos)

    def updata_callback(self, world:World):
        crashed_list = []
        reach_list = []
        for agent, landmark in zip(world.vehicle_list,world.landmark_list):
            if agent.state.crashed:
                crashed_list.append(agent)
            dist = coord_dist(agent.state.coordinate, landmark.state.coordinate)
            if dist < landmark.radius:
                reach_list.append(landmark)
        for agent in crashed_list:
            reset_state(agent)
            agent.state.theta = np.random.uniform(0,2*3.14159)
        check_confilict(world)
        safe_count = 1000
        while True:
            if safe_count == 0:
                print('can not place objects with no conflict')
            safe_count -= 1
            for landmark in reach_list:
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-1,+1)
                    landmark.state.coordinate[idx] = norm_pos * scale + trans

            for agent in crashed_list:
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-1,+1)
                    agent.state.coordinate[idx] = norm_pos * scale + trans
            
            conflict = check_confilict(world)
            if not conflict:
                break
            

    def info(self, agent:Vehicle, world:World):
        agent_info:dict = {}
        return agent_info
