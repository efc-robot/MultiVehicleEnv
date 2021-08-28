import numpy as np
from MultiVehicleEnv.basic import World, Vehicle, Entity
from MultiVehicleEnv.scenario import BaseScenario
from MultiVehicleEnv.utils import coord_dist, hsv2rgb


def encode_direction(coord):
    x = coord[0]
    y = coord[1]
    code = (2 if x < y else 0) + (1 if x < -y else 0)
    return code


def default_vehicle():
    agent = Vehicle()
    agent.r_safe = 0.17
    agent.L_car = 0.250
    agent.W_car = 0.185
    agent.L_axis = 0.20
    agent.K_vel = 0.361
    agent.K_phi = 0.561
    agent.dv_dt = 2.17
    agent.dphi_dt = 2.10
    agent.color = [[1.0, 0.0, 0.0], 1.0]
    agent.discrete_table = None
    return agent


def place_wo_conflict(world: World, agent_list, landmark_list, obstacle_list):
    for agent_idx in agent_list:
        state = world.vehicle_list[agent_idx].state
        state.reset()
        state.theta = np.random.uniform(0, 2*3.14159)
    safe_count = 1000
    while True:
        if safe_count < 0:
            raise RuntimeError
        safe_count -= 1
        for agent_idx in agent_list:
            # place all agent around the ground
            x = np.random.uniform(-2, +2)
            y = 2.25*(np.random.randint(2)*2-1)
            if np.random.randint(2) == 1:
                x, y = y, x
            world.vehicle_list[agent_idx].state.coordinate[0] = x
            world.vehicle_list[agent_idx].state.coordinate[1] = y

        for landmark_idx in landmark_list:
            # place all landmark around the ground
            x = np.random.uniform(-2, +2)
            y = 2.25*(np.random.randint(2)*2-1)
            if np.random.randint(2) == 1:
                x, y = y, x
            world.landmark_list[landmark_idx].state.coordinate[0] = x
            world.landmark_list[landmark_idx].state.coordinate[1] = y

        for obstacle_idx in obstacle_list:
            for idx in range(2):
                scale = world.field_half_size[idx]
                trans = world.field_center[idx]
                norm_pos = np.random.uniform(-1, +1)
                state = world.obstacle_list[obstacle_idx].state
                state.coordinate[idx] = norm_pos * scale + trans

        no_conflict_flag = True
        for landmark_idx in landmark_list:
            landmark = world.landmark_list[landmark_idx]
            agent = world.vehicle_list[landmark_idx]
            code1 = encode_direction(landmark.state.coordinate)
            code2 = encode_direction(agent.state.coordinate)
            if code1 == code2:
                no_conflict_flag = False
            for obstacle in world.obstacle_list:
                dis = coord_dist(landmark.state.coordinate,
                                 obstacle.state.coordinate)
                if dis < landmark.radius + obstacle.radius:
                    no_conflict_flag = False
            for agent in world.vehicle_list:
                dis = coord_dist(agent.state.coordinate,
                                 landmark.state.coordinate)
                if dis < landmark.radius + agent.r_safe:
                    no_conflict_flag = False

        for obstacle_idx in obstacle_list:
            obstacle = world.obstacle_list[obstacle_idx]
            for landmark in world.landmark_list:
                dis = coord_dist(obstacle.state.coordinate,
                                 landmark.state.coordinate)
                if dis < obstacle.radius + landmark.radius:
                    no_conflict_flag = False
            for agent in world.vehicle_list:
                dis = coord_dist(agent.state.coordinate,
                                 obstacle.state.coordinate)
                if dis < obstacle.radius + agent.r_safe:
                    no_conflict_flag = False

        for agent_idx in agent_list:
            agent = world.vehicle_list[agent_idx]
            for obstacle in world.obstacle_list:
                dis = coord_dist(agent.state.coordinate,
                                 obstacle.state.coordinate)
                if dis < agent.r_safe + obstacle.radius:
                    no_conflict_flag = False
            landmark = world.landmark_list[agent_idx]
            dis = coord_dist(agent.state.coordinate, landmark.state.coordinate)
            if dis < agent.r_safe + landmark.radius:
                no_conflict_flag = False

            code1 = encode_direction(landmark.state.coordinate)
            code2 = encode_direction(agent.state.coordinate)
            if code1 == code2:
                no_conflict_flag = False
        if no_conflict_flag:
            break


def check_conflict(world: World):
    obstacle_coord_list = []
    obstacle_radius_list = []
    for obstacle in world.obstacle_list:
        obstacle_coord_list.append(obstacle.state.coordinate)
        obstacle_radius_list.append(obstacle.radius)

    landmark_coord_list = []
    landmark_radius_list = []
    for landmark in world.landmark_list:
        landmark_coord_list.append(landmark.state.coordinate)
        landmark_radius_list.append(landmark.radius)

    agent_coord_list = []
    agent_radius_list = []
    for agent in world.vehicle_list:
        agent_coord_list.append(agent.state.coordinate)
        agent_radius_list.append(agent.r_safe)

    for idx_a in range(len(agent_coord_list)):
        for idx in range(len(obstacle_coord_list)):
            dis = coord_dist(agent_coord_list[idx_a], obstacle_coord_list[idx])
            if dis < agent_radius_list[idx_a] + obstacle_radius_list[idx]:
                return True
        for idx_b in range(idx_a+1, len(agent_coord_list)):
            dis = coord_dist(agent_coord_list[idx_a], agent_coord_list[idx_b])
            if dis < agent_radius_list[idx_a] + agent_radius_list[idx_b]:
                return True
    return False


class Scenario(BaseScenario):
    def make_world(self, args):
        self.reward_coef = 1.0
        self.control_coef = 1.0
        if 'reward_coef' in args.__dict__.keys():
            self.reward_coef = args.reward_coef
        if 'reward_coef' in args.__dict__.keys():
            self.control_coef = args.control_coef
        # init a world instance
        world = World()

        # for simulate real world
        world.step_t = args.step_t
        world.sim_step = args.sim_step
        world.field_range = [-2.5, -2.5, 2.5, 2.5]

        # set world.GUI_port as port dir when usegui is true
        if args.usegui:
            world.GUI_port = args.gui_port
        else:
            world.GUI_port = None

        # add 3 agents
        agent_number = 1
        world.vehicle_list = []
        for idx in range(agent_number):
            vehicle_id = 'AKM_'+str(idx+1)
            agent = default_vehicle()
            agent.vehicle_id = vehicle_id
            agent.color = [hsv2rgb((idx/agent_number)*360, 1.0, 1.0), 0.8]
            world.vehicle_list.append(agent)
            world.vehicle_id_list.append(vehicle_id)
            world.data_interface[vehicle_id] = {}
            world.data_interface[vehicle_id]['x'] = 0.0
            world.data_interface[vehicle_id]['y'] = 0.0
            world.data_interface[vehicle_id]['theta'] = 0.0

        # add landmark_list
        landmark_number = agent_number
        world.landmark_list = []
        for idx in range(landmark_number):
            entity = Entity()
            entity.radius = 0.2
            entity.collideable = False
            entity.color = [hsv2rgb((idx/agent_number)*360, 1.0, 1.0), 0.8]
            world.landmark_list.append(entity)

        # add obstacle_list
        obstacle_number = 1
        world.obstacle_list = []
        for idx in range(obstacle_number):
            entity = Entity()
            entity.radius = 0.2
            entity.collideable = True
            entity.color = [[0.0, 0.0, 0.0], 1.0]
            world.obstacle_list.append(entity)

        world.data_slot['max_step_number'] = 40
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world: World):
        world.data_slot['total_step_number'] = 0
        # set random initial states
        agent_list = [idx for idx in range(len(world.vehicle_list))]
        landmark_list = [idx for idx in range(len(world.landmark_list))]
        obstacle_list = [idx for idx in range(len(world.obstacle_list))]
        place_wo_conflict(world, agent_list, landmark_list, obstacle_list)

    def done(self, agent: Vehicle, world: World):
        return False

    def reward(self, agent: Vehicle, world: World, old_world: World = None):
        rew: float = 0.0
        zip_list = zip(old_world.vehicle_list,
                       world.vehicle_list,
                       world.landmark_list)
        for agent_old, agent_new, landmark in zip_list:
            dist_old = coord_dist(agent_old.state.coordinate,
                                  landmark.state.coordinate)
            dist_new = coord_dist(agent_new.state.coordinate,
                                  landmark.state.coordinate)
            rew += self.reward_coef*(dist_old-dist_new)
            if dist_new < agent_new.r_safe + landmark.radius:
                rew += 10.0

        for agent_a in world.vehicle_list:
            rew += self.control_coef * agent_a.state.ctrl_vel_b**2

        # collision reward
        if agent.state.crashed:
            rew -= 10.0

        world.data_slot['total_step_number'] += 1
        return rew

    def observation(self, agent: Vehicle, world: World):
        agent_data = world.data_interface[agent.vehicle_id]

        def get_pos(obj, main_agent: dict = None):
            if isinstance(obj, Entity):
                obj_x = obj.state.coordinate[0]
                obj_y = obj.state.coordinate[1]
            else:
                obj_x = obj['x']
                obj_y = obj['y']

            main_shift_x = 0.0 if main_agent is None else main_agent['x']
            main_shift_y = 0.0 if main_agent is None else main_agent['y']
            x = obj_x - main_shift_x
            y = obj_y - main_shift_y

            if isinstance(obj, Entity):
                return [x, y]
            elif isinstance(obj, dict):
                ctheta = np.cos(obj['theta'])
                stheta = np.sin(obj['theta'])
                return [x, y, ctheta, stheta]
            else:
                raise TypeError

        # get positions of all obstacle_list in this agent's reference frame
        obstacle_pos = []
        for obstacle in world.obstacle_list:
            obstacle_pos.append(get_pos(obstacle, agent_data))

        agent_pos = [get_pos(agent_data)]

        # check in view
        landmark_pos = []
        idx = world.vehicle_list.index(agent)
        landmark_pos.append(get_pos(world.landmark_list[idx]))

        # communication of all other agents
        other_pos = []

        for other in world.vehicle_list:
            if other is agent:
                continue
            other_data = world.data_interface[other.vehicle_id]
            other_pos.append(get_pos(other_data, agent_data))
        return np.concatenate(agent_pos + landmark_pos +
                              obstacle_pos + other_pos)

    def updata_callback(self, world: World):
        crashed_list = []
        reach_list = []
        for idx in range(len(world.vehicle_list)):
            agent = world.vehicle_list[idx]
            landmark = world.landmark_list[idx]
            if agent.state.crashed:
                crashed_list.append(idx)
            dist = coord_dist(agent.state.coordinate,
                              landmark.state.coordinate)
            if dist < landmark.radius:
                reach_list.append(idx)
        place_wo_conflict(world, crashed_list, reach_list, [])

    def ros_updata_callback(self, world: World):
        crashed_list = []
        reach_list = []
        for idx in range(len(world.vehicle_list)):
            agent = world.vehicle_list[idx]
            landmark = world.landmark_list[idx]
            if agent.state.crashed:
                crashed_list.append(idx)
            dist = coord_dist(agent.state.coordinate,
                              landmark.state.coordinate)
            if dist < landmark.radius:
                reach_list.append(idx)
        place_wo_conflict(world, [], reach_list, [])

    def info(self, agent: Vehicle, world: World):
        agent_info: dict = {}
        return agent_info
