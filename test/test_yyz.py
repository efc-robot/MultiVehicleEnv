import random
import numpy as np
import argparse
from MultiVehicleEnv.evaluate import EvaluateWrap
import time
from MultiVehicleEnv.utils import naive_inference

def make_env(scenario_name, args):
    from MultiVehicleEnv.environment import MultiVehicleEnv
    import MultiVehicleEnv.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment

    env = MultiVehicleEnv(world, scenario.reset_world, scenario.reward, scenario.observation,scenario.info,None,None)
    return env

parser = argparse.ArgumentParser(description="GUI for Multi-VehicleEnv")
parser.add_argument('--gui-port',type=str,default='/dev/shm/gui_port')
parser.add_argument('--fps',type=int,default=24)
parser.add_argument('--usegui', action='store_true', default=False)
parser.add_argument('--step-t',type=float,default=0.1)
parser.add_argument('--sim-step',type=int,default=100)
parser.add_argument('--direction_alpha', type=float, default=1.0)
parser.add_argument('--num_agents', type=int, default=3)
parser.add_argument('--ideal_side_len', type=float, default=5.0)
parser.add_argument('--num_landmarks', type=int, default=1)
parser.add_argument('--num_obstacles', type=int, default=0)

parser.add_argument('--add_direction_encoder',type=str, default='train')

discrete_table = {0:( 0.0, 0.0),
                  1:( 1.0, 0.0), 2:( 1.0, 1.0), 3:( 1.0, -1.0),
                  4:(-1.0, 0.0), 5:(-1.0, 1.0), 6:(-1.0, -1.0)}
args = parser.parse_args()

env = make_env('yyz2', args)
while True:
    obs = env.reset()
    print('reset env')
    for idx in range(100):
        action = []
        for i in range(len(obs)):

            action.append(random.randint(0,6))
        obs,reward,done,info = env.step(action)
        env.render()
        #time.sleep(0.1)

