import time
import random
import argparse
from MultiVehicleEnv.utils import naive_inference
import numpy as np

def make_env(scenario_name, args):
    from MultiVehicleEnv.environment import MultiVehicleEnv
    import MultiVehicleEnv.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment

    env = MultiVehicleEnv(world, scenario.reset_world, scenario.reward, scenario.observation,scenario.info)
    return env

parser = argparse.ArgumentParser(description="GUI for Multi-VehicleEnv")
parser.add_argument('--gui-port',type=str,default='/dev/shm/gui_port')
parser.add_argument('--usegui', action='store_true', default=False)
parser.add_argument('--step-t',type=float,default=1.0)
parser.add_argument('--sim-step',type=int,default=100)
parser.add_argument('--direction_alpha', type=float, default=1.0)
parser.add_argument('--add_direction_encoder',type=str, default='train')

discrete_table = {0:( 0.0, 0.0),
                  1:( 1.0, 0.0), 2:( 1.0, 1.0), 3:( 1.0, -1.0),
                  4:(-1.0, 0.0), 5:(-1.0, 1.0), 6:(-1.0, -1.0)}
args = parser.parse_args()

env = make_env('3p1t2f', args)
while True:
    obs = env.reset()
    print('reset env')
    for idx in range(100):
        action = []
        for i in range(3):
            tx = obs[i][16]
            ty = obs[i][17]
            theta =  np.arctan2(obs[i][7],obs[i][6])
            c_action = naive_inference(tx,ty,theta)
            for idx in range(7):
                if c_action[0] == discrete_table[idx][0] and c_action[1] == discrete_table[idx][1]:
                    action.append(idx)
                    break
        obs,reward,done,info = env.step(action)
        #time.sleep(0.5)
    #train RL
