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

    env = MultiVehicleEnv(world, scenario.reset_world, scenario.reward, scenario.observation,scenario.info,None,scenario.updata_callback)
    return env

step_t = 1.0
parser = argparse.ArgumentParser(description="GUI for Multi-VehicleEnv")
parser.add_argument('--gui-port',type=str,default='/dev/shm/gui_port')
parser.add_argument('--fps',type=int,default=24)
parser.add_argument('--usegui', action='store_true', default=False)
parser.add_argument('--step-t',type=float,default=step_t)
parser.add_argument('--sim-step',type=int,default=100)
parser.add_argument('--direction_alpha', type=float, default=1.0)
parser.add_argument('--add_direction_encoder',type=str, default='train')

discrete_table = {0:( 0.0, 0.0),
                  1:( 1.0, 0.0), 2:( 1.0, 1.0), 3:( 1.0, -1.0),
                  4:(-1.0, 0.0), 5:(-1.0, 1.0), 6:(-1.0, -1.0)}
args = parser.parse_args()

env = make_env('multi_reach_road_sensor', args)
while True:
    obs = env.reset()
    print('reset env')
    #start_time = time.time()
    for idx in range(10000):
        action = []
        for i in range(len(obs)):
            tx = obs[i][4] - obs[i][0]
            ty = obs[i][5] - obs[i][1]
            theta =  np.arctan2(obs[i][3],obs[i][2])
            c_action = naive_inference(tx,ty,theta,dist= 0.4,min_r = 0.3,delta_theta=step_t * 1.1342)
#            c_action = np.random.rand(2)*2-1
            action.append(c_action)
        obs,reward,done,info = env.step(action)
        env.render()
    #end_time = time.time()
    #print(end_time-start_time)
        
        #time.sleep(0.1)
