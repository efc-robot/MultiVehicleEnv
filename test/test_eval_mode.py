import random
import numpy as np
import argparse
from MultiVehicleEnv.GUI import GUI
from MultiVehicleEnv.evaluate import EvaluateWrap


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
parser.add_argument('--fps',type=int,default=24)
parser.add_argument('--usegui', action='store_true', default=False)
parser.add_argument('--step-t',type=float,default=1.0)
parser.add_argument('--sim-step',type=int,default=100)
parser.add_argument('--direction_alpha', type=float, default=1.0)
parser.add_argument('--add_direction_encoder',type=str, default='keyboard')


args = parser.parse_args()
def policy(obs:np.ndarray)->int:
    return random.randint(0,6)
env = make_env('3p2t2f', args)
policy_list = [policy,policy,policy]
wrap = EvaluateWrap(env,policy_list)
GUI_instance = GUI(dir = args.gui_port, fps = args.fps)
GUI_instance.spin()

wrap.main()
