from MultiVehicleEnv.GUI import GUI
import argparse


parser = argparse.ArgumentParser(description="GUI for Multi-VehicleEnv")
parser.add_argument('--gui-port',type=str,default='/dev/shm/gui_port')
parser.add_argument('--fps',type=int,default=24)
args = parser.parse_args()

GUI_instance = GUI(dir = args.gui_port, fps = args.fps)
GUI_instance.spin()