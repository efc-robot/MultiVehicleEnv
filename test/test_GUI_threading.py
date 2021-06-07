from MultiVehicleEnv.GUI import GUI
import argparse
import time
import threading

parser = argparse.ArgumentParser(description="GUI for Multi-VehicleEnv")
parser.add_argument('--gui-port',type=str,default='/dev/shm/gui_port')
parser.add_argument('--fps',type=int,default=24)
args = parser.parse_args()

GUI_instance = GUI(port_type = 'file',gui_port = '/dev/shm/gui_port' , fps = 24)
GUI_t = threading.Thread(target=GUI_instance._render_target())
GUI_t.setDaemon(True)
GUI_t.start()
GUI_t.join()