import time
import pickle
import pyglet
import os
import threading
from . import rendering 


class GUI(object):
    def __init__(self,dir:str = '/dev/shm/gui_port' , fps:float = 24):
        self.field_range = None
        self.cam_bound = None
        self.viewer = None
        self.file_handle = None

        self.gui_port = dir
        self.fps = fps


    def _read_data(self):
        while(True):
            if os.path.exists(self.gui_port):
                self.file_handle = open(self.gui_port, "rb")
                self.file_handle.seek(0)
                gui_data = pickle.load(self.file_handle)
                self.file_handle.close
                return gui_data
            else:
                print('%s does not exist, wait for Simulator dump GUI data'%self.gui_port)
                time.sleep(1)

    def init_viewer(self):
        gui_data = self._read_data()
        self.field_range = gui_data['field_range']
        if self.cam_bound is None:
            center_x = (self.field_range[0]+self.field_range[2])/2.0
            center_y = (self.field_range[1]+self.field_range[3])/2.0
            length_x = (self.field_range[2]-self.field_range[0])
            length_y = (self.field_range[3]-self.field_range[1])
            # 1.4 times of field range for camera range
            gap = 0.1
            self.cam_bound = [center_x - (0.5+gap)*length_x, (1+gap*2)*length_x, center_y - (0.5+gap)*length_y, (1+gap*2)*length_y ]

        screen = pyglet.canvas.get_display().get_default_screen()
        max_width = int(screen.width * 0.9) 
        max_height = int(screen.height * 0.9)
        
        if self.cam_bound[1]/self.cam_bound[3]>max_width/max_height:
            screen_width = max_width
            screen_height  = max_width/(self.cam_bound[1]/self.cam_bound[3])
        else:
            screen_height = max_height
            screen_width  = max_height*(self.cam_bound[1]/self.cam_bound[3])

        self.viewer = rendering.Viewer(int(screen_width),int(screen_height))
        self.viewer.set_bounds(self.cam_bound[0],
                               self.cam_bound[0]+self.cam_bound[1],
                               self.cam_bound[2],
                               self.cam_bound[2]+self.cam_bound[3])
    
    def init_object(self):
        self.viewer.geoms = []
        gui_data = self._read_data()
        
        self.vehicle_list = gui_data['vehicle_list']
        self.landmark_list = gui_data['landmark_list']
        self.obstacle_list = gui_data['obstacle_list']

        self.obstacle_geom_list = []
        for obstacle in self.obstacle_list:
            obstacle_geom = {}
            total_xform = rendering.Transform()
            obstacle_geom['total_xform'] = total_xform
            geom = rendering.make_circle(radius=obstacle.radius, filled=True)
            geom.set_color(*obstacle.color[0],alpha = obstacle.color[1])
            xform = rendering.Transform()
            geom.add_attr(xform)
            geom.add_attr(total_xform)
            obstacle_geom['base']=(geom,xform)
            self.obstacle_geom_list.append(obstacle_geom)
        for obstacle_geom in self.obstacle_geom_list:
            self.viewer.add_geom(obstacle_geom['base'][0])

        self.landmark_geom_list = []
        for landmark in self.landmark_list:
            landmark_geom = {}
            total_xform = rendering.Transform()
            landmark_geom['total_xform'] = total_xform
            geom = rendering.make_circle(radius=landmark.radius, filled=True)
            geom.set_color(*landmark.color[0],alpha = landmark.color[1])
            xform = rendering.Transform()
            geom.add_attr(xform)
            geom.add_attr(total_xform)
            landmark_geom['base']=(geom,xform)
            self.landmark_geom_list.append(landmark_geom)
        for landmark_geom in self.landmark_geom_list:
            self.viewer.add_geom(landmark_geom['base'][0])

        self.vehicle_geom_list = []
        for vehicle in self.vehicle_list:
            vehicle_geom = {}
            total_xform = rendering.Transform()
            vehicle_geom['total_xform'] = total_xform
            
            half_l = vehicle.L_car/2.0
            half_w = vehicle.W_car/2.0
            geom = rendering.make_polygon([[half_l,half_w],[-half_l,half_w],[-half_l,-half_w],[half_l,-half_w]])
            geom.set_color(*vehicle.color[0],alpha = vehicle.color[1])
            xform = rendering.Transform()
            geom.add_attr(xform)
            geom.add_attr(total_xform)
            vehicle_geom['base']=(geom,xform)

            geom = rendering.make_line((0,0),(half_l,0))
            geom.set_color(1.0,0.0,0.0,alpha = 1)
            xform = rendering.Transform()
            geom.add_attr(xform)
            geom.add_attr(total_xform)
            vehicle_geom['front_line']=(geom,xform)
            
            geom = rendering.make_line((0,0),(-half_l,0))
            geom.set_color(0.0,0.0,0.0,alpha = 1)
            xform = rendering.Transform()
            geom.add_attr(xform)
            geom.add_attr(total_xform)
            vehicle_geom['back_line']=(geom,xform)

            self.vehicle_geom_list.append(vehicle_geom)

        for vehicle_geom in self.vehicle_geom_list:
            self.viewer.add_geom(vehicle_geom['base'][0])
            self.viewer.add_geom(vehicle_geom['front_line'][0])
            self.viewer.add_geom(vehicle_geom['back_line'][0])

    def _render(self):
        self.init_viewer()
        self.init_object()

        while True:
            gui_data = self._read_data()
            self.vehicle_list = gui_data['vehicle_list']
            self.landmark_list = gui_data['landmark_list']
            self.obstacle_list = gui_data['obstacle_list']
            self.total_time = gui_data['total_time']
            self.info = str(gui_data['info'])
            for obstacle, obstacle_geom in zip(self.obstacle_list, self.obstacle_geom_list):
                obstacle_geom['total_xform'].set_translation(obstacle.state.coordinate[0],obstacle.state.coordinate[1])
            for landmark, landmark_geom in zip(self.landmark_list, self.landmark_geom_list):
                landmark_geom['total_xform'].set_translation(landmark.state.coordinate[0],landmark.state.coordinate[1])
            for vehicle, vehicle_geom in zip(self.vehicle_list,self.vehicle_geom_list):
                vehicle_geom['front_line'][1].set_rotation(vehicle.state.phi)
                vehicle_geom['total_xform'].set_rotation(vehicle.state.theta)
                vehicle_geom['total_xform'].set_translation(vehicle.state.coordinate[0],vehicle.state.coordinate[1])
            mode = 'human'
            if self.viewer.closed:
                self.init_viewer()
                self.init_object()
            self.viewer.render(time = '%.1f'%(self.total_time),info = self.info,return_rgb_array = mode=='rgb_array')
            time.sleep(1.0/self.fps)
    
    def spin(self):
        t= threading.Thread(target=self._render)
        t.setDaemon(True)
        t.start()
        #t.join()