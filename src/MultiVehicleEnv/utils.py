import numpy as np
from typing import *

def coord_dist(coord_a:List[float],coord_b:List[float]) -> float:
    return ((coord_a[0]-coord_b[0])**2 + (coord_a[1]-coord_b[1])**2)**0.5

def naive_inference(target_x:float, target_y:float, theta:float,
                    dist:float=0.0, min_r:float=0.0) -> Tuple[float,float]:
    r = (target_x**2+target_y**2)**0.5
    relate_theta = np.arctan2(target_y,target_x)-theta
    yt = np.sin(relate_theta)*r
    xt = np.cos(relate_theta)*r
    if abs(np.tan(relate_theta)*r) < dist * 0.5:
        vel = np.sign(xt)
        phi = 0
    else:
        in_min_r = (xt**2+(abs(yt)-min_r)**2)< min_r**2
        vel = -1 if (bool(in_min_r) ^ bool(xt<0)) else 1
        phi = -1 if (bool(in_min_r) ^ bool(yt<0)) else 1
    return vel,phi