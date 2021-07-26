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

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = np.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    return r, g, b