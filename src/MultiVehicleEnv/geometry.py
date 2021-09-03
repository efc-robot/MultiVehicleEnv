import numpy as np

def intersection_line_circle(point, angle, circle_c, circle_r):
    r = circle_r
    moved_p1_coord = point - circle_c
    angle = angle%(2*np.pi)
    if angle == np.pi/2 or angle == -np.pi/2:
        x = moved_p1_coord[0]
        if abs(x) > r:
            intersections = []
        elif abs(x) == r:
            intersections = [np.array([x, 0.0])]
        else:
            temp = (r**2 - x**2)**0.5
            intersections = [np.array([x, temp]), np.array([x, -temp])]
    else:
        k = np.tan(angle)
        b = moved_p1_coord[1] - k * moved_p1_coord[0]
        delta = 4 * (r**2+r**2*k**2-b**2)
        if delta < 0:
            intersections = []
        elif delta == 0:
            x0 = -k*b / (k**2+1)
            y0 = k*x0 + b
            intersections = [np.array([x0,y0])]
        else:
            x0 = -k*b / (k**2+1)
            temp = delta**0.5 / (2*(k**2+1))
            x1 = x0 - temp 
            y1 = k*x1 + b
            x2 = x0 + temp
            y2 = k*x2 + b
            intersections = [np.array([x1, y1]), np.array([x2, y2])]
    return [np.array([p[0]+circle_c[0], p[1]+circle_c[1]]) for p in intersections] 
            
def laser_circle_dist(center, angle, max_range, circle_c, circle_r):
    moved_circle_c = circle_c - center
    intensities = 0.0
    interserction = intersection_line_circle(np.array([0.0,0.0]), angle, moved_circle_c, circle_r)
    if len(interserction) == 0:
        return max_range, intensities
    final_dist = max_range
    for point in interserction:
        vec = np.array([np.cos(angle), np.sin(angle)])
        dist = np.dot(point, vec)
        if (dist > 0) and (dist < final_dist):
            final_dist = dist
            intensities = 1.0
    return final_dist, intensities
