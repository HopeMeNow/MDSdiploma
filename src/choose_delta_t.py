import math
import numpy as np
from data_generation import function_next_value, derivative_next_value, generate_noise

def hit_boundary_probability(x_init: float = 0, h: float = 0.1, f_t: float = 0, alpha: float = 1):
    D = h
    x_0 = x_init
    # print('x_0', x_0)
    x_h = function_next_value(x_0, delta=h, f_t=f_t, alpha=alpha)
    # print('x_h', x_h)
    x_1 = x_0 + h*derivative_next_value(x_0, f_t, alpha)
    # x_1 = x_h
    # print('x_1', x_1)
    x_b = x_h - math.pi
    # print('x_b', x_b)
    F_b = np.sin(x_b)
    # print('F_b', F_b)
    F_b_deriv = np.cos(x_b)
    # print('F_b_deriv', F_b_deriv)
    F_0 = np.sin(x_0)
    # print('F_0', F_0)
    F_h = np.sin(x_h)
    # print('F_h', F_h)

    part1 = -F_b_deriv/(2*D*(np.exp(2*h*F_b_deriv) - 1))
    print('part1', part1)
    part2 = math.pow((x_h - x_b + (x_0 - x_b)*np.exp(h*F_b_deriv) - (F_b/F_b_deriv)), 2)
    print('part2', part2)
    part3 = (1/(4*D*h))*math.pow((x_1 - (x_0 + (h*(F_0 + F_h)/2))), 2)
    print('part3', part3)
    print('part1*part2', part1*part2)

    return np.exp(part1*part2 + part3)


f = generate_noise(1)
print(f)
print(hit_boundary_probability(0, h = 1, f_t=f[0]))