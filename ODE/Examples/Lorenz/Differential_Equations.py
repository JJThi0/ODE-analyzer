import numpy as np

class d_functions(object):
    """docstring for d_functions."""

    def __init__(self):
        pass

    @staticmethod
    def simple_harmonic_motion():
        def simple_harmonic_motion_function(input):
            return np.dot(np.array([[0,1],[-1,0]]),input)
        return simple_harmonic_motion_function

    @staticmethod
    def damped_harmonic_motion(gamma, omega_0):
        def damped_harmonic_motion_function(input):
            return np.dot(np.array([[0,1],[-gamma,-omega_0**2]]), input)
        return damped_harmonic_motion_function

    @staticmethod
    def lorenz(a, b, r):
        def lorenz_function(input):
            output = np.array([0,0,0])
            output[0] = a*(input[1]-input[0])
            output[1] = r*input[0] - input[1] -input[0]*input[2]
            output[2] = input[0]*input[1] - b*input[2]
            return output
        return lorenz_function

class Interval_Generators():
    """docstring for Interval_Generators."""

    def __init__(self):
        pass

    @staticmethod
    def linear_sampling(size, step):
        def linear_sampling_function(INITIAL, yi):
            return np.arange(INITIAL[yi]-size,INITIAL[yi]+size,step)
        return linear_sampling_function

    @staticmethod
    def gaussian_sampling(sigma):
        def gaussian_sampling_function(INITIAL,yi):
            return np.array([INITIAL[yi]-2*sigma,INITIAL[yi]-sigma,INITIAL[yi],INITIAL[yi]+sigma,INITIAL[yi]+2*sigma])
        return gaussian_sampling_function
