import numpy as np

class differential_equations(object):
    """Class containing constructors for a number of differential equations in simultaneous
    first order form"""

    def __init__(self):
        pass

    @staticmethod
    def simple_harmonic_motion():
        """1D simple harmonic motion according to y'' = - y """
        def simple_harmonic_motion_function(input):
            return np.dot(np.array([[0,1],[-1,0]]),input).astype(np.float)
        #return the function itself
        return simple_harmonic_motion_function

    @staticmethod
    def damped_harmonic_motion(gamma, omega_0):
        """1D damped harmonic motion according to y'' +gamma * y' + omega_0**2 *y = 0

        Parameters
        ----------
        gamma : float
            dampening factor of oscillator
        omega_0 : float
            Natural frequency of oscillator
        """

        def damped_harmonic_motion_function(input):
            return np.dot(np.array([[0,1],[-gamma,-omega_0**2]]), input).astype(np.float)
        #return the function itself
        return damped_harmonic_motion_function

    @staticmethod
    def lorenz(a, b, r):
        """ODE constructor accoring to Lorenz ODEs

        Parameters
        ----------
        a : float
            (corrected) Prandtl number
        b : float
            physical constant depending on the system
        r : float
            (corrected) Rayleigh number
        """

        def lorenz_function(input):
            output = [0,0,0]
            output[0] = a*(input[1]-input[0])
            output[1] = r*input[0] - input[1] -input[0]*input[2]
            output[2] = input[0]*input[1] - b*input[2]
            return np.array(output, dtype = float)
        #return the function itself
        #BUGFIX: The algorithm assumes output is array with float, not ints!!!
        return lorenz_function

    @staticmethod
    def TDSE(U):
        """Time dependent schrodinger equation in positions representation constructor

        Parameters
        ----------
        U : function
            Potential energy function minus energy (V(X)-E)
        """

        def TDSE_function(input):
            return np.dot(np.array([[0,1],[U(input[0]),0]]),input).astype(np.float)
        return TDSE_function

    def double_pendulum(m,l, g=9.81): #Could be waaaaay more efficient
        def double_pendulum_function(input):
            theta1dot = (6/(m*l**2))*(2*input[2] - 3*np.cos(input[0]-input[1])*input[3])/(16-9*(np.cos(input[0]-input[1])**2))
            theta2dot = (6/(m*l**2))*(8*input[3] - 3*np.cos(input[0]-input[1])*input[2])/(16-9*(np.cos(input[0]-input[1])**2))
            p1dot = (-0.5*m*l**2) *(theta1dot*theta2dot*np.sin(input[0]-input[1]) + 3*g*np.sin(input[0])/l)
            p2dot = (-0.5*m*l**2) *(-theta1dot*theta2dot*np.sin(input[0]-input[1]) + g*np.sin(input[1])/l)
            return np.array([theta1dot, theta2dot, p1dot, p2dot], dtype = float)

        return double_pendulum_function

class Interval_Generators():
    """Class containing all generators for intervals around a certain point"""

    def __init__(self):
        pass

    @staticmethod
    def linear_sampling(size, step):
        """constructs function for sampling linearly around an intial condtions.

        Parameters
        ----------
        size : float
            amplitude of the sampling range
        step : float
            distance between two sampling points
        """
        def linear_sampling_function(INITIAL, yi):
            return np.arange(INITIAL[yi]-size,INITIAL[yi]+size,step, dtype = float)
        #return the function itself
        return linear_sampling_function

    @staticmethod
    def gaussian_sampling(sigma):
        """constructs function for sampling around an intial condtions according
        over the [-2sigma, 2sigma] range.

        Parameters
        ----------
        sigma : float
            standard deviation
        """
        def gaussian_sampling_function(INITIAL,yi):
            return np.array([INITIAL[yi]-2*sigma,INITIAL[yi]-sigma,INITIAL[yi],INITIAL[yi]+sigma,INITIAL[yi]+2*sigma])
        #return the function itself
        return gaussian_sampling_function
