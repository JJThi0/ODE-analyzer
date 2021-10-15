import numpy as np
from ODE import ODE1
from functions import differential_equations, Interval_Generators
import matplotlib.pyplot as plt

#TO DO:code more ode's in differential_equations, write test cases

#define some constants for Lorenz ODE
a = 10.0
b = 8.0/3.0
r = 28.0

#define some constants for damped harmonic motion
gamma = 1
omega_0 = 0.6

#initial conditions for functions
INITIAL_LORENZ = np.array([4,5,6], dtype = float)
INITIAL = np.array([4,5], dtype = float)
TIMES = np.arange(0,20,0.01)


def main():
    #demonstrate module using damped harmonic motion
    eqn = ODE1(differential_equations.damped_harmonic_motion(gamma,omega_0), INITIAL, TIMES)
    eqn.solve()
    eqn.plot(0, title = "Numerical Solution for Damped Harmonic Motion", y_label = "displacement / m")
    #calculate and plot over large range of y0 values
    eqn.sample_space(0,Interval_Generators.linear_sampling(3,0.5))
    eqn.plot_sample_space(0, title = "Numerical Solutions for Damped Harmonic Motion", y_label = "displacement / m")

    #demonstrate module using lorenz equation
    eqn = ODE1(differential_equations.lorenz(a,b,5), INITIAL_LORENZ, TIMES)
    eqn.solve()
    eqn.plot(0)

    y = np.transpose(eqn.y)
    eqn.sample_space(0,Interval_Generators.gaussian_sampling(0.2))
    eqn.plot_sample_space(0, title = "Numerical solutions for Lorenz Equation")

    eqn.plot3(0,1,2)


    #calculate and plot over large range of y0 values
    eqn.sample_space(0,Interval_Generators.linear_sampling(3,0.5))
    eqn.plot_sample_space(0)

    #demonstrate phase space functionality
    eqn = ODE1(differential_equations.lorenz(a,b,r), INITIAL_LORENZ, np.arange(0,8,0.01))
    eqn.sample_space(0,Interval_Generators.gaussian_sampling(0.1))
    eqn.plot_sample_space(0)
    eqn.plot_sample_space(0, 1)
    eqn.plot_sample_space(1, 2)
    eqn.plot_sample_space(2, 0)


if __name__ == "__main__":
    main()
