import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from functions import differential_equations, Interval_Generators
from ODE import ODE1

INITIAL = np.array([1,2,-0.1,3], dtype = float)
TIMES = np.arange(0,50,0.01)
dt=0.01

eqn = ODE1(differential_equations.double_pendulum(1,1),INITIAL, TIMES )
eqn.solve()

solns = np.transpose(eqn.y)

x1 = np.sin(solns[0])
y1 = -np.cos(solns[0])

x2 = np.sin(solns[0]) + np.sin(solns[1])
y2 = -np.cos(solns[0]) - np.cos(solns[1])

import matplotlib.animation as animation


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text


ani = animation.FuncAnimation(fig, animate, range(1, len(y1)),
                              interval=dt*1000, blit=True, init_func=init)
plt.show()
