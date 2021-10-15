from ODE import ODE2
# from functions import differential_equations
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite as H

#TODO: writeup, macro's for templates, fix bug in psi5, fix the whole init problem

class QHO(ODE2):
    """
    A class used to analyze the Quantum Harmonic Oscillator for different energies

    Inherits from general ODE class

    ...

    Attributes
    ----------
    function : function
        a function that when multiplied by the state vector gives the SECOND derivative
    INITIAL : numpy.ndarray
        Initial conditions to be analyzed (must be floats!)
    y : numpy.ndarray
        solution as determined by solve method #inherited
    ys : numpy.ndarray
        solutions as determined by solve_sample_space method #inherited
    derivative : function
        derivative of function
    second_derivative : function
        second derivative of function
    times : numpy.ndarray
        independent parameter (=time) range to be analyzed
    y0dot : float
        first derivative of soln evaluated at zero
    delta : float
        time step in times vector (must be constant!)

    Methods
    -------
     Psi5(self, E):
        Find limiting value of the generating function for certain energy

    find_eigen_energy(self, guesses):
        find eigen energy near guesses using secant method to find roots of psi5

    brute_force_eigen_energy(self, left, delta = 0.01):
        brute force solution to find the minimum of abs(psi5) near left. Run
        time in O(1/delta), so pretty horrible. Recommended use to use secant
        method to find a left value close enough to limit run time.

    draw_Psi5(self, Es = np.arange(0,10,0.01)):
        draw psi5 function against energy

    plot_analytical(self):
        Plot numerical solution against analytical solution for the QHO

    solve(self, CHANGE_SOLUTION = True)
        solve the ODE with initial condition specified using Numerov's method #inherited

    find_second_y(self):
        find another value of y to initiate numerovs methods #inherited

    plot(yi, x = None)
        plot solution against x value (default x = time) #inherited

    plot_sample_space(yi, x = None)
        plot sample space solutions against x value (default x = time) #inherited
        NOTE: This is currently useless as Numerov does NOT return derivatives

    plot3(self,xi,yi,zi, title = None, labels = [None, None, None]):
        Plots the found sol'ns in 3D phase space using pyplot #inherited
        NOTE: This is currently useless as Numerov does NOT return derivatives

    """
    def __init__(self, energy, INITIAL = np.array([1,0], dtype= float), times = np.arange(0,5,0.01)):
        """
        Parameters
        ----------
        energy : float
            energy of QHO
        INITIAL : numpy.ndarray
            Initial conditions (i.e. [psi(0),psi'(0)])
        times : numpy.ndarray
            independent parameter (=time) range to be analyzed
        """
        #Define potential function for QHO
        self.energy = energy
        V = lambda x: float(x**2 - energy)
        Vprime = lambda x: float(2*x)
        Vdoubleprime = lambda x: float(2)

        #all variables are handled by ODE2
        super().__init__(V, Vprime, Vdoubleprime, INITIAL, times)

    def Psi5(self, E):
        """Find limiting value of the generating function for certain energy

        Parameters
        ----------
        E : float
            energy of QHO to be analyzed with
        """

        #solve with different potential function
        self.function = lambda x: float(x**2 - E)
        temp_soln = self.solve(CHANGE_SOLUTION = False)
        #self.function = lambda x: float(x**2 - self.energy) Okay so this ensures you can solve the function again properly but slows psi5 down and thats bad as it get called often
        return temp_soln[-1][0]

    def find_eigen_energy(self, guesses):
        """find eigen energy near guesses using secant method to find roots of psi5

        Parameters
        ----------
        guesses : list
            list of initial guesses to be used
        """

        #implement secant method
        while abs(guesses[0]-guesses[1])>1e-6:
            temp = (guesses[0]*self.Psi5(guesses[1]) - guesses[1]*self.Psi5(guesses[0]))/(self.Psi5(guesses[1]) - self.Psi5(guesses[0]))
            guesses[0] = guesses[1]
            guesses[1] = temp
        return guesses[1]

    def brute_force_eigen_energy(self, left, delta = 0.01):
        """brute force solution to find the minimum of abs(psi5) near left. Run
        time in O(1/delta), so pretty horrible. Recommended use to use secant
        method to find a left value close enough to limit run time.


        Parameters
        ----------
        left : float
            Initial guess for the mimimum
        delta : float
            double the unvertainty on found mimimum
        """
        #initiate right and psis
        right = left+delta
        psileft = abs(self.Psi5(left))
        psiright = abs(self.Psi5(right))

        #Check what direction to step
        if psileft < psiright :
            direction = -1
        elif psileft > psiright:
            direction = 1
        else:
            return 0.5*(left+right)

        #walk direction untill direction change is detected
        while direction*(psileft-psiright) > 0:
            left += direction*delta
            right += direction*delta
            psiright = abs(self.Psi5(right))
            psileft = abs(self.Psi5(left))

        return 0.5*(left+right)


    def draw_Psi5(self, Es = np.arange(0,10,0.01)):
        """draw psi5 function against energy

        Parameters
        ----------
        Es : iterable
            list of energies to be analyzed over
        """

        #initialize list of y values
        ys = []

        #find all corresponding psis
        for E in Es:
            ys.append(abs(self.Psi5(E)))

        #plot E against Psi
        plt.plot(Es,ys)
        plt.xlabel("Energy")
        plt.ylabel("limit of Psi")
        plt.show()

    def plot_analytical(self, x_label = "x / m", y_label = "Psi", title = "Analytical and Numerical solution to QHO"):
        """Plot numerical solution against analytical solution for the QHO
        """
        #find corresponding quantum number
        n = int(round((self.energy-1)/2))

        #solutions
        ys_analytical = H(n, self.times)*np.exp(-0.5* self.times**2)
        ys_numerical = np.transpose(self.y)[0]

        #normalize analytical solution
        if abs(ys_numerical[0]) > 1e-2:
            ys_analytical =  ys_analytical*ys_numerical[0]/ys_analytical[0]
        elif abs(ys_numerical[100]) > 1e-2:
            ys_analytical =  ys_analytical*ys_numerical[100]/ys_analytical[100]
        else:
            print("Everything is too close to zero to normalize properly")

        #plot everything
        plt.plot(self.times, ys_analytical)
        plt.plot(self.times, ys_numerical)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()
