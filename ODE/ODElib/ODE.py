import numpy as np
import matplotlib.pyplot as plt
#import sys
from mpl_toolkits import mplot3d

class ODE():
    """
    A class used to define and analyze an ODE/Cauchy Problem with a particular
    focus on chaotic behaviour

    ...

    Attributes
    ----------
    INITIAL : numpy.ndarray
        Initial conditions to be analyzed (must be floats!)
    times : numpy.ndarray
        independent parameter (=time) range to be analyzed
    y : numpy.ndarray
        solution as determined by solve method
    ys : numpy.ndarray
        solutions as determined by solve_sample_space method

    Methods
    -------

    plot(yi, x = None)
        plot solution against x value (default x = time)

    plot_sample_space(yi, x = None)
        plot sample space solutions against x value (default x = time)

    plot3(self,xi,yi,zi, title = None, labels = [None, None, None]):
        Plots the found sol'ns in 3D phase space using pyplot

    """

    def __init__(self, INITIAL, times):
        """
        Parameters
        ----------
        INITIAL : numpy.ndarray
            Initial conditions to be analyzed (must be floats!)
        times : numpy.ndarray
            independent parameter (=time) range to be analyzed
        """

        self.INITIAL = INITIAL
        self.times = times
        self.y = np.zeros((len(times),len(INITIAL))) #I swapped x an y as its easier to acces numpy arrays this way
        self.ys = []


        #Check if all initial conditions are floats
        for i, yi in enumerate(self.INITIAL):
            if not isinstance(yi,float):
                print("Initial condition " + str(i) + " is not a float!")
                #sys.exit(1) NOTE also commented out library


    def plot(self, yi, xi = None, title = None, y_label = None):
        """Plots the found sol'n using pyplot

        Parameters
        ----------
        yi : int
            Choose what input should be plotted on y-axis

        xi : int, optional
            what should be plotted on x-axis (default is time)

        title: str, optional
            choose title of plot (defaults to time evolution or phase space)

        y_label: str, optional
            choose label of y axis (defaults to yi)
        """

        #set up default x
        if xi is None:
            x = self.times
            x_label = "time / sec"
            if title == None:
                title = "time evolution of y" + str(yi)
        #set up for other x
        else:
            x_label = "y" + str(xi)
            x = np.transpose(self.y)[xi]
            if title == None:
                title = "phase portrait of y" + str(xi) + " and " + str(yi)

        #plot y against x
        plt.plot(x, np.transpose(self.y)[yi])
        plt.xlabel(x_label)
        if y_label == None:
            y_label = "y" + str(yi)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

    def plot_sample_space(self, yi, xi = None, title = None, y_label = None):
        """Plots the found sol'ns in sample space using pyplot

        Parameters
        ----------
        yi : int
            Choose what input should be plotted on y-axis

        xi : int
            Choose what input should be plotted on y-axis (default is time)

        title: str, optional
            choose title of plot (defaults to time evolution or phase space)

        y_label: str, optional
            choose label of y axis (default is yi)
        """
        #if we want to plot against time
        if xi is None:
            x = self.times
            x_label = "time / sec"
            if title == None:
                title = "time evolution of y" + str(yi) + " (sampled around input)"
            #plot every sol'n
            for y in self.ys:
                plt.plot(x, np.transpose(y)[yi])
        #if we want to plot against another yi (so in phase space)
        else:
            x_label = "y" + str(xi)
            if title == None:
                title = "phase portrait of y" + str(xi) + " and y" + str(yi) +" (sampled around input)"
            #plot every sol'n
            for y in self.ys:
                plt.plot(np.transpose(y)[xi], np.transpose(y)[yi])

        #label and show
        plt.xlabel(x_label)
        if y_label == None:
            y_label = "y" + str(yi)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

    def plot3(self,xi,yi,zi, title = None, labels = [None, None, None]):
        """Plots the found sol'ns in 3D phase space using pyplot

        Parameters
        ----------
        yi : int
            Choose what input should be plotted on y-axis

        xi : int
            Choose what input should be plotted on x-axis

        zi : int
            Choose what input should be plotted on z-axis

        title: str, optional
            choose title of plot (defaults to time evolution or phase space)

        labels: str, optional
            choose label of axes (default is yi's)
        """

        #make figure
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        y = np.transpose(self.y)
        ax.plot3D(y[xi], y[yi], y[zi])

        #check labels
        indeces = (xi,yi,zi)
        for i,label in enumerate(labels):
            if label == None:
                labels[i] = "y" + str(indeces[i])

        #set labels
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])

        #show
        plt.show()


class ODE1(ODE):
    """
    A class used to define and analyze a first order ODE/Cauchy Problem with a particular
    focus on chaotic behaviour

    Inherits from general ODE class

    ...

    Attributes
    ----------
    function : function
        a function that takes in the state vector and returns the derivative vector
    INITIAL : numpy.ndarray
        Initial conditions to be analyzed (must be floats!)
    times : numpy.ndarray
        independent parameter (=time) range to be analyzed
    y : numpy.ndarray
        solution as determined by solve method
    ys : numpy.ndarray
        solutions as determined by solve_sample_space method

    Methods
    -------
    take_step(dt,y0)
        Calculate next value of the conditions given time step and intial condition

    solve(self, y0 = None, CHANGE_SOLUTION = True)
        solve the ODE with initial condition y0

    sample_space(self, yi, sampling_function)
        solve ODE over distribution of yi

    plot(yi, x = None)
        plot solution against x value (default x = time) #inherited

    plot_sample_space(yi, x = None)
        plot sample space solutions against x value (default x = time) #inherited

    plot3(self,xi,yi,zi, title = None, labels = [None, None, None]):
        Plots the found sol'ns in 3D phase space using pyplot #inherited

    """
    def __init__(self, function, INITIAL, times):
        """
        Parameters
        ----------
        function : function
            a function that takes in the state vector and returns the derivative vector
        INITIAL : numpy.ndarray
            Initial conditions to be analyzed (must be floats!) #inherited
        times : numpy.ndarray
            independent parameter (=time) range to be analyzed #inherited
        """
        super().__init__(INITIAL, times)
        self.function = function

        #Check if number of initial conditions matches function
        try:
            self.function(self.INITIAL)
        except ValueError:
            print("Number of initial conditions does not match function!")
            #sys.exit(1) NOTE also commented out library


    def take_step(self,dt, y0):
        """Calculate next value of the conditions given time step and intial
        condition according to Runge-Kutta algorithm

        Parameters
        ----------
        dt : float
            Total length of time step to be considered

        y0 : numpy.ndarray
            Conditions at start of time step (must be floats!)
        """

        #Runge-Kutta4 algorithm
        f0 = self.function(y0)
        y1 = y0 + 0.5*f0*dt
        f1 = self.function(y1)
        y2 = y0 + 0.5*f1*dt
        f2 = self.function(y2)
        y3 = y0 + f2*dt
        f3 = self.function(y3)

        return y0 + dt*(f0+2*f1+2*f2+f3)/6

    def solve(self, y0 = None, CHANGE_SOLUTION = True):
        """Solve Cauchy problem over the entire given range of times

        Parameters
        ----------
        y0 : numpy.ndarray, optional
            Initial condition to be solved (default is INITIAL)

        CHANGE_SOLUTION : bool, optional
            Flag to indicate if self.y (attribute sol'n) should be updated
        """

        #Set initial condition
        if y0 is None:
            y0 = self.INITIAL

        #setup if sol'n should not be changed
        if not CHANGE_SOLUTION:
            previous_y = self.y.copy()

        #Calculate every y using take_step
        #assumes dt is constant
        dt = self.times[1] - self.times[0]
        for i in range(len(self.times)):
            self.y[i] = y0
            y0 = self.take_step(dt, y0)
        self.y[-1] = y0 #finish last step

        #return correct sol'n depending on CHANGE_SOLUTION flag
        if CHANGE_SOLUTION:
            return self.y
        elif not CHANGE_SOLUTION:
            temp_y = self.y.copy()
            self.y = previous_y.copy()
            return temp_y

    def sample_space(self, yi, sampling_function):
        """Solve Cauchy problem sampled over range of inputs

        Parameters
        ----------
        yi : int
            Choose what input should be sampled over

        sampling_function : function
            Function used to determine sampling distribution
        """

        #clear ys
        self.ys = []

        #create range of inputs to be considered
        start_condition = self.INITIAL
        interval = sampling_function(self.INITIAL, yi)

        #solve Cauchy-Problem for every initial condition
        for element in interval:
            start_condition[yi] = element
            temp_y = self.solve(y0 = start_condition, CHANGE_SOLUTION = False)

            #append sol'n to ys (array of sol'ns)
            self.ys.append(temp_y.copy())

        return self.ys

class ODE2(ODE):
    """
    A class used to define and analyze a second order ODE/Cauchy Problem with a particular
    focus on chaotic behaviour

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
    solve(self, CHANGE_SOLUTION = True)
        solve the ODE with initial condition specified using Numerov's method

    find_second_y(self):
        find another value of y to initiate numerovs methods

    plot(yi, x = None)
        plot solution against x value (default x = time) #inherited

    plot_sample_space(yi, x = None)
        plot sample space solutions against x value (default x = time) #inherited
        NOTE: This is currently useless as Numerov does NOT return derivatives

    plot3(self,xi,yi,zi, title = None, labels = [None, None, None]):
        Plots the found sol'ns in 3D phase space using pyplot #inherited
        NOTE: This is currently useless as Numerov does NOT return derivatives

    """

    def __init__(self, function, derivative, second_derivative, INITIAL, times):
        """
        Parameters
        ----------
        function : function
            a function that when multiplied by the state vector gives the SECOND derivative
        derivative : function
            derivative of function
        second_derivative : function
            second derivative of function
        INITIAL : numpy.ndarray
            Initial conditions to be analyzed (must be floats!)
        times : numpy.ndarray
            independent parameter (=time) range to be analyzed
        """
        #intial conditions and times initialized in general ODE
        super().__init__(INITIAL, times)

        #initialize a variables needed
        self.function = function
        self.derivative = derivative
        self.second_derivative = second_derivative
        self.y0dot = self.INITIAL[1]
        self.delta = times[1] - times[0]
        self.y[0][0] = self.INITIAL[0]


    def solve(self, CHANGE_SOLUTION = True):
        """Solve the ODE with initial condition specified using Numerov's method

        Parameters
        ----------
        CHANGE_SOLUTION : bool
            boolean indicating whether self.y should be changed
        """

        #keep record if not to change solution
        if not CHANGE_SOLUTION:
            copy = self.y.copy()

        #Apply Numerov's method using the recursive relation
        self.find_second_y()
        for j in range(len(self.times)-2):
            self.y[j+2][0] = ((2+5*self.delta**2*self.function((j+1)*self.delta)/6)*self.y[j+1][0] - (1-self.delta**2 * self.function(self.delta*j)/12)*self.y[j][0])/(1-self.delta**2*self.function((j+2)*self.delta)/12)

        #if not to change solution, swap found and old and return new
        if not CHANGE_SOLUTION:
            self.y , copy = copy, self.y #i thought this was wrong but it works
            return copy

    def find_second_y(self):
        """find another value of y to initiate Numerovs method
        """

        #cache intermediate values
        f0 = self.function(0)
        f0dot = self.derivative(0)
        f0doubledot = self.second_derivative(0)

        #fourth order taylor expansion around 0
        self.y[1][0] = self.y[0][0] + self.delta*self.y0dot + self.delta**2 * f0 * self.y[0][0]/2 + self.delta**3 * (f0*self.y0dot + f0dot*self.y[0][0])/6 + self.delta**4 * (f0doubledot*self.y[0][0]+2*f0dot*self.y0dot+f0**2*self.y[0][0])/24
