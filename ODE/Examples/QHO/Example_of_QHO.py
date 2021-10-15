from QHO import QHO
import numpy as np

#Show that QHO can find numeric solution to QHO
for E in [0.95, 1.0, 1.05]:
    eqn = QHO(E, INITIAL = np.array([1,0], dtype= float), times = np.arange(0,5,0.05))
    eqn.solve()
    eqn.plot(0, title = "Solution for E = " + str(E),x_label = "x / m", y_label = "Psi")

#show how it compares to analytical solution
eqn = QHO(5.0, INITIAL = np.array([1,0], dtype= float), times = np.arange(0,6.4,0.05))
eqn.solve()
eqn.plot_analytical()

#shows how it can find eigen energies of QHO
eigen_energy = eqn.find_eigen_energy([33,31])
eigen_energy = eqn.brute_force_eigen_energy(eigen_energy, delta = 0.00001)
print("eigen energy found: ",eigen_energy)
eqn = QHO(eigen_energy)
eqn.solve()
eqn.plot_analytical()

#find eigen energies near 1,3,5
eqn = QHO(1., INITIAL = np.array([1,0], dtype= float), times = np.arange(0,6.4,0.05))

eigen_energy = eqn.brute_force_eigen_energy(1.5, delta = 0.01)
print("start from 1.5: ", eigen_energy)

eigen_energy = eqn.brute_force_eigen_energy(5.2, delta = 0.01)
print("start from 5.2: ", eigen_energy)

eqn = QHO(1., INITIAL = np.array([0,1], dtype= float), times = np.arange(0,6.4,0.05))

eigen_energy = eqn.brute_force_eigen_energy(3.1, delta = 0.01)
print("start from 3.1: ", eigen_energy)
