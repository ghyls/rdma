

from plotter import Plotter
import numpy as np


M = np.loadtxt("linearTest.dat")

size = np.array([elem[0] for elem in M])
t_AB = np.array([elem[1] for elem in M])
t_BA = np.array([elem[2] for elem in M])


p = Plotter((12, 6))

p.createAxis(111)

p.addSubplot(size, t_AB-t_BA, size=6)
#p.addSubplot(size, t_BA, size=6, color="crimson")
p.setProperties("test", "size (MB9)", "time")

p.saveFig("test.png", dpi=300)