
import numpy as np
from plotter import Plotter

M = np.loadtxt("emptyFile_IB_OneWay.dat")

AB = np.array([k[0] for k in M])
BA = np.array([k[1] for k in M])





meanAB = np.mean(AB)
meanBA = np.mean(BA)

stdAB = np.std(AB)
stdBA = np.std(BA)

print("IB, One Way, 1000 runs")
print(meanAB, meanBA)
print(stdAB, stdBA)

# ----------------------------------------------------------------

M = np.loadtxt("emptyFile_ETH_OneWay.dat")

AB = np.array([k[0] for k in M])
BA = np.array([k[1] for k in M])

meanAB = np.mean(AB)
meanBA = np.mean(BA)

stdAB = np.std(AB)
stdBA = np.std(BA)



p = Plotter((12, 7))
x = np.linspace(0, 1, len(AB))
p.createAxis(111)
p.addSubplot(x, AB, size=6, dataLabel=r"A$\rightarrow$ B")
p.addSubplot(x, BA, size=6, dataLabel=r"B$\rightarrow$ A", color="crimson")
p.setProperties("Empty File transference over ethernet", "elapsed time from the beggining of the run (a.u.)", 
            "time (s)", m = 1.2)
p.saveFig("emptyFile_ETH_OneWay.png", dpi = 300)

print("ETH, One Way, 230 runs")
print(meanAB, meanBA)
print(stdAB, stdBA)

# ----------------------------------------------------------------

M = np.loadtxt("emptyFile_IB_Send_Barriers_GoAndBack.dat")

data = np.array([k[3] for k in M])

meandata = np.mean(data)

stddata = np.std(data)
dataB = data

print("IB, Barriers, GonBack, 1000 runs")
print(meandata, stddata)

# ----------------------------------------------------------------

M = np.loadtxt("emptyFile_IB_Ssend_NoBarriers_GoAndBack.dat")

data = np.array([k[3] for k in M])

meandata = np.mean(data)

stddata = np.std(data)

print("IB, NoBarriers, GonBack, 1000 runs")
print(meandata, stddata)


p = Plotter((12, 7))
x = np.linspace(0, 1, len(data))
p.createAxis(111)
p.addSubplot(x, 1e6*np.array(dataB), dataLabel="MPI_Send() + Barriers", dataStyle="-", width=0.7)
p.addSubplot(x, 1e6*np.array(data), dataLabel="MPI_Ssend()", color="crimson", size = 3)
p.setProperties(r"Empty File transference over IB, A$\rightarrow$ B $\rightarrow$ A",
             "elapsed time from the beggining of the run (a.u.)", "time ($\mu s$)",
                 xlim=[0, 1], ylim=[2, 8])
p.saveFig("Send_VS_Ssend.png", dpi=300)