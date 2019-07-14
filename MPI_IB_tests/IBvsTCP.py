

from plotter import Plotter
import numpy as np


rawTimeTCP = [[]]  # [each size][num. run]
rawTimeIB = [[]]  # [each size][num. run]

numSizes = len(np.loadtxt("./benchmarkResults/dataIB_0.dat"))

sizePackage = [k[1] for k in np.loadtxt("./benchmarkResults/dataIB_0.dat")]
timeIB = []
sigmaTimeIB = []
timeOverIB = []
sigmaTimeOverIB = []
timeTCP = []
sigmaTimeTCP = []
timeFtF = []        # Felk to Felk
sigmaTimeFtF = []

IB = np.loadtxt("./benchmarkResults/dataIB_" + str(0) + ".dat")
overIB = np.loadtxt("./benchmarkResults/overIB_" + str(0) + ".dat")

print(IB[-1])
print(overIB[-1])

for i in range(numSizes): # fill time arrays and their std

    timeEachRun_IB = []
    timeEachRun_TCP = []
    timeEachRun_FtF = []
    timeEachRun_overIB = []
    for j in range(5):

        IB = np.loadtxt("./benchmarkResults/dataIB_" + str(j) + ".dat")
        TCP = np.loadtxt("./benchmarkResults/dataTCP_" + str(j) + ".dat")
        FtF = np.loadtxt("./benchmarkResults/dataFelkToFelk_" + str(j) + ".dat")
        overIB = np.loadtxt("./benchmarkResults/overIB_" + str(j) + ".dat")

        timeEachRun_IB.append(IB[i][2])
        #timeEachRun_TCP.append(TCP[i][2])
        timeEachRun_FtF.append(FtF[i][2])
        timeEachRun_overIB.append(overIB[i][2])
    


    timeIB.append(np.mean(timeEachRun_IB))
    sigmaTimeIB.append(np.std(timeEachRun_IB))

    timeOverIB.append(np.mean(timeEachRun_overIB))
    sigmaTimeOverIB.append(np.std(timeEachRun_overIB))

    timeTCP.append(np.mean(timeEachRun_TCP))
    #sigmaTimeTCP.append(np.std(timeEachRun_TCP))

    timeFtF.append(np.mean(timeEachRun_FtF))
    sigmaTimeFtF.append(np.std(timeEachRun_FtF))

p = Plotter((10, 7))

p.createAxis(111)
#p.addSubplot(sizePackage, timeTCP, dataLabel="TCP over Eth", dataStyle=".-", width=0.5, size=5)
#p.drawErrorBars(sigmaTimeTCP, 0, 0)

p.addSubplot(sizePackage, timeOverIB, dataLabel="openib over IB", dataStyle=".-", width=0.5, size=5, color="green")
p.drawErrorBars(sigmaTimeOverIB, 0, 0)

p.addSubplot(sizePackage, timeIB, dataLabel="UCX over IB", dataStyle=".-", width=0.5, size=5, color="crimson")
p.drawErrorBars(sigmaTimeIB, 0, 0)

p.addSubplot(sizePackage, timeFtF, dataLabel="Felk to Felk, UCX", dataStyle=".-", width=0.5, size=5, color="orange")
p.drawErrorBars(sigmaTimeFtF, 0, 0)

p.setProperties("IB vs TCP performance", "size of the package (Mb)", 
                "transference speed (MB/s)")

p.saveFig("IBvsTCP.png")
p.showGraph()
