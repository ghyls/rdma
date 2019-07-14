

from plotter import Plotter
import numpy as np



def GenerateData(dataFile):

    #rawTimeIB = [[]]  # [each size][num. run]

    numSizes = len(np.loadtxt("./benchmarkResults/" + dataFile + "_0.dat"))

    sizePackage = [k[0] for k in np.loadtxt("./benchmarkResults/" + dataFile + "_0.dat")]
    t_AB = []
    sigma_t_AB = []
    t_BA = []
    sigma_t_BA = []
    t_ABA = []
    sigma_t_ABA = []

    for i in range(numSizes): # fill time arrays and their std

        timeEachRun_AB = []
        timeEachRun_BA = []
        timeEachRun_ABA = []

        for j in range(5):

            data = np.loadtxt("./benchmarkResults/" + dataFile + "_" + str(j) + ".dat")

            timeEachRun_AB.append(data[i][1])
            timeEachRun_BA.append(data[i][2])
            timeEachRun_ABA.append(data[i][3])



        t_AB.append(np.mean(timeEachRun_AB))
        sigma_t_AB.append(np.std(timeEachRun_AB))

        t_BA.append(np.mean(timeEachRun_BA))
        sigma_t_BA.append(np.std(timeEachRun_BA))

        t_ABA.append(np.mean(timeEachRun_ABA))
        sigma_t_ABA.append(np.std(timeEachRun_ABA))

    return sizePackage, t_AB, t_BA, t_ABA, sigma_t_AB, sigma_t_BA, sigma_t_ABA



sizePackage_IB, t_AB_IB, t_BA_IB, t_ABA_IB, sigma_t_AB_IB, sigma_t_BA_IB, sigma_t_ABA_IB = GenerateData("latency")
sizePackage_FF, t_AB_FF, t_BA_FF, t_ABA_FF, sigma_t_AB_FF, sigma_t_BA_FF, sigma_t_ABA_FF = GenerateData("latencyFtF")

p = Plotter((12, 7))

p.createAxis(111)

p.addSubplot(sizePackage_IB, t_AB_IB, dataLabel=r"Different machine: A $\rightarrow$ B", dataStyle=".-", width=0.5, size=5, color="darkred")
p.drawErrorBars(sigma_t_AB_IB, 0, 0, color="darkred")

p.addSubplot(sizePackage_IB, np.array(t_BA_IB), dataLabel=r"Different machine: B $\rightarrow$ A", dataStyle=".-", width=0.5, size=5, color="red")
p.drawErrorBars(sigma_t_BA_IB, 0, 0, color="red")

p.addSubplot(sizePackage_IB, t_ABA_IB, dataLabel=r"Different machine: A $\rightarrow$ B $\rightarrow$ A", dataStyle=".-", width=0.5, size=5, color="orange")
p.drawErrorBars(sigma_t_ABA_IB, 0, 0, color="orange")

p.addSubplot(sizePackage_FF, t_AB_FF, dataLabel=r"Same machine: A $\rightarrow$ B", dataStyle=".-", width=0.5, size=5, color="midnightblue")
p.drawErrorBars(sigma_t_AB_FF, 0, 0, color="midnightblue")

p.addSubplot(sizePackage_FF, np.array(t_BA_FF), dataLabel=r"Same machine: B $\rightarrow$ A", dataStyle=".-", width=0.5, size=5, color="deepskyblue")
p.drawErrorBars(sigma_t_BA_FF, 0, 0, color="deepskyblue")

p.addSubplot(sizePackage_FF, t_ABA_FF, dataLabel=r"Same machine: A $\rightarrow$ B $\rightarrow$ A", dataStyle=".-", width=0.5, size=5, color="seagreen")
p.drawErrorBars(sigma_t_ABA_FF, 0, 0, color="seagreen")


p.setProperties("transfer rates using UCX over Infiniband / shared memory", "size of the package (MB)", 
                "transference time (s)", xlim=[0.000008000, 0.829527974], ylim=[1.6e-7, 0.002])
p.saveFig("transferRates.png")

p.setProperties("transfer rates using UCX over Infiniband / shared memory", "size of the package (MB)", 
                "transference time (s)", doXlog=1, doYlog=1)
p.setProperties("transfer rates using UCX over Infiniband / shared memory", "size of the package (MB)", 
                "transference time (s)", doXlog=1, doYlog=1)
p.saveFig("transferRates_log.png")
#p.showGraph()

# -----------------------------------

p2 = Plotter((12, 7))

p2.createAxis(111)

latencyArray_IB = np.array(t_ABA_IB)-(np.array(t_AB_IB)+np.array(t_BA_IB))
latencyArray_FF = np.array(t_ABA_FF)-(np.array(t_AB_FF)+np.array(t_BA_FF))

p2.addSubplot(sizePackage_IB, latencyArray_IB*1e6, dataLabel=r"Different machines: (A $\rightarrow$ B$\rightarrow$A) - ([A$\rightarrow$B] + [B$\rightarrow$A])", dataStyle=".-", width=0.5, size=5, color="crimson")
p2.drawErrorBars(np.array(sigma_t_AB_IB)*1e6, 0, 0, color="crimson")

p2.addSubplot(sizePackage_FF, latencyArray_FF*1e6, dataLabel=r"Same machine: (A $\rightarrow$ B$\rightarrow$A) - ([A$\rightarrow$B] + [B$\rightarrow$A])", dataStyle=".-", width=0.5, size=5, color="deepskyblue")
p2.drawErrorBars(np.array(sigma_t_AB_FF)*1e6, 0, 0, color="deepskyblue")

p2.setProperties("Mean latency using UCX over Infiniband / shared memory", "size of the package (Mb)", 
                r"Latency ($\mu$s)", doXlog=1, doYlog=1)

p2.saveFig("latency.png")

#p2.showGraph()
