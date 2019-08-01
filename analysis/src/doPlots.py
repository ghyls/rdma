import numpy as np

from plotter import Plotter


p = Plotter((12,7))

dataFolder = "./throughput_step/"
dataFolder = "./step_256KB/"
dataFolder = "./step_syncronous/"
dataFolderTEMP = "./ib_ucx/"

#ETH_110 = [k[1] for k in np.loadtxt("ETH_110.dat")][1:]
#noUCX_001 = [k[1] for k in np.loadtxt("noUCX_001.dat")][1:]
#noUCX_110 = [k[1] for k in np.loadtxt("noUCX_110.dat")][1:]
#UCX_001_Y =   [k[1] for k in np.loadtxt("UCX_001_Y.dat")][1:]
#UCX_101_Y =   [k[1] for k in np.loadtxt("UCX_101_Y.dat")][1:]
#UCX_110_Y =   [k[1] for k in np.loadtxt("UCX_110_Y.dat")][1:]
UCX_001_N =   [k[1] * 1e6 for k in np.loadtxt(dataFolder + "UCX_001_N.dat")][1:]
UCX_101_N =   [k[1] * 1e6 for k in np.loadtxt(dataFolder + "UCX_101_N.dat")][1:]
UCX_110_N =   [k[1] * 1e6 for k in np.loadtxt(dataFolder + "UCX_110_N.dat")][1:]
#Memcpy =   [k[1] for k in np.loadtxt("Memcpy.dat")][1:]

UCX_001_NTEMP =   [k[1] * 1e6 for k in np.loadtxt(dataFolderTEMP + "UCX_001_N.dat")][1:]
UCX_101_NTEMP =   [k[1] * 1e6 for k in np.loadtxt(dataFolderTEMP + "UCX_101_N.dat")][1:]
UCX_110_NTEMP =   [k[1] * 1e6 for k in np.loadtxt(dataFolderTEMP + "UCX_110_N.dat")][1:]


pSize = [k[0] for k in np.loadtxt(dataFolder + "UCX_001_N.dat")][1:]

p.createAxis(111)

#p.addSubplot(pSize, ETH_110, dataLabel=r"ETH: H$\rightarrow$H", dataStyle=".-", width=2, size=8)
#p.addSubplot(pSize, noUCX_001, dataLabel=r"noUCX: H$\rightarrow$H$\rightarrow$D", dataStyle=".-", width=2, size=8)
#p.addSubplot(pSize, noUCX_110, dataLabel=r"noUCX: H$\rightarrow$H", dataStyle=".-", width=2, size=8)
#p.addSubplot(pSize, UCX_001_Y, dataLabel=r"UCX_Y: H$\rightarrow$H$\rightarrow$D", dataStyle=".-", width=2, size=8)
#p.addSubplot(pSize, UCX_101_Y, dataLabel=r"UCX_Y: H$\rightarrow$D", dataStyle=".-", width=2, size=8)
#p.addSubplot(pSize, UCX_110_Y, dataLabel=r"UCX_Y: H$\rightarrow$H", dataStyle=".-", width=2, size=8)
p.addSubplot(pSize, UCX_001_N, dataLabel=r"UCX_N: H$\rightarrow$H$\rightarrow$D", dataStyle=".-", width=1, size=4)
p.addSubplot(pSize, UCX_101_N, dataLabel=r"UCX_N: H$\rightarrow$D", dataStyle=".-", width=1, size=4)
p.addSubplot(pSize, UCX_110_N, dataLabel=r"UCX_N: H$\rightarrow$H", dataStyle=".-", width=1, size=4)

p.addSubplot(pSize, UCX_001_NTEMP, dataLabel=r"UCX_N + IB: H$\rightarrow$H$\rightarrow$D", dataStyle=".-", width=1, size=4)
p.addSubplot(pSize, UCX_101_NTEMP, dataLabel=r"UCX_N + IB: H$\rightarrow$D", dataStyle=".-", width=1, size=4)
p.addSubplot(pSize, UCX_110_NTEMP, dataLabel=r"UCX_N + IB: H$\rightarrow$H", dataStyle=".-", width=1, size=4)

p.drawVerticalLine(4.2e4 * 4. / 1048576)
p.drawVerticalLine(2e6 * 4. / 1048576)

#p.addSubplot(pSize, Memcpy, dataLabel=r"cudaMemcpy", dataStyle=".-", width=2, size=8)

p.setTitles("MPI_Ssend()", "size (MB)", r"time ($\mu$s)")
p.setProperties(grid=True, doXlog=1, doYlog=1, doWhite=0)

p.saveFig(dataFolder + "throughput.png", dpi=200, transparent=0)