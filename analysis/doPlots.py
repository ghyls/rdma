import numpy as np

from plotter import Plotter


p = Plotter((12,7))

# Sample names
sample = "fullRange/async"


# input and output folders
outputFolder = "./images/"
dataFolder = "./data/" + sample + "/"



UCX_001_N =   [k[1] * 1e6 for k in np.loadtxt(dataFolder + "UCX_001_N.dat")][1:]
UCX_101_N =   [k[1] * 1e6 for k in np.loadtxt(dataFolder + "UCX_101_N.dat")][1:]
UCX_110_N =   [k[1] * 1e6 for k in np.loadtxt(dataFolder + "UCX_110_N.dat")][1:]


# every transfer method, host to host
#aSync =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/async/UCX_101_N.dat")][1:]
#NoUCXoverNoIB =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/NoUCXoverNoIB/UCX_101_N.dat")][1:]
#ob1overTCP =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/ob1overTCP/UCX_101_N.dat")][1:]
#UCXoverIB =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/UCXoverIB/UCX_101_N.dat")][1:]
#UCXoverTCP =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/UCXoverTCP/UCX_101_N.dat")][1:]


#pSize = [k[0] for k in np.loadtxt(dataFolder + "UCX_001_N.dat")][1:]
pSize = [k[0] for k in np.loadtxt("./data/fullRange/UCXoverIB/UCX_110_N.dat")][1:]

p.createAxis(111)

width=2
size=8

#p.addSubplot(pSize, aSync, dataLabel=r"aSync: H$\rightarrow$D", dataStyle=".-", width=width, size=size)
#p.addSubplot(pSize, NoUCXoverNoIB, dataLabel=r"NoUCXoverNoIB: H$\rightarrow$D", dataStyle=".-", width=width, size=size)
#p.addSubplot(pSize, ob1overTCP, dataLabel=r"ob1overTCP: H$\rightarrow$D", dataStyle=".-", width=width, size=size)
#p.addSubplot(pSize, UCXoverIB, dataLabel=r"UCXoverIB: H$\rightarrow$D", dataStyle=".-", width=width, size=size)
#p.addSubplot(pSize, UCXoverTCP, dataLabel=r"UCXoverTCP: H$\rightarrow$D", dataStyle=".-", width=width, size=size)



p.addSubplot(pSize, UCX_001_N, dataLabel=r"UCX_N: H$\rightarrow$H$\rightarrow$D", dataStyle=".-", width=width, size=size)
p.addSubplot(pSize, UCX_101_N, dataLabel=r"UCX_N: H$\rightarrow$D", dataStyle=".-", width=width, size=size)
p.addSubplot(pSize, UCX_110_N, dataLabel=r"UCX_N: H$\rightarrow$H", dataStyle=".-", width=width, size=size)




#p.drawVerticalLine(4.2e4 * 4. / 1048576)
#p.drawVerticalLine(2e6 * 4. / 1048576)

#p.addSubplot(pSize, Memcpy, dataLabel=r"cudaMemcpy", dataStyle=".-", width=2, size=8)

p.setTitles("MPI_Send()", "size (MB)", r"time ($\mu$s)")

# save the figure
p.setProperties(grid=1, doXlog=1, doYlog=1, doWhite=0)
p.saveFig(outputFolder + sample + ".png", dpi=200, transparent=0)
p.showGraph()

p.setProperties(doWhite=1)
p.saveFig(outputFolder + sample + "_v2.png", dpi=200, transparent=1)

