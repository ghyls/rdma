import numpy as np

from plotter import Plotter


p = Plotter((12,7))

ETH_110 = [k[1] for k in np.loadtxt("ETH_110.dat")][1:]
noUCX_001 = [k[1] for k in np.loadtxt("noUCX_001.dat")][1:]
noUCX_110 = [k[1] for k in np.loadtxt("noUCX_110.dat")][1:]
UCX_001 =   [k[1] for k in np.loadtxt("UCX_001.dat")][1:]
UCX_101 =   [k[1] for k in np.loadtxt("UCX_101.dat")][1:]
UCX_110 =   [k[1] for k in np.loadtxt("UCX_110.dat")][1:]
Memcpy =   [k[1] for k in np.loadtxt("Memcpy.dat")][1:]

pSize = [k[0] for k in np.loadtxt("noUCX_001.dat")][1:]

p.createAxis(111)

p.addSubplot(pSize, ETH_110, dataLabel=r"ETH: H$\rightarrow$H", dataStyle=".-", width=2, size=8)
p.addSubplot(pSize, noUCX_001, dataLabel=r"noUCX: H$\rightarrow$H$\rightarrow$D", dataStyle=".-", width=2, size=8)
p.addSubplot(pSize, noUCX_110, dataLabel=r"noUCX: H$\rightarrow$H", dataStyle=".-", width=2, size=8)
p.addSubplot(pSize, UCX_001, dataLabel=r"UCX: H$\rightarrow$H$\rightarrow$D", dataStyle=".-", width=2, size=8)
p.addSubplot(pSize, UCX_101, dataLabel=r"UCX: H$\rightarrow$D", dataStyle=".-", width=2, size=8)
p.addSubplot(pSize, UCX_110, dataLabel=r"UCX: H$\rightarrow$H", dataStyle=".-", width=2, size=8)
p.addSubplot(pSize, Memcpy, dataLabel=r"cudaMemcpy", dataStyle=".-", width=2, size=8)

p.setTitles("OneStep HtoH HtoDev", "size (MB)", "time (s)")
p.setProperties(grid=True, doXlog=1, doYlog=1)

p.saveFig("throughput.png", dpi=200)