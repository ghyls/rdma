import numpy as np

from plotter import Plotter


p = Plotter((12,7))

ucxtHH = 1

# Sample names
if ucxtHH: sample = "fullRange/UCX_transports"
else: sample = "fullRange/fullComparison_HtoH"


# input and output folders
outputFolder = "./images/"
if ucxtHH: dataFolder = "./data/fullRange/UCX_transports"
else: dataFolder = "./data/fullRange/"


if ucxtHH:
    UCX_t_all_110 =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/UCX_transports/UCX_t_all_110.dat")][1:]
    UCX_t_none_110 =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/UCX_transports/UCX_t_none_110.dat")][1:]
    UCX_t_rc_110 =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/UCX_transports/UCX_t_rc_110.dat")][1:]
    UCX_t_tcp_110 =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/UCX_transports/UCX_t_tcp_110.dat")][1:]
    UCX_t_ud_110 =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/UCX_transports/UCX_t_ud_110.dat")][1:]
    UCX_t_all_101 =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/UCX_transports/UCX_t_all_101.dat")][1:]
    UCX_t_none_101 =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/UCX_transports/UCX_t_none_101.dat")][1:]
else:
    # every transfer method, host to host
    ob1_noOpenIB =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/ob1_noOpenIB_110.dat")][1:]
    ob1_openIB =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/ob1_openIB_110.dat")][1:]
    ob1_tcp =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/ob1_tcp_110.dat")][1:]
    none_none =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/none_none_110.dat")][1:]
    ucx_none =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/ucx_none_110.dat")][1:]
    ucx_none_101 =   [k[1] * 1e6 for k in np.loadtxt("./data/fullRange/ucx_none_101.dat")][1:]


pSize = [k[0] for k in np.loadtxt("./data/fullRange/none_none_110.dat")][1:]

p.createAxis(111)


width=2; size=8; style = ".-"

if ucxtHH:
    p.addSubplot(pSize, UCX_t_all_110, dataLabel=r"all, H$\rightarrow$H", dataStyle=style, width=width, size=size)
    #p.addSubplot(pSize, UCX_t_none_110, dataLabel=r"( ), H$\rightarrow$H", dataStyle=style, width=width, size=size)
    p.addSubplot(pSize, UCX_t_rc_110, dataLabel=r"rc, H$\rightarrow$H", dataStyle=style, width=width, size=size)
    p.addSubplot(pSize, UCX_t_tcp_110, dataLabel=r"tcp, H$\rightarrow$H", dataStyle=style, width=width, size=size)
    p.addSubplot(pSize, UCX_t_ud_110, dataLabel=r"ud, H$\rightarrow$H", dataStyle=style, width=width, size=size)
    p.addSubplot(pSize, UCX_t_all_101, dataLabel=r"all, H$\rightarrow$D", dataStyle=".:", width=width, size=size)
    #p.addSubplot(pSize, UCX_t_none_101, dataLabel=r"( ), H$\rightarrow$D", dataStyle=".:", width=width, size=size)
else:
    p.addSubplot(pSize, ob1_noOpenIB, dataLabel=r"PML=ob1; BTL=^openIB: H$\rightarrow$H", dataStyle=style, width=width, size=size)
    p.addSubplot(pSize, ob1_tcp, dataLabel=r"PML=ob1; BTL=tcp H$\rightarrow$H", dataStyle=".:", width=width, size=size)
    p.addSubplot(pSize, ob1_openIB, dataLabel=r"PML=ob1; BTL=openIB H$\rightarrow$H", dataStyle=style, width=width, size=size)
    p.addSubplot(pSize, ucx_none_101, dataLabel=r"PML=UCX; BTL=( ) H$\rightarrow$D", dataStyle=style, width=width, size=size)
    p.addSubplot(pSize, none_none, dataLabel=r"PML=( ); BTL=( ) H$\rightarrow$H", dataStyle=style, width=width, size=size)
    p.addSubplot(pSize, ucx_none, dataLabel=r"PML=UCX; BTL=( ) H$\rightarrow$H", dataStyle=".:", width=width, size=size)


#p.drawVerticalLine(4.2e4 * 4. / 1048576)
#p.drawVerticalLine(2e6 * 4. / 1048576)

#p.addSubplot(pSize, Memcpy, dataLabel=r"cudaMemcpy", dataStyle=".-", width=2, size=8)

p.setTitles("MPI_Send()", "size (MB)", r"time ($\mu$s)")

# save the figure
p.setProperties(grid=1, doXlog=1, doYlog=1, doWhite=0)
p.saveFig(outputFolder + sample + ".png", dpi=200, transparent=0)

p.setProperties(doWhite=1)
p.saveFig(outputFolder + sample + "_v2.png", dpi=200, transparent=1)

