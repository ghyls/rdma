import numpy as np

from plotter import Plotter
from plotter import meanAndStd



def getFilePaths(fileName, numFiles = 5):

    filePaths = []

    for i in range(numFiles):
        if ucxtHH:  filePaths.append("./data/fullRange/" + str(i) + "/UCX_transports/" + fileName)
        else:       filePaths.append("./data/fullRange/" + str(i) + "/" + fileName)
    
    return filePaths


ucxtHH = 0

# Sample names
if ucxtHH: sample = "fullRange/UCX_transports_5"
else: sample = "fullRange/fullComparison_HtoH_5"


# input and output folders
outputFolder = "./images/"
if ucxtHH: dataFolder = "./data/fullRange/UCX_transports"
else: dataFolder = "./data/fullRange/"


if ucxtHH:

    UCX_t_all_110 = meanAndStd("UCX_t_all_110.dat")[0][1:]
    UCX_t_all_110_err = meanAndStd("UCX_t_all_110.dat")[1][1:]

    UCX_t_rc_110 = meanAndStd("UCX_t_rc_110.dat")[0][1:]
    UCX_t_rc_110_err = meanAndStd("UCX_t_rc_110.dat")[1][1:]

    UCX_t_ud_110 = meanAndStd("UCX_t_ud_110.dat")[0][1:]
    UCX_t_ud_110_err = meanAndStd("UCX_t_ud_110.dat")[1][1:]

    UCX_t_tcp_110 = meanAndStd("UCX_t_tcp_110.dat")[0][1:]
    UCX_t_tcp_110_err = meanAndStd("UCX_t_tcp_110.dat")[1][1:]

    UCX_t_all_101 = meanAndStd("UCX_t_all_101.dat")[0][1:]
    UCX_t_all_101_err = meanAndStd("UCX_t_all_101.dat")[1][1:]

else:
    # every transfer method, host to host

    ob1_openIB_110 =   meanAndStd(getFilePaths("ob1_openIB_110.dat"))[0][1:]
    ob1_openIB_err_110 = meanAndStd(getFilePaths("ob1_openIB_110.dat"))[1][1:]

    ob1_tcp_110 =   meanAndStd(getFilePaths("ob1_tcp_110.dat"))[0][1:]
    ob1_tcp_err_110 = meanAndStd(getFilePaths("ob1_tcp_110.dat"))[1][1:]

    none_noOpenIB_101 =   meanAndStd(getFilePaths("none_noOpenIB_101.dat"))[0][1:]
    none_noOpenIB_101_err = meanAndStd(getFilePaths("none_noOpenIB_101.dat"))[1][1:]

    none_noOpenIB_110 =   meanAndStd(getFilePaths("none_noOpenIB_110.dat"))[0][1:]
    none_noOpenIB_110_err = meanAndStd(getFilePaths("none_noOpenIB_110.dat"))[1][1:]

    none_OpenIB_101 =   meanAndStd(getFilePaths("none_OpenIB_101.dat"))[0][1:]
    none_OpenIB_101_err = meanAndStd(getFilePaths("none_OpenIB_101.dat"))[1][1:]

    ucx_none_110 =   meanAndStd(getFilePaths("ucx_none_110.dat"))[0][1:]
    ucx_none_110_err = meanAndStd(getFilePaths("ucx_none_110.dat"))[1][1:]

    ucx_none_101 =   meanAndStd(getFilePaths("ucx_none_101.dat"))[0][1:]
    ucx_none_101_err = meanAndStd(getFilePaths("ucx_none_101.dat"))[1][1:]

    ucx_openIB_110 =   meanAndStd(getFilePaths("ucx_openIB_110.dat"))[0][1:]
    ucx_openIB_110_err = meanAndStd(getFilePaths("ucx_openIB_110.dat"))[1][1:]

    ucx_openIB_101 =   meanAndStd(getFilePaths("ucx_openIB_101.dat"))[0][1:]
    ucx_openIB_101_err = meanAndStd(getFilePaths("ucx_openIB_101.dat"))[1][1:]



pSize = [k[0] for k in np.loadtxt("./data/fullRange/0/ucx_none_110.dat")][1:]

p = Plotter((12,7))
p.createAxis(111)

width=2; size=8; style = ".-"

if ucxtHH:
    #p.addSubplot(pSize, UCX_t_none_110, dataLabel=r"( ), H$\rightarrow$H", dataStyle=style, width=width, size=size)

    p.addSubplot(pSize, UCX_t_tcp_110, color="C2", dataLabel=r"UCX_TLS=tcp, H$\rightarrow$H", dataStyle=style, width=width, size=size)
    p.drawErrorBars(UCX_t_tcp_110_err, 0, 0, color="C2")

    p.addSubplot(pSize, UCX_t_rc_110,  color="C0", dataLabel=r"UCX_TLS=rc,  H$\rightarrow$H", dataStyle=style, width=width, size=size)
    p.drawErrorBars(UCX_t_rc_110_err, 0, 0, color="C0")

    p.addSubplot(pSize, UCX_t_ud_110,  color="C8", dataLabel=r"UCX_TLS=ud,  H$\rightarrow$H", dataStyle=style, width=width, size=size)
    p.drawErrorBars(UCX_t_ud_110_err, 0, 0, color="C8")
   
    p.addSubplot(pSize, UCX_t_all_110, color="C1", dataLabel=r"UCX_TLS=all, H$\rightarrow$H", dataStyle=style, width=width, size=size)
    p.drawErrorBars(UCX_t_all_110_err, 0, 0, color="C1")
    
    p.addSubplot(pSize, UCX_t_all_101, color="C3", dataLabel=r"UCX_TLS=all, H$\rightarrow$D", dataStyle=".:", width=width, size=size)
    p.drawErrorBars(UCX_t_all_101_err, 0, 0, color="C3")
    
    #p.addSubplot(pSize, UCX_t_none_101, dataLabel=r"( ), H$\rightarrow$D", dataStyle=".:", width=width, size=size)
else:
    #p.addSubplot(pSize, ob1_noOpenIB, dataLabel=r"PML=ob1; BTL=^openIB: H$\rightarrow$H", dataStyle=style, width=width, size=size)
    
    
    
    p.addSubplot(pSize, ob1_openIB_110,   dataLabel=r"PML=ob1, BTL=openIB, H$\rightarrow$H", color="C2", dataStyle=style, width=width, size=size)
    p.drawErrorBars(ob1_openIB_err_110, 0, 0, color="C2")
    
    p.addSubplot(pSize, ob1_tcp_110,   dataLabel=r"PML=ob1, BTL=tcp, H$\rightarrow$H", color="C9", dataStyle=style, width=width, size=size)
    p.drawErrorBars(ob1_tcp_err_110, 0, 0, color="C9")
    
    p.addSubplot(pSize, none_noOpenIB_101,    dataLabel=r"PML=( ), BTL=^openIB,    H$\rightarrow$D", color="C5", dataStyle=style, width=width, size=size)
    p.drawErrorBars(none_noOpenIB_101_err, 0, 0, color="C5")

    p.addSubplot(pSize, none_noOpenIB_110,    dataLabel=r"PML=( ), BTL=^openIB,    H$\rightarrow$H", color="C4", dataStyle=style, width=width, size=size)
    p.drawErrorBars(none_noOpenIB_110_err, 0, 0, color="C4")

    p.addSubplot(pSize, none_OpenIB_101,    dataLabel=r"PML=( ), BTL=openIB,    H$\rightarrow$H", color="C6", dataStyle=style, width=width, size=size)
    p.drawErrorBars(none_OpenIB_101_err, 0, 0, color="C6")

    p.addSubplot(pSize, ucx_none_110,     dataLabel=r"PML=UCX, BTL=( ),    H$\rightarrow$H", color="C1", dataStyle=style,  width=width, size=size)
    p.drawErrorBars(ucx_none_110_err, 0, 0, color="C1")
    
    p.addSubplot(pSize, ucx_none_101, dataLabel=r"PML=UCX, BTL=( ),    H$\rightarrow$D", color="C3", dataStyle=".:", width=width, size=size)
    p.drawErrorBars(ucx_none_101_err, 0, 0, color="C3")

    p.addSubplot(pSize, ucx_openIB_110,     dataLabel=r"PML=UCX, BTL=openIB,    H$\rightarrow$H", color="C7", dataStyle=style,  width=width, size=size)
    p.drawErrorBars(ucx_openIB_110_err, 0, 0, color="C7")
    
    p.addSubplot(pSize, ucx_openIB_101, dataLabel=r"PML=UCX, BTL=openIB,    H$\rightarrow$D", color="C8", dataStyle=".:", width=width, size=size)
    p.drawErrorBars(ucx_openIB_101_err, 0, 0, color="C8")



#p.drawVerticalLine(12138 * 4. / 1048576)
#p.drawVerticalLine(61447 * 4. / 1048576)

#p.addSubplot(pSize, Memcpy, dataLabel=r"cudaMemcpy", dataStyle=".-", width=2, size=8)

p.setTitles("", "size (MB)", r"latency ($\mu$s)")

# save the figure
p.setProperties(grid=1, doXlog=1, doYlog=1, doWhite=0)
#p.saveFig(outputFolder + sample + ".pdf", dpi=200, transparent=0)
#p.saveFig(outputFolder + sample + ".png", dpi=200, transparent=0)

p.setProperties(doWhite=1)
p.saveFig(outputFolder + sample + "_v2.pdf", dpi=200, transparent=1)
p.saveFig(outputFolder + sample + "_v2_1.png", dpi=200, transparent=1)

