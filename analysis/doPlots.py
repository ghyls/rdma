import numpy as np

from plotter import Plotter

def meanAndStd2(path, fileName, num_files):

    mean = []
    error = []

    # This will contain all the data.
    matrix = np.array([None]*num_files)   # matrix[len(data)][0...num_files]

    for i in range (num_files):

        if ucxtHH: filePath = "./data/fullRange/" + str(i) + "/UCX_transports/" + file
        else: filePath = "./data/fullRange/" + str(i) + "/" + fileName

        # We only load it $num_file times. The prize is the memory.
        data = np.loadtxt(filePath)

        #tempColumn = []
        for j in range(len(data)):
            #tempColumn.append(data[j][1])
            matrix[i][j] = data[j][1]

        #matrix[i] = tempColumn

    for i in range(len(matrix[0])):

        means = [matrix[k][i]*1e6 for k in range(num_files)]
        mean.append(np.mean(means))
        error.append(np.std(means))

    return [mean, error]

def meanAndStd(file):

    mean = []
    error = []

    matrix = np.array([None]*10)   # matrix[len(data)][sample(0...9)]

    for i in range (10):

        if ucxtHH: filePath = "./data/fullRange/" + str(i) + "/UCX_transports/" + file
        else: filePath = "./data/fullRange/" + str(i) + "/" + file

        data = np.loadtxt(filePath)

        tempColumn = []
        for j in range(len(data)):
            tempColumn.append(data[j][1])

        matrix[i] = tempColumn

    for i in range(len(matrix[0])):

        means = [matrix[k][i]*1e6 for k in range(10)]
        mean.append(np.mean(means))
        error.append(np.std(means))

    return [mean, error]


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
###    ob1_noOpenIB =   meanAndStd("ob1_noOpenIB_110.dat")[0][1:]
###    ob1_noOpenIB_err = meanAndStd("ob1_noOpenIB_110.dat")[1][1:]
###
###    ob1_openIB =   meanAndStd("ob1_openIB_110.dat")[0][1:]
###    ob1_openIB_err = meanAndStd("ob1_openIB_110.dat")[1][1:]
###
###    ob1_tcp =   meanAndStd("ob1_tcp_110.dat")[0][1:]
###    ob1_tcp_err = meanAndStd("ob1_tcp_110.dat")[1][1:]
###
###    none_none =   meanAndStd("none_none_110.dat")[0][1:]
###    none_none_err = meanAndStd("none_none_110.dat")[1][1:]

    ucx_none =   meanAndStd("ucx_none_110.dat")[0][1:]
    ucx_none_err = meanAndStd("ucx_none_110.dat")[1][1:]

    ucx_none_101 =   meanAndStd("ucx_none_101.dat")[0][1:]
    ucx_none_101_err = meanAndStd("ucx_none_101.dat")[1][1:]




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
    
    
###    p.addSubplot(pSize, ob1_tcp,      dataLabel=r"PML=ob1, BTL=tcp,    H$\rightarrow$H", color="C2", dataStyle=style, width=width, size=size)
###    p.drawErrorBars(ob1_tcp_err, 0, 0, color="C2")
###    
###    p.addSubplot(pSize, ob1_openIB,   dataLabel=r"PML=ob1, BTL=openIB, H$\rightarrow$H", color="C9", dataStyle=style, width=width, size=size)
###    p.drawErrorBars(ob1_openIB_err, 0, 0, color="C9")
###    
###    p.addSubplot(pSize, none_none,    dataLabel=r"PML=( ), BTL=( ),    H$\rightarrow$H", color="C5", dataStyle=style, width=width, size=size)
###    p.drawErrorBars(none_none_err, 0, 0, color="C5")
    
    p.addSubplot(pSize, ucx_none,     dataLabel=r"PML=UCX, BTL=( ),    H$\rightarrow$H", color="C1", dataStyle=style,  width=width, size=size)
    p.drawErrorBars(ucx_none_err, 0, 0, color="C1")
    
    p.addSubplot(pSize, ucx_none_101, dataLabel=r"PML=UCX, BTL=( ),    H$\rightarrow$D", color="C3", dataStyle=".:", width=width, size=size)
    p.drawErrorBars(ucx_none_101_err, 0, 0, color="C3")

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
p.saveFig(outputFolder + sample + "_v2.png", dpi=200, transparent=1)

