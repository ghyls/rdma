

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
#------------------------------------------------------------------


# global variables

dataFile = "results_small.txt"
#dataFile = "results.txt"

mult = 1000     # factor for z data



data = np.loadtxt(dataFile)  #recordar tener cuidado con la ubicacion del 'Phi_2.x'
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
x = []  # N doubles
y = []  # us
z = []  # s

# we exclude the upper left element of the matrix
x = [elem[0] * 8 / 1024 for elem in data[1:]]   # first column, size
y = data[0][1:] / 1000                                 # first row, time, microseconds

for elem in data[1:,1:]:
    for i in range(len(elem)):
        elem[i] *= mult
        if elem[i] > 2: elem[i] = 0 # <- DELETEME
    z.append(elem) 
z = np.transpose(z)
# At this point we already have x, y and z.
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
cmap = 'bone'
cmap = 'PRGn'
cmap = 'PuOr'
cmap = 'RdYlBu'
cmap = 'twilight_shifted'
grid = 0           #do you want a grid? 
grid_color = 'k'


plot_title = u"Total elapsed time (ms)"
#plot_title = "auto"

text_scale = 1.2      #set it for fit the resolution of your screen
only_contours = False #Do you want the full plot colored or only the contours? 
#cmap can be, amongst others: ocean, bone, coolwarm, autumn, inferno
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#let's produce the plot
x, y = np.meshgrid(x, y)





figure = plt.figure(figsize=(16,8))
ax1 = figure.add_subplot(111)

#generate labels for the contours
if only_contours:
    cont = ax1.contour(y,x,z, 20,cmap=cmap,rstride=1,cstride=1)
    figure.colorbar(cont, shrink=0.9) #the colorbar
else:

    cont = ax1.pcolormesh(x, y, z, cmap=cmap)
    tmp_min = [min(elem) for elem in z]
    tmp_max = [max(elem) for elem in z]
    #levels = MaxNLocator(nbins=1000).tick_values(min(tmp_min), max(tmp_max))
    #cont = ax1.contourf(y,x,z, 20,cmap=cmap, levels=levels)    # the fill
    #cont = ax1.contourf(y,x,z, 20,cmap=cmap, levels=levels)    # the fill
    cbar = figure.colorbar(cont, shrink=0.9)           #the colorbar
    cbar.ax.tick_params(labelsize=12*text_scale) 

    #ax1.contour(x,y,z, 20, colors='w', linewidths=0.5, alpha=0.7)
    grid=0


if grid: ax1.grid(linestyle='solid', axis='x', color=grid_color)

ax1.set_ylabel (u"sleeping time (ms)", fontsize = 16*text_scale)
#ax1.set_xlabel ("Number of doubles", fontsize = 16*text_scale)
ax1.set_xlabel ("package size (KB)", fontsize = 16*text_scale)

ax1.set_title(plot_title, fontsize=20*text_scale)

ax1.tick_params(axis='x', which='both', labelsize=14)
ax1.tick_params(axis='y', which='both', labelsize=14)





#set the number of ticks you want in each axis
plt.locator_params(axis='x', nbins=11)
#plt.locator_params(axis='y', nbins=20)


#Do you want to save the fig?
plt.savefig('a0.png', transparent=False)
plt.savefig('2d0_2.pdf', transparent=True)

#show the fig, enabling clicking option
#ax1.clabel(cont, inline = 1, manual=True, colors='w', fmt="%0.3f")
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

