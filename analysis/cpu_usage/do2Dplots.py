

import numpy as np
#------------------------------------------------------------------


# global variables

dataFile = "results.txt"
mult = 1000     # factor for z data




data = np.loadtxt(dataFile)  #recordar tener cuidado con la ubicacion del 'Phi_2.x'
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
x = []
y = []
z = []

# we exclude the upper left element of the matrix
x = data[0][1:]                     # first row
y = [elem[0] for elem in data[1:]]  # first column

for elem in data[1:,1:]:
    for i in range(len(elem)):
        elem[i] *= mult
        if elem[i] > 20: elem[i] = 0 # <- DELETEME
    z.append(elem)
# At this point we already have x, y and z.
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
cmap = 'magma'
grid = 1           #do you want a grid? 
grid_color = 'k'


plot_title = u"running time (ms)"
#plot_title = "auto"

text_scale = 0.9      #set it for fit the resolution of your screen
only_contours = False #Do you want the full plot colored or only the contours? 
#cmap can be, amongst others: plasma, bone, coolwarm, autumn, inferno
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#let's produce the plot
x, y = np.meshgrid(x, y)



#matplotlib stuff
import matplotlib.pyplot as plt

figure = plt.figure(figsize=(16,9.5))
ax1 = figure.add_subplot(111)

#generate labels for the contours
if only_contours:
    cont = ax1.contour(y,x,z, 20,cmap=cmap,rstride=1,cstride=1)
    figure.colorbar(cont, shrink=0.9) #the colorbar
else:
    
    cont = ax1.contourf(y,x,z, 20,cmap=cmap)    # the fill

    figure.colorbar(cont, shrink=0.9)           #the colorbar

    cont = ax1.contour(y,x,z, 20,\
         colors='w', linewidths=0.5, alpha=0.7)



if grid: ax1.grid(linestyle='solid', axis='x', color=grid_color)

ax1.set_ylabel ("Sleeping time (us)", fontsize = 16*text_scale)
ax1.set_xlabel ("Number of doubles", fontsize = 16*text_scale)

ax1.set_title(plot_title, fontsize=18)

#set the number of ticks you want in each axis
plt.locator_params(axis='x', nbins=22)
plt.locator_params(axis='y', nbins=20)

#Do you want to save the fig?
plt.savefig('a0.png', transparent=True)

#show the fig, enabling clicking option
ax1.clabel(cont, inline = 1, manual=True, colors='w', fmt="%0.3f")
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

