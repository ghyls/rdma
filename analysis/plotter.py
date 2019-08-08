#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



class Plotter:

    m = 1.2

    def __init__(self, size):
        self.fig1 = plt.figure(figsize=size)

    def createAxis(self, pos):
        self.ax1 = self.fig1.add_subplot(pos, aspect='auto', facecolor='aliceblue')

    def setTitles(self, title='', xTitle='', yTitle=''):
        self.ax1.set_title(title, size = 14*self.m)
        if xTitle != '': self.ax1.set_xlabel(xTitle, size=12*self.m)
        if yTitle != '': self.ax1.set_ylabel(yTitle, size=12*self.m)

    def setProperties(self, grid = False, legLoc = '', rmX = False, rmY = False, 
                        xlim = [], ylim = [], doXlog = False, doYlog = False, 
                        doWhite = False):

        # axix limits
        if xlim != []: self.ax1.set_xlim(xlim)
        if ylim != []: self.ax1.set_ylim(ylim)

        # toogle logarithm
        if doXlog: self.ax1.set_xscale('log')
        if doYlog: self.ax1.set_yscale('log')

        # remove axis ticks
        if rmX: self.ax1.tick_params(axis='x', which='both', bottom=False, \
                labelbottom=False)
                
        if rmY: self.ax1.tick_params(axis='y', which='both', left=False, \
                labelleft=False)        

        # grid
        if grid != False: 
            if grid==True: self.ax1.grid(which="major")
            else: self.ax1.grid(which=grid)
        
        # paint everything white
        if doWhite:
            self.ax1.tick_params(axis='x', which='both', colors='white')
            self.ax1.tick_params(axis='y', which='both', colors='white')
            self.ax1.title.set_color('white')
            self.ax1.xaxis.label.set_color('white')
            self.ax1.yaxis.label.set_color('white')
            self.ax1.spines['bottom'].set_color('white')
            self.ax1.spines['top'].set_color('white')
            self.ax1.spines['left'].set_color('white')
            self.ax1.spines['right'].set_color('white')            

        # legend
        self.ax1.legend(loc=legLoc) if legLoc != '' else self.ax1.legend()
        self.ax1.legend(prop={'family': 'DejaVu Sans Mono'})
            # [ 2    1 ]
            # [ 6    5 ]
            # [ 3    4 ]   

    def getPoptAndPerr(self, func, x, y, p0):

        popt, pcov = curve_fit(func, x, y, p0=p0)
        perr = np.sqrt(np.diag(pcov))               

        return popt, perr


    x = []; y = []
    def addSubplot(self, x, y, fit = False, func = '', p0 = [], 
        dataLabel='', dataStyle='.', fitStyle='-', fitLabel = '', color=None, 
        size = 12, width = 12,  fitColor='c'):

        self.x = x; self.y = y

        self.ax1.plot(x, y, dataStyle, linewidth=width, label=dataLabel, 
                color = color, markersize = size)

        if fit: 
               
            popt = self.getPoptAndPerr(func, x, y, p0=p0)[0]
            
            xAux = np.linspace(min(x), max(x), 200)
            self.ax1.plot(xAux, func(xAux, *popt), fitStyle,
                                             label=fitLabel, color = fitColor)


    def drawErrorBars(self, errors, linewidth = 0.1, markerwidth = 0, color = "darkolivegreen"):
        
        self.ax1.errorbar(self.x, self.y, yerr = errors, capsize = 0,
                        elinewidth = 1, markeredgewidth = markerwidth,
                        fmt = 'none', ecolor = color, capthick = 0, ms=20, mew=4)

        yplus = [self.y[j] + errors[j] for j in range(len(self.x))]
        yless = [self.y[j] +-errors[j] for j in range(len(self.x))]

        self.ax1.fill_between(self.x, yless, yplus, color = color, alpha = 0.3)


    def drawVerticalLine(self, coord, label="", color = "crimson", size = 2, 
                                                            style = "-"):
        self.ax1.axvline(x=coord, label = label, color = color, 
                            linestyle = style, linewidth = size)
    
    def drawHorizonralLine(self, coord, label="", color = "crimson", size = 2, 
                                                            style = "-"):
        self.ax1.axhline(y = coord, label = label, color = color, 
                            linestyle = style, linewidth = size)

    def showGraph(self):
        plt.show()

    def saveFig(self, outputName, dpi = None, transparent = False):
        self.fig1.savefig(outputName, bbox_inches = 'tight', 
                            dpi = dpi, transparent = transparent)




def getR2(func, x, y, p0):
     
    popt = curve_fit(func, x, y, p0=p0)[0]
    residuals = y - func(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    R2 = 1-(ss_res/ss_tot)
    
    return R2


