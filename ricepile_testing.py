# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:24:25 2023

@author: lizzi
"""
import numpy as np
import matplotlib.pyplot as plt
from ricepile import RicePile
import seaborn as sns
import pandas as pd
import scipy.stats as sp
import scipy.optimize as so
from logbin_2020 import logbin
sns.set()
sns.set_style("whitegrid")
sns.set_context("paper", font_scale = 1.5)
sns.set_palette("winter", 3)
#%% some useful functions 

def smootheHeight(repeatData):
    repeats = len(repeatData.columns) - 1
    colList = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    smootheHeight = 1/repeats * repeatData[colList].sum(axis = 1)
    return np.array(smootheHeight)

def timeAvHeight(heightData):
    time = len(heightData)
    averageHeight = 1/time * np.sum(heightData)
    return averageHeight

def stdevHeight(heightData):
    time = len(heightData)
    var = 1/time * np.sum(heightData**2) - (1/time* np.sum(heightData))**2
    return np.sqrt(var)

def heightProb(heightData):
    dfHeight = {}
    total = len(heightData)
    for i in range(len(heightData)):
        if int(heightData[i]) in dfHeight.keys():
            dfHeight[int(heightData[i])] += 1/total
        else:
            dfHeight[int(heightData[i])] = 1/total
    return dfHeight

def firstOrder(length, a0):
    return a0*length

def secondOrder(length, a1, w1):
    return 1 - a1*length**(-w1)

def firstSecond(length, a0, a1, w1):
    return a0*length*(1 - a1*length**(-w1))

def sigmaDist(length, mul, exp):
    return mul*length**exp

def gaussian(x, mean, sigma, amp):
    return amp*np.exp(-(x-mean)**2/(2*sigma**2))

def linear(x, gradient, intercept):
    return x*gradient + intercept

def calculateMoment(avalancheData, moment):
    T = len(avalancheData)
    sK = np.array(avalancheData)**moment
    sKSum = np.sum(sK)/T
    return sKSum

def exponential(x, a, b):
    return a*x**b

def sizeProb(sizeData):
    dfSize = {}
    total = len(sizeData)
    for i in range(len(sizeData)):
        if int(sizeData[i]) in dfSize.keys():
            dfSize[int(sizeData[i])] += 1/total
        else:
            dfSize[int(sizeData[i])] = 1/total
    return dfSize
    
#%% Task 1: testing the basic functions - no iterations or threshold testing implemented yet
#testPile = RicePile(length = 128) #each run of this line will change the value of the thresholds so has to be kept separate - alter this later?
"""
testPile.reset()
print(testPile.getHeights(), testPile.getSlopes(), testPile.getThresholdSlopes())
testPile.drive()
print(testPile.getHeights(), testPile.getSlopes(), testPile.getThresholdSlopes())
testPile.relax(0)
print(testPile.getHeights(), testPile.getSlopes(), testPile.getThresholdSlopes())
testPile.updateThresholdSlope(3)
print(testPile.getHeights(), testPile.getSlopes(), testPile.getThresholdSlopes())
print(testPile.calcSlopes())
testPile.relax(3)
print(testPile.getHeights(), testPile.getSlopes(), testPile.getThresholdSlopes())
testPile.relax(4)
print(testPile.getHeights(), testPile.getSlopes(), testPile.getThresholdSlopes())
testPile.drive()
testPile.drive()
testPile.drive()
print(testPile.getHeights(), testPile.getSlopes(), testPile.getThresholdSlopes())
print(testPile.aboveThreshold())
print(testPile.pileSize())
"""

#%%
boringPile16 = RicePile(length = 16, probability = 1) 
boringPile32 = RicePile(length = 32, probability = 1)
boringData16 = boringPile16.startIteration(10000)
boringData32 = boringPile32.startIteration(10000)

boringPile32.visualise(max_height = 32)
boringPile16.visualise(max_height = 32)
plt.xlim(-0.4, 32)
plt.xlabel(r"$i$")
plt.savefig("visualised.svg")
#%%
plt.plot(boringData16[2], label = "L = 16")
plt.plot(boringData32[2], label = "L = 32")
plt.xlabel(r"$t$")
plt.ylabel(r"$Number \ of \ grains \ in  \ pile$")
plt.legend()
plt.savefig("boring_height.svg")

#%% testing that the system can reach a steady state for lengths 16 and 32
steadyPile16 = RicePile(length = 16)
steadyPile32 = RicePile(length = 32)
steadyData16 = steadyPile16.startIteration(10000)
steadyData32 = steadyPile32.startIteration(10000)

#%% plotting the number of grains in the system against time - this does not account for transient states as this would just introduce more noise
plt.plot(steadyData16[2], label = "L = 16")
plt.plot(steadyData32[2], label = "L = 32")
plt.xlabel(r"$t$")
plt.ylabel(r"$Number \ of \ grains \ in  \ pile$")
plt.legend()
plt.savefig("reaches_steady.svg")
#%% calculating the average height after steady state is reached
#print(len(steadyData16[3]), len(steadyData32[3])) finding where it goes into steady state
steadyTotal16 = steadyData16[2][-29768:]
steadyTotal32 = steadyData32[2][-29125:]
print(f"the average heights after reaching the steady state are %.5f and %.5f for the L = 16 and L = 32 systems, respectively" %(np.mean(steadyTotal16), np.mean(steadyTotal32)))

#%% Task 2: Investigating height of the pile - creating the systems 
lengths = [4, 8, 16, 32, 64, 128, 256]
heightPile4 = RicePile(length = 4)
heightPile8 = RicePile(length = 8)
heightPile16 = RicePile(length = 16)
heightPile32 = RicePile(length = 32)
heightPile64 = RicePile(length = 64)
heightPile128 = RicePile(length = 128)
heightPile256 = RicePile(length = 256) #there is method to the madness here, it'll be easier to analyise individually and find bugs if they're implemented individually 
#%% iterating the systems - starting with the same bagSize for each system but updating as needed

heightPile4.reset()
heightData4 = heightPile4.startIteration(60000) 
#%%
heightPile8.reset()
heightData8 = heightPile8.startIteration(60000)
#%%
heightPile16.reset()
heightData16 = heightPile16.startIteration(60000)
#%%
heightPile32.reset()
heightData32 = heightPile32.startIteration(60000)
#%%
heightPile64.reset()
heightData64 = heightPile64.startIteration(60000)
#%%
heightPile128.reset()
heightData128 = heightPile128.startIteration(60000)
#%%
heightPile256.reset()
heightData256 = heightPile256.startIteration(60000)
#%% saving the data
heightDataNames = [["listData4.csv", "totalsData4.csv", "heightData4.csv", "crossOverData4.csv"], ["listData8.csv", "totalsData8.csv", "heightData8.csv", "crossOverData8.csv"],
                   ["listData16.csv", "totalsData16.csv", "heightData16.csv", "crossOverData16.csv"],["listData32.csv", "totalsData32.csv", "heightData32.csv", "crossOverData32.csv"],
                   ["listData64.csv", "totalsData64.csv", "heightData64.csv", "crossOverData64.csv"],["listData128.csv", "totalsData128.csv", "heightData128.csv", "crossOverData128.csv"],
                   ["listData256.csv", "totalsData256.csv", "heightData256.csv", "crossOverData256.csv"]]
heightDataFiles = [heightData4, heightData8, heightData16, heightData32, heightData64, heightData128, heightData256]
#%%
for i in range(len(heightDataFiles)):
   np.savetxt(heightDataNames[i][0], heightDataFiles[i][1])
   np.savetxt(heightDataNames[i][1], heightDataFiles[i][2])
   np.savetxt(heightDataNames[i][2], heightDataFiles[i][3])
   np.savetxt(heightDataNames[i][3], heightDataFiles[i][4])

#%% Task 2a plotting height as a function of time 
plt.scatter(np.arange(0, 60002, 1), heightData4[3], marker = ".")
plt.scatter(np.arange(0, 60002, 1), heightData8[3], marker = ".")
plt.scatter(np.arange(0, 60002, 1), heightData16[3], marker = ".")
plt.scatter(np.arange(0, 60002, 1), heightData64[3], marker = ".")
plt.scatter(np.arange(0, 60002, 1), heightData128[3], marker = ".")
plt.scatter(np.arange(0, 60002, 1), heightData256[3], marker = ".")
#%% measuring crossover time
# find the first crossover for each from previous measurement
initialCrossovers = []
for data in heightDataFiles:
    initialCrossovers.append(data[4][0])
#%% run until just over this point M times for each L
piles = [heightPile4, heightPile8, heightPile16, heightPile32, heightPile64, heightPile128, heightPile256]
crossOverTimes = []
for pile in piles:
    crossOverL = []
    for i in range(10):
        pile.reset()
        data = pile.startIteration(60000, crossOver = True)
        crossOverL.append(data)
    crossOverTimes.append(crossOverL)
    print("round complete")
    
#%%
np.savetxt("crossOverData.csv", crossOverTimes)
for i in range(len(crossOverTimes)):
    print(np.mean(crossOverTimes[i]))

#%% getting the multiple readings for average h

repeatDataNames = [["reAvs4.csv", "reTotals4.csv", "reheights4.csv"], ["reAvs8.csv", "reTotals8.csv", "reheights8.csv"], ["reAvs16.csv", "reTotals16.csv", "reheights16.csv"],
                   ["reAvs32.csv", "reTotals32.csv", "reheights32.csv"], ["reAvs64.csv", "reTotals64.csv", "reheights64.csv"], ["reAvs128.csv", "reTotals128.csv", "reheights128.csv"],
                   ["reAvs256.csv", "reTotals256.csv", "reheights256.csv"],]

repeatAvalanches = []
repeatTotals = []
repeatHeights = []
piles = [heightPile4, heightPile8, heightPile16, heightPile32, heightPile64, heightPile128, heightPile256]
#times = [5015, 5054, 5218, 6862, 8476, 18972, 61138]
for i in range(len(piles)):
    avalanches = {}
    totals = {}
    heights = {}
    for j in range(10):
        piles[i].reset()
        data = piles[i].startIteration(61150)
        avalanches[j] = data[0]
        totals[j] = data[1]
        heights[j] = data[2]
    avFrame = pd.DataFrame(avalanches)
    totalsFrame = pd.DataFrame(totals)
    heightsFrame = pd.DataFrame(heights)
    avFrame.to_csv(repeatDataNames[i][0])
    totalsFrame.to_csv(repeatDataNames[i][1])
    heightsFrame.to_csv(repeatDataNames[i][2])
    repeatAvalanches.append(avalanches)
    repeatTotals.append(totals)
    repeatHeights.append(heights)
    print("round complete")
  
#%% reading in the repeat height data
repeatHeightData = []
for i in range(len(repeatDataNames)):
    heightData = pd.read_csv(repeatDataNames[i][2], names = ["time", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], skiprows = 1)
    repeatHeightData.append(heightData)
#%% plotting the smoothed heights
smootheHeights = []
systemSizes = np.array([4, 8, 16, 32, 64, 128, 256])
#%%
for i in range(len(repeatDataNames)):
    smoothie = smootheHeight(repeatHeightData[i])
    smootheHeights.append(smoothie)
    plt.xlabel("Grains Added")
    plt.ylabel("Height")
    plt.loglog(repeatHeightData[i]["time"], smoothie, label = systemSizes[i])
    plt.legend()
    
#%% finding tc relationship
crossOverTimes = pd.read_csv("crossOverData.csv")
lengthList = ["4", "8", "16", "32", "64", "128", "256"]
crossOverAvs = []
for i in range(len(lengthList)):
    crossOverAvs.append(int(np.round(np.mean(np.array(crossOverTimes[lengthList[i]])), 0)))
   
plt.scatter(lengths, np.sqrt(crossOverAvs), color = "red")
plt.xlabel(r"$L$")
plt.ylabel(r"$\sqrt{\langle t_c(L) \rangle}$")
#plt.savefig("root_crossovers.svg")
linearFit = sp.linregress(lengths, np.sqrt(crossOverAvs))
crossOverConst = linearFit[0]**2

#%% finding tau t
linearRegions = []
linearFits = []
for i in range(len(smootheHeights)):
    linear = np.log10(smootheHeights[i][1:crossOverAvs[i]])
    linearRegions.append(linear)
    time = np.array(repeatHeightData[i]["time"])[1:crossOverAvs[i]]
    plt.plot(np.log10(time), linear)
    fit = sp.linregress(np.log10(time), linear)
    linearFits.append(fit)
    
#%% plotting data collapse
colours = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]

for i in range(len(repeatDataNames)):
    plt.xlabel(r"$t/t_c$")
    plt.ylabel(r"$t^{-\tau_t}\tilde{h}(t)$")
    time = repeatHeightData[i]["time"] 
    plt.loglog(time/(crossOverConst*systemSizes[i]**2), smootheHeights[i]*(time**(-linearFits[6][0])), label = systemSizes[i], color = colours[i])
    plt.legend()
    plt.savefig("data_collapse.svg", bbox_inches = "tight")
#%%
for i in range(len(repeatDataNames)):
    plt.xlabel(r"$t$")
    plt.ylabel(r"$t^{-\tau_t}\tilde{h}(t)$")
    time = repeatHeightData[i]["time"] 
    plt.loglog(time, smootheHeights[i]*(time**(linearFits[6][0])), label = systemSizes[i], color = colours[i])
    plt.legend()
    #plt.savefig("vertical_align.svg", bbox_inches = "tight")
#%%
for i in range(len(repeatDataNames)):
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\tilde{h}(t)$")
    time = repeatHeightData[i]["time"] 
    plt.loglog(time, smootheHeights[i], label = systemSizes[i], color = colours[i])
    plt.legend()
    plt.savefig("original_height.svg", bbox_inches = "tight")
#%%  obtaining height data for t > tc

recurrData = []
crossOvers = [15, 52, 230, 912, 3388, 13970, 56092]
for i in range(len(repeatHeightData)):
    data = np.array(repeatHeightData[i]["7"])[crossOvers[i]:]
    recurrData.append(data)


#%% measuring averages and stdev

timeAvHeightdf = {}
stdevdf = {}
#timeAvProbdf = {}
for i in range(len(recurrData)):
    timeAvHeightdf[lengthList[i]] = timeAvHeight(recurrData[i])
    stdevdf[lengthList[i]] = stdevHeight(recurrData[i])
timeAvFrame = pd.DataFrame(timeAvHeightdf, index = ["<h(t)>"])
stdevFrame = pd.DataFrame(stdevdf, index = ["sigma"])
 
heightDataFrame = pd.concat([timeAvFrame, stdevFrame])
heightDataFrame.to_csv("height_calcs.csv")


#%% measuring height probability
heightProbNames = ["heightProb4.csv", "heightProb8.csv", "heightProb16.csv", "heightProb32.csv", "heightProb64.csv", "heightProb128.csv", "heightProb256.csv"]
heightProbs = []
for i in range(len(recurrData)):
    probsdf = heightProb(recurrData[i])
    probsFrame = pd.DataFrame(probsdf, index = [0])
    heightProbs.append(probsFrame)
    probsFrame.to_csv(heightProbNames[i])

#%% Task 2e: Corrections to scaling - step one, plot <h> against system size
heightDataFrame = pd.read_csv("height_calcs.csv")
#plt.scatter(systemSizes, np.array(heightDataFrame["<h(t)>"]), marker = ".")
firstOrderFit = so.curve_fit(firstOrder, systemSizes, np.array(heightDataFrame["<h(t)>"]), p0 = [2])
#%% plotting the first order fit
plt.scatter(systemSizes, np.array(heightDataFrame["<h(t)>"]), marker = ".", color = "darkblue")
plt.plot(systemSizes, firstOrder(np.array(systemSizes), firstOrderFit[0][0]))
#%% calculating next order
h0 = firstOrder(np.array(systemSizes), firstOrderFit[0][0])
#residuals = np.array(heightDataFrame["<h(t)>"]) - h0
divs = np.array(heightDataFrame["<h(t)>"])/h0
#print(h0)
plt.scatter(systemSizes[0:4], divs[0:4], marker = ".")
#%% 
x = np.linspace(1, 256, 1000)
secondOrderFit = so.curve_fit(secondOrder, systemSizes, divs, p0 = [0.5, 0.5])
plt.plot(systemSizes, divs, color = "red")
plt.plot(x, secondOrder(x, *secondOrderFit[0]))
#%% 
plt.scatter(systemSizes, np.array(heightDataFrame["<h(t)>"]), label =  r"$\langle h(t)\rangle$", color = "red")
plt.plot(x, firstSecond(x, firstOrderFit[0][0], secondOrderFit[0][0], secondOrderFit[0][1]), label = r"$a_0L(1 - a_1L^{-\omega_1}$)", color = "cyan")
plt.plot(x, firstOrder(x, firstOrderFit[0][0]), label = r"$a_0L$")
plt.legend()
plt.ylabel(r"$L$")
plt.xlabel(r"$\langle h(t;L)\rangle$")
plt.savefig("corr_scaling_fullrange.svg", bbox_inches = "tight")
#%%axes1.scatter(systemSizes[0:3], np.array(heightDataFrame["<h(t)>"])[0:3], label = r"$\langle h(t)\rangle$", color = "red")
plt.scatter(systemSizes[4:], np.array(heightDataFrame["<h(t)>"])[4:], label = r"$\langle h(t)\rangle$", color = "red")
plt.plot(x, firstSecond(x, firstOrderFit[0][0], secondOrderFit[0][0], secondOrderFit[0][1]), label = r"$a_0L(1 - a_1L^{-\omega_1}$)", color = "cyan")
plt.plot(x, firstOrder(x, firstOrderFit[0][0]), label = r"$a_0L$")
plt.ylim(80, 450)
plt.xlim(50, 260)
plt.xlabel(r"$\langle h(t;L)\rangle$")
plt.ylabel(r"$L$")
#plt.set_xticks([2, 4, 6 ,8, 10, 12, 14, 16, 18, 20])
plt.legend()
#plt.savefig("corr_scaling_zoomed.svg", bbox_inches = "tight")

#%% making an inlayed zoom plot
fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.9, 0.9]) # main axes
axes2 = fig.add_axes([0.25, 0.65, 0.4, 0.3]) # inset axes


# insert
axes2.scatter(systemSizes, np.array(heightDataFrame["<h(t)>"]), color = "red", marker = ".")
axes2.plot(systemSizes, np.array(heightDataFrame["<h(t)>"]), color = "red")
axes2.plot(x, firstSecond(x, firstOrderFit[0][0], secondOrderFit[0][0], secondOrderFit[0][1]), color = "cyan")
axes2.plot(x, firstOrder(x, firstOrderFit[0][0]), label = r"$a_0L$")
axes2.set_ylabel(r"$\langle h(t;L)\rangle$")
axes2.set_xlabel(r"$L$")


#axes1.legend()

# main figure
axes1.scatter(systemSizes[0:3], np.array(heightDataFrame["<h(t)>"])[0:3], label = r"$\langle h(t)\rangle$", color = "red")
axes1.plot(x, firstSecond(x, firstOrderFit[0][0], secondOrderFit[0][0], secondOrderFit[0][1]), label = r"$a_0L(1 - a_1L^{-\omega_1}$)", color = "cyan")
axes1.plot(x, firstOrder(x, firstOrderFit[0][0]), label = r"$a_0L$")
axes1.set_ylim(0, 40)
axes1.set_xlim(1, 20)
axes1.set_ylabel(r"$\langle h(t;L)\rangle$")
axes1.set_xlabel(r"$L$")
axes1.set_xticks([2, 4, 6 ,8, 10, 12, 14, 16, 18, 20])
axes1.legend(loc = "lower right")

fig.savefig("corr_scaling_overlay.svg", bbox_inches = "tight")
#%%
y12 = firstSecond(np.array(systemSizes), firstOrderFit[0][0], secondOrderFit[0][0], secondOrderFit[0][1])
linChi = sp.chisquare(np.array(heightDataFrame["<h(t)>"]), h0)
curveChi = sp.chisquare(np.array(heightDataFrame["<h(t)>"]), y12)
"""
print(linChi, curveChi)
#%%
print(firstOrderFit[0], secondOrderFit[0])
print(np.sqrt(np.diag(firstOrderFit[1])), np.sqrt(np.diag(secondOrderFit[1])))
"""
#%% Task 2f: standard dev scale with system size
xS = np.linspace(2, 256, 1000)
plt.scatter(systemSizes, np.array(heightDataFrame["sigma"]), color = "red")
sigmaFit = so.curve_fit(sigmaDist, systemSizes, heightDataFrame["sigma"], p0 = [0.1, 0.5])
plt.plot(xS, sigmaDist(xS, *sigmaFit[0]))
plt.xlabel(r"$L$")
plt.ylabel(r"$\sigma_h(L)$")
plt.savefig("standard_dev_L.svg", bbox_inches = "tight")
print(sigmaFit)
print(np.sqrt(np.diag(sigmaFit[1])))
#%% Task 2g: Height probability 
heightProbdfs = []
for i in range(len(systemSizes)):
    data = pd.read_csv(heightProbNames[i])
    heightProbdfs.append(data)
for i in range(len(systemSizes)):
    plt.scatter(heightProbdfs[i]["height"], heightProbdfs[i]["prob"], color = colours[i], marker = "x", label = systemSizes[i])
    plt.legend(fontsize = "x-small")
    plt.xlabel(r"$h$")
    plt.ylabel(r"$P(h;L)$")
    plt.savefig("height_prob_L.svg", bbox_inches = "tight")

#%% attempting a data collapse given an assumed normal distribution 
hPrimes = []
scaleProbs = []
for i in range(len(systemSizes)):
    hPrime = (np.array(heightProbdfs[i]["height"]) - np.array(heightDataFrame["<h(t)>"])[i])/np.array(heightDataFrame["sigma"])[i]
    scaleProb = np.array(heightProbdfs[i]["prob"])*np.array(heightDataFrame["sigma"])[i]
    hPrimes.append(hPrime)
    scaleProbs.append(scaleProb)
    
hPrimex = np.linspace(-3, 5, 500)
gaussY = gaussian(hPrimex, 0, 1, 1/np.sqrt(2*np.pi))
for i in range(len(systemSizes)):
    plt.scatter(hPrimes[i], scaleProbs[i], color = colours[i], label = systemSizes[i], marker = "x")
    #plt.plot(hPrimex, gaussY, linestyle = "dotted", color = "black")
    plt.ylabel(r"$\sigma_hP(h;L)$")
    plt.xlabel(r"$(h - \langle h \rangle)/\sigma_h$")
    plt.legend(fontsize = "x-small")
    #plt.savefig("height_prob_collapse_no_gauss.svg", bbox_inches = "tight")
    plt.show()
#%% testing the Gaussian 
lessZero = []
moreZero = []
for i in range(len(systemSizes)):
    smallProbs = scaleProbs[i][0:int(len(scaleProbs[i])/2)]
    largeProbs = scaleProbs[i][int(len(scaleProbs[i])/2):]
    lessZero.append(np.sum(smallProbs)/np.array(heightDataFrame["sigma"])[i])
    moreZero.append(np.sum(largeProbs)/np.array(heightDataFrame["sigma"])[i])
    
print([len(data) for data in hPrimes])
"""
if heightData[j] < 0:
            smallProbs.append(probData[j]/np.array(heightDataFrame["sigma"])[i])
        else:
            largeProbs.append(probData[j]/np.array(heightDataFrame["sigma"])[i])
"""

    
#%% running for 10000 past tc for task 3

avalanchedf = {}
heightdf = {}
piles = [heightPile4, heightPile8, heightPile16, heightPile32, heightPile64, heightPile128, heightPile256]
times = [20, 60, 250, 900, 3500, 14000, 56200]
for i in range(len(piles)):
    piles[i].reset()
    data = piles[i].startIteration(times[i] + 20000)
    avalanchedf[systemSizes[i]] = data[0]
    heightdf[systemSizes[i]] = data[2]
    
#%%
avalancheFrame = pd.DataFrame.from_dict(avalanchedf, orient = "index", dtype = int).transpose()
avalancheFrame.to_csv("20000_avalanches.csv")
heightFrame = pd.DataFrame.from_dict(heightdf, orient = "index", dtype = int).transpose()
heightFrame.to_csv("20000_heights.csv")
#%%
print(avalancheFrame)
    
#%% Task 3: reading in avalanche data and logbinning
times = [20, 60, 250, 900, 3500, 14000, 56200]
avalancheRawData = []
avalancheBinData = []
names = ["time", "4", "8", "16", "33", "64", "128", "256"]
crossOvers = [15, 52, 230, 912, 3388, 13970, 56092]
for i in range(len(systemSizes)):
    dfAvalanche = pd.read_csv("20000_avalanches.csv", names = ["time", "4", "8", "16", "33", "64", "128", "256"], skiprows = 1)
    data = np.array(dfAvalanche[names[i+1]])[times[i]:times[i]+20000]
    intData = np.array(dfAvalanche[names[i+1]], dtype = int)[times[i]:times[i]+20000]
    avalancheRawData.append(data)
    avalancheBinData.append(logbin(intData, scale = 1.2))
#%% plotting unbinned data
rawProbs = []
rawSizes = []
for i in range(len(systemSizes)):
    df = sizeProb(avalancheRawData[i])
    rawProbs.append(list(df.values()))
    rawSizes.append(list(df.keys()))
#%% plotting log-binned data
limits = [1e2, 1e3, 1e3, 1e4, 1e5, 1e5, 1e5]

for i in range(len(systemSizes)):
    plt.plot(avalancheBinData[i][0], avalancheBinData[i][1], label = systemSizes[i], color = colours[i])
    plt.xscale("log")
    plt.yscale("log")
    #plt.xlim(1, limits[i])
    #plt.scatter(rawSizes[i], rawProbs[i], label = r"Raw avalanche probability, $N = 10^4 $", color = "red")
    plt.xlabel(r"$s$")
    plt.ylabel(r"$P(s;L)$")
    #plt.title(systemSizes[i])
    plt.legend(fontsize = "x-small")
    #if i == 2:
    #    plt.savefig("raw_vs_binned.svg")
    plt.savefig("all_binned.svg")
    #plt.show()

#%% finding tau s by fitting to the linear region 
linearIndexes = [2, 11, -11, -14, -14, -13, -14]
linearBinData = []
for data in avalancheBinData:
    linearBinData.append((data[0][0:linearIndexes[i]], data[1][0:linearIndexes[i]]))
allLinearX = np.concatenate([data[0] for data in linearBinData])
allLinearY = np.concatenate([data[1] for data in linearBinData])
tauSFit = sp.linregress(np.log10(allLinearX), np.log10(allLinearY))
xFit = np.linspace(1, 10000, 500)
yFit = exponential(xFit, 10**tauSFit[1], tauSFit[0])
plt.scatter(allLinearX, allLinearY, marker = "x", color = "red")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$s$")
plt.ylabel(r"$P(s;L)$")
plt.plot(xFit, yFit) #yay it worked
plt.savefig("tauSFit.svg", bbox_inches = "tight")

#%% finding D and doing data collapse
dExponent = 1/(2+tauSFit[0])
for i in range(len(systemSizes)):
    rescaledProb = (avalancheBinData[i][0]**(-tauSFit[0]))*avalancheBinData[i][1]
    rescaledS = avalancheBinData[i][0]/(systemSizes[i]**dExponent)
    plt.loglog(rescaledS, rescaledProb, color = colours[i], label = systemSizes[i])
    plt.xlabel(r"$s / L^D$")
    plt.ylabel(r"$s^{\tau_s}P(s;L)$")
    plt.legend(fontsize = "x-small")
   # plt.savefig("av_prob_datacollapse.svg", bbox_inches = "tight")
    
#%% Task 3b: final one babeyyy
moments = [1, 2, 3, 4]
avMoments = []
for i in range(len(moments)):
    avMoment = []
    for j in range(len(systemSizes)):
        avMoment.append(calculateMoment(avalancheRawData[j], moments[i]))
    avMoments.append(avMoment)
    print(np.log10(avMoment))

#%% fitting to find D(1 + k - taus)
logSKFits = []
for i in range(len(avMoments)):
    fit = sp.linregress(np.log10(systemSizes), np.log10(avMoments[i]))
    logSKFits.append(fit)
    plt.scatter(systemSizes, avMoments[i], color = "red", label = r"$\langle s^k \rangle $")
    plt.xscale("log")
    plt.yscale("log")
    #plt.legend(fontsize = "x-small")
    y = exponential(systemSizes, 10**fit[1], fit[0])
    plt.plot(systemSizes, y, label = "Linear Fit")
    plt.xlabel(r"$L$")
    plt.ylabel(r"$\langle s^k \rangle $")
    if i == 2:
        plt.savefig("logL_logk_fit.svg")
    plt.show()
    
#%% plotting D(1 + k - taus) against k

gradients = [fit[0] for fit in logSKFits]
momentFit = sp.linregress(moments, gradients)
plt.scatter(moments, gradients, color = "red")
plt.plot(moments, linear(np.array(moments), momentFit[0], momentFit[1]))
plt.xlabel(r"$k$")
plt.ylabel(r"$D(1 \ + \ k \ - \tau_s)$")
plt.xticks([1, 2, 3,4])
plt.savefig("moment_fit.svg", bbox_inches = "tight")
#%%
dMoment = momentFit[0]
tauMoment = 1-(momentFit[1]/momentFit[0])
tauEst = -tauSFit[0]
#%%
print((momentFit[4]/momentFit[0])*tauMoment)
print(tauEst, dExponent, tauMoment, dMoment)
print(momentFit)
#%% redo data collapse with new values
for i in range(len(systemSizes)):
    rescaledProb = (avalancheBinData[i][0]**(tauMoment))*avalancheBinData[i][1]
    rescaledS = avalancheBinData[i][0]/(systemSizes[i]**dMoment)
    plt.loglog(rescaledS, rescaledProb, color = colours[i], label = systemSizes[i])
    plt.xlabel(r"$s / L^D$")
    plt.ylabel(r"$s^{\tau_s}P(s;L)$")
    plt.legend(fontsize = "x-small")
    #plt.savefig("av_prob_datacollapse_moment.svg", bbox_inches = "tight")
    plt.legend()
    
    
#%% making plots
#%% System size vs time 
#reading in data
totalsData = []
for i in range(len(systemSizes)):
    totalsData.append(pd.read_csv(heightDataNames[i][1], names = ["total"]))
#%%
#plotting
for i in range(len(systemSizes)):
    data = totalsData[i]
    plt.plot(data, color = colours[i], label = systemSizes[i])
    plt.legend(fontsize = "x-small")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$N^o \ grains \ in \ system$")
    #plt.savefig("number_grains_time.svg", bbox_inches = "tight")

#%% Height vs time
#reading in data
heightsData = []
for i in range(len(systemSizes)):
    heightsData.append(pd.read_csv(heightDataNames[i][2], names = ["total"]))
#%%
#plotting
for i in range(len(systemSizes)):
    data = heightsData[i]
    plt.plot(data, color = colours[i], label = systemSizes[i])
    plt.legend(fontsize = "x-small")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$h(t;L)$")
    plt.savefig("height_time.svg",  bbox_inches = "tight")


