# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:22:11 2023

@author: lizzi
"""

import numpy as np
import matplotlib.pyplot as plt

class RicePile:
    
    def __init__(self, length, probability = 0.5):
        
        if type(length) != int:
            raise TypeError("length argument must be an integer")
        self._continue = True
        self._length = length
        self._heights = np.zeros(length)
        self._slopes = np.zeros(length)
        self._thresholdSlopes = np.random.choice([1, 2], length, [probability, 1-probability])
        self._probability = probability
        
        
    def reset(self):
        length = self.getLength()
        self._heights = np.zeros(length)
        self._slopes = np.zeros(length)
        self._thresholdSlopes = np.random.choice([1, 2], length, [self._probability, 1-self._probability])
        
        
    def getHeights(self):
        
        return self._heights
    
    def getSlopes(self):
        
        return self._slopes
    
    def getThresholdSlopes(self):
        
        return self._thresholdSlopes
    
    def getLength(self):
        
        return self._length
    
    def pileSize(self):
        
        return np.sum(self.getHeights())
    
    def pileHeights(self):
        
        return np.sum(self.getSlopes())
    
    def calcSlopes(self):
        
        slopes = []
        
        for i in range(self.getLength()-1):
            slopes.append(self.getHeights()[i] - self.getHeights()[i+1])
        slopes.append(self.getHeights()[-1])
        
        return np.array(slopes)
    
    def visualise(self, max_height):
        
        heights = self.getHeights()
        sites = list(range(0, self.getLength()))
        plt.bar(sites, heights, width = 0.8, label = "Total in system = {total}".format(total = int(self.pileSize())))
        plt.ylim(0, max_height)
        plt.legend()
        plt.xlim(-0.4, self.getLength())        
                
    def relax(self, location):
        
        """
        implements the changes in slope and height that occur when a given site relaxes
        
        """
        
        if type(location) != int:
            raise TypeError("location index must be an integer")
        
        if location == 0:
            
            self._slopes[0] = self._slopes[0] - 2
            self._slopes[1] = self._slopes[1] + 1
            self._heights[0] = self._heights[0] - 1
            self._heights[1] = self._heights[1] + 1
            
        elif location == self.getLength() - 1:
            
            self._slopes[location] = self._slopes[location] - 1
            self._slopes[location - 1] = self._slopes[location - 1] + 1
            self._heights[location] = self._heights[location] - 1
            
        else:
            
            self._slopes[location] = self._slopes[location] - 2
            self._slopes[location + 1] = self._slopes[location + 1] + 1
            self._slopes[location - 1] = self._slopes[location - 1] + 1
            self._heights[location] = self._heights[location] - 1
            self._heights[location + 1] = self._heights[location + 1] + 1
            
              
    def drive(self):
        
        self._heights[0] = self._heights[0] + 1
        self._slopes[0] = self._slopes[0] + 1
        
    
    def updateThresholdSlope(self, location):
        
        if type(location) != int:
            raise TypeError("location index must be an integer")
        
        self._thresholdSlopes[location] = np.random.choice([1, 2], p = [self._probability, 1 - self._probability])
        #print(self._thresholdSlopes[location])
        
    def aboveThreshold(self):
        
        checks = []
        slopes = self.getSlopes()
        thresholds = self.getThresholdSlopes()
        
        for i in range(self.getLength()):
            
            if slopes[i] > thresholds[i]:
                checks.append(True)
                
            else:
                checks.append(False)
        
        return np.array(checks)
    
    def startIteration(self, bagSize, crossOver = False):
        
        if type(bagSize) != int:
            raise TypeError("Only integer number of rice grains please")
        
        grainsAdded = 0
        grainsOut = 0
        avalancheDict = {}
        avalancheList = []
        systemTotals = [0]
        heightTracker = [0.0]
        crossOverTime = []

        while grainsAdded <= bagSize:
            if self._continue == False:
                print("iteration stopped")
                break
            self.drive()
            #print(self.getHeights(), self.getSlopes(), "drivin")
            grainsAdded += 1
            avalancheSize = 0
            while self.aboveThreshold().any() == True:
                site = np.where(self.aboveThreshold() == True)[0][0]
                if site == self.getLength() - 1:
                    grainsOut += 1
                    crossOverTime.append(self.pileSize() - 1)
                    if crossOver == True:
                        return self.pileSize() - 1
                    else:
                        pass
                self.relax(int(site)) 
                #print(self.getHeights(), self.getSlopes())
                self.updateThresholdSlope(int(site))
                avalancheSize += 1
                
            if avalancheSize in avalancheDict.keys():
                avalancheDict[avalancheSize] += 1
            else:
                avalancheDict[avalancheSize] = 1
            
            #if self.pileSize() < grainsAdded:
            #    heightTracker.append(self.getHeights()[0]) #starts recording after steady state reached
                
            avalancheList.append(avalancheSize)
            systemTotals.append(self.pileSize())
            heightTracker.append(self.getHeights()[0]) #records all heights
            print(self.pileSize(), grainsAdded, self.getHeights()[0])
            
                
        print("run out of rice!")
        
        return  np.array(avalancheList), np.array(systemTotals), np.array(heightTracker), np.array(crossOverTime)
                
            
        
        
            
                
        