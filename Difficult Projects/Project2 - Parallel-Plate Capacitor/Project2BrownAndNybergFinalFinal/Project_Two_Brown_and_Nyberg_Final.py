#Haleigh Brown and Douglas Nyberg 4/12/2023 PROJECT 2 PROBLEM 1 Electric Field of a Parallel-Plate Capacitor
########################################################
# This code aims to calculate the electric potential in a 2D box-like region with grounded sides 
# (V=0) and parallel plates held at opposite potentials (V=-1 and V=+1) inside it. The code 
# showcases the differences between using the basic Jacobi method and simultaneous 
# over-relaxation (SOR) techniques for approaching this problem. Moreover, this code employs 
# finite difference methods to compute the electric field within the box region. 
#
# 
# Definition of variables: 
# ep = epsilon AKA the value we are using to detect convergence
# phiprime = array that is used to hold the most recently calculated potentials 
# phi = array that is used to hold calculated potentials from one iteration ago
# 
# current_error = value that stores our programs current error between iterations (starts with default value of 1)
# iterations = value that stores the number of iterations that are completed before convergence is detected
# l = metaphorical length of our 2D box that is used for finding the proper placing of the parallel-plates in our box
#     (derived from our number of grid points)
#
# x = converted value (derived from i in the nested for loop) that represents our x position relative to our 2D grid space 
# y = converted value (derived from j in the nested for loop) that represents our y position relative to our 2D grid space 
#
# weights = np.array that stores what weight values we consider when looking for the optimal SOR weighting
# iterChangingW = python dictionary that stores our potential optimal weight values as keys with the number of iterations  
#                 using them leads to as their values
# 
# n or m = number of grid points we want in our 2D grid (defulat value is 100)
# potentials = resulting potentials found from running our functions when they detect convergence and return phiprime
# iterationsTaken = the resulting number of iterations our function took to find "potentials"
# optimalW = the optimal weighting that we find (identified through using all weights in "weights" and finding the value
#            that leads to the lowest number of iterations           
#
# nValues = similar to "weights" this is a list that stores various that we  that stores the results of using the values contained in nValues and
#                             a weighting 0.0 AKA just using basic Jacobi
#
# itersChangingNSOR = python list like "itersChangingNBasicJacobi" but stores values obtianed with using optimalW
#
# even and odd = bolean values used to figure out which checkerboard we are looking at in our GS method
# start1 and start2 = related to even and odd values these variables store where we are starting our checkerboard in our GS method
########################################################

import numpy as np
import matplotlib.pyplot as plt


#This is our main function that can be utilized to find convergence using basic Jacobi (by feeding this function w = 0.0)
#or SOR methods (by using w != 0.0)
def Jacobi(M):
    #initializing pertinent values 
    ep = 1e-6     
    phi = np.zeros([M+1,M+1],float)            
    phiprime = np.zeros([M+1,M+1],float)       
    current_error = 1.0
    iterations = 0
    l = M/100
    
    #this while loop runs until convergence criteria is met 
    while current_error > ep:

        #These two for loops are used to calculate new values of the potential everywhere in our 2D grid
        for i in range(1, M):        
            for j in range(1, M):    

                x = j*0.01
                y = i*0.01
                 
                if x == round(l/3, 2) and y >= l/4 and y <= (l - l/4):
                    phiprime[i,j] = -1
                
                elif x == 2*round(l/3, 2) and y >= l/4 and y <= (l - l/4):
                    phiprime[i,j] = 1
                
                else:
                    phiprime[i,j] = (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])/4 

        #Calculate maximum difference from old potentials and current ones to see if we have converged yet
        current_error = np.max(abs(phi-phiprime))
        
        #Swap the two arrays around and count the iteration as complete
        phi,phiprime = phiprime,phi
        iterations += 1

    #returning the resulting potentials and how long it took to get them
    return phiprime, iterations


def sorGS(w, M, d):
    #initializing pertinent values 
    ep = 1e-6
    phi = np.zeros([M+1,M+1],float)            
    phiprime = np.zeros([M+1,M+1],float)       
    current_error = 1.0
    iterations = 0
    l = M/100

    #values for detecting which "checkerboard" we should look at 
    even = True 
    odd = False

    #this while loop runs until convergence criteria is met 
    while current_error > ep:

        #these if statements determine where we should start looking at alternate cells (AKA which checkerboard we are looking at)
        if even:
            start1 = 1
            start2 = 2
        if odd:
            start1 = 2
            start2 = 1

            
        #these nested for loops insure that we don't skip any rows but we skip every other column 
        for i in range(1, M):        
            for j in range(start1, M, 2):    
                x = j*0.01
                y = i*0.01
    
                #once we have landed on an appropiate grid point we then update its values using the known
                #potential of the plates or by calculating it using phi (our list of past potentials)
                if x == round(l/3+ d*0.01, 2) and y >= l/4 and y <= (l - l/4):
                    phiprime[i,j] = -1
                
                elif x == 2*round(l/3- d*0.01, 2)  and y >= l/4 and y <= (l - l/4):
                    phiprime[i,j] = 1
                
                else:
                    phiprime[i,j] = (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])/4 


        #these nested for loops insure that we don't skip any rows but we skip every other column inversely when compared
        #to the loops above
        for i in range(1, M):        
            for j in range(start2, M, 2):    
                x = j*0.01
                y = i*0.01
    
                #once we have landed on an appropiate grid point we then update its values using the known
                #potential of the plates or by calculating it using the values that we just got from the for 
                #loops above!
                if x == round(l/3+ d*0.01, 2) and y >= l/4 and y <= (l - l/4):
                    phiprime[i,j] = -1
                
                elif x == 2*round(l/3- d*0.01, 2)  and y >= l/4 and y <= (l - l/4):
                    phiprime[i,j] = 1
     
                else:
                    phiprime[i,j] = (phiprime[i+1,j] + phiprime[i-1,j] + phiprime[i,j+1] + phiprime[i,j-1])/4


        #we can then finally find weighted phiprime using our un-weighted phiprime and phi
        phiprime = w*phiprime + (1-w)*phi

        #we can also find the current error between our new potentials and our past to see if convergence has occured
        current_error = np.max(abs(phi-phiprime))
        
        #swapping arrays and counting a full iteration
        phi,phiprime = phiprime,phi
        iterations += 1
        
        #these statements swap the even and odd boolean values so that the next iteration of this while loop
        #pulls the opposing checkerboard as the current iteration first 
        if even:
            even = False
            odd = True
        else: 
            odd = False
            even = True

    #returning the resulting potentials and how long it took to get them
    return phiprime, iterations 



#testing different weights ranging form 1 to 1.5 
weights = np.linspace(1.0, 1.5, 10)
iterChangingW = {}
n = 100   

#running through the weights we want to try
for w in weights:
    print(w)
    potentials, iterationsTaken = sorGS(w, n, 0)
    iterChangingW[w] = iterationsTaken
    #this is code that can be uncommented to demonstrate what our potentials look like for our nxn grid
    #plt.imshow(potentials, origin = "lower")
    #plt.show()

#printing the optimal weight
optimalW = min(iterChangingW, key=iterChangingW.get)
print("The optimal weighting is: ", optimalW)



#testing different numbers of grid points
nValues = [50, 100, 200, 400]
itersChangingNBasicJacobi = []
itersChangingNSOR = []

for n in nValues:
    print(n)
    potentials, iterationsTakenBasic = Jacobi(n)
    itersChangingNBasicJacobi.append(iterationsTakenBasic)

    potentials, iterationsTakenSOR = sorGS(optimalW, n, 0)
    itersChangingNSOR.append(iterationsTakenSOR)

#ploting 
plt.scatter(nValues, itersChangingNBasicJacobi, label = "scaling of the basic Jacobi convergence")
plt.scatter(nValues, itersChangingNSOR, label = "scaling of the optimal SOR convergence")
plt.xlabel("Number of Grid Points")
plt.ylabel("Number of Iterations")
plt.title("Scaling of Convergence Methods")
plt.legend()
plt.show()


m = 50 #number of grid points (50 is utilized here for speed)
distances = [0, 1, 2, 3] #distances we want to look at

#running through our distances
for d in distances: 
    #finding our potentials
    optimalV, iters = sorGS(optimalW, m, d)
    
    #preparing to plot
    x = np.linspace(0, m/100, m+1)
    y = np.linspace(0, m/100, m+1)
    
    #finding E field using finite difference
    Ex = np.zeros([m+1,m+1],float)
    Ey = np.zeros([m+1,m+1],float)
    Ex[1:-1, 1:-1] = (-1)*(optimalV[1:-1, 2:] - optimalV[1:-1, :-2])/(2*1/m)
    Ey[1:-1, 1:-1] = (-1)*(optimalV[2:, 1:-1] - optimalV[:-2, 1:-1])/(2*1/m)

    #ploting
    fig, ax = plt.subplots()

    ax.quiver(x, y, Ex.reshape(m+1, m+1), Ey.reshape(m+1, m+1))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("E Field When Plates are " + str(round((.5 - 2*(2*round(m/300+ d*0.01, 2) - round(m/300+ d*0.01, 2))), 2)) + " apart")
    plt.show()
