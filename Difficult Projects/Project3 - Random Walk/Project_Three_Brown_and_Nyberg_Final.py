#Haleigh Brown and Douglas Nyberg 5/9/2023 PROJECT 3 PROBLEM 2 SELF-AVOIDING RANDOM WALK
########################################################
# This code models the average length of a polymer chain made up of N monomers by conducting many 
# self-avoiding walks (SAW's). These SAW's are not completely random as the walkers used are
# not allowed to walk anywhere they have already been. Later in this code weighting is also
# implemented to demonstrate how SAW's could be programmed to last longer. 
########################################################

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
from matplotlib.animation import FuncAnimation, PillowWriter



def walk(N):
    #this size of array allows for the max dis of N to be traveled in any direction 
    history = np.zeros((4*N, 4*N), dtype=int)
   
    #creating x and y arrays with their first values set to 0
    x = [0]
    y = [0]

    #marking that (0,0) has been visited
    history[2*N, 2*N] = 1
    
    #taking a step upwards and marking it as visited
    x.append(0)
    y.append(1)
    history[2*N+1, 2*N] = 1

    #initial direction
    pastchoice = (0, 1)  
    choices = []
    for i in range(2, N+1):

        #loop for finding viable step choices 
        for i in range (4):
            if pastchoice == (1,0):
                 choices = [(1, 0), (0, 1), (0, -1)]
            elif pastchoice == (-1,0):
                choices = [(-1, 0), (0, 1), (0, -1)]
            elif pastchoice == (0,1):
                choices = [(1, 0), (-1, 0), (0, 1)]
            else:
                choices = [(1, 0), (-1, 0), (0, -1)] 
            
        dx, dy = random.choice(choices)
        
        #if the step the polymer chain is trying to take has been visited
        #in the past then the for loop breaks
        if history[2*N+y[-1]+dy, 2*N+x[-1]+dx] == 1:
            break

        #otherwise we assign our step's history to 1 to signify that we have been here 
        else:
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)

            history[2*N+y[-1], 2*N+x[-1]] = 1
            
        pastchoice = (dx, dy)  
        choices.clear()

    #calc the mean squared end-to-end distance
    dis = x[-1] ** 2 + y[-1] ** 2
    
    #return all of the x and y points the polymer chain has visited before either
    #completing N steps or visiting a site that it has already been
    return x, y, dis



walks = 10000  #number of walks we try to conduct
f = 10000     #variable used to keep track of how many polymer chain's make it to N steps
n = 4         #number of steps the polymer chain is trying to complete

mean_squared_list = []  #list for keeping track of the mean square end-to-end distances
                        #for differing n values

fs = []     #array for storing number of successes for each n that we look at
N = [] #array for storing all the values of n we look at

#this loop runs while it is still reasonable for us to expect some fraction of
#polymer chain's at a certain n to actually reach n  instead of running into themselves
while f > 10:
    #store what n we are looking at
    N.append(n)
    
    #start the count of sucessful chains at zero
    f = 0
    #variable for keeping track of the sum
    dis_sum = 0


    #loop for conducting walks
    for i in range(walks):
        x, y, dis = walk(n)
    
        #store it 
        dis_sum += dis
        
        #if the walk was a success (AKA we took exactly n steps)
        if len(x) - 1 == n:
            #count the success
            f += 1
            
    #store fraction of successes
    fs.append(f/walks)

    
    #calcuating the mean squared end-to-end distance by taking avg of our results for all
    #walks in this run
    mean_squared = dis_sum/walks

    #storing our value
    mean_squared_list.append(mean_squared)

    #adding it to our plot
    plt.scatter(n, mean_squared, label=n)
    plt.xlabel("N")
    plt.ylabel("Mean Squared End-to-End Distance")
    plt.title("Mean Squared End-to-End Distance as a Function of N")

    #incrementing n 
    n = n + 1
    


#printing and showing results
plt.show()
print("The last 'resonable' n that we should consider is: ", n)
print("for at this value the fraction of sucessful walks is only: ", str(f) + "/" + str(walks))

plt.plot(N, fs, label=n)
plt.xlabel("N")
plt.ylabel("Fraction of Sucesses")
plt.title("Fraction of Sucessful Chain Creation vs. N")
plt.show()










#function for performing 2D random walk to attempt to creat a polymer chain of length N using
#Rosenbluth and Rosenbluth method
def RRwalk(N):
    #same exact intial process as earlier function
    history = np.zeros((4*N, 4*N), dtype=int)
    x = [0]
    y = [0]
    history[2*N, 2*N] = 1
    x.append(0)
    y.append(1)
    history[2*N+1, 2*N] = 1

    #defining intial weighting to be 1 
    W = 1

    #array for storing possible steps the chain can take in order to avoid visiting sites
    #it has already been 
    choices = []

    #loop for taking steps
    for i in range(2, N):
        #variables for finding and storing the point values surrounding the chains current location
        x0 = x[i-1]; xp = x0 + 1; xm = x0 - 1
        y0 = y[i-1]; yp = y0 + 1; ym = y0 - 1

        #variable for storing how many possible steps the chain can consider
        numPossible = 0


        #all of these if loops check if a point is vible (hasn't been visited before)
        #if the particular step is not possible that option is ignored
        #if it is, however, that particular step is then appended to our choices array
        #and we count that option by incrementing imPossible
        
        if history[2*N+y0, 2*N+xp] == 0:  # A step right is okay
            numPossible += 1; choices.append((1, 0))
            
        if history[2*N+y0, 2*N+xm] == 0:  # A step left is okay
            numPossible += 1; choices.append((-1, 0))
            
        if history[2*N+yp, 2*N+x0] == 0:  # A step up is okay
            numPossible += 1; choices.append((0, 1))
            
        if history[2*N+ym, 2*N+x0] == 0:  # A step down is okay
            numPossible += 1; choices.append((0, -1))


        #no possible choices-terminate the walk and set W = 0
        if numPossible == 0:  
            W = 0.0
            break;

        #one possible choice set W(N) = 1/3*W(N-1)
        elif numPossible == 1: 
            W *= 1/3.0

        #two possible choices set W(N) = 2/3*W(N-1)
        elif numPossible == 2: 
            W *= 2.0/3.0

        #three possible choices keep W(N) as is  
        else:
            W = W

        #finally take one of the possible steps and store the change in x and y
        dx, dy = random.choice(choices)
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)

        #indicat that our new location has now been visited
        history[2*N+y[i], 2*N+x[i]] = 1

        #reset our possible choices 
        choices.clear()
        
    #return x abd y history along with our final weighting W
    return x, y, W




#this is a small function that helps make the curve fit for our v estimate
def power(n, a, b):
    return a*n**(2*b)




#number of walks
walks = 1000

#list of step sizes we want to consider
N = np.array([4, 8, 16, 32])

#array for storing weighted mean squared end-to-end distances for each value in N
ws = []


#loop for running through our N list
for n in N:
    #value to keep track the sum of the weighted means for each walk
    weightedMean = 0
    #value to keep track the sum of the weightings for each walk
    wSum = 0

    #conducting walks AKA trials
    for i in range(walks):
        x, y, W = RRwalk(n)

        #adding W*end-to-end mean dis and W to our current summations
        weightedMean += W*(x[-1]**2+y[-1]**2)
        wSum += W
    #storing end value for weighted mean of R**2(N)
    ws.append(weightedMean/wSum)

#creating the curve fit 
params, cov = curve_fit(power, N, ws)
a, b = params

#array for storing power values at each value of n in N
pows = []

#populating pows
for n in N:
     pows.append(power(n, a, b))


#ploting both the basic mean weighted squared end-to-end distances and fiting a power lot curve to the data
plt.scatter(N, ws, label="Mean Weighted Squared End-to-End Dis's")
plt.plot(N, pows, label="Power Fit")
plt.xlabel("N")
plt.ylabel("Mean Weighted Squared End-to-End Dis")
plt.title("Mean Weighted Squared End-to-End Dis vs. N")
plt.legend()
plt.show()


#printing our estimate for v
print("Estimate for v:", b)




#making the movie of RRwalk working for N 
N = 50
def plot(frame):
    x, y, W = RRwalk(N)
    plt.plot(x, y, linewidth=0.8)
    plt.xlim(-N, N)
    plt.ylim(-N, N)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f'Frame {frame + 1}')


fig = plt.figure()
ani = FuncAnimation(fig, plot, frames=walks)
writer = PillowWriter(fps=30)
ani.save("RRWalkN10.gif", writer=writer)




