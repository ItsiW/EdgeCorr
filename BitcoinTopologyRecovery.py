#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import random as rd
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import time
from scipy.stats import betaprime
from scipy.optimize import fsolve
from scipy import optimize
from progressbar import ProgressBar
from time import sleep
from functools import partial


# In[2]:


def DTPpmf(x,alpha,beta):
    return ((1/(1-(1/(beta + 1))**alpha))*(1/(x**alpha) - 1/((x+1)**alpha)))
    
def discTruncPareto(n,xc,c):
    # generate a discretised pareto distribution truncated at n 
    # that accounts for c proportion of blocks from xc proportion of nodes
    
    func = lambda alpha : (1-(1/(xc*n+1))**alpha) / (1-(1/(n+1))**alpha) - c

    # Use the numerical solver to find the roots

    alpha_initial_guess = 0.5
    alpha_solution = fsolve(func, alpha_initial_guess)
    alpha_solution
    
    # generate pmf
    pmf = np.zeros(n)
    for i in range(n):
        pmf[i] = DTPpmf(i+1,alpha_solution,n)
    return(pmf)


# In[3]:


def createGraph(degree,normal,testing,validation):
    # number of nodes
    n = normal
    # mean degree
    deg = degree


    # number of abberant nodes
    numtestnodes = testing
    numvalnodes = validation

    nodes = range(n)
    valnodes = range(n,n+numvalnodes)
    testnodes = range(n + numvalnodes,n+numvalnodes+numtestnodes)
    total = n+numvalnodes+numtestnodes

    # number of edges
    m = total*deg/2
    
    # normal nodes
    G=nx.gnm_random_graph(total,m)


    
    # parameters for each node's queuing and processing times
    expParameters = np.zeros(total)
    unifParameters = np.zeros(total)
    for i in range(total):
        # exponential delay parameter from Gamma(20,5)
        expParameters[i] = np.random.gamma(20,5)
        # uniform processing parameter from Pareto(1.25,20)
        unifParameters[i] = (np.random.pareto(1.25)+1)*20

    return(G,nx.to_dict_of_lists(G),expParameters,unifParameters)


# In[4]:


def cascade(n,total,pmf,delaypar,processpar,AdjDict):
    # create arrival times matrix
    label = np.arange(total)
    arrival = np.ones(total)*1000000
    measure = np.zeros(total)
    times = np.stack([label,arrival,measure],axis=1)
    # generate source node
    source = np.random.choice(np.arange(n),p = pmf)
    times[source,1] = 0
    # sort arrival times
    times = np.array(sorted(times, key=lambda x: x[1]))

    for i in range(total):
        node = times[i,0]
        basetime = times[i,1]
        peers = AdjDict[times[i,0]]
        # create measurement
        times[i,2] = basetime + np.random.exponential(delaypar[int(node)])
        # calculate arrival times at peers
        times = np.array(sorted(times, key=lambda x: x[0]))
        for j in peers:
            # ignore if arrival time for peer is earlier
            if times[j,1] > basetime:
                delaytime = np.random.exponential(delaypar[int(node)])
                processingtime = np.random.uniform(20,processpar[j])
                newarrivaltime = basetime + delaytime + processingtime
                # update arrival time
                if newarrivaltime < times[j,1]:
                    times[j,1] = newarrivaltime
        # sort times
        times = np.array(sorted(times, key=lambda x: x[1]))
    
    return(times, source)


# In[5]:


def multiThresholdOrder(corrmatrix,total):
    # generate order structure
    corrmatrixcopy = np.copy(corrmatrix)
    for i in range(total):
        corrmatrixcopy[i,i] = -1
    
    # create ordered links by correlation coefficient
    order1 = np.zeros([total, total, 3])
    order = np.zeros([total, total-1, 3])
    pbar = ProgressBar()
    for i in pbar(range(total)):
        for j in range(total):
            order1[i][j][0]=i
            order1[i][j][1]=j
            order1[i][j][2]=corrmatrixcopy[i,j]
        order1[i] = np.array(sorted(order1[i], key=lambda x: x[2], reverse = True))
        order[i] = order1[i][:-1]
    return(order)


# In[6]:


def maxcorr(corrmatrix,total):
    guess = {i:[] for i in range(total)}
    for i in range(total):
        corrmatrix[i,i]=0
        j = np.argmax(corrmatrix[i])
        if j not in guess[i] and j != i:
            guess[i].append(j)
            guess[j].append(i)
    return(guess)


# In[7]:


def oneThreshold(corrmatrix,total,meandegree,truth,n,v,t):
    tot = n+v+t
    possibleEdges = int((tot**2-tot)/2)

    order = np.zeros([possibleEdges,3])
    k = 0
    # convert correlation matrix into list of edges and correlations
    for i in range(tot - 1):
        for j in range(i+1,tot):
            order[k][0] = i
            order[k][1] = j
            order[k][2] = corrmatrix[i,j]
            k += 1

    # sort by correlation
    order = np.array(sorted(order, key=lambda x: x[2], reverse = True))

    # plot F1 value from k
    x = np.arange(0,min(2*total*meandegree,possibleEdges))
    y = [evaluateThresholdk(i,corrmatrix,order,truth,n,v,t)[1][2] for i in x]
    return(x,y,order)


# In[8]:


def globalCoordinateAscent(order,total,startingks,AdjDict):
    # initialise individual k values
    ks = startingks

    # a check to break the loop once there are no changes
    check = 1

    # record ascent
    y = []
    
    # break if there is no ascent for one loop
    while check == 1:
        check = 0
        pbar = ProgressBar()
        for i in pbar(range(total)):
            search = np.zeros(total)
            oldki = ks[i]
            for j in range(total):
                ks[i] = j
                search[j] = evaluateMultiThresholdk(ks,order,AdjDict)[1][2]
            newki = np.argmax(search)
            y.append(search[newki])
            ks[i] = newki
            if oldki != newki:
                check = 1

    bestguess, stats = evaluateMultiThresholdk(ks,order,AdjDict)
    return(bestguess, stats, y)

def localCoordinateAscent(radius, order, startingks, truth,n,v,t):
    # local coordinate ascent with a neighbourhood radius
    total = n+v+t
    # initialise individual k values
    ks = np.array(edges/total * np.ones(total), dtype='int')

    # a check to break the loop once there are no changes
    check = 1

    # record ascent
    y = []

    # break if there is no ascent for one loop
    while check == 1:

        check = 0
        pbar = ProgressBar()
        for i in pbar(range(total)):
            search = np.zeros(2*radius + 1)
            oldki = ks[i]
            for j in range(2*radius + 1):
                ks[i] = min(max(0,oldki + j - radius),total-1)
                search[j] = evaluateMultiThresholdk(ks,order,truth,n,v,t)[1][2]
            newki = min(max(0,np.argmax(search) + oldki - radius),total-1)
            y.append(search[newki-oldki + radius])
            ks[i] = newki
            if oldki != newki:
                check = 1


    bestguess, stats = evaluateMultiThresholdk(ks,order,truth,n,v,t)
    plt.plot(y)

    return(ks, bestguess, stats, y)


# In[9]:


# Simmulated Annealing implementation

def SA(iterations, meandegree, startingTemp, startingks, order, truth,n,v,t):
    tot = n + v + t
    probs = []
    costs = []
    T = startingTemp
    a = iterations/T
    ks = startingks
    cost = evaluateMultiThresholdk(ks,order,truth,n,v,t)[1][2]
    bestks = ks
    bestcost = cost
    pbar = ProgressBar()
    for i in pbar(range(iterations)):
        # generate neighbour
        changer = rd.sample(range(tot),1)
        distance = np.floor(np.random.uniform(-meandegree/2,meandegree/2+1))
        newks = np.copy(ks)
        newks[changer] = min(tot-1,max(0,ks[changer]+distance))
        newcost = evaluateMultiThresholdk(newks,order,truth,n,v,t)[1][2]
        # if better, new point is accepted
        if newcost >= cost:
            cost = newcost
            ks = newks
            costs.append(cost)
            if cost > bestcost:
                bestks = ks
                bestcost = cost
        # if worse, new point is accepted with some probability
        else:
            r = rd.uniform(0, 1)
            prob = np.exp((newcost - cost)/T)
            probs.append(prob)
            if r < prob:
                cost = newcost
                ks = newks
            costs.append(cost)
        T -= 1/a
    return(bestks,bestcost,costs,probs)   


# In[10]:


def evaluateUniversalk(k,corrmatrix):
    # for correlation threshold k, what is the graph and its statistic
    n = corrmatrix.shape[1]
    guess = {i:[] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if (corrmatrix[i,j] > k) and (i!= j) and (j not in guess[i]):
                guess[i].append(j)
                guess[j].append(i)
    return(guess, evaluateGuess(guess,AdjDict,total))
    
    
def evaluateThresholdk(k, corrmatrix, order,truth,n,v,t):
    # for k total edge inclusions, what is the graph and its statistics?
    total = corrmatrix.shape[1]
    guess = {i:[] for i in range(total)}
    for i in range(k):
        guess[order[i][0]].append(order[i][1])
        guess[order[i][1]].append(order[i][0])
    return(guess, evaluateGuessTesting(guess,truth,n,v,t))

def evaluateMultiThresholdk(ks,order,truth,n,v,t):
    # for ki edge inclusions for node i, what is the graph and its statistic
    guess = {i:[] for i in range(n+v+t)}
    for i in range(n+v+t):
        for j in range(ks[i]):
            if order[i][j][1] not in guess[order[i][j][0]]:
                guess[order[i][j][0]].append(order[i][j][1])
                guess[order[i][j][1]].append(order[i][j][0])
    return(guess, evaluateGuessTesting(guess,truth,n,v,t))


# In[11]:


def evaluateGuessTesting(guess,truth,n,v,t):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(n + v, n + v + t):
        for j in range(n+v+t):
            if j in guess[i] and j in truth[i]:
                TP += 1
            elif j in guess[i] and j not in truth[i]:
                FP += 1
            elif j not in guess[i] and j in truth[i]:
                FN += 1
            else :
                TN += 1
       

    recall = TP/(np.maximum(TP + FN,1))
    precision = TP/(np.maximum(TP + FP,1))
    F1 = 2*(precision*recall)/(np.maximum(precision+recall,1))
    return(recall,precision,F1)        


# In[12]:


def ValidateGuess(guess,validationtruth,n,v):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(n, n + v):
        for j in range(n+v):
            if j in guess[i] and j in validationtruth[i]:
                TP += 1
            elif j in guess[i] and j not in validationtruth[i]:
                FP += 1
            elif j not in guess[i] and j in validationtruth[i]:
                FN += 1
            else :
                TN += 1
       
    recall = TP/(np.maximum(TP + FN,1))
    precision = TP/(np.maximum(TP + FP,1))
    F1 = 2*(precision*recall)/(np.maximum(precision+recall,1))
    return(recall,precision,F1)  


# In[22]:


# cascades to perform
tests = 10000

# graph details
meandegree = 8
normal = 500
validation = 5
testing = 5

total = normal+testing+validation

n = normal
v = validation
t = testing
tot = n+v+t

# block generation distribution xc proportion of nodes accounts for c proportion of blocks
xc = 0.10
c = 0.50
pmf = np.array(discTruncPareto(normal,xc,c))

# generate Graph
Graph, AdjDict, delaypar, processpar = createGraph(meandegree,normal,testing,validation)

ValAdj = {k:AdjDict[k] for k in range(normal, normal + validation)}
TestAdj = {k:AdjDict[k] for k in range(normal + validation, total)}

edges = 0
for i in AdjDict:
    edges += len(AdjDict[i])
edges = edges/2


# In[23]:


# record cascade data
data = np.zeros([total,tests])
sources = np.zeros(tests)
pbar = ProgressBar()
for i in pbar(range(tests)):
    times, source = cascade(normal,total,pmf,delaypar,processpar,AdjDict)
    sources[i] = source
    times = np.array(sorted(times, key=lambda x: x[0]))
    newdata = times[:,2]
    data[:,i] = newdata

# create correlation matrix
corrmatrix = np.corrcoef(data)


# In[24]:


# MaxCorr
guess1 = maxcorr(corrmatrix,total)
recall1,precision1,F11 = evaluateGuessTesting(guess1,TestAdj,n,v,t)
print("Individual Maximum Correlation method")
print("F1 score: " + str(F11))


# In[25]:


# k Threshold
ks1, F1s1, order1 = oneThreshold(corrmatrix,total,meandegree,TestAdj,n,v,t)

print("Total k edges correlation threshold method")
plt.plot(ks1,F1s1)
plt.xlabel('edges')
plt.ylabel('F1 score')
plt.savefig('Threshold_learning_blend.png')
print("Maximum at k = " + str(np.argmax(F1s1)))
print("F1 = " + str(np.max(F1s1)))


guess2 = evaluateThresholdk(np.argmax(F1s1),corrmatrix,order1,TestAdj,n,v,t)[0]


# In[26]:


order = multiThresholdOrder(corrmatrix, tot)


print("Individual k thresholds local coordinate descent")
ks = np.array(meandegree*np.ones(tot), dtype='int')
bestks1, guess3, stats, F1s3 = localCoordinateAscent(3,order,ks,TestAdj,n,v,t)
print("F1 = " + str(stats[2]))
plt.plot(F1s3)
plt.xlabel('iterations')
plt.ylabel('F1 score')
plt.savefig('localAscent_learning_blend.png')


# In[27]:


print(evaluateGuessTesting(guess3,TestAdj,n,v,t))
print()


# In[37]:


order = multiThresholdOrder(corrmatrix, tot)
ks = np.array(edges/(tot)*np.ones(tot), dtype='int')
bestks2, F1new, costs, probs = SA(10000, edges/tot, 0.012, ks, order, TestAdj,n,v,t)
plt.plot(costs)
plt.xlabel('iterations')
plt.ylabel('F1 score')
plt.savefig('SA_learning_blend.png')
guess4 = evaluateMultiThresholdk(bestks2,order,TestAdj,n,v,t)[0]
print("F1 = " + str(F1new))


# In[35]:


plt.plot(probs)
plt.xlabel('iterations')
plt.ylabel('acceptance probability')
plt.savefig('SA_probs_learning_blend.png')


# In[36]:


# recall, precision, F1


print(np.round(100*np.array(evaluateGuessTesting(guess1,TestAdj,n,v,t)),1))
print(np.round(100*np.array(evaluateGuessTesting(guess2,TestAdj,n,v,t)),1))
print(np.round(100*np.array(evaluateGuessTesting(guess3,TestAdj,n,v,t)),1))
print(np.round(100*np.array(evaluateGuessTesting(guess4,TestAdj,n,v,t)),1))

print(np.round(100*np.array(ValidateGuess(guess1,ValAdj,n,v)),1))
print(np.round(100*np.array(ValidateGuess(guess2,ValAdj,n,v)),1))
print(np.round(100*np.array(ValidateGuess(guess3,ValAdj,n,v)),1))
print(np.round(100*np.array(ValidateGuess(guess4,ValAdj,n,v)),1))


# In[31]:


bestks2

