import numpy as np
from pylab import *
from copy import deepcopy
from config import *
from torch import randperm


def MARO(dim,max_it, npop,trainX, testX, trainy, testy , pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):

    best_acc = best_acc_init
    best_cols=best_cols_init
    pop_pos = pop_pos_init
    pop_fit = pop_fit_init
    best_f = best_fit_init
    best_x = best_pop_init
    his_best_fit = [best_fit_init]
    allCols=[]
    allFitList=[]
    repuduce=[]
    for i in range(npop):
        allFitList.append([])
    for it in range(max_it-1):
        print('MARO,Itration : '+str(roound) +'-'+str(it)+'  Fitness: '+str(best_f)+'  Acc: '+str(best_acc)+
              '  NumF: '+str(len(best_cols))+'  Features: '+str(best_cols))

        direct1=np.zeros((npop, dim))
        direct2=np.zeros((npop, dim))
        theta = 2 * (1 - (it+1) / max_it)
        k=0
        for i in range(npop):
            newPopPos=[]
            z=0
            while (getFeatures(newPopPos) in allCols):
                if z>0:
                    print("*****************************************")
                L = (np.e - np.exp((((it+1) - 1) / max_it) ** 2)) * (np.sin(2 * np.pi * np.random.rand())) # Eq.(3)
                rd = np.floor(np.random.rand() * (dim))
                rand_dim = randperm(dim)
                direct1[i, rand_dim[:int(rd)]] = 1
                c = direct1[i,:]  #Eq.(4)
                R = L * c # Eq.(2)
                A = 2 * np.log(1 / np.random.rand()) * theta #Eq.(15)
                if A>1:
                    K=np.r_[0:i,i+1:npop]
                    RandInd=jIndexRecognizer(it,max_it,allFitList)#(K[np.random.randint(0,npop-1)])#
                    newPopPos = pop_pos[RandInd, :] + R * (pop_pos[i, :] - pop_pos[RandInd, :])+round(0.5 * (0.05 +np.random.rand())) * np.random.randn() # Eq.(1)
                else:
                    ttt=int(np.floor(np.random.rand() * dim))
                    # ttt=jIndexRecognizer(it,max_it,allFitList)
                    direct2[i, ttt] = 1
                    gr = direct2[i,:] #Eq.(12)
                    H = ((max_it - (it+1) + 1) / max_it) * np.random.randn() # % Eq.(8)
                    b = pop_pos[i,:]+H * gr * pop_pos[i,:] # % Eq.(13)
                    newPopPos = pop_pos[i,:]+ R* (np.random.rand() * b - pop_pos[i,:]) #Eq.(11)
                z+=1
            k=k+z-1
               

                #newPopPos = space_bound(newPopPos, ub, lb)
            newPopFit,tempacc,tempcols = Fit_KNN(newPopPos,trainX, testX, trainy, testy)
            allCols.append(tempcols)
         
            if newPopFit < pop_fit[i]:
                pop_fit[i] = newPopFit
                pop_pos[i, :] = newPopPos
                allFitList[i].append(newPopFit)

            if pop_fit[i] < best_f:
                best_f = pop_fit[i]
                best_x = pop_pos[i, :]
                best_acc=tempacc
                best_cols=tempcols
        repuduce.append(k)
        his_best_fit.append(best_f.tolist())
    repuduce_save(repuduce)
    return best_x, best_f, his_best_fit, best_acc,best_cols

def getFeatures(x):
    cols=[]
    for i in range(len(x)):
        zz=sigmoid(x[i])
        if zz>=np.random.rand():
            cols.append(i)
    return (cols)

def sigmoid2(gamma):
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))
    
def sumRank(posIndex,RKList,CS,it):
    rank=0
    for index in range(len(posIndex)): #
        k=CS[index]+1
        k1=sigmoid2(RKList[posIndex[index]])
        k2=sigmoid2(k)
        rank+=(2*it/100*k1)+(100/(it+1)*k2)
        # rank+=k


    kt=[i for i, e in enumerate(posIndex) if e != 0]
    if len(kt)<2:
        return 10000
    rank/=len(kt)
    return rank
def jIndexRecognizer(it,maxIt,allFitList):
    if (it/maxIt)<np.random.rand():
        min=10000
        out=0
        for i in range(len(allFitList)):
            if min>len(allFitList[i]):
                min=len(allFitList[i])
                out=i
        return out
    else:
        for i in range(len(allFitList)):
            min=100000
            out=0
            if min>sum(allFitList[i])/len(allFitList):
                min=sum(allFitList[i])/len(allFitList)
                out=i
        return out
def repuduce_save(r):
    f=open("repuduce.txt", "a")
    f.write(str(r))
    f.write('\n')
    f.close()
