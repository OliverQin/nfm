#/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Mon May  9 18:42:13 2016

@author: oliver
"""

import numpy as np
import theano

import theano.tensor as T
import time

import scipy.optimize

theano.config.floatX = 'float32'
theano.config.optimizer = 'fast_run'
theano.config.nvcc.fastmath = True

class NonNegativeDecomposition:
    def __init__(self, mat, K):
        self.matShape = matShape = mat.shape
        assert len(self.matShape) == 2
        self.K = K
        
        matAInit = ( np.random.rand(matShape[0], K) ).astype(theano.config.floatX)
        self.matA = theano.shared( matAInit )
        
        matxInit = ( np.random.rand(K, matShape[1]) ).astype(theano.config.floatX)
        self.matx = theano.shared( matxInit )
        
        self.b = T.fmatrix('b')
        
        self.innerCost = self.cost(self.b)
        self.setTrainingSet(mat)
        
    def L2(self):
        return T.sum( self.matx**2 ) + T.sum( self.matA**2 )

    def L1(self):
        return T.sum( abs(self.matx) ) + T.sum( abs(self.matA) )
        
    def cost(self, b):
        return T.sum((T.dot(self.matA, self.matx) - b)**2) + self.L2() * 1e-3
    
    def setTrainingSet(self, b):
        if hasattr(self, 'bb'):
            self.bb.set_value(b)
        else:
            self.bb = theano.shared(b, borrow=True)

        self.optCoreFunc = theano.function(
            inputs=[],
            outputs=self.innerCost,
            givens={
                self.b: self.bb
            }
        )
        
        self.grad_all = T.grad(cost=self.innerCost, wrt=[self.matA, self.matx] )

        self.grad_all_func  = theano.function([], self.grad_all, givens={self.b: self.bb})

    def optSetPara(self, x):
        x = x.astype(np.float32)
        
        a1 = x[:self.matShape[0] * self.K].reshape( self.matShape[0], self.K )
        a2 = x[self.matShape[0] * self.K:].reshape( self.K, self.matShape[1] )
        a = [ a1, a2 ]
        self.setParameters( a )
        
    def optFunc(self, x):
        self.optSetPara(x)
        
        #bp = time.time()        
        a = self.optCoreFunc()
        #ep = time.time()
        #print 'Seconds: %.2f' % (ep-bp), '\t\tCost: %.7f' % a
        return np.float64(a)
        
    def optFuncPrime(self, x):
        self.optSetPara(x)
        ret = map( lambda x: x.flatten(), self.grad_all_func( ) )
        return np.concatenate(ret).astype(np.float64)

    def callCost(self, b):
        tt = theano.function([self.b], self.innerCost)
        return tt(b)
        
    def callL2(self):
        tt = theano.function([], self.L2())
        return tt()
        
    def getParameters(self):             
        return map(lambda x:x.get_value(), [self.matA, self.matx])
    
    def setParameters(self, params):
        self.matA.set_value(params[0])
        self.matx.set_value(params[1])
        return 
        
    def runOpt(self, m=40, maxiter=500):
        x0, f, y = scipy.optimize.fmin_l_bfgs_b( self.optFunc, np.random.rand(self.matShape[0]*self.K+self.K*self.matShape[1]), self.optFuncPrime, m=m, pgtol=1e-5, maxiter=maxiter, bounds=[(0, None)] * (self.matShape[0]*self.K+self.K*self.matShape[1]) )
        return self.getParameters()
        
if __name__ == '__main__':
    A = ( np.random.rand(100, 2)).astype(np.float32)
    x = ( np.random.rand(2, 2000)).astype(np.float32)
    
    b = np.dot(A, x) + np.random.randn(100, 2000).astype(np.float32) * 0.5
    
    dcp = NonNegativeDecomposition(b, 2)
    
    nA, nx = dcp.runOpt()
    
    print 'RMSE: ', ((np.dot(nA, nx) - b)**2).mean() ** 0.5
    #print nndcp.getParameters()
    
    
    