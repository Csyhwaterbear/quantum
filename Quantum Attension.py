# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 16:12:02 2024
Quantum Attension
@author: yinghao chensheng
"""

print("\033[H\033[J",end = "")

import os
import numpy as np
import random
import math

import torch
from torch import nn

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

class SelfAttention(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.scale = math.sqrt(ni)
        self.norm  = nn.BatchNorm2d(ni) 
        self.qkv   = nn.Linear(ni, ni * 3)
        self.proj  = nn.Linear(ni, ni)
        
    def forward(self,x):
        inp = x
        n,c,h,w = x.shape
        x = self.norm(x).view(n, c, -1).transpose(1,2)
        q, k, v = torch.chunk(self.qkv(x), 3, dim = -1)
        s = (q@k.transpose(1,2))/self.scale
        x = s.softmax(dim = -1)@v
        x = self.proj(x).transpose(1,2).reshape(n,c,h,w)
        return x + inp
    
class QuantumSelfAttension(nn.Module):
    def __init__(self, num_inp):
        super().__init__()
        self.scale = math.sqrt(num_inp)
        self.norm  = nn.BatchNorm2d(num_inp) 
        self.q     = self.qnn(num_inp)
        self.k     = self.qnn(num_inp)
        self.v     = self.qnn(num_inp)
        self.proj  = nn.Linear(num_inp, num_inp)
        
    def qnn(self, num_inp):
        observable = []
        for I in range(num_inp):
            observable.append(
                SparsePauliOp.from_sparse_list(
                    [("Z", [I], 1)],
                    num_inp
                    )
                )
        
        feature_map = ZFeatureMap(num_inp)
        ansatz = RealAmplitudes(num_inp)
        qc = QuantumCircuit(num_inp)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        
        qnn = EstimatorQNN(
            circuit=qc,
            observables = observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,
        )
        qnn_torch = TorchConnector(
            neural_network = qnn,
            initial_weights = algorithm_globals.random.random(qnn.num_weights)
            )
        return qnn_torch
    
    def forward(self, x):
        inp = x
        q = self.q(x).reshape((-1,1))
        k = self.k(x).reshape((1,-1))
        v = self.v(x).reshape((-1,1))
        s = q@k@v
        return x + s.reshape(1,-1)
        
num = 8
QSA = QuantumSelfAttension(num)
x = x = torch.randn(1,num) 
print(x)
print(QSA(x))