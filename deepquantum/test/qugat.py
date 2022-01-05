import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from deepquantum import Circuit
from deepquantum.utils import dag, ptrace,multi_kron,encoding, Gram, measure
from typing import List
import math


class init_cir(nn.Module):
    # 初始化U_query
    def __init__(self, n_qubits=2,
                 gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(
            nn.init.uniform_(torch.empty(n_qubits * 3), a=0.0, b=2 * np.pi) * init_std)  # theta_size=5

        self.n_qubits = n_qubits

    def layer(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits):
            cir.rx(which_q, w[which_q * 3 + 0])
            cir.ry(which_q, w[which_q * 3 + 1])
            cir.rz(which_q, w[which_q * 3 + 2])
        # ring cnot gates
        for which_q in range(0, self.n_qubits - 1):
            cir.cnot(which_q, which_q + 1)
        if self.n_qubits > 1:
            cir.cnot(self.n_qubits - 1, 0)
        return cir.get()

    def forward(self, x):
        E_out = self.layer()
        out = E_out @ x @ dag(E_out)
        return out


class QuGAT(nn.Module):
    def __init__(self, n_qubits):
        super(QuGAT, self).__init__()
        self.n_qubits = n_qubits
        self.w_cir = init_cir(n_qubits=n_qubits)
        self.a_cir = init_cir(n_qubits=n_qubits*2)

    def forward(self, state_list: List[torch.Tensor], adj):
        w_s = []
        output = []
        n = len(state_list)
        init_a = torch.zeros(n,n)
        zero_vec = -9e15*torch.ones_like(init_a)
        for state in state_list:
            w_state = self.w_cir(state)
            w_s.append(w_state)
        for i in range(n):
            for j in range(n):
                temp = torch.kron(w_s[i], w_s[j])
                a_state = self.a_cir(temp)
                e = measure(a_state, self.n_qubits*2)
                e = e.mean()
                init_a[i][j] = e
        attention = torch.where(adj > 0, init_a, zero_vec)
        attention = F.softmax(attention, dim=1)
        for i in range(n):
            temp = attention[i][0]*w_s[0]
            for j in range(1,n):
                temp = temp + attention[i][j]*w_s[j]
            output.append(temp)
        return output
