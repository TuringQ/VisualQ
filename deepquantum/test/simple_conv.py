# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from deepquantum.circuit import Circuit
from deepquantum.utils import dag, encoding


class QuConv(nn.Module):
    """Quantum Conv layer with equalized learning rate and custom learning rate multiplier.
       放置5个量子门，也即有5个参数。
    """
    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()
        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(5), a=0.0, b=2 * np.pi) * init_std)
        self.n_qubits = n_qubits

    def qconv(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits, 2):
            cir.rx(which_q, w[0])
            cir.rx(which_q, w[1])
            cir.ryy(which_q, which_q + 1, w[2])
            cir.rz(which_q, w[3])
            cir.rz(which_q + 1, w[4])
        U = cir.get()
        return U

    def forward(self, x):
        E_qconv0 = self.qconv()
        qconv0_out = E_qconv0 @ x @ dag(E_qconv0)
        return qconv0_out


if __name__ == '__main__':
    conv = QuConv(2)
    a = torch.rand(4, 4)
    aq = encoding(a)
    u = conv.qconv()
    out = conv(aq)