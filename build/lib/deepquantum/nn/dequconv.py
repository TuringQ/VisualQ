from deepquantum.circuit import Circuit
from deepquantum.utils import dag
import torch
import torch.nn as nn
import numpy as np

class DeQuConvXYZ(nn.Module):
    """
    Quantum Conv layer.
    """

    def __init__(self, n_qubits,
                 gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()
        assert n_qubits%2 == 0, \
            "nqubits应该为2的倍数"
        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(15), a=0.0, b=2 * np.pi) * init_std)
        self.n_qubits = n_qubits

    def qconv0(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits, 2):
            cir.rx(which_q, w[0])
            cir.rx(which_q+1, w[1])
            cir.ry(which_q, w[2])
            cir.ry(which_q+1, w[3])
            cir.rz(which_q, w[4])
            cir.rz(which_q+1, w[5])
            cir.rxx(which_q, which_q + 1, w[6])
            cir.ryy(which_q, which_q + 1, w[7])
            cir.rzz(which_q, which_q + 1, w[8])
            cir.rx(which_q, w[9])
            cir.rx(which_q+1, w[10])
            cir.ry(which_q, w[11])
            cir.ry(which_q+1, w[12])
            cir.rz(which_q, w[13])
            cir.rz(which_q+1, w[14])
        U = cir.get()
        U = dag(U)
        return U

    def forward(self, x):
        E_qconv0 = self.qconv0()
        qconv0_out = dag(E_qconv0) @ x @ E_qconv0
        return qconv0_out


class DeQuConvSXZ(nn.Module):
    """
    Simple Quantum Conv layer.
    """

    def __init__(self, n_qubits,
                 gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()
        assert n_qubits%2 == 0, \
            "nqubits应该为2的倍数"
        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(5), a=0.0, b=2 * np.pi) * init_std)
        self.n_qubits = n_qubits

    def qconv0(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits, 2):
            cir.rx(which_q, w[0])
            cir.rx(which_q+1, w[1])
            cir.ryy(which_q, which_q + 1, w[2])
            cir.rz(which_q, w[3])
            cir.rz(which_q+1, w[4])
        U = cir.get()
        U = dag(U)
        return U

    def forward(self, x):
        E_qconv0 = self.qconv0()
        qconv0_out = dag(E_qconv0) @ x @ E_qconv0
        return qconv0_out