from deepquantum import Circuit
from deepquantum.utils import dag
import torch
import torch.nn as nn
import numpy as np

class DeQuPoolXYZ(nn.Module):
    """Quantum Pool layer.
       放置4个量子门，2个参数。
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
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(6), a=0.0, b=2 * np.pi) * init_std)
        self.n_qubits = n_qubits
    def qpool(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits, 2):
            cir.rx(which_q, w[0])
            cir.rx(which_q + 1, w[1])
            cir.ry(which_q, w[2])
            cir.ry(which_q + 1, w[3])
            cir.rz(which_q, w[4])
            cir.rz(which_q + 1, w[5])
            cir.cnot(which_q, which_q + 1)
            cir.rz(which_q + 1, -w[5])
            cir.ry(which_q + 1, -w[3])
            cir.rx(which_q + 1, -w[1])
        U = cir.get()
        U = dag(U)
        return U
    def forward(self, x):
        E_qpool = self.qpool()
        qpool_out = E_qpool @ x @ dag(E_qpool)
        return qpool_out

class DeQuPoolSX(nn.Module):
    """Quantum Pool layer.
       放置4个量子门，2个参数。
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
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(2), a=0.0, b=2 * np.pi) * init_std)
        self.n_qubits = n_qubits
    def qpool(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits, 2):
            cir.rx(which_q, w[0])
            cir.rx(which_q + 1, w[1])
            cir.cnot(which_q, which_q + 1)
            cir.rx(which_q + 1, -w[1])
        U = cir.get()
        U = dag(U)
        return U
    def forward(self, x):
        E_qpool = self.qpool()
        qpool_out = E_qpool @ x @ dag(E_qpool)
        return qpool_out