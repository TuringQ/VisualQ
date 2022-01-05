import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from deepquantum import Circuit
from deepquantum.utils import dag,measure_state,ptrace,multi_kron,encoding,expecval_ZI,measure


# 量子线路模块（Encoder,Decoder,Discriminator):
class QuEn(nn.Module):
    """
    根据量子线路图摆放旋转门以及受控门
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

        self.n_qubits = n_qubits

        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(3 * self.n_qubits), a=0.0, b=2 * np.pi) * init_std)

    def layer(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)

        # 旋转门
        for which_q in range(0, self.n_qubits):
            cir.rx(which_q, w[which_q])
            cir.ry(which_q, w[which_q + 6])
            cir.rz(which_q, w[which_q + 12])

        # cnot门
        for which_q in range(1, self.n_qubits):
            cir.cnot(which_q - 1, which_q)

        # 旋转门
        for which_q in range(0, self.n_qubits):
            cir.rx(which_q, -w[which_q])
            cir.ry(which_q, -w[which_q + 6])
            cir.rz(which_q, -w[which_q + 12])

        U = cir.get()
        return U

    def forward(self, x):
        E_qlayer = self.layer()

        qdecoder_out = E_qlayer @ x @ dag(E_qlayer)

        return qdecoder_out


class Q_Encoder(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()

        # 6比特的编码线路
        self.n_qubits = n_qubits

        self.encoder = QuEn(n_qubits)

    def forward(self, molecular, dimA):
        # type: (torch.Tensor, int) ->torch.Tensor
        x = molecular

        x = self.encoder(x)

        dimB = self.n_qubits - dimA

        # 偏迹运算
        x_out = ptrace(x, dimA, dimB)

        return x_out


class QuDe(nn.Module):
    """
    根据量子线路图摆放旋转门以及受控门
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

        self.n_qubits = n_qubits

        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(3 * self.n_qubits), a=0.0, b=2 * np.pi) * init_std)

    def layer(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        # print(self.n_qubits)

        # 旋转门
        for which_q in range(0, self.n_qubits):
            cir.rx(which_q, w[which_q])
            cir.ry(which_q, w[which_q + 10])
            cir.rz(which_q, w[which_q + 20])

        # cnot门
        for which_q in range(1, self.n_qubits):
            cir.cnot(which_q - 1, which_q)

        # 旋转门
        for which_q in range(0, self.n_qubits):
            cir.rx(which_q, -w[which_q])
            cir.ry(which_q, -w[which_q + 10])
            cir.rz(which_q, -w[which_q + 20])

        U = cir.get()
        return U

    def forward(self, x):
        E_qlayer = self.layer()
        qdecoder_out = E_qlayer @ x @ dag(E_qlayer)
        return qdecoder_out


class Q_Decoder(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()

        # 10比特量子解码器
        self.n_qubits = n_qubits
        self.decoder = QuDe(n_qubits)

    def forward(self, molecular, gene, dimA):
        # type: (torch.Tensor, torch.Tensor, int) ->torch.Tensor
        m = molecular
        g = gene

        x = torch.kron(m, g)

        x = self.decoder(x)

        dimB = self.n_qubits - dimA

        # 偏迹运算
        x_out = ptrace(x, dimA, dimB)

        return x_out


class QuDis(nn.Module):
    """
    根据量子线路图摆放旋转门以及受控门
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

        self.n_qubits = n_qubits

        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(3 * self.n_qubits), a=0.0, b=2 * np.pi) * init_std)

    def layer(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)

        # 旋转门
        for which_q in range(0, self.n_qubits):
            cir.rx(which_q, w[which_q])
            cir.ry(which_q, w[which_q + 4])
            cir.rz(which_q, w[which_q + 8])

        # cnot门
        for which_q in range(1, self.n_qubits):
            cir.cnot(which_q - 1, which_q)
        cir.cnot(self.n_qubits - 2, self.n_qubits-1)

        U = cir.get()
        return U

    def forward(self, x):
        E_qlayer = self.layer()
        qdiscriminator = E_qlayer @ x @ dag(E_qlayer)
        # 测量
        qdiscriminator_out = measure(qdiscriminator, self.n_qubits)

        return qdiscriminator_out


class Q_Discriminator(nn.Module):
    def __init__(self, n_qubit):
        super().__init__()

        # 4比特量子判别器
        self.n_qubit = n_qubit
        self.discriminator = QuDis(self.n_qubit)

    def forward(self, molecular):
        x = molecular

        x_out = self.discriminator(x)

        return x_out


# QE=Q_Encoder(6)
# scripted_qe=torch.jit.script(QE)
# print(scripted_qe.code)
#
# QD=Q_Decoder(4)
# scripted_de =torch.jit.script(QD)
# print(scripted_de.code)
#
# QDis=Q_Discriminator(10)
# scripted_dis=torch.jit.script(QDis)
# print(scripted_dis.code)