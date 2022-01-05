import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from deepquantum import Circuit
from deepquantum.utils import dag, ptrace,multi_kron,encoding, Gram
from typing import List


def measure_state(state, M):
    # type: (Tensor, Tensor) -> Tensor
    if len(state.shape) != 2:  # state必须是二维张量，即便只有1个态矢也要view成(n,1)
        raise ValueError("state必须是二维张量,即便batch只有1个态矢也要view成(n,1)")
    else:  # state为batch_size个态矢，即二维张量
        m1 = (dag(state) @ M @ state)
        rst = torch.diag(m1).view(-1, 1)  # 取对角元变成1维张量，在被view成2维张量
        rst = rst.real
        return rst


def measure(state,n_qubits):
    # type: (Tensor, int) -> Tensor
    cir=Circuit(n_qubits)
    for i in range(n_qubits):
        cir.z_gate(i)
    m=cir.get()
    return measure_state(state,m)


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


def cal_query_key(queryQ_out, keyQ_out, dim_q, dim_k):
    # type: (Tensor, Tensor, int, int) -> Tensor
    """queryQ_out: type torch.Tensor
       keyQ_out: torch.Tensor
    """
    """计算query与key的interaction score

    """
    out = torch.kron(queryQ_out, keyQ_out)
    n_qubits = dim_q + dim_k

    # 对称cnot门
    cir = Circuit(n_qubits)
    for t in range(0, dim_k, 1):
        cir.cnot(t, n_qubits - dim_k + t)
    for t in range(dim_k - 1, -1, -1):
        cir.cnot(n_qubits - dim_k + t, t)
    U = cir.get()

    out = U @ out @ dag(U)

    quantum_score = measure(out, n_qubits)

    return quantum_score


def cal_src_value(quantum_src, valueQ_out, dim_v):
    # type: (Tensor, Tensor, int) -> Tensor
    """input torch.Tensor
    """
    """计算经过attention score加权作用后的value
    """
    src = quantum_src.mean()
    # src=(src+1.0)/2.0     #[-1,1] -> [0,1]
    # phi=(src-0.5)*2*np.pi #phi=[-pi,pi]
    phi = src * np.pi  # phi=[-pi,pi]

    # rx-ringCnot-ry-RingCnot-rz
    cir = Circuit(dim_v)
    for i in range(dim_v):
        cir.rx(i, phi * 0.5)

    for which_q in range(0, dim_v - 1):
        cir.cnot(which_q, which_q + 1)
    if dim_v > 1:
        cir.cnot(dim_v - 1, 0)

    for i in range(dim_v):
        cir.ry(i, phi * 0.5)

    for which_q in range(0, dim_v - 1):
        cir.cnot(which_q, which_q + 1)
    if dim_v > 1:
        cir.cnot(dim_v - 1, 0)

    for i in range(dim_v):
        cir.rz(i, phi)
    U = cir.get()

    quantum_weighted_value = U @ valueQ_out @ dag(U)

    return quantum_weighted_value


def cal_output(qwv_list, dim):
    # type: (List[torch.Tensor], int) -> Tensor
    """计算weighted values的“和”（通过多个cnot门将信息融合）
    """
    # out = multi_kron(qwv_list)
    # n_qubits=2*dim
    cir = Circuit(2 * dim)
    for t in range(dim):
        cir.cnot(t, dim + t)
    U = cir.get()

    # 为避免线路上比特数过多，两个两个处理
    attnQ = qwv_list[-1]
    for i in range(len(qwv_list) - 1):
        out = torch.kron(qwv_list[i], attnQ)
        out = U @ out @ dag(U)
        attnQ = ptrace(out, dim, dim)

    return attnQ


class QuAttention(nn.Module):
    def __init__(self, n_qubits):
        super(QuAttention, self).__init__()
        self.n_qubits = n_qubits
        self.init_q = init_cir(n_qubits=n_qubits)
        self.init_k = init_cir(n_qubits=n_qubits)
        self.init_v = init_cir(n_qubits=n_qubits)

    def forward(self, state_list: List[torch.Tensor]):
        qs, ks, vs = [], [], []
        output = []
        for state in state_list:
            q = self.init_q(state)
            k = self.init_k(state)
            v = self.init_v(state)
            qs.append(q)
            ks.append(k)
            vs.append(v)
        for q in range(len(qs)):
            qwv_list = []
            for k in range(len(ks)):
                score = cal_query_key(qs[q], ks[k], self.n_qubits, self.n_qubits)
                s_value = cal_src_value(score, vs[k], self.n_qubits)
                qwv_list.append(s_value)
            output.append(cal_output(qwv_list, self.n_qubits))
        return output

# """
# temp = torch.rand(200)
# data = temp.reshape(1,-1) # 输入数据拉长为一维向量
# qubits = 2 # 设定比特数
# k = int(2**qubits) # 设定切片维度
# state_list = []
# for i in range(0, data.shape[1], k):
#     state = data[0,i:i+k].reshape(1,-1)
#     x = Gram(state)
#     state_list.append(x)
# quattention = QuAttention(qubits)
# output = quattention(state_list)
# quattention = QuAttention(qubits)
# print(torch.jit.script(quattention).code)
