import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


from deepquantum import Circuit
from deepquantum.utils import dag


class Policy(nn.Module):
    """Quantum Conv layer with equalized learning rate and custom learning rate multiplier.
       放置n个量子门，也即有x个参数。
    """
    def __init__(self, n_qubits=3, layer=3, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight1 = nn.Parameter(nn.init.uniform_(torch.empty(14*n_qubits*layer), a=0.0, b=2*np.pi))
        self.theta1 = nn.Parameter(torch.ones(14*n_qubits*layer))
        self.n_qubits = n_qubits
        self.layer = layer
        self.q1 = nn.Linear(2**self.n_qubits, 14)

        self.saved_log_probs = []
        self.rewards = []

        # # 输入状态向量的数组
        # c = np.random.randint(0,5,size=[3472,14])
        #
        # self.c = c

    # decision circuit to replace neural network. We use 4 qubits and get a density matrix of 16x16.
    def layers(self, x):
        # w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)   # circuit list]
        x = x
        # place the gate

        for k in range(self.layer):
            for i in range(self.n_qubits):
                cir.rz(i, self.weight1[0 + 14 * i + 14*self.n_qubits * k] * x[0] + self.theta1[0 + 14 * i + 14*self.n_qubits * k])
                cir.ry(i, self.weight1[1 + 14 * i + 14*self.n_qubits * k] * x[1] + self.theta1[1 + 14 * i + 14*self.n_qubits * k])
                cir.rz(i, self.weight1[2 + 14 * i + 14*self.n_qubits * k] * x[2] + self.theta1[2 + 14 * i + 14*self.n_qubits * k])

                cir.rz(i, self.weight1[3 + 14 * i + 14*self.n_qubits * k] * x[3] + self.theta1[3 + 14 * i + 14*self.n_qubits * k])
                cir.ry(i, self.weight1[4 + 14 * i + 14*self.n_qubits * k] * x[4] + self.theta1[4 + 14 * i + 14*self.n_qubits * k])
                cir.rz(i, self.weight1[5 + 14 * i + 14*self.n_qubits * k] * x[5] + self.theta1[5 + 14 * i + 14*self.n_qubits * k])

                cir.rz(i, self.weight1[6 + 14 * i + 14*self.n_qubits * k] * x[6] + self.theta1[6 + 14 * i + 14*self.n_qubits * k])
                cir.ry(i, self.weight1[7 + 14 * i + 14*self.n_qubits * k] * x[7] + self.theta1[7 + 14 * i + 14*self.n_qubits * k])
                cir.rz(i, self.weight1[8 + 14 * i + 14*self.n_qubits * k] * x[8] + self.theta1[8 + 14 * i + 14*self.n_qubits * k])

                cir.rz(i, self.weight1[9+14*i+14*self.n_qubits*k]*x[9]+self.theta1[9+14*i+14*self.n_qubits*k])
                cir.ry(i, self.weight1[10+14*i+14*self.n_qubits*k]*x[10]+self.theta1[10+14*i+14*self.n_qubits*k])
                cir.rz(i, self.weight1[11+14*i+14*self.n_qubits*k]*x[11]+self.theta1[11+14*i+14*self.n_qubits*k])

                cir.rz(i, self.weight1[12+14*i+14*self.n_qubits*k]*x[12]+self.theta1[12+14*i+14*self.n_qubits*k])
                cir.ry(i, self.weight1[13+14*i+14*self.n_qubits*k]*x[13]+self.theta1[13+14*i+14*self.n_qubits*k])
                cir.rz(i, torch.tensor(0))
            if k != self.layer - 1:
                if self.n_qubits > 1:
                    for j in range(self.n_qubits-1):
                        cir.Hcz(j, j+1)
                if self.n_qubits > 2:
                    cir.Hcz(self.n_qubits-1, 0)
                # cir.Hcz(0, 1)
                # cir.Hcz(1, 2)
                # cir.Hcz(2, 0)
        U = cir.get()
        state = U @ cir.state_init
        return state.reshape(-1,1)

    # Calculate the output of the Quantum unitary transformation. Contrast the module length of diagonal elements of density matrix to choose the action
    def forward(self, x):
        temp = x
        output = self.layers(temp)
        output1 = dag(output)
        q = (output1[0] * output.reshape(1, int(2**self.n_qubits))[0]).real
        q = self.q1(q)
        return F.softmax(q, dim=0)


# policy = Policy()
# scripted_module=torch.jit.script(policy)
# print(scripted_module.code)