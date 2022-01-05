from deepquantum.utils import ptrace
import torch
import torch.nn as nn
from .quconv import QuConvXYZ
from .qupool import QuPoolXYZ
from .dequconv import DeQuConvXYZ
from .dequpool import DeQuPoolXYZ


class Qu_AEnet(nn.Module):
    def __init__(self, n_qubits):
        super(Qu_AEnet, self).__init__()
        self.qconv1 = QuConvXYZ(n_qubits)
        self.pool = QuPoolXYZ(n_qubits)
        self.depool = DeQuPoolXYZ(n_qubits)
        self.deqconv = DeQuConvXYZ(n_qubits)
    def forward(self, x, y, n, n_trash):
        # type: (tensor, tensor, int, int) -> tensor
        x = self.qconv1(x)
        x = self.pool(x)
        x = ptrace(x, n, n_trash)
        de_input = torch.kron(x, y)
        de_out = self.depool(de_input)
        de_out = self.deqconv(de_out)
        return de_out