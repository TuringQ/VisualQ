import math
from deepquantum.circuit import Circuit
import torch
import torch.nn as nn
import numpy as np


class QuLinear(nn.Module):
    """
    This layer can be used to replace pytorch's nn.Linear(in_features, out_features)
    except that the batch_size can only be one, 
    out_features must be 2**1 <= 2**i <= 2**in_features
    in_features must be greater than 2

    args:
      in_features
      out_features 

    input: tensor of shape (1, in_features) 
    output: tensor of shape (1, out_features) 
    
    """
    def __init__(self, in_features, out_features, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.n_para = in_features*3
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(self.n_para), a=0.0, b=2 * np.pi) * init_std)

        self.n_qubits = in_features
        self.n_part = int(math.log(out_features, 2)) # 2**n_part=out_features

        self.cir = Circuit(self.n_qubits)
        


    def get_zero_state(self, n_qubits):
        """
        returns:
            |0⟩, the lowest computatinal basis state for a n qubits circuit
        """
        zero_state = torch.zeros(2**n_qubits, dtype=torch.cfloat)
        zero_state[0] = 1.+0j
        return zero_state


    def partial_measurements(self, final_state, n_qubits, n_part):
        """
        https://quantum.country/teleportation#background_partial_measurement
        
        args:
            final_state: a statevector of dimension 2**n_qubits
            n_qubits: the number of qubits of the circuit
            n_part:  the first n_part qubits to be meausred
        returns:
            a vector of prbabilities of dimension 2**n_part
        
        Example:
        ```
         final_state = tensor([0.5000+0.j, 0.0000+0.j, 0.5000+0.j,
        0.0000+0.j, 0.5000+0.j, 0.0000+0.j, 0.5000+0.j, 0.0000+0.j])
         n_qubits = 3
         n_part = 2
         t = partial_measurements(final_state, n_qubits, n_part)
        tensor([0.2500+0.j, 0.2500+0.j, 0.2500+0.j, 0.2500+0.j])
        ```

        each element in final_state is the probability amplitude of measuring all qubits with outputs 
        '000', '001', '010', '011', '100', '101', '110', '111' respectively.

        each element in t is the probability of measuring first two qubits with outputs 
        '00', '01', '10', '11' respectively.
        """


        # Generate all possible bitstrings of outputs, 2**n_qubits
        outputs_decimal = list(range(2**n_qubits))
        format_spec = '{0:0'+ str(n_qubits) + 'b}'
        outputs_binary =  [format_spec.format(x) for x in outputs_decimal]

        # Generate subset of bitstrings of outputs, 2**n_part)
        outputs_decimal_part = list(range(2**n_part))
        format_spec = '{0:0'+ str(n_part) + 'b}'
        outputs_binary_part =  [format_spec.format(x) for x in outputs_decimal_part]

        # ouput vector: the probabilities distribution of first  n_part qubits measurement
        t = torch.zeros(2**n_part, dtype=torch.cfloat)
        
        # This loop could take exponential time when n_qubits becomes large
        for i, x in enumerate(outputs_binary_part):
          for j, y in enumerate(outputs_binary):
            if y.startswith(x):
              t[i] += torch.square(torch.abs(final_state[j]))

        return t.real


    def encoding_layer(self, data):
        for which_q in range(0, self.n_qubits, 1):
            self.cir.Hadamard(which_q)
            self.cir.ry(which_q, torch.arctan(data[which_q]))
            self.cir.rz(which_q, torch.arctan(torch.square(data[which_q])))


    def variational_layer(self):
        w = self.weight * self.w_mul

        
        for which_q in range(0, self.n_qubits, 1):
            self.cir.cnot(which_q, (which_q+1) % self.n_qubits)
            self.cir.cnot(which_q, (which_q+2) % self.n_qubits)


        for which_q in range(0, self.n_qubits, 1):
            self.cir.rx(which_q, w[3*which_q+0])
            self.cir.rz(which_q, w[3*which_q+1])
            self.cir.rx(which_q, w[3*which_q+2])



    def forward(self, x):
        self.cir.clear()
        zero_state = self.get_zero_state(self.n_qubits) 
        
        
        x = x.clone().detach()
        x = torch.squeeze(x)
        self.encoding_layer(x)

        self.variational_layer()
        U = self.cir.get()
  
        final_state = U @ zero_state

        hidden = self.partial_measurements(final_state, self.n_qubits, self.n_part)

        return hidden.unsqueeze(0)
        


class QuGRUCell(nn.Module):

    """
    自定义QGRU单元, 其中经典线性变换被量子线性变换替代。

    参数:
    input_size: 输入x的特征数
    hidden_size: 隐藏状态的特征数

    输入:
    `x`: 张量形状是(N, input_size)， 输入矢量， 代表一个词语或字母的数字化表示
    `h_prev`: 张量形状是 (N, hidden_size)， 隐藏状态矢量， 代表之前输入到模型信息的数字化表示。
    其中N是Batch Size，用于把多个序列中同时间步的输入进行批量计算。

    输出:
    `h_new`: 张量形状是 (N, hidden_size)，隐藏状态矢量，代表考虑到当前输入后，
    隐藏状态的更新。
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # nn.Linear(in_features, out_features) => QLinear(in_features, out_features)
        self.linear_x_r = QuLinear(input_size, hidden_size)   # change
        self.linear_x_u = QuLinear(input_size, hidden_size)   # change
        self.linear_x_n = QuLinear(input_size, hidden_size)   # change
        self.linear_h_r = QuLinear(hidden_size, hidden_size)  # change
        self.linear_h_u = QuLinear(hidden_size, hidden_size)  # change
        self.linear_h_n = QuLinear(hidden_size, hidden_size)  # change

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)  


    def forward(self, x, h_prev):

        x_r = self.linear_x_r(x)
        x_u = self.linear_x_u(x)
        x_n = self.linear_x_n(x)
        h_r = self.linear_h_r(h_prev)
        h_u = self.linear_h_u(h_prev)
        h_n = self.linear_h_n(h_prev)

        resetgate = torch.sigmoid(x_r + h_r)
        updategate = torch.sigmoid(x_u + h_u)
        newgate = torch.tanh(x_n + (resetgate * h_n))
        h_new = newgate - updategate * newgate + updategate * h_prev

        return h_new





class QuGRU(nn.Module):
    """
    args:
        input_dim,  the size of embedding vector, i.e. embedding_dim
        hidden_dim, the size of GRU's hidden vector
        output_dim, the size of predicted vector 

    input:
        tensor of shape (batch_size, seq_length, input_dim)

        e.g. input_dim is 

    output:
        tensor of shape (batch_size, output_dim)


    """
    def __init__(self, input_dim, hidden_dim, output_dim, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.qgru_cell = QuGRUCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        outputs = []

        # RNN 循环
        # 初始化隐藏状态为零矢量
        h = torch.zeros(x.size(0), self.hidden_dim)
        for seq in range(x.size(1)):
            h = self.qgru_cell(x[:,seq,:], h) 
            outputs.append(h)

        output = outputs[-1]
        output = self.fc(output) 

        return output 







