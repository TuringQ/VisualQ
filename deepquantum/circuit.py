# -*- coding: utf-8 -*-
import torch
from deepquantum.gates import *
# from deepquantum.utils import *


class Circuit(object):
    def __init__(self, n):
        self.U = None
        self.n_qubits = n  # 总QuBit的个数
        self.gate_list = []  # 顺序保存门结构
        self.u = []    # 顺序保存酉矩阵
        self.M_lst = []  # 保存测量算子矩阵

        #线路的初始态，默认全为|0>态
        self.state_init = torch.zeros(2**self.nqubits)
        self.state_init[0] = 1
        self.state_init = self.state_init + 0j

#   内置函数 添加分拆资源
    def _add_gate(self, gate_name: str, target_qubit, gate_params):
        """add gate and its feature to the circuit by sequence.

        """
        # assert gate_name in list[]  #todo 创建一个可信池子
        self.gate_list.append({'gate': gate_name, 'theta': gate_params, 'which_qubit': target_qubit})

    def _add_u(self, u_matrix):
        """add u_matrix to the circuit by sequence.

        """
        # assert u_name in list[]  #todo 创建一个可信池子
        self.u.append(u_matrix)

#   实例调用部分可使用的函数 添加门、run线路
    def rx(self, target_qubit, phi):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('rx', target_qubit, phi)
        if type(phi) == float or type(phi) == int:
            phi = torch.tensor(phi)
            self._add_u(gate_expand_1toN(rx(phi), self.n_qubits, target_qubit))
        else:
            self._add_u(gate_expand_1toN(rx(phi), self.n_qubits, target_qubit))

    def ry(self, target_qubit, phi):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('ry', target_qubit, phi)
        if type(phi) == float or type(phi) == int:
            phi = torch.tensor(phi)
            self._add_u(gate_expand_1toN(ry(phi), self.n_qubits, target_qubit))
        else:
            self._add_u(gate_expand_1toN(ry(phi), self.n_qubits, target_qubit))

    def rz(self, target_qubit, phi):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('rz', target_qubit, phi)
        if type(phi) == float or type(phi) == int:
            phi = torch.tensor(phi)
            self._add_u(gate_expand_1toN(rz(phi), self.n_qubits, target_qubit))
        else:
            self._add_u(gate_expand_1toN(rz(phi), self.n_qubits, target_qubit))

    def cnot(self, control_qubit: int, target_qubit: int):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert isinstance(control_qubit, int), \
            "control qubit is not integer"
        assert control_qubit <= self.n_qubits
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('cnot', control_qubit, target_qubit)
        self._add_u(two_qubit_control_gate(x_gate(), self.n_qubits, control_qubit, target_qubit))

    def x_gate(self, target_qubit):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('X', target_qubit, None)
        self._add_u(gate_expand_1toN(x_gate(), self.n_qubits, target_qubit))

    def y_gate(self, target_qubit):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('Y', target_qubit, None)
        self._add_u(gate_expand_1toN(y_gate(), self.n_qubits, target_qubit))

    def z_gate(self, target_qubit):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('Z', target_qubit, None)
        self._add_u(gate_expand_1toN(z_gate(), self.n_qubits, target_qubit))

    def Hcz(self, control_qubit, target_qubit):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert isinstance(control_qubit, int), \
            "control qubit is not integer"
        assert control_qubit <= self.n_qubits
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"

        self._add_gate('cz', control_qubit, target_qubit)
        self._add_u(two_qubit_control_gate(z_gate(), self.n_qubits, control_qubit, target_qubit))

    def Hadamard(self, target_qubit):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('H', target_qubit, None)
        self._add_u(gate_expand_1toN(Hadamard(), self.n_qubits, target_qubit))

    def rxx(self, target_qubit01, target_qubit02, phi):
        assert isinstance(target_qubit01, int), \
            "target qubit is not integer"
        assert isinstance(target_qubit02, int), \
            "target qubit is not integer"
        if not target_qubit02:
            target_qubit02 = target_qubit01 + 1
        assert target_qubit01 <= self.n_qubits
        assert target_qubit02 <= self.n_qubits

        self._add_gate('rxx', target_qubit01, phi)
        self._add_gate('rxx', target_qubit02, phi)
        if type(phi) == float or type(phi) == int:
            phi = torch.tensor(phi)
            self._add_u(two_qubit_rotation_gate(phi, self.n_qubits, target_qubit01, target_qubit02, way='rxx'))
        else:
            self._add_u(two_qubit_rotation_gate(phi, self.n_qubits, target_qubit01, target_qubit02, way='rxx'))

    def ryy(self, target_qubit01, target_qubit02, phi):
        assert isinstance(target_qubit01, int), \
            "target qubit is not integer"
        assert isinstance(target_qubit02, int), \
            "target qubit is not integer"

        if not target_qubit02:
            target_qubit02 = target_qubit01 + 1
        assert target_qubit01 <= self.n_qubits
        assert target_qubit02 <= self.n_qubits
        assert target_qubit01 != target_qubit02, \
            "target qubit should not be the same"

        self._add_gate('ryy', target_qubit01, phi)
        self._add_gate('ryy', target_qubit02, phi)
        if type(phi) == float or type(phi) == int:
            phi = torch.tensor(phi)
            self._add_u(two_qubit_rotation_gate(phi, self.n_qubits, target_qubit01, target_qubit02, way='ryy'))
        else:
            self._add_u(two_qubit_rotation_gate(phi, self.n_qubits, target_qubit01, target_qubit02, way='ryy'))

    def rzz(self, target_qubit01, target_qubit02, phi):
        assert isinstance(target_qubit01, int), \
            "target qubit is not integer"
        assert isinstance(target_qubit02, int), \
            "target qubit is not integer"
        if not target_qubit02:
            target_qubit02 = target_qubit01 + 1
        assert target_qubit01 <= self.n_qubits
        assert target_qubit02 <= self.n_qubits

        self._add_gate('rzz', target_qubit01, phi)
        self._add_gate('rzz', target_qubit02, phi)
        if type(phi) == float or type(phi) == int:
            phi = torch.tensor(phi)
            self._add_u(two_qubit_rotation_gate(phi, self.n_qubits, target_qubit01, target_qubit02, way='rzz'))
        else:
            self._add_u(two_qubit_rotation_gate(phi, self.n_qubits, target_qubit01, target_qubit02, way='rzz'))

    def multi_control_cnot(self, control_lst, target):
        self._add_u(multi_control_gate(x_gate(), self.n_qubits, control_lst, target))

    def swap(self, target_qubit01, target_qubit02):
        self.cnot(target_qubit01, target_qubit02)
        self.cnot(target_qubit02, target_qubit01)
        self.cnot(target_qubit01, target_qubit02)

    def cswap(self, control_qubit, target_qubit01, target_qubit02):
        zero_zero = torch.tensor([[1, 0], [0, 0]]) + 0j
        one_one = torch.tensor([[0, 0], [0, 1]]) + 0j

        lst = [torch.eye(2, 2)] * self.n_qubits
        lst[control_qubit] = zero_zero

        swap = two_qubit_control_gate(x_gate(), self.n_qubits, target_qubit01, target_qubit02)
        swap = swap @ two_qubit_control_gate(x_gate(), self.n_qubits, target_qubit02, target_qubit01)
        swap = swap @ two_qubit_control_gate(x_gate(), self.n_qubits, target_qubit01, target_qubit02)

        self._add_gate('cswap', control_qubit, [target_qubit01, target_qubit02])
        self._add_u(multi_kron(lst) + swap @ gate_expand_1toN(one_one, self.n_qubits, control_qubit))

    def get(self):
        self.U = gate_sequence_product(self.u, self.n_qubits)
        return self.U

    def measure_operator(self, operator='Z'):
        # 生成测量力学量的列表
        if operator == 'Z':
            for i in range(self.n_qubits):
                gate = gate_expand_1toN(z_gate(), self.n_qubits, i)
                Mi = gate
                self.M_lst.append(Mi)
        elif operator == 'X':
            for i in range(self.n_qubits):
                gate = gate_expand_1toN(x_gate(), self.n_qubits, i)
                Mi = gate
                self.M_lst.append(Mi)
        elif operator == 'Y':
            for i in range(self.n_qubits):
                gate = gate_expand_1toN(y_gate(), self.n_qubits, i)
                Mi = gate
                self.M_lst.append(Mi)

        return self.M_lst

    def clear(self):
        # 清空
        self.gate_list = []
        self.u = []
        self.M_lst = []
        self.U = torch.eye(2**self.n_qubits) + 0j


