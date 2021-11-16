# -*- coding: utf-8 -*-
import torch
from deepquantum.utils import multi_kron

__all__ = ["I",
           "rx",
           "ry",
           "rz",
           "x_gate",
           "y_gate",
           "z_gate",
           "Hadamard",
           "gate_expand_1toN",
           "two_qubit_control_gate",
           "two_qubit_rotation_gate",
           "multi_control_gate",
           "gate_sequence_product",
           ]


#   内置函数
def I():
    """Single-qubit Identification gate
    -------
    result : torch.tensor for operator describing Identity matrix.
    """
    return torch.eye(2) + 0j


def rx(phi):
    """Single-qubit rotation for operator sigmax with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """
    return torch.cat((torch.cos(phi / 2).unsqueeze(dim=0), -1j * torch.sin(phi / 2).unsqueeze(dim=0),
                      -1j * torch.sin(phi / 2).unsqueeze(dim=0), torch.cos(phi / 2).unsqueeze(dim=0)),
                     dim=0).reshape(2, -1) + 0j


def ry(phi):
    """Single-qubit rotation for operator sigmay with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """
    return torch.cat((torch.cos(phi / 2).unsqueeze(dim=0), -1 * torch.sin(phi / 2).unsqueeze(dim=0),
                      torch.sin(phi / 2).unsqueeze(dim=0), torch.cos(phi / 2).unsqueeze(dim=0)),
                     dim=0).reshape(2, -1) + 0j


def rz(phi):
    """Single-qubit rotation for operator sigmaz with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """
    return torch.cat((torch.exp(-1j * phi / 2).unsqueeze(dim=0), torch.zeros(1),
                      torch.zeros(1), torch.exp(1j * phi / 2).unsqueeze(dim=0)),
                     dim=0).reshape(2, -1) + 0j


def x_gate():
    """
    Pauli x
    """
    return torch.tensor([[0, 1], [1, 0]]) + 0j


def y_gate():
    """
    Pauli y
    """
    return torch.tensor([[0, -1j], [1j, 0]]) + 0j


def z_gate():
    """
    Pauli z
    """
    return torch.tensor([[1, 0], [0, -1]]) + 0j


def Hadamard():
    H = torch.sqrt(torch.tensor(0.5)) * torch.tensor([[1, 1], [1, -1]]) + 0j
    return H


def gate_expand_1toN(U, N, target):
    """
    representing a one-qubit gate that act on a system with N qubits.
    """
    if N < 1:
        raise ValueError("integer N must be larger or equal to 1")

    if target >= N:
        raise ValueError("target must be integer < integer N")
    lst1 = [I()] * N
    lst1[target] = U
    # return multi_kron([torch.eye(2)] * target + [U] + [torch.eye(2)] * (N - target - 1))
    return multi_kron(lst1)


def two_qubit_control_gate(U, N, control, target):
    if N < 1:
        raise ValueError("integer N must be larger or equal to 1")
    if control >= N:
        raise ValueError("control must be integer < integer N")
    if target >= N:
        raise ValueError("target must be integer < integer N")
    if target == control:
        raise ValueError("control cannot be equal to target")

    zero_zero = torch.tensor([[1, 0], [0, 0]]) + 0j
    one_one = torch.tensor([[0, 0], [0, 1]]) + 0j
    list1 = [torch.eye(2)] * N
    list2 = [torch.eye(2)] * N
    list1[control] = zero_zero
    list2[control] = one_one
    list2[target] = U
    return multi_kron(list1) + multi_kron(list2)


def two_qubit_rotation_gate(theta, N, qbit1, qbit2, way):
    # if type(theta) != type(torch.tensor(0.1)):
    #     theta = torch.tensor(theta)
    if N < 1:
        raise ValueError("number of qubits N must be >= 1")
    if qbit1 < 0 or qbit1 > N - 1 or qbit2 < 0 or qbit2 > N - 1:
        raise ValueError("index must between 0~N-1")
    if qbit1 == qbit2:
        raise ValueError("qbit1 cannot be equal to qbit2")
    lst1 = [torch.eye(2)] * N
    lst2 = [torch.eye(2)] * N
    if way == 'rxx':
        lst2[qbit1] = x_gate()
        lst2[qbit2] = x_gate()
    elif way == 'ryy':
        lst2[qbit1] = y_gate()
        lst2[qbit2] = y_gate()
    elif way == 'rzz':
        lst2[qbit1] = z_gate()
        lst2[qbit2] = z_gate()
    else:
        raise ValueError("Error gate")
    rst = torch.cos(theta / 2) * multi_kron(lst1) - 1j * torch.sin(theta / 2) * multi_kron(lst2)
    return rst + 0j


def multi_control_gate(U, N, control_lst, target):
    """
    多控制比特受控门，比如典型的toffoli gate就是2个控制1个受控
    control_lst:一个列表，内部是控制比特的索引号
    """
    if N < 1:
        raise ValueError("number of qubits(interger N) must be >= 1")

    if max(max(control_lst), target) > N - 1:
        raise ValueError("control&target must <= number of qubits - 1")

    if min(min(control_lst), target) < 0:
        raise ValueError("control&target must >= 0")

    for each in control_lst:
        if each == target:
            raise ValueError("control cannot be equal to target")

    U = U + 0j
    one_one = torch.tensor([[0, 0], [0, 1]]) + 0j
    lst1 = [torch.eye(2)] * N
    for each in control_lst:
        lst1[each] = one_one
    lst1[target] = U

    lst2 = [torch.eye(2)] * N

    lst3 = [torch.eye(2)] * N
    for each in control_lst:
        lst3[each] = one_one
    # multi_kron(lst2) - multi_kron(lst3)对应不做操作的哪些情况
    return multi_kron(lst2) - multi_kron(lst3) + multi_kron(lst1)


def gate_sequence_product(U_list, n_qubits, left_to_right=True):
    """
    Calculate the overall unitary matrix for a given list of unitary operations.
    return: Unitary matrix corresponding to U_list.
    """
    U_overall = torch.eye(2 ** n_qubits, 2 ** n_qubits) + 0j
    for U in U_list:
        if left_to_right:
            U_overall = U @ U_overall
        else:
            U_overall = U_overall @ U
    return U_overall