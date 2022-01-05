#!/usr/bin/env python
# coding: utf-8

# # Attention - Qutorch

# In[10]:


import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from deepquantum import Circuit
from deepquantum.utils import dag,measure_state,ptrace,multi_kron,encoding


# In[11]:


def measure(state,n_qubits):
    cir=Circuit(n_qubits)
    for i in range(n_qubits):
        cir.z_gate(i)
    m=cir.get()
    return measure_state(state,m,rho=False)


class init_cir_q(nn.Module):
    #初始化U_query
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
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(n_qubits*3), a=0.0, b=2*np.pi) * init_std)# theta_size=5
        
        self.n_qubits = n_qubits


    def queryQ(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits):
            cir.rx(which_q,w[which_q*3+0])
            cir.ry(which_q,w[which_q*3+1])       
            cir.rz(which_q,w[which_q*3+2])
        #ring cnot gates
        for which_q in range(0,self.n_qubits-1):
            cir.cnot(which_q,which_q+1)
        if self.n_qubits>1:
            cir.cnot(self.n_qubits-1,0)
        return cir.get()

    def forward(self, x):
        E_out = self.queryQ()
        queryQ_out = E_out@ x @ dag(E_out)
        return queryQ_out


# In[15]:


class init_cir_k(nn.Module):
    #初始化U_key
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
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(n_qubits*3), a=0.0, b=2*np.pi) * init_std)# theta_size=5
        
        self.n_qubits = n_qubits


    def keyQ(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits):
            cir.rx(which_q,w[which_q*3+0])
            cir.ry(which_q,w[which_q*3+1])       
            cir.rz(which_q,w[which_q*3+2])
        #ring cnot gates
        for which_q in range(0,self.n_qubits-1):
            cir.cnot(which_q,which_q+1)
        if self.n_qubits>1:
            cir.cnot(self.n_qubits-1,0)
        return cir.get()


    def forward(self, x):
        E_out = self.keyQ()
        keyQ_out = E_out @ x @ dag(E_out)
        return keyQ_out


# In[16]:


class init_cir_v(nn.Module):
    #初始化U_value
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
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(n_qubits*3), a=0.0, b=2*np.pi) * init_std)# theta_size=5
        
        self.n_qubits = n_qubits


    def valueQ(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits):
            cir.rx(which_q,w[which_q*3+0])
            cir.ry(which_q,w[which_q*3+1])       
            cir.rz(which_q,w[which_q*3+2])
        #ring cnot gates
        for which_q in range(0,self.n_qubits-1):
            cir.cnot(which_q,which_q+1)
        if self.n_qubits>1:
            cir.cnot(self.n_qubits-1,0)
        return cir.get()


    def forward(self, x):
        E_out = self.valueQ()
        valueQ_out = E_out @ x @ dag(E_out)
        return valueQ_out


# In[17]:


def cal_query_key(queryQ_out, keyQ_out, dim_q, dim_k):
    """queryQ_out: type torch.Tensor
       keyQ_out: torch.Tensor
    """
    """计算query与key的interaction score
    
    """
    out = torch.kron(queryQ_out, keyQ_out)
    n_qubits = dim_q + dim_k
    
    #对称cnot门
    cir=Circuit(n_qubits)
    for t in range(0,dim_k,1):
        cir.cnot(t,n_qubits-dim_k+t)
    for t in range(dim_k-1,-1,-1):
        cir.cnot(n_qubits-dim_k+t,t)
    U=cir.get()
    
    out=U @ out @ dag(U)
    
    quantum_score = measure(out, n_qubits)
    
    return quantum_score


# In[26]:


def cal_src_value(quantum_src, valueQ_out, dim_s, dim_v):
    """input torch.Tensor
    """
    """计算经过attention score加权作用后的value
    """
    src=quantum_src.mean()
    #src=(src+1.0)/2.0     #[-1,1] -> [0,1]
    #phi=(src-0.5)*2*np.pi #phi=[-pi,pi]
    phi=src*np.pi   #phi=[-pi,pi]
    
    #rx-ringCnot-ry-RingCnot-rz
    cir=Circuit(dim_v)
    for i in range(dim_v):
        cir.rx(i,phi*0.5)
        
    for which_q in range(0,dim_v-1):
        cir.cnot(which_q,which_q+1)
    if dim_v>1:
        cir.cnot(dim_v-1,0)
        
    for i in range(dim_v):
        cir.ry(i,phi*0.5)
        
    for which_q in range(0,dim_v-1):
        cir.cnot(which_q,which_q+1)
    if dim_v>1:
        cir.cnot(dim_v-1,0)
        
    for i in range(dim_v):
        cir.rz(i,phi)
    U=cir.get()
    
    quantum_weighted_value = U @ valueQ_out @ dag(U)
    
    return quantum_weighted_value


# In[27]:


def cal_output(qwv_list, dim):
    """计算weighted values的“和”（通过多个cnot门将信息融合）
    """
    #out = multi_kron(qwv_list)
    #n_qubits=2*dim
    cir=Circuit(2*dim)
    for t in range(dim):
        cir.cnot(t,dim+t)            
    U=cir.get()
    
    #为避免线路上比特数过多，两个两个处理
    attnQ=qwv_list[-1]
    for i in range(len(qwv_list)-1):
        out=torch.kron(qwv_list[i],attnQ)
        out=U @ out @ dag(U)
        attnQ = ptrace(out, dim, dim)
        
    return attnQ


# In[28]:


def q_attention(query, key, value, mask=None, dropout=None):
    # "Compute 'Scaled Dot Product Attention'"
    query_input = query.squeeze(0)
    key_input = key.squeeze(0)
    value_input = value.squeeze(0)
    # print(query_input.size(-1))
    n_qubits = math.ceil(math.log2(query_input.size(-1)))
    # print(n_qubits)

    qqs = []
    qks = []
    qvs = []

    init_q = init_cir_q(n_qubits=n_qubits)
    init_k = init_cir_k(n_qubits=n_qubits)
    init_v = init_cir_v(n_qubits=n_qubits)

    for x in query_input.chunk(query_input.size(0), 0):
        # expand to 2**n_qubits length vector
        qx = nn.ZeroPad2d((0, 2 ** n_qubits - query_input.size(-1), 0, 0))(x)
        # l2-regularization
        if qx.dim() > 2:
            qx = qx.squeeze()
        qinput = encoding(qx.T @ qx)
        qqs.append(init_q(qinput))

    for x in key_input.chunk(key_input.size(0), 0):
        # expand to 2**n_qubits length vector
        qx = nn.ZeroPad2d((0, 2 ** n_qubits - key_input.size(-1), 0, 0))(x)
        # l2-regularization
        if qx.dim() > 2:
            qx = qx.squeeze()
        qinput = encoding(qx.T @ qx)
        qks.append(init_k(qinput))

    for x in value_input.chunk(value_input.size(0), 0):
        # expand to 2**n_qubits length vector
        qx = nn.ZeroPad2d((0, 2 ** n_qubits - query_input.size(-1), 0, 0))(x)
        # l2-regularization
        if qx.dim() > 2:
            qx = qx.squeeze()
        qinput = encoding(qx.T @ qx)
        qvs.append(init_v(qinput))

    outputs = []
    for i in range(len(qqs)):
        qwvs_i = []
        for j in range(len(qks)):
            score_ij = cal_query_key(qqs[i], qks[j], n_qubits, n_qubits)
            qwvs_i.append(cal_src_value(score_ij, qvs[j], n_qubits, n_qubits))
        out_i = measure(cal_output(qwvs_i, n_qubits), n_qubits).squeeze().unsqueeze(0)
        outputs.append(out_i)
        # print(out_i)

    return torch.cat(outputs)


# In[29]:


class Q_MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(Q_MultiHeadedAttention, self).__init__()
        #assert d_model % h == 0
        # We assume d_v always equals d_k
        #self.d_k = d_model // h
        self.n_qubits=math.ceil(math.log2(d_model))
        self.h = h
        self.linear = nn.Linear(2**self.n_qubits*h,d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Apply attention on all the projected vectors in batch.
        x = q_attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        # multi-head
        for i in range(self.h-1):
            x=torch.cat((x,q_attention(query, key, value, mask=mask, dropout=self.dropout)),-1)
        #print(x.size())
        x=x.unsqueeze(0)
        #print(self.n_qubits)
        #print(self.linear)
        return self.linear(x)






