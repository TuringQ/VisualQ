#!/usr/bin/env python
# coding: utf-8

# This impl is meant to quantize graph attention networks (GATs).
# 
# Basically, this impl is  just Transformer's encoder. If you set `edge_index=None`, you get the quantized Transformer's encoder without positional encoding. I also provide the class `PositionalEncoding` so you can easily inject positional information into input.
# 
# TODO：
# 
# 3. parallel implemntation (for loop -> tensor dim)
# 
# 4. GPU加速
# 
# 
# 
# 
# 教训：
# 尽量不要定义全局变量，所有测试封装起来。
# 如果非要定义全局变量，把它们全部放在Train的第一个cell里，避免冲突。
# 

# **Loss:**
# 
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=cross#torch.nn.CrossEntropyLoss  (softmax included, expecting model's output to be raw logits)
# 
# https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html?highlight=cross
# 
# https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e
# 
# https://discuss.pytorch.org/t/how-exactly-should-i-understand-the-cross-entropy-loss-function/61183/2

# ## Prerequisites

# In[1]:


# In[2]:


import math, copy, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from deepquantum import Circuit
from deepquantum.utils import dag, measure_state, ptrace


# In[3]:


def amplitudes_encoding(x):
    """
    x is a input vector with dim 2**i
    returns a vector with probabilities summed to 1, which can be interpreted as a ket
    """
    x = x.clone().detach()
    x /= torch.sqrt(torch.sum(x*torch.conj(x))) # normalization

    return x


# In[4]:


def get_zero_state(n_qubits):
    """
    returns |0⟩, the lowest computatinal basis state for a n qubits circuit
    """
    zero_state = torch.zeros(2**n_qubits, dtype=torch.cfloat)
    zero_state[0] = 1.+0j
    return zero_state


def get_random_state(n_qubits):
    r = torch.rand(2**n_qubits) + 0j
    normalization = torch.sqrt(torch.sum(r*torch.conj(r)))
    return r/normalization


# In[5]:


def purestate_density_matrix(pure_state):
    """
    create a pure state density matrix form a pure state
    """

    dim = len(pure_state)

    density_matrix = torch.zeros(dim, dim, dtype=torch.cfloat)
    for i in range(dim):
        density_matrix[i] = pure_state[i]*torch.conj(pure_state)

    return density_matrix


# In[6]:


def measure(state, n_qubits, rho=True):
    "measure in Pauli Z basis, i.e. get its expectation values"
    cir=Circuit(n_qubits)
    for i in range(n_qubits):
        cir.z_gate(i)
    m=cir.get()
    return measure_state(state, m, rho=rho)


# In[7]:


def test_measure():
    "batched state test"
    print('### measuring a batch of 1 qubit state in Pauli Z basis ###')

    state1 = torch.tensor([math.sqrt(0.3), math.sqrt(0.7)])+0j
    state2 = torch.tensor([math.sqrt(0.4), math.sqrt(0.6)])+0j
    state3 = torch.tensor([math.sqrt(0.2), math.sqrt(0.8)])+0j

    print('state1:', state1)
    print('state2:', state2)
    print('state3:', state3)

    state1 = state1.unsqueeze(1)
    state2 = state2.unsqueeze(1)
    state3 = state3.unsqueeze(1)
    batched_state = torch.cat((state1, state2, state3), 1)
    print('shape of batched_state = (num_dim, batch_size):', batched_state.shape)
    print('batched_state:\n', batched_state)

    exp_val = measure(batched_state , 1, rho=False)

    print('expectation value of Pauli z:\n', exp_val)

    assert abs(exp_val.mean() - torch.tensor([[-0.4000], [-0.2000], [-0.6000]]).mean())  < 1e-4, 'test failed'

    print('### test passed ###')


# In[8]:


test_measure()


# ## Layer Normalization
# 
# 
# 
# Normalization of features before activation
#  
# 
# 
# 

# In[9]:


class LayerNorm(nn.Module):
    "Construct a layernorm module https://arxiv.org/abs/1607.06450"
    "features: hidden vector's dimension"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# ## Neighbor nodes selection

# In[10]:


def get_neighbors(i, edge_index):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.tolist()

    neighbors = []
    for e in edge_index: # this takes time
        if i in e:
            if e[0] != i:
                neighbors.append(e[0])
            else:
                neighbors.append(e[1])
    return neighbors


# In[ ]:





# ## Define Model

# In[11]:


class PQC(nn.Module):
    """
    parameterized quatum circuit, counterpart to linear transformation
    
    arg:
        n_qubits: number of qubits of a parameterized quantum circuit.
        if input tensor has F features, you need log2(F) qubits 
        assuming amplitude data encoding.
    input:
        density matrix representing classical input feature vector with dimension F
        shape = (F, F)
    output:
        evolved density matrix which has learnable parameters 
        shape = (F, F)
        
    """
    def __init__(self, n_qubits, 
                 gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(n_qubits*3), a=0.0, b=2*np.pi) * init_std)
        self.n_qubits = n_qubits
        self.unitary = None


    def circuit(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits):
            cir.rx(which_q,w[which_q*3+0])
            cir.ry(which_q,w[which_q*3+1])       
            cir.rz(which_q,w[which_q*3+2])
        return cir.get()

    def forward(self, x):
        self.unitary = self.circuit()
        out = self.unitary@ x @ dag(self.unitary)
        return out


# In[12]:


def test_PQC():
    # input0 (4 features/2 qubits) -> query0, key0, value0 (2 qubits) -> output0 (2 qubits)
    # input1 (4 features/2 qubits) -> query1, key1, value1 (2 qubits) -> output0 (2 qubits)


    # Suppose this is the input0's desity matrix
    rho0 = purestate_density_matrix(get_random_state(n_qubits=2))
    print('rho0', rho0)

    # Suppose this is the input1's desity matrix
    rho1 = purestate_density_matrix(get_random_state(n_qubits=2))
    print('rho1', rho1)

    # query
    cir_q = PQC(n_qubits=2)
    rho0_q = cir_q(rho0)
    rho1_q = cir_q(rho1)
    print('rho0_q', rho0_q)
    print('rho1_q', rho1_q)


    # key
    cir_k = PQC(n_qubits=2)
    rho0_k = cir_k(rho0)
    rho1_k = cir_k(rho1)
    print('rho0_k', rho0_k)
    print('rho1_k', rho1_k)


    # value
    cir_v = PQC(n_qubits=2)
    rho0_v = cir_v(rho0)
    rho1_v = cir_v(rho1)
    print('rho0_v', rho0_v)
    print('rho1_v', rho1_v)


    return [rho0_q, rho1_q], [rho0_k, rho1_k], [rho0_v, rho1_v]


# In[13]:


#test_PQC()
None


# In[14]:


def cal_query_dot_key(query, key):
    n_qubits = math.ceil(math.log2(query.size(-1))) * 2
    out = torch.kron(query, key)
    
    cir = Circuit(n_qubits)
    for t in range(n_qubits):
        cir.cnot(t, (t+1) % n_qubits)
    U = cir.get() 
    out = U @ out @ dag(U)
    
    out = measure(out, n_qubits)

    return (out + 1) / 2


# In[15]:


def test_cal_query_dot_key():
    qs, ks, vs = test_PQC()

    score00 = cal_query_dot_key(qs[0], ks[0])
    print('score00', score00)

    score01 = cal_query_dot_key(qs[0], ks[1])
    print('score01', score01)

    score10 = cal_query_dot_key(qs[1], ks[0])
    print('score10', score10)

    score11 = cal_query_dot_key(qs[1], ks[1])
    print('score11', score11)


    return [[score00, score01], [score10, score11]], vs


# In[16]:


#test_cal_query_dot_key()
None


# In[17]:


def cal_weighted_value(score, value):
    """
    calculate value weighted by the attention score
    """
    n_qubits = math.ceil(math.log2(value.size(-1)))
    phi=(score-0.5)*2*np.pi # phi=[-pi,pi]
    
    cir=Circuit(n_qubits)
    for i in range(n_qubits):
        cir.rx(i,phi*0.5)
        cir.ry(i,phi*0.5)
        cir.rz(i,phi)
    U=cir.get()
    
    weighted_value = U @ value @ dag(U)
    
    return weighted_value


# In[18]:


def test_cal_weighted_value():
    scores, vs = test_cal_query_dot_key()

    wv00 = cal_weighted_value(scores[0][0], vs[0])
    print('wv00', wv00)

    wv01 = cal_weighted_value(scores[0][1], vs[1])
    print('wv01', wv01)

    wv10 = cal_weighted_value(scores[1][0], vs[0])
    print('wv10', wv10)

    wv11 = cal_weighted_value(scores[1][1], vs[1])
    print('wv11', wv11)

    return [[wv00, wv01], [wv10, wv11]]
    


# In[19]:


#test_cal_weighted_value()
None


# In[20]:


def cal_sum(wv_list, rho=False):
    """
    rho = True returns final density matrix
    rho = False returns probabilities of measurements in computational basis, i.e. diagonal elements of density matrix 
    sum weighted values
    """
    n_qubits_v = math.ceil(math.log2(wv_list[0].size(-1)))
    n_qubits = 2 * n_qubits_v

    cir = Circuit(n_qubits)
    for t in range(n_qubits):
        cir.cnot(t, (t+1) % n_qubits) 
    U = cir.get()

    rho_A = wv_list[0]
    for i in range(1, len(wv_list)):
        rho_B = wv_list[i]
        rho_AB = torch.kron(rho_A, rho_B)
        
        out = U @ rho_AB @ dag(U)
        rho_A = ptrace(out, n_qubits_v, n_qubits-n_qubits_v)

    sum = rho_A if rho else rho_A.diag().real

    return sum  


# In[21]:


def test_cal_sum():

    wv_list = test_cal_weighted_value()

    output0 = cal_sum(wv_list[0])
    output1 = cal_sum(wv_list[1])

    print('output0', output0)
    print('output1', output1)


# In[22]:


#test_cal_sum()


# In[23]:


def get_query_key_value(x, cir_q, cir_k, cir_v):
    """
    for each word in sequence x, this function performs 
    word -> rho_word -> rho_q, rho_k, rho_v
    x is calssical data, shape of x = (N, seq_len, emb_dim), N=1 for quantum circuit
    returns quantum data, i.e. density matrices 
    """
    qs = [] # generate query for each word
    ks = [] # generate key for each word
    vs = [] # generate value for each word

    # squeeze batch dim if its 1
    x = x.squeeze(0)

    for word in x: # loop through seq dim, for loop is not parallel impl, this can be parallel 
        # amplitudes encoding
        rho = purestate_density_matrix(amplitudes_encoding(word)) 
        qs.append(cir_q(rho))
        ks.append(cir_k(rho))
        vs.append(cir_v(rho))
        
 

    return qs, ks, vs

    


# In[24]:


def test_get_query_key_value():
    input = torch.rand(1, 4, 4) # dummy input 
    cir_q = PQC(n_qubits=2)
    cir_k = PQC(n_qubits=2)
    cir_v = PQC(n_qubits=2)

    qs, ks, vs = get_query_key_value(input, cir_q, cir_k, cir_v)

    print('qs', qs)
    print('ks', ks)
    print('vs', vs)

    return qs, ks, vs


# In[25]:


#test_get_query_key_value()
None


# In[26]:


def q_attention(qs, ks, vs, edge_index=None):
    # qs is a list of query of each word, similarly for ks, vs
    graph = False if edge_index is None else True
    scores = torch.zeros(len(qs), len(ks))  # zero meaning no connection in the graph
    
    outputs = []
    for i in range(len(qs)):
        wvs_i = []
        range_ks = get_neighbors(i, edge_index) if graph else range(len(ks))
        for j in range_ks:
            # 1) score
            score_ij = cal_query_dot_key(qs[i], ks[j])
            scores[i, j] = score_ij
            #print('debug:', f's_{i}{j}:', score_ij)
            # 2) multiplication
            wvs_i.append(cal_weighted_value(score_ij, vs[j]))
        # 3) sum
        out_i = cal_sum(wvs_i)
        outputs.append(out_i)
    
    z = (torch.stack(outputs)).unsqueeze(0) # output of attn layer
        
    return z, scores  # shape of z = (N, seq_len, 2^n)


# In[27]:


def test_q_attention():


    qs, ks, vs = test_get_query_key_value()
    z, scores = q_attention(qs, ks, vs)
    print('z', z)
    print('scores', scores)


# In[28]:


#test_q_attention()
None


# In[29]:


class MultiHeadedAttention(nn.Module):
    """
    nn.Linear transform last dim of x into  h*d_k h is number of heads. then feed into q_attention
    finally recombine/concat the results  
    
    args:
        h = number of heads 
        d_model = embedding_dim
    input:
        x is calssical data, shape of x = (N, seq_len, emb_dim), N=1 for quantum circuit

    outpt:
        shape = (1, seq_len, d_model)
    """
   
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()

        self.n_qubits = math.ceil(math.log2(d_model))  # amplitudes encoding ⌈log2(x)⌉
        self.h = h
        self.d_model = 2**self.n_qubits
        assert self.d_model % self.h == 0, 'd_model % h!=0'
        self.d_k = self.d_model // self.h  #assume d_v always equals d_k
        self.attn_scores = [] # record attn scores for each head
       
        self.linear1 = nn.Linear(self.d_model, self.d_k)
        self.linear2 = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(p=dropout)

        n = math.ceil(math.log2(self.d_k))  # amplitudes encoding ⌈log2(x)⌉

        # we need h*3 PQC
        self.PQC_list = nn.ModuleList([PQC(n_qubits=n) for _ in range(self.h*3)])

    def forward(self, x, edge_index):
        self.attn_scores = [] # only record the most recent scores
        x = F.pad(x, (0, self.d_model-x.size(-1)), "constant", 0) 

        batch_size = x.size(0)
        zs = []   # outputs of each head
        # 1) decrease dim
        x = self.linear1(x)
        
        # 2) parallel heads, each head is an independent encoder, for loop is not parallel impl, this can be parallel
        for head in range(self.h):
            cir_q = self.PQC_list[head*3+0]
            cir_k = self.PQC_list[head*3+1]
            cir_v = self.PQC_list[head*3+2]

            qs, ks, vs = get_query_key_value(x, cir_q, cir_k, cir_v)
            z, scores = q_attention(qs, ks, vs, edge_index)
            zs.append(z)
            self.attn_scores.append(scores) 
        
        # 3) concat results from different heads
        zs = torch.stack(zs, 1)   # (1, h, seq_len, d_k)
        zs = zs.transpose(1, 2).contiguous()              .view(batch_size, -1, self.h * self.d_k)

        return self.dropout(self.linear2(zs)) # (1, seq_len, h*d_k)


# Position-Wise Feed-Forward Layer is a type of feed-forward layer consisting of two dense layers that applies to the last dimension, which means the same dense layers are used for each position item in the sequence, so called position-wise.  

# In[30]:


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# In[31]:


class LayerNorm(nn.Module):
    "Construct a layernorm module https://arxiv.org/abs/1607.06450"
    "features: hidden vector's dimension"
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# In[32]:


class SublayerConnection(nn.Module):
    """
    Sublayer with residual connnection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, features, dropout):
        super().__init__()
        self.norm = LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same shape."
        return x + self.dropout(sublayer(self.norm(x)))


# In[33]:


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# In[34]:


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, features, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(features, dropout), 2)
        self.features = features

    def forward(self, x, edge_index):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, edge_index))
        return self.sublayer[1](x, self.feed_forward)


# In[35]:


class Encoder(nn.Module):
    "Core encoder is a stack of n_encoder_layer layers"
    def __init__(self, encoder_layer, n_encoder_layer):
        super().__init__()
        self.layers = clones(encoder_layer, n_encoder_layer)
        self.norm = LayerNorm(encoder_layer.features)
        
    def forward(self, x, edge_index):
        "Pass the input through each layer in turn."
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.norm(x)


# In[36]:


class QGAT(nn.Module):
    def __init__(self, encoder, linear):
        super().__init__()
        self.encoder = encoder
        self.linear = linear
        
    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        return self.linear(x)


# In[37]:


def make_model(vocab_size, n_encoder_layer, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    model = QGAT(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_encoder_layer),
        nn.Linear(in_features=d_model, out_features=vocab_size)
        )

    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ## Positional Encoding (TODO)

# In[ ]:





# ## Train

# ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAHMCAIAAADnA2MBAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAACbKADAAQAAAABAAABzAAAAABHdP2uAABAAElEQVR4Ae2deeBMVf/HK3skJEtpEVHZsqYIIWTJEi3ahDySpLK0CpVCkuRRljZttiQiRJJIhSyptEuSFlKeouX5veo8v9NtZr7znblzZ+bOnff3D87cOcvnvM6d+7nnnM/5fA7+73//e5D+REAEREAEREAE4idwSPxFVEIEREAEREAEROBPAlKiug9EQAREQAREwCUBKVGX4FRMBERABERABKREdQ+IgAiIgAiIgEsCUqIuwamYCIiACIiACEiJ6h4QAREQAREQAZcEpERdglMxERABERABEZAS1T0gAiIgAiIgAi4JSIm6BKdiIiACIiACIiAlqntABERABERABFwSkBJ1CU7FREAEREAEREBKVPeACIiACIiACLgkICXqEpyKiYAIiIAIiICUqO4BERABERABEXBJQErUJTgVEwEREAEREAEpUd0DIiACIiACIuCSgJSoS3AqJgIiIAIiIAJSoroHREAEREAERMAlASlRl+BUTAREQAREQASkRHUPiIAIiIAIiIBLAlKiLsGpmAiIgAiIgAhIieoeEAEREAEREAGXBKREXYJTMREQAREQARGQEtU9IAIiIAIiIAIuCUiJugSnYiIgAiIgAiIgJap7QAREQAREQARcEpASdQlOxURABERABERASlT3gAiIgAiIgAi4JCAl6hKciomACIiACIiAlKjuAREQAREQARFwSUBK1CU4FRMBERABERABKVHdAyIgAiIgAiLgkoCUqEtwKiYCIiACIiACUqK6B0RABERABETAJQEpUZfgVEwEREAEREAEpER1D4iACIiACIiASwJSoi7BqZgIiIAIiIAISInqHhABERABERABlwSkRF2CUzEREAEREAERkBLVPSACIiACIiACLglIiboEp2IiIAIiIAIiICWqe0AEREAEREAEXBKQEnUJTsVEQAREQAREQEpU94AIiIAIiIAIuCQgJeoSnIqJgAiIgAiIgJSo7gEREAEREAERcElAStQlOBUTAREQAREQASlR3QMiIAIiIAIi4JKAlKhLcComAiIgAiIgAlKiugdEQAREQAREwCUBKVGX4FRMBERABERABKREdQ+IgAiIgAiIgEsCUqIuwamYCIiACIiACEiJ6h4QAREQAREQAZcEpERdglMxERABERABEZAS1T0gAiIgAiIgAi4JSIm6BKdiIiACIiACIiAlqntABERABERABFwSkBJ1CU7FREAEREAEREBKVPeACIiACIiACLgkICXqEpyKiYAIiIAIiICUqO4BERABERABEXBJQErUJTgVEwEREAEREAEpUd0DIiACIiACIuCSgJSoS3AqJgIiIAIiIAJSoroHREAEREAERMAlASlRl+BUTAREQAREQASkRHUPiIAIiIAIiIBLAlKiLsGpmAiIgAiIgAhIieoeEAEREAEREAGXBKREXYJTMREQAREQARGQEtU9IAIiIAIiIAIuCUiJugSnYiIgAiIgAiIgJap7QAREQAREQARcEpASdQlOxURABERABERASlT3gAiIgAiIgAi4JCAl6hKciomACIiACIiAlKjuAREQAREQARFwSUBK1CU4FRMBERABERABKVHdAyIgAiIgAiLgkoCUqEtwKiYCIiACIiACUqK6B0RABERABETAJQEpUZfgVEwEREAEREAEpER1D4iACIiACIiASwJ5XZZTMREQAREQAX8Q+OOPP377649EwYIF8+bVgz11AyPWqWOtlkRABEQgEQLbt2/fuHHj1q1bP/3rj4/f//X3448/OqvNnz9/4cKFjzzyyLJ//VWoUOGkv/6qVq2KinXmVDpxAgf/97//TbwW1SACIiACIuA5gZ07d65evXrVqlVr167dsGEDGjORJvLly1ejRo369es3b968WbNmRYoUSaQ2lTUEpER1J4iACIiAjwh88803L7/88pIlS5YvX86EM0mSMVtt1KjRJZdc0rlzZ6atSWolG6qVEs2GUVYfRUAE/E7gnXfemTNnzrx580jEskCYJ08eFmuL/fV32GGHHXLIIQcf/OfzfN++fXv37v3hhx927Njx888/59rtEiVK9O3b94YbbihatGiumZUhnICUaDgTXREBERCBFBHYtGnTE088MWvWrM8++yxKk0cccUS1atXY1OTvlFNOOe64444++mj0aJQifPXdd99t27Zty5Yt7777LrqZleE9e/ZELMIG6qhRo7p16xbxW12MQkBKNAocfSUC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Ak1ASjTQw+u2cwRUWrx48b///W/jBBjzekyNqlevzjzVbZUqJwIiIAIBJCAlGsBB9apLV1111ZYtW3DEbyr8+OOPcROIL/soERC9alr1iIAIiEBGEJASzYhhSpuQ+JdYtGgRkVaPOOIIhMABxbhx4zhFysW0yaSGRUAERMA3BKREfTMUPhYEF8FMSW0k823btuH1sHfv3saXk48Fl2giIAIikFwCUqLJ5RuY2vG2T6Tx2bNnly5d2nTq4Ycf5oBplChygem7OiICIiACORGQEs2JjK5HINCpUyempOeff7757quvvsJX8EUXXRRLDOEI1emSCIiACGQ4ASnRDB/AlItfokQJYsa9+OKL5cqVM40/++yzTEmff/75lMuiBkVABEQgzQSkRNM8ABnafOvWrYlIetlllxn5v/nmm45//RFlIkN7JLFFQAREwAUBKVEX0FTkTwJFixZ9/PHHCbxavnx5Q4TJKFNSJqYCJAIiIAJZQkBKNEsGOlndbNas2aZNm3r16oWrI9rYvXs3W6RslBJgNVlNql4REAER8A0BKVHfDEXGClK4cGEsdVesWFGpUiXTCUx2q1SpQlxSzpVmbLckuAiIgAjkTkBKNHdGyhELAfzrbtiw4dprr82TJw/58WrUvXt35qmffPJJLMWVRwREQAQykYCUaCaOmk9lLliw4P3330/IF1waGRFfeeWVatWqTZw4Ee+7PhVaYomACIhAAgSkRBOAp6KRCNStW3ft2rWDBw/Oly8f3xP7pU+fPmeeeeYHH3wQKbuuiYAIiEAGE5ASzeDB863o+fPnv+eee956661atWoZIVetWlWjRo2xY8cSo9S3YkswERABEYiXgJRovMSUP1YCaM01a9bcfvvtBQoUoMz+/fuJVHraaadhzRtrFconAiIgAv4mICXq7/HJcOny5s07dOjQ9evX169f33SFld7atWvffffdv/76a4Z3TuKLgAiIwEFSoroJkk7g5JNPfv3111GchQoVojHU580334wqRaEmvW01IAIiIALJJCAlmky6qvv/CRxyyCE33ngjC7mNGzc210iztMtiL8u8/59L/4uACIhAhhGQEs2wActocStUqMChF8yLihQpQkcwMho+fDhbp5gdZXS/JLwIiEDWEpASzdqhT0/H8Q7Yv39/nNe3aNHCSMDRFw7ADBo0iMMw6ZFJrYqACIiA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WrQoWX7//fd77rmnRo0ar732Wg4ldFkERCDNBKRE0zwAal4EQgj06dOHaWjr1q3N9Q8//LBx48Ycg8E/Q0hOfRQBEUg7ASnRtA+BBBCBUALlypV78cUXH3nkkRIlSvAd3gFxyFCtWrWXX345NKs+i4AIpJWAlGha8atxEciZwBVXXMGUlABqJstnn3129tln4ywQL/Y5F9I3IiACKSUgJZpS3GpMBOIiUKZMmblz5xLQu1SpUqYgbuurVKkyf/78uOpRZhEQgSQRkBJNElhVKwKeEejSpQtOAc8//3xT45dfftmuXbtu3boR69uzNlSRCIiAKwJSoq6wqZAIpJbAEUccMX369BdeeOGoo44yLT/++OOE+H7++edTK4haEwER+AcBKdF/4NAHEfAzASag7JJedtllRshdu3Z17Nixc+fOX3/9tZ/FlmwiEGACUqIBHlx1LYAEihUrxhx00aJFxx13nOne7NmzmZI+++yzAeytuiQCvicgJer7IZKAIhBGoEWLFps3b+7Vq9fBBx/Ml99///1FF13Utm1btkvD8uqCCIhAEglIiSYRrqoWgeQRKFKkCJa6r7zySsWKFU0rHC3FcPexxx5LXqOqWQREIISAlGgIEH0UgUwigDOjjRs39uvXD++7yM0RUk6XcpyUQ6WZ1A3JKgIZS0BKNGOHToKLwF8EChUqhD+j119//eSTTzZIcGxEsNKJEyfi6kiQREAEkkpASjSpeFW5CKSIQP369YlFOnjw4Lx589Lkvn378MHLPBXXuymSQM2IQFYSkBLNymFXp4NIoECBAkR9IQopgV9M/wj/Qnrs2LEEhAlij9UnEUg/ASnR9I+BJBABDwnUrFnzrbfeuv322/Pnz0+1hCMlUimhSTlg6mErqkoERMAQkBLVnSACQSOQL1++oUOHrlu3rl69eqZvTE9r1ap19913//bbb0HrrfojAmklICWaVvxqXASSRoDjLqtWrUJxFixYkEYOHDhw8803161bl63TpLWpikUg6whIiWbdkKvD2UMgT548N95444YNGxo2bGh6/c477zA9ZbF3//792cNBPRWB5BGQEk0eW9UsAr4gUKlSpRUrVmBeVLhwYQRiRXf48OGs7q5Zs8YX8kkIEchkAlKimTx6kl0EYiOAd8D+/fvjKbBZs2amBLHVsDbiSAyWR7HVoVwiIAIRCEiJRoCiSyIQSALHH388fhhwwlC0aFE6+Mcff4waNap69erMUwPZX3VKBFJAQEo0BZDVhAj4iEDv3r2ZhrZp08bI9NFHHzVp0gTHgT/99JOPpJQoIpAhBKREM2SgJKYIeEfg6KOPnj9/Pq7qS5QoQa14Bxw/fjyeApcsWeJdI6pJBLKCgJRoVgyzOikC4QQuv/xypqQdOnQwX33++edEWCO8Gl7swzPrigiIQEQCUqIRseiiCGQFgdKlS8+ZM2fmzJmlSpUyHZ48eTIhvufNm5cV/VcnRSBhAlKiCSNUBSKQ4QQ6d+7MlPTCCy80/dixY8e555572WWXfffddxneM4kvAkknICWadMRqQAT8T+CII4545plnmICyXWqknTZtGlPS5557zv/CS0IRSCMBKdE0wlfTIuAvAm3btsVPfbdu3YxYu3btOu+vv6+//tpfgkoaEfANASlR3wyFBBEBHxA4/PDDH3300cWLFx933HFGHCajTEmffvppH0gnEUTAdwSkRH03JBJIBNJO4Oyzz8a9ESdKcXWEMN9///3FF1/M0dLt27enXTYJIAK+IiAl6qvhkDAi4BcCRYoUwbfR8uXLTzzxRCPTggULiAzzyCOP+EVEySECPiAgJeqDQZAIIuBXAo0aNSIIDH53Dznkz2fF3r17e/To0bx5808//dSvIksuEUgpASnRlOJWYyKQcQQKFSpEBBhCk7IzaoRfunRptWrVJkyYgKujjOuOBBYBbwlIiXrLU7WJQDAJnHbaaUTzvummm/LmzUsP9+3b17dvX+apW7duDWaH1SsRiI2AlGhsnJRLBLKeQP78+UeMGPHWW2+deuqpBsbKlStr1KgxZsyY33//PevxCECWEpASzdKBV7dFwB0BNCh6dNiwYehUavjll18GDBhw+umnY83rrkKVEoGMJiAlmtHDJ+FFIA0EWNEdMmQIq7v16tUzzaNWa9eufdddd/36669pEEhNikD6CEiJpo+9WhaBTCaAnRHWRiNHjsTyiH4cOHDg1ltvrVu37rp16zK5W5JdBOIjICUaHy/lFgERsATy5MkzaNAgzsCceeaZ5iJpTJBuu+22/fv322xKiECACUiJBnhw1TURSAUBvDG8+uqr48aNwz8D7f3222933nlnzZo133jjjVQ0rzZEIK0EpETTil+Ni0AgCOAdsF+/fps2bcIPg+nQe++916BBg4EDB/7nP/8JRBfVCRGITEBKNDIXXRUBEYiXwPHHH79kyZKHH34YL/aU/eOPP+69997q1aszT423KuUXgUwhICWaKSMlOUUgMwj06tWLeGpEVTPifvzxx2eddRaeGX788cfM6ICkFIF4CEiJxkNLeUVABGIgQGRv4ns/8cQTxPomO94B8RFYtWpVIqzFUFpZRCCTCEiJZtJoSVYRyCACl1566ZYtWzp16mRk3rZtW8uWLXv27Llnz54M6oVEFYHoBKREo/PRtyIgAu4JlCpVavbs2bNmzSpdurSpZerUqcRTe+GFF9xXqpIi4CcCUqJ+Gg3JIgJBJHDeeecxJe3atavp3I4dO9q3b3/JJZd8++23Qeyu+pRdBKREs2u81VsRSAuBEiVKPPXUU/Pnz2e71AjAR3weMUlNizxqVAS8IiAl6hVJ1SMCIpALgTZt2mC42717d5Pvm2++6dKlC5umO3fuzKWkvhYBvxKQEvXryEguEQgiAY6Qsi3KcVIOlZr+zZkzhynpk08+GcTuqk/BJyAlGvwxVg9FwG8EcGxE6LQ+ffrg6gjZdu/ejSlv69att2/f7jdRJY8IRCcgJRqdj74VARFICoHChQtzeBRnRrjeNQ0sXLgQw90pU6ZwrjQpTapSEUgCASnRJEBVlSIgArERIPwLgV+uv/56AsJQYu/evVdeeSXz1E8//TS2CpRLBNJMQEo0zQOg5kUgywkQjnTMmDGEJmUaalAsW7asWrVq48ePx/tulsNR9/1PQErU/2MkCUUg+ATq1atHNO9bbrklb9689Hbfvn2EhWnUqNEHH3wQ/M6rh5lMQEo0k0dPsotAgAjkz5+fQKRvv/02sUhNt15//fVTTz119OjRv//+e4A6qq4EioCUaKCGU50RgUwnUKNGjTfffPOOO+4oUKAAffnll18GDRpUv359gpVmetckfyAJSIkGcljVKRHIYAKs6N56662s7p522mmmG0xP69Spg2b99ddfM7hjEj2IBKREgziq6pMIZD4BPDBgbcRaLpZH9ObAgQNDhgxBla5duzbzO6ceBIeAlGhwxlI9EYGAETjkkEMGDBiwceNGLIxM10iztIv90f79+wPWWXUnQwlIiWbowElsEcgWAhUrVly+fDknXooUKUKff/vttxEjRmBwtHr16mxBoH76mICUqI8HR6KJgAj8RQDvgH379sVTYIsWLQyS999/v2HDhjfccMN//vMfQRKBNBKQEk0jfDUtAiIQB4Hjjjtu0aJFkydPLlasGMVwxXDfffdVr16deWoctSirCHhKQErUU5yqTAREIMkEevbsSTy1du3amXY+/vjjpk2b4sv+xx9/THLLql4EIhCQEo0ARZdEQAT8TOCoo4564YUXiJ52xBFHICcO6ydOnFi1alXmqX4WW7IFkoCUaCCHVZ0SgeATuPjii7ds2dK5c2fT1W3btrVq1YqI33v27Al+59VD3xCQEvXNUEgQERCBOAmUKlVq5syZs2fPLl26tCn66KOPcsB07ty5cdak7CLgkoCUqEtwKiYCIuATAp06dWJKeskllxh5vvrqqw4dOnTt2vXbb7/1iYQSI8AEpEQDPLjqmghkC4ESJUpMmzbtxRdfLFeunOnzh1WicgAAIutJREFUM888w5R0xowZ2YJA/UwTASnRNIFXsyIgAl4TaN26NYa7mO+air/55psLLrigY8eOzE29bkr1icD/CEiJ6lYQAREIDoGiRYtykPTll18uX7686dXzzz9PuO8nnngiOJ1UT/xEQErUT6MhWURABLwg0KxZM0Kn4eQI77vUt3v37ssvv/ycc8754osvvKhedYjA3wSkRP9moZQIiEBgCBQuXBh3uytWrKhUqZLp1EsvvcSUdNKkSZwrDUw31ZG0E5ASTfsQSAAREIFkEWjQoMGGDRsIBZMnTx7awKvRv/71L+apn3zySbKaVL1ZRkBKNMsGXN0VgSwjULBgQYKSEvIFl0am66+88kq1atXGjRuH990sg6Huek9AStR7pqpRBETAbwTq1q1LNO/bbrstX758yEbsl/79+5955plEg/GbqJInswhIiWbWeElaERABlwTy588/fPjwt956q1atWqaKVatWEZd05MiRv//+u8tKVSzrCUiJenMLsGRkKipQoIA3NaoWERCBJBCoUaPGmjVr7rrrLvNT3b9//4033njaaadhzZuE1lRl8AlIiXozxgQHJrLEZZddVqFCBW9qVC0iIALJIZA3b96bb755/fr19evXNy2w0lu7du1hw4b9+uuvyWlTtQaWwMGy9k58bDH527x580cffYSvTv727dvHvgs/VIzsy5Qpc8IJJ+B+jETiDakGERABDwlgWHT//fezUcoWqakWg6NHHnmkTp06HrbiSVWPP/54t27d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)

# In[ ]:





# In[38]:


# Global variabes

# train for one data point to test convergence



N, seq_len, emb_dim = 1, 6, 64   # d_model=emb_dim
N, seq_len, vocab_size = 1, 6, 121

"""
edge_index = torch.tensor([[0, 0],
                           [0, 1],
                           [0, 2],
                           [0, 3],
                           [1, 1],
                           [1, 2],
                           [2, 2],
                           [2, 3],
                           [3, 3]])
"""

edge_index = None

input = torch.rand(N, seq_len, emb_dim)




label = torch.randint(0, vocab_size, (N, seq_len))    # nodes mutilclass classification, nodes/words

print('dummy input.shape', input.shape)
print('dummy label.shape', label.shape)
print('dummy label', label)


# In[39]:


q_model = make_model(vocab_size=vocab_size, n_encoder_layer=2,
               d_model=emb_dim, d_ff=2048, h=8, dropout=0.1)

q_loss_fn = nn.CrossEntropyLoss()
q_optimizer = torch.optim.Adadelta(q_model.parameters(), lr=10)
q_scheduler = torch.optim.lr_scheduler.StepLR(q_optimizer, step_size=30, gamma=0.1, verbose=True)  # very important trick

#q_model


# In[40]:


def train(input, edge_index, label, model, loss_fn, optimizer):
    # train one step with dummy data

    optimizer.zero_grad()

    output = model(input, edge_index)
    print('output.shape', output.shape)
    output = output.transpose(1, 2)  # swap dims for nn.CrossEntropyLoss()

    loss = loss_fn(output, label)
    print('loss', loss)

    loss.backward()

    optimizer.step()

    q_scheduler.step()

    return loss.item()


# In[41]:
#
#
# history = []
# for i in range(120):
#     print(f'===step{i+1}===')
#     history.append(train(input, edge_index, label, q_model, q_loss_fn, q_optimizer))
#
#
# # In[42]:
#
#
# import matplotlib.pyplot as plt
#
# plt.plot(history)
# plt.show()
#
#
# # In[43]:
#
#
# output = q_model(input, edge_index)
# torch.argmax(F.softmax(output, dim=-1), dim=-1)
#
#
# # In[44]:
#
#
# label
#
#
# # ![q-transformer-encoder.png](data:image/png;base64,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)
#
# # In[ ]:
#
#
# # layers[0] first layer N=0 refer to above figure
#
# q_model.encoder.layers[0].self_attn.attn_scores
#
#
# # In[ ]:




