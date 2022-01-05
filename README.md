# DeepQuantum 
---

## 编写目的
本文档介绍了DeepQuantum量子计算软件库的使用方法，包括对量子计算的一般性知识进行介绍以及软件库的功能函数的介绍，指导用户自行搭建模拟的通用型量子逻辑门线路以及已定义的量子神经网络层。

---
## 运行环境
| 硬件环境 |               |  软件环境  |                   |
| :--: | :-----------: | :----: | :---------------: |
|  名称  |      型号       |   名称   |        版本         |
| CPU  | Intel/AMD/ARM |  操作系统  | Windows/Linux/MAC |
|  内存  |      2GB      | 深度学习框架 |   Pytorch 1.10    |
|  硬盘  |      8GB      |   语言   |   Python 3.6及以上   |

***
## 目录
* [使用说明](#1)
    - [依赖库安装](#1.1)
    - [运行步骤](#1.2)
* [子程序作用举例](#2)
    - [Circuit类的作用](#2)
    - [类外调用函数](#2.2)
    - [基于DeepQuantum的量子线路](#2.3)
* [量子神经网络](#3)
    - [ QuConv](#3.1)
    - [DeQuConv](#3.2)
    - [QuPool](#3.3)
    - [DeQuPool](#3.4)
    - [QuLinear](#3.5)
    - [QuAE](#3.6)
    - [QuGRU](#3.7)
    - [QuSAAE](#3.8)
    - [QuML](#3.9)
    - [QuAttention](#3.10)
* [TorchScript IR](#4)
    - [TorchScript简介](#4)
    - [DeepQuantum](#4.2)
* [组织交流方式](#5)
* [贡献者](#6)
* [鸣谢](#7)
* [许可证](#8)
* [To-List](#9)

<span id="1"></span>
***
## 使用说明

<span id="1.1"></span>
### 依赖库安装
* Pytorch 1.10版本以上
* Numpy
* Scipy
* typing

### 安装命令
以deepquantum 0.0.2版本为例：
`pip install deepquantum==0.02`
上述的依赖包在安装DeepQuantum时，会附带安装，无须另外安装。

<span id="1.2"></span>
### 运行步骤
运行代码库代码后，首先可以实例化一个量子线路Circuit，参数为线路的比特数，随后可以选择添加门，例如添加一个rx旋转门，参数中第一位为目标	比特位置，第二位为旋转角度；

```python
cir=Circuit(2)
cir.rx(1,torch.tensor(0.2))
```
添加量子逻辑门完毕后，可以查看已经添加的门的酉矩阵；
```python
cir.U
tensor([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])
```
随后选择再添加一个受控非门，其控制比特位是0，受控比特位是1，再次打印已添加的门的酉矩阵；
```python
cir.cnot(0,1)
cir.U
tensor([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])

```
最后，可以通过get()函数得到整个量子线路上所有逻辑门的整体酉矩阵。
```python
cir.get()
tensor([[0.9950+0.0000j, 0.0000-0.0998j, 0.0000+0.0000j, 0.0000+0.0000j],
        [0.0000-0.0998j, 0.9950+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
        [0.0000+0.0000j, 0.0000+0.0000j, 0.0000-0.0998j, 0.9950+0.0000j],
        [0.0000+0.0000j, 0.0000+0.0000j, 0.9950+0.0000j, 0.0000-0.0998j]])
```

<span id="2"></span>
***
## 子程序作用举例
### Circuit类的作用	
Circuit类定义了量子线路包含的信息，通过实例化Circuit类可以模拟搭建量子线路。Circuit类的初始化函数中包含三个空列表，分别用于保存放置门的资源信息、放置门的酉矩阵以及用于测量的酉矩阵，还提供一个初始化的|00...0>态。
```python
class Circuit(object):
    def __init__(self, n:int):
        self.n_qubits = n  # 总QuBit的个数
        self.U = torch.eye(int(2**self.n_qubits)) + 0j
        self.u = []    # 顺序保存酉矩阵

        #线路的初始态，默认全为|0>态
        self.state_init = torch.zeros(int(2**self.n_qubits))
        self.state_init[0] = 1
        self.state_init = self.state_init + 0j
```
Circuit的函数包括rx、rxx、x_gate、cnot等添加门的函数，
```python
    def cnot(self, control_qubit: int, target_qubit: int):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert isinstance(control_qubit, int), \
            "control qubit is not integer"
        assert control_qubit <= self.n_qubits
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_u(two_qubit_control_gate(x_gate(), self.n_qubits, control_qubit, target_qubit))

    def x_gate(self, target_qubit:int):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_u(gate_expand_1toN(x_gate(), self.n_qubits, target_qubit))
```
还包括用于计算整体线路酉矩阵的get函数，
```python
    def get(self):
        self.U = gate_sequence_product(self.u, self.n_qubits)
        return self.U
```
以及重置线路的clear函数。
```python
    def clear(self):
        # 清空
        self.u = []
        self.U = torch.eye(int(2**self.n_qubits)) + 0j
```

<span id="2.2"></span>
### 类外调用函数
|           函数名称            |                    功能                    |
| :-----------------------: | :--------------------------------------: |
|       multi_kron 函数       | 此函数用于对多个酉矩阵进行求克罗内克积的操作。数学上，克罗内克积是两个任意大小的矩阵间的运算。克罗内克积是张量积的特殊形式 |
|        rx、ry、rz函数         | 此类函数用于初始化一个单比特的旋转门酉矩阵。配和gate_expand_1toN函数，可以将单比特酉矩阵扩张成在整个线路上的酉矩阵 |
|          cont函数           | 此函数用于初始化一个两比特的受控非门。配和two_qubit_control_gate函数，可以将两比特酉矩阵扩张成在整个线路上的酉矩阵 |
|  x_gate、y_gate、z_gate函数   |             此类函数用于初始化一个泡利矩阵              |
|     gate_expand_1toN      |        此函数用于将一个单比特门的酉矩阵扩张成n比特酉矩阵         |
| two_qubit_control_gate函数  |        此函数用于将两比特受控门的酉矩阵扩张成n比特酉矩阵         |
| two_qubit_rotation_gate函数 |        此函数用于将两比特旋转门的酉矩阵扩张成n比特酉矩阵         |
|   multi_control_gate 函数   |        此函数用于将多比特控制门的酉矩阵扩张成n比特酉矩阵         |
| gate_sequence_product 函数  |   此函数被Circuit类内部get函数调用，用于计算整个线路的整体酉矩阵   |
|           dag函数           |            此函数用于求某个酉矩阵的共轭转置矩阵            |
|         ptrace 函数         | 此函数用于对某个密度矩阵做偏迹测量，定义与克罗内克积运算相反的运算叫做偏迹运算  |
|         measure函数         | 此函数使用泡利z矩阵作为测量算子，对某个密度矩阵测量求期望，获得n比特个测量结果 |
|            I函数            |               此函数用于生成单位矩阵                |
|        Hadamard函数         |             此函数用于生成一个单比特哈达玛门             |
|        encoding函数         |               此函数用于编码密度矩阵                |
|        IsUnitary函数        |             此函数用于判断矩阵是否为酉矩阵              |
|      IsNormalized函数       |             此函数用于判断一个矢量是否归一              |
|       IsHermitian函数       |             此函数用于判断矩阵是否是厄密矩阵             |
|      get_fidelity函数       |                此函数用于保真度计算                |
|   get_trace_distance函数    |                此函数用于计算迹距离                |

<span id="2.3"></span>
### 基于DeepQuantum的量子线路
量子线路是由量子线互连的量子门的集合。在基于DeepQuantum的量子计算中，线路的实际结构、门的数量和类型以及互连方案由所要执行的幺正变换U决定。

#### 酉矩阵维度
以3比特线路为例：
1. 量子比特是二维复向量空间中的向量
2. 若考虑一个由n个量子比特组成的系统，则其空间是一个$2^n$维的希伯特复空间，即$2^3=8$

##### 酉矩阵
1. 量子比特的演化是线性的，所以量子门可以用矩阵进行表示（酉矩阵）
2. 量子门矩阵的条件：经过量子门作用后得到的状态也要满足基态系数平方和为1
3. 单量子比特门的相应矩阵U（酉矩阵）要满足的条件是“酉性”，即 $U^\dagger U=I$, 其中$U^\dagger$是U的共轭转置，I是单位矩阵。
  酉性限制是对量子门的唯一限制，每个酉矩阵都定义一个有效的量子门

#### 纠缠态
对于一个量子系统，处于纠缠状态的子系统之间会相互影响，对一个子系统的测量行为会改变另外一个子系统的状态

#### CNOT门（受控非门）
将控制量子比特和目标量子比特作异或运算，并将结果储存在目标量子比特中

#### Toffoli门
1. Toffoli接受三个比特作为输入，其中两个控制比特（不受门的影响），一个目标比特；当两个控制比特都为1时，对目标比特进行翻转，否则目标比特保持不变。
2. 若连续进行两次Toffoli门操作，结果不发生变化，因此Toffoli门是可逆门，逆是它本身

#### 泡利X门
单比特量子门，相当于逻辑门中的非门，量子非门的作用是线性的。
$$
X=\begin{bmatrix}
   0 & 1 \\
   1 & 0
\end{bmatrix}
$$
$$\alpha \lvert{0}\rangle+\beta \lvert{1}\rangle \xrightarrow{X} \alpha \lvert{1}\rangle+\beta \lvert{0}\rangle​$$

#### 泡利Y门
单比特量子门


$$
Y=\begin{bmatrix}
   0 & -i \\
   i & 0
\end{bmatrix}
$$


#### 泡利Z门
单比特量子门，它保持$\lvert{0}\rangle$不变，翻转$\lvert{1}\rangle$的符号变为$-\lvert{1}\rangle$

$$
Z=\begin{bmatrix}
   1 & 0 \\
   0 & -1
\end{bmatrix}
$$

$$\alpha \lvert{0}\rangle+\beta \lvert{1}\rangle \xrightarrow{Z} \alpha \lvert{0}\rangle-\beta \lvert{1}\rangle$$


#### 交换门
$$
SWAP=\begin{bmatrix}
   1 & 0 & 0 & 0 \\
   0 & 0 & 1 & 0 \\
   0 & 1 & 0 & 0 \\
   0 & 0 & 0 & 1 \\
\end{bmatrix}
$$



#### 哈达玛门
$$
H=\frac{1}{\sqrt{2}}\begin{bmatrix}
   1 & 1 \\
   1 & -1
\end{bmatrix}
$$


哈达玛门可以将$\lvert{0}\rangle$变为$(\lvert{0}\rangle+\lvert{1}\rangle)/\sqrt{2}$，把$\lvert{1}\rangle$变到同样的$(\lvert{0}\rangle-\lvert{1}\rangle)/\sqrt{2}$（连续两次应用哈达玛门等于什么也没做）

#### 张量积

$$
\begin{bmatrix}


1 & 0 \\
0 & 1 \\


\end{bmatrix} 


\bigotimes 


  \begin{bmatrix} 


2 & 3 \\ 
4 & 5 \\ 


\end{bmatrix} = \begin{bmatrix}
   1\times2 & 1\times3 & 0\times2 & 0\times3 \\
   1\times4 & 1\times5 & 0\times4 & 0\times5 \\
   0\times2 & 0\times3 & 1\times2 & 1\times3 \\
   0\times4 & 0\times5 & 1\times4 & 1\times5 \\
\end{bmatrix}
$$


##### 示例
```python
from deepquantum.utils import multi_kron

a = torch.tensor([[1, 0], [0, 1]])
b = torch.tensor([[1, 0], [0, 1]])

result = multi_kron([a, b])
```

#### 共轭转置矩阵
- 共轭复数：实部相同，虚部互为相反数的复数互为共轭复数
- 共轭转置：先进行共轭操作，然后对矩阵进行转置
##### 示例
```python
from deepquantum.utils import dag

a = torch.randint(0,4,(2,2)) + 0j
a_dag = dag(a)
```

#### 密度矩阵
当量子系统处于纯态时，系统的状态通常用波函数或态矢量表示；当系统处于混合态时，系统的状态用密度矩阵表示。
##### 密度矩阵计算
1. 纯态：$\rho = \lvert{\psi}\rangle\langle{\psi}\lvert$
2. 混态：$\rho_{mix} = \sum_{i} p_{i} \lvert{\psi}\rangle\langle{\psi}\lvert$
##### 示例
```python
from deepquantum.utils import encoding

a = torch.randint(0,4,(2,2))
a_enc = encoding(a)
```


#### 偏迹
对于一个包含多个子系统的量子系统，应用偏迹运算可以求解不同子系统的约化密度矩阵
##### 示例
```python
import torch
from deepquantum.utils import ptrace, encoding
    
a = torch.ones(8,8)
a_enc = encoding(a)

a_ptr = ptrace(a_enc,1,2)
```

#### 测量
使用DeepQuantum中的measure()函数对某个子系统进行测量求期望
##### 示例
```python
from deepquantum.utils import expecval_ZI, measure, encoding

a = torch.rand(8,8)   # 3个量子比特
a_enc = encoding(a)   # 计算密度矩阵
a_exp = expecval_ZI(a_enc, 3, 0)
a_measure = measure(a_enc, 3)
```

#### 保真度
表示信息在传输和处理过程中保持原来状态的程度，这里用来衡量输入输出量子态之间的相似程度。
##### 示例
```python
from deepquantum.utils import get_fidelity,encoding

a = torch.rand(8,8)
b = torch.rand(8,8)
a = encoding(a)
b = encoding(b)

fidelity = get_fidelity(a,b)
```

<span id="3"></span>
***
## 量子神经网络

<span id="3.1"></span>
### QuConv
DeepQuantum为方便用户使用提供了量子卷积核，量子卷积之后可以是量子层或经典层。与经典卷积的主要区别在于，量子电路可以生成高度复杂的内核，其计算至少在原则上是经典上难以处理的。

#### QuConv提供的接口
* QuConvXYZ：在量子线路上放置了单比特泡利旋转门以及双比特旋转门，如$R_x(\theta_1)$、$R_y(\theta_3)$、$R_z(\theta_5)$以及$R_{xx}(\theta_6)$等，该量子卷积核有14个参数
* QuConvSXZ：组成和QuConvXYZ相同，但是所包含的参数较少，线路较简单

#### QuConv使用方法
以QuConvXYZ为例，首先输入一个数据实例化量子卷积核，这里注意实例化数据必须为2的倍数
```python
QuConv=QuConvXYZ(4)  #定义了4比特量子线路
```
接下来将输入数据编码成量子态，并输入到定义的卷积核中
```python
X=torch.rand(16,16)  #随机生成一个符合输入维度的矩阵
X=encoding(X)      #将经典数据编码成量子态密度矩阵

out=QuConv(X)
```
QuConvSXZ的使用方法与QuConvXYZ方法类似

#### 示例
```python
import torch
import torch.nn as nn
import deepquantum.nn as Qnn
from deepquantum.utils import encoding

class My_QuConv(nn.Module):
    def __init__(self,n_qubits):
        super(My_QuConv, self).__init__()
        self.n_qubits = n_qubits
        self.linear = nn.Linear(250,2**self.n_qubits)
        self.QuConvX = Qnn.QuConvXYZ(self.n_qubits)
        self.QuConvS = Qnn.QuConvSXZ(self.n_qubits)
        
    def forward(self,x):
        x = self.linear(x)
        x = Gram(x)      #将向量x输入到Gram函数中编码为对应的量子态密度矩阵
        x = self.QuConvX(x)
        out = self.QuConvS(x)
        return out
    
#初始化模型时输入需要是2的倍数    
module = My_QuConv(4)
x = torch.rand(1,250)
result = module(x)
```

<span id="3.2"></span>
### DeQuConv
DeepQuantum提供了DeQuConv是上述QuConv的$U^\dagger$，在量子计算过程中相当于数据在QuConv的量子线路中逆演化。

#### DeQuConv提供的接口
* DeQuConvXYZ：在量子线路上放置了单比特泡利旋转门以及双比特旋转门同QuConv，然后使用dag()操作得到$U$的共轭转置矩阵$U^\dagger$
* DeQuConvSXZ：与上述类似

#### DeQuConv使用方法
DeQuConv使用方法与QuConv相同，请参照QuConv的使用方法。

#### 示例
```python
import torch
import torch.nn as nn
import deepquantum.nn as Qnn
from deepquantum.utils import Gram

class My_DeQuConv(nn.Module):
    def __init__(self,n_qubits):
        super(My_DeQuConv, self).__init__()
        self.n_qubits = n_qubits
        self.linear = nn.Linear(250,2**self.n_qubits)
        self.DeQuConvX = Qnn.DeQuConvXYZ(self.n_qubits)
        
    def forward(self,x):
        x = self.linear(x)
        x = Gram(x)
        out = self.DeQuConvX(x)
        return out

module = My_DeQuConv(4)
x = torch.rand(1,250)
result = module(x)
```

<span id="3.3"></span>
### QuPool
DeepQuantum提供了量子池化核方便用户直接使用，引入受控门将相邻线路的量子态纠缠起来，模拟经典池化提取数据更深层次特征的作用。

#### QuPool提供的接口
* QuPoolXYZ：在量子线路上放置了单比特泡利旋转门以及受控门cnot()，其中单比特泡利旋转门如$R_x(\theta_1)$、$R_y(\theta_3)$、$R_z(\theta_5)$
* QuPoolSX：只包含了单比特x泡利旋转门和受控门cnot()

#### QuPool使用方法
以QuPoolXYZ为例，首先输入一个数据实例化量子池化核，这里注意示例化时的数必须为2的倍数
```python
QuPool=QuPoolXYZ(4)  #定义了4比特量子线路
```
接下来将输入数据编码成量子态，并输入到定义的卷积核中
```python
X=torch.rand(16,16)  #随机生成一个符合输入维度的矩阵
X=encoding(X)      #将经典数据编码成量子态密度矩阵

out=QuPool(X)
```
QuPoolSX的使用方法与QuPoolXYZ方法类似

#### 示例
```python
import torch
import torch.nn as nn
import deepquantum.nn as Qnn
from deepquantum.utils import Gram

class My_QuPool(nn.Module):
    def __init__(self,n_qubits):
        super(My_QuPool, self).__init__()
        self.n_qubits = n_qubits
        self.linear = nn.Linear(250,2**self.n_qubits)
        self.QuConvX = Qnn.QuConvXYZ(self.n_qubits)
        self.QuPoolX = Qnn.QuPoolXYZ(self.n_qubits)
        self.QuPoolS = Qnn.QuPoolSX(self.n_qubits)
        
    def forward(self,x):
        x = self.linear(x)
        x = Gram(x)
        x = self.QuConvX(x)
        x = self.QuPoolX(x)
        out = self.QuPoolS(x)
        return out

module = My_QuPool(4)
x = torch.rand(1,250)
result = module(x)
```

<span id="3.4"></span>
### DeQuPool
同理DeQuPool是上述QuPool量子线路$U$酉矩阵矩阵的$U^\dagger$。

#### DeQuPool提供的接口
* DeQuPoolXYZ：在量子线路上放置了单比特泡利旋转门以及受控门同QuPool，然后使用dag()得到$U$酉矩阵的$U^\dagger$
* DeQuPoolSX：与上述类似

#### DeQuPool使用方法
DeQuPool使用方法与QuPool相同，请参照QuPool的使用方法。

#### 示例
```python
import torch
import torch.nn as nn
import deepquantum.nn as Qnn
from deepquantum.utils import Gram

class My_DeQuPool(nn.Module):
    def __init__(self,n_qubits):
        super(My_DeQuPool, self).__init__()
        self.n_qubits = n_qubits
        self.linear = nn.Linear(250,2**self.n_qubits)
        self.QuConvX = Qnn.QuConvXYZ(self.n_qubits)
        self.QuPoolX = Qnn.QuPoolXYZ(self.n_qubits)
        self.DeQuPoolX = Qnn.DeQuPoolXYZ(self.n_qubits)
        self.DeQuPoolS = Qnn.DeQuPoolSX(self.n_qubits)
        
    def forward(self,x):
        x = self.linear(x)
        x = Gram(x)
        x = self.QuConvX(x)
        x = self.QuPoolX(x)
        x = self.DeQuPoolX(x)
        out = self.DeQuPoolS(x)
        return out
    
module = My_DeQuPool(4)
x = torch.rand(1,250)
result = module(x)
```

<span id="3.5"></span>
### QuLinear
DeepQuantum中的QuLinear是量子神经网络中被用来替代PyTorch中的nn.Linear,增强量子神经网络的表达能力的。

#### QuLinear提供的接口
* QuLinear:先定义量子线形成的输入和输出维度，然后将数据输入到QuLinear中得到输出结果

#### QuLinear使用方法
首先定义输入特征和输出特征维度，这里需要注意输出维度out_features需要满足$2^1\leq2^i\leq2^{in-features}$
```python
QL=QuLinear(6,32)  #定义了输入和输出特征
```
接下来生成输入数据，用户这里应该使用自己的数据，上一步的定义也需要根据需要自行定义
```python
X=torch.rand(64,1)  #随机生成一个符合输入维度的矩阵

out=QL(X)
print('结果：',out)

#输出结果
结果： tensor([[0.0834, 0.0257, 0.0419, 0.0088, 0.0652, 0.0187, 0.0434, 0.0105, 0.1252,
         0.0317, 0.0511, 0.0113, 0.0973, 0.0239, 0.0466, 0.0113, 0.0141, 0.0157,
         0.0067, 0.0228, 0.0223, 0.0134, 0.0236, 0.0210, 0.0284, 0.0137, 0.0218,
         0.0143, 0.0145, 0.0254, 0.0122, 0.0339]],
       grad_fn=<UnsqueezeBackward0>)
```
这里输入的数据X是用户需要做线性激活的经典数据，输入QuLinear后会自动将其编码为量子态，经过演化后对其进行测量并输出结果

#### 示例
```python
import torch
import torch.nn as nn
import deepquantum.nn as Qnn

class My_QuLinear(nn.Module):
    def __init__(self,n_qubits,out_qubits):
        super(My_QuLinear, self).__init__()
        self.n_qubits = n_qubits
        self.out_features = 2**out_qubits
        self.linear = nn.Linear(250,self.n_qubits)
        self.Qulinear = Qnn.QuLinear(self.n_qubits,self.out_features)
        
    def forward(self,x):
        x = self.linear(x)
        out = self.Qulinear(x)
        return out
    
module = My_QuLinear(6,5)
x=torch.rand(1,250)
result = module(x)
print('结果：',result)

#结果如下
结果： tensor([[0.0298, 0.0751, 0.0795, 0.0231, 0.0986, 0.0326, 0.0480, 0.0834, 0.0207,
         0.0434, 0.0542, 0.0129, 0.0597, 0.0230, 0.0282, 0.0551, 0.0113, 0.0191,
         0.0258, 0.0065, 0.0273, 0.0128, 0.0130, 0.0264, 0.0053, 0.0163, 0.0125,
         0.0058, 0.0198, 0.0059, 0.0105, 0.0143]],
       grad_fn=<UnsqueezeBackward0>)
```

<span id="3.6"></span>
### QuAE
量子自编码网络来压缩和重构输入量子态，为此需要用量子线路实现编码器encoder网络和解码器decode网络。DeepQuantumy提供已定义好的AE网络只需将数据输入即可重构出量子态。

#### QuAE 提供的接口
* Qu_AEnet：实例化量子神经网络层后，输入压缩数据X,单位矩阵I，编码后保留的比特数N和丢弃的比特数N_Trash

#### QuAE使用方法
首先实例化量子自编码网络的比特数
```python
QuAE=Qu_AEnet(8) #实例化一个8比特的量子自编码网络
```
这里对应的输入数据应为$2^8\times2^8$的数据，假如保留7比特的数据那么单位矩阵为$2^{8-7}\times2^{8-7}$
```python
x=torch.rand(256,256)  #随机生成一个符合输入维度的矩阵
X=encoding(x)       #将经典数据编码为量子态密度矩阵
I=torch.eye(2)      #生成符合要求的单位矩阵I
```
将数据输入到量子自编码网络中
```python
out=QuAE(X,I,7,1)
print('结果',out)

#输出结果
结果 tensor([[ 1.4972e-01+1.4203e-08j,  2.3368e-03+6.4389e-02j,
         -6.2205e-02+1.0121e-02j,  ...,
          1.7100e-03+1.3212e-03j, -1.2927e-03+2.9093e-03j,
          3.4853e-04+3.0846e-03j],
        [ 2.3368e-03-6.4389e-02j,  1.2847e-01+6.9849e-09j,
         -7.1227e-02+1.1383e-02j,  ...,
          2.5565e-03-3.5899e-03j, -2.1651e-03+2.4778e-03j,
         -1.0873e-03-2.9004e-03j],
        [-6.2205e-02-1.0121e-02j, -7.1227e-02-1.1383e-02j,
          1.3469e-01-6.2864e-09j,  ...,
         -1.5109e-03+1.5907e-03j,  1.0315e-03-2.6573e-03j,
          2.4558e-03+1.2072e-04j],
        ...,
        [ 1.7100e-03-1.3212e-03j,  2.5565e-03+3.5899e-03j,
         -1.5109e-03-1.5907e-03j,  ...,
          1.6151e-04+5.8208e-10j, -1.2107e-04-1.3201e-05j,
          9.0529e-05-6.1642e-05j],
        [-1.2927e-03-2.9093e-03j, -2.1651e-03-2.4778e-03j,
          1.0315e-03+2.6573e-03j,  ...,
         -1.2107e-04+1.3195e-05j,  3.0566e-04+9.3714e-09j,
          8.7468e-05+7.1358e-05j],
        [ 3.4853e-04-3.0846e-03j, -1.0873e-03+2.9004e-03j,
          2.4558e-03-1.2071e-04j,  ...,
          9.0531e-05+6.1647e-05j,  8.7472e-05-7.1363e-05j,
          3.6828e-04+3.1723e-09j]], grad_fn=<MmBackward0>)
```
用户可以参照以上方法，根据自己的数据定义量子线路数量以及保留比特数N等

#### 示例
```python
import torch
import torch.nn as nn
import deepquantum.nn as Qnn
from deepquantum.utils import Gram

class My_QuAE(nn.Module):
    def __init__(self,n_qubits,n_retain):
        super(My_QuAE, self).__init__()
        self.n_qubits = n_qubits
        self.n_retain = n_retain
        self.n_trash = self.n_qubits-self.n_retain
        self.QuAE = Qnn.Qu_AEnet(self.n_qubits)
        self.I = torch.eye(2**self.n_trash)
        
    def forward(self,x):
        x = Gram(x)
        out = self.QuAE(x, self.I, self.n_retain, self.n_trash)
        return out
 
module = My_QuAE(8,7)
x = torch.rand(1,256)
result = module(x)

```
注意初始化时需要满足$0\leq{n retain}\leq{n qubits}$

<span id="3.7"></span>
### QuGRU
GRU(Gate Recurrent Unit)是循环神经网络的一种，是为了解决长期记忆和反向传播中的梯度等问题而提出来的。QuGRU原理在经典GRU上做了逻辑上的映射。

#### QuGRU提供的接口
* QuGRU：实例化的时候需要输入三个数，分别是输入维度、隐藏维度和输出维度，然后将数据x输入模型，得到QuGRU计算后结果

#### QuGRU使用方法
首先定义输入维度、隐藏维度和输出维度，这里需要注意隐藏维度$2^1\leq2^i\leq2^{input-dim}$
```python
QG=QuGRU(3,8,3)  #定义了输入、隐藏以及输出维度
```
接下来生成输入数据，用户这里应该使用自己的数据，上一步的定义也需要根据需要自行定义。注意这里张量输入维度为(batch_size, seq_length, input_dim)
```python
X=torch.rand(1,8,3)  #随机生成一个符合输入维度的矩阵

out=QG(X)
print('结果',out)

#输出结果
结果 tensor([[-0.0015, -0.0575,  0.1222]], grad_fn=<AddmmBackward0>)
```
这里输入数据x是用户需要输入QuGRU的经典数据，输入QuGRU后会自动将其编码为量子态数据经过演化后输入符合要求的结果

#### 示例
```python
import torch
import torch.nn as nn
import deepquantum.nn as Qnn

class My_QuGRU(nn.Module):
    def __init__(self,input_dim, hidden_dim, Quoutput_dim):
        super(My_QuGRU, self).__init__()
        self.n_qubits = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = Quoutput_dim
        self.QuGRU = Qnn.QuGRU(self.n_qubits,self.hidden_dim, self.output_dim)
        self.linear = nn.Linear(self.output_dim,4)
        
    def forward(self,x):
        x = self.QuGRU(x)
        out = self.linear(x)
        return out
    
module = My_QuGRU(3,8,3)
x=torch.rand(1,8,3)
result = module(x)
print('结果：',result)

#输出结果
结果： tensor([[-0.0485,  0.3078,  0.5784, -0.1791]], grad_fn=<AddmmBackward0>)
```

<span id="3.8"></span>
### QuSAAE
在量子对抗自编码网络中，编码器对输入数据X进行压缩，主要保留与标签数据Y无关的数据$Z_x$。解码器输入$Z_x$和Y的混合数据，重构输入数据$R_x$。判别器输入$Z_x$及其正态分布数据用于后续的训练和优化。

#### QuSAAE 算法逻辑图

<img src="./pictures/SAAE.png" style="zoom:45%">

#### QuSAAE提供接口
* Q_Encoder：先实例化量子编码器后，输入数据和保留比特数

* Q_Decoder：先实例化后，输入$Z_x$和Y

* Q_Discriminator：根据$Z_x$实例化，然后输入判别数据

#### QuSAAE使用方法
调用DeepQuantum 0.0.2版本及以上中的量子编码器
```python
QE=Q_Encoder(6)    #实例化一个6比特的量子编码器
```
这里使用6比特实例化量子编码器，对应的输入数据X的维度应为$2^6\times2^6$
```python
X=torch.rand(64,64) #随机生成一个符合输入维度的矩阵
X=encoding(X)      #将经典数据编码成量子态密度矩阵
Zx=QE(X,4)        #输入量子判别器中并输入保留的比特数

```
将数据X输入到实例化的量子编码器中，并编码压缩得到$Z_x$维度为$2^4\times2^4$,解码器和判别器的使用方法类似，均可根据输入数据维度对量子编码器，量子解码器、量子判别器的线路数进行定义。
```python
Y=torch.rand(64,64) #随机生成一个标签数据
Y=encoding(Y)     #将经典数据编码成量子态密度矩阵
QD=Q_Decoder(10)   #实例化一个10比特的量子解码器
Rx=QD(Zx,Y,6)     #将编码得到的Zx和Y输入，解码重建得到2^6*2^6的Rx
QDis=Q_Discriminator(4) #根据Zx定义量子判别器
QDisS=QDis(Zx)  #量子判别器示例输入
```

#### 示例
```python
import torch
import torch.nn as nn
import deepquantum.nn as Qnn
from deepquantum.utils import Gram

class My_QuSAAE(nn.Module):
    def __init__(self,n_qubits, n_retain, ):
        super(My_QuSAAE, self).__init__()
        self.n_qubits = n_qubits
        self.n_retain = n_retain
        self.linearE = nn.Linear(250, 2**self.n_qubits)
        self.QuEncoder = Qnn.Q_Encoder(self.n_qubits)
        self.linearD = nn.Linear(33,2**self.n_qubits)
        self.QuDecoder = Qnn.Q_Decoder(self.n_qubits+self.n_retain)
        self.QuDiscriminator = Qnn.Q_Discriminator(self.n_retain)
        
    def forward(self, x, y):
        x = self.linearE(x)
        x = Gram(x)
        x = self.QuEncoder(x, self.n_retain)
        y = self.linearD(y)
        y = Gram(y)
        rx = self.QuDecoder(x,y,self.n_qubits)
        out = self.QuDiscriminator(x)
        return rx,out
    
module = My_QuSAAE(6,4)
x = torch.rand(1,250)
y = torch.rand(1,33)
Rx,result = module(x, y)
print('重建数据',Rx)
print('量子判别器结果：',result)

#输出结果
重建数据 tensor([[ 0.0089+3.8299e-09j,  0.0014+4.3687e-03j,  0.0005+4.3297e-03j,
          ..., -0.0031+1.5534e-03j, -0.0011-1.4202e-03j,
         -0.0010-5.6579e-04j],
        [ 0.0014-4.3687e-03j,  0.0109+5.4006e-09j,  0.0031+5.9140e-04j,
          ...,  0.0014+8.9393e-03j,  0.0006-7.0626e-04j,
         -0.0008-2.0498e-03j],
        [ 0.0005-4.3297e-03j,  0.0031-5.9140e-04j,  0.0076-8.1491e-10j,
          ...,  0.0007+1.7250e-03j,  0.0006+4.0059e-03j,
          0.0010+1.1066e-03j],
        ...,
        [-0.0031-1.5534e-03j,  0.0014-8.9393e-03j,  0.0007-1.7250e-03j,
          ...,  0.0160-2.7931e-09j,  0.0009-1.2880e-03j,
         -0.0046-1.0336e-03j],
        [-0.0011+1.4202e-03j,  0.0006+7.0627e-04j,  0.0006-4.0059e-03j,
          ...,  0.0009+1.2880e-03j,  0.0156+7.2760e-12j,
          0.0029-4.8818e-03j],
        [-0.0010+5.6579e-04j, -0.0008+2.0498e-03j,  0.0010-1.1066e-03j,
          ..., -0.0046+1.0336e-03j,  0.0029+4.8818e-03j,
          0.0188-1.9008e-10j]], grad_fn=<AddBackward0>)
量子判别器结果： tensor([[0.6479],
        [0.3726],
        [0.4541],
        [0.4984]])
```
注意这里中间引入的数据Y用户可以根据自己的需求自行修改

<span id="3.9"></span>
### QuML(Mutual Learning)
主要思想是中间使用一个量子互信息的过程，交互后的信息各自都包含着自己和对方的特征，再分别将各自量子互信息后的信息与经典卷积得到的信息拼接，再进行后面的卷积及全连接过程，最后得到包含双方信息的结构。


#### QuML结构图
<img src="./pictures/QuML.png" style="zoom:70%">

#### QuML提供的接口
* Qu_conv_mutual：先实例化模型，然后将需要得到互信息的两个输入输入到模型中即可得到两者间的特征结果

#### QuML使用方法
调用DeepQuantum 0.0.2版本中的Qu_conv_mutual模型
```python
QuCM=Qu_conv_mutual(64,25)    #实例化互信息模型
```
实例化后将需要进行互学习的数据输入到模型中，注意这里可直接输入原始数据不需要编码为量子态数据
```python
X=torch.linspace(1,10,steps=150,dtype=int)  #将1~10数据平均分成150个数据，组成1*150的一维向量
Y=torch.linspace(1,10,steps=1000,dtype=int)  #将1~10数据平均分成1000个数据，组成1*1000的一维向量
```
将数据X和Y输入到QuCM中即可得到互学习后的结果
```python
out=QuCM(x,y)  #将需要互学习的结果输入到模型中
print('结果：',out)  #打印结果

#输出结果
结果： tensor([-298.0587], grad_fn=<ViewBackward0>)
```
用户可根据自己的需要定义输入数据并定义训练过程

#### 示例
```python
import torch
import torch.nn as nn
import deepquantum.nn as Qnn

class My_QuML(nn.Module):
    def __init__(self,embedding_num_x, embedding_num_y):
        super(My_QuML, self).__init__()
        self.embedx = embedding_num_x
        self.embedy = embedding_num_y
        self.QuML = Qnn.Qu_conv_mutual(self.embedx,self.embedy)
        self.linearx = nn.Linear(250, 150)
        self.lineary = nn.Linear(33,1000)
        
    def forward(self, x, y):
        x = self.linearx(x).int()
        x = abs(x)
        y = self.lineary(y).int()
        y = abs(y)
        out = self.QuML(x,y)
        return out
    
module = My_QuML(64,25)
x = torch.linspace(1,10,steps=250)
y = torch.linspace(1,10,steps=33)
result = module(x, y)
print('结果：',result)

#输出结果
结果： tensor([-653.8476], grad_fn=<ViewBackward0>)
```
注意输入到模型中的数据集需要为正整数组成的向量

<span id="3.10"></span>
### QuAttention
量子注意力机制是经典注意力机制的量子版本，用量子比特的方式进行编码和信息处理。量子线路的是对量子信息的操作方法，通过量子线路，我们可以对含有量子信息的量子比特进行旋转，演化等操作。

#### QuAttention提供的接口
* QuAttention：先初始化量子注意力机制比特数n，然后将输入数据按定义的比特数分割成数份,每份的维度为$1\times2^n$，将其编码为量子态密度矩阵并添加到同一列表中，最后将整个列表输入到模型中得到结果

#### QuAtttention使用方法
首先实例化模型
```python
qubits = 2 # 设定比特数
quattention = QuAttention(qubits)
```
注意这里使用的是2比特线路的注意力机制模型，这里的输入数据需要为$2^2$的整数倍，并将数据切成数份长度为$2^2$的向量放入列表中
```python
temp = torch.rand(200)
data = temp.reshape(1,-1) # 输入数据拉长为一维向量
k = int(2**qubits) # 设定切片维度
state_list = []
for i in range(0,data.shape[1],k):
    state = data[0,i:i+k].reshape(1,-1)
    x = Gram(state)
    state_list.append(x)
```
最后将得到的列表输入到模型中得到最后的结果
```python
output = quattention(state_list)
```

#### 示例
```python
import torch
import torch.nn as nn
import deepquantum.nn as Qnn
from deepquantum.utils import Gram

class My_QuAttention(nn.Module):
    def __init__(self, n_qubits, slice_n):
        super(My_QuAttention, self).__init__()
        self.n_qubits = n_qubits
        self.slice_n = slice_n
        self.linear = nn.Linear(100,2**self.n_qubits*self.slice_n )
        self.QuAttn = Qnn.QuAttention(self.n_qubits)
        
        
    def forward(self,x):
        x = self.linear(x)
        state_list = []
        k = 2**self.n_qubits
        for i in range(0,k*self.slice_n, k):
            state = x[0, i:i+k].reshape(1,-1)
            data = Gram(state)
            state_list.append(data)
        out = self.QuAttn(state_list)
        return out
    
module = My_QuAttention(2,10)
x = torch.rand(1,100)
result = module(x)
```


<span id="4"></span>
***
## TorchScript IR
### TorchScript简介
<p>  PyTorch框架提供了一种即时（just-in-time, JIT）编译内联加载方式，即在一个Python文件中，将C++代码作为字符串传递给PyTorch中负责内联编译的函数。在运行Python文件时，即时编译出动态链接文件，并导入函数进行后续运算。torch.jit.torchscript是PyTorch提供的即时编译模块，它是Python语言的一个静态类型子集，并且支持Python语法同时也从Python中分离出来，只专注于Python中用于在Pytorch中表示神经网络模型所需的特性。</p>

<span id="4.2"></span>
### DeepQuantum
DeepQuantum是基于PyTorch框架开发的，量子经典混合神经网络的训练和部署是基于同一有向无环图(DAG)进行的，DeepQuantum 0.0版本是支持TorchScript进行跟踪和编译的，可使用该版本的对量子神经网络层算子进行定义和梯度计算，并对计算过程重写编译器做到将量子神经网络的DAG结构储存在CPU中，而推导过程在QPU或模拟QPU上进行计算提升运算效率。

<span id="5"></span>
***
## 组织交流方式
<img src="./pictures/DeepQuantum交流群.png" style="zoom:100%">

**QQ群：727013373**

**交流邮箱：Algorithms-Applications@turingq.com**

<span id="6"></span>
***
 ## 贡献者
 **图灵量子 算法应用部门**

<span id="7"></span>
***
## 鸣谢
### 参考资料
[1] M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum Information: 10th Anniversary Edition, 10th ed. (Cambridge University Press, New York, NY, USA, 2011).</br>

[2]Alireza Makhzani,Jonathon Shlens,Navdeep Jaitly,Ian J. Goodfellow. Adversarial Autoencoders.[J]. CoRR,2015,abs/1511.05644:

[3]Shayakhmetov Rim,Kuznetsov Maksim,Zhebrak Alexander,Kadurin Artur,Nikolenko Sergey,Aliper Alexander,Polykovskiy Daniil. Erratum: Addendum: Molecular Generation for Desired Transcriptome Changes With Adversarial Autoencoders.[J]. Frontiers in pharmacology,2020,11:

[4]Maxwell Henderson, Samriddhi Shakya, Shashindra Pradhan, Tristan Cook. “Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits.”arXiv:1904.04767, 2019.

[5] PyTorch官方文档：[https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

<span id="8"></span>
***
## 许可证
Copyright 2021 TuringQ

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

<span id="9"></span>
***
## To-do List

- [x] QuConv
- [x] DeQuConv
- [x] QuPool
- [x] DeQuPool
- [x] QuLinear
- [x] QuAE
- [x] QuGRU
- [x] QuSAAE
- [x] QuQML
- [x] QuAttention
- [ ] GAT 
- [ ] QuBiAAE
- [ ] QuReinforce
- [ ] Qustyle-GAN
