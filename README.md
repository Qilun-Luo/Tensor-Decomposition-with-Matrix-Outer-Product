# Tensor-Decomposition-with-Matrix-Outer-Product
Matlab Implementation for the paper:

Luo, Q., Yang, M., Li, W., & Xiao, M. (2023). Multi-Dimensional Data Processing with Bayesian Inference via Structural Block Decomposition. *IEEE Transactions on Cybernetics*, 10.1109/TCYB.2023.3234356.

- **Matrix Outer Product**

The matrix outer product among $U\in \mathbb{R}^{n_1\times n_2}, V\in \mathbb{R}^{n_2\times n_3}$ and $W\in \mathbb{R}^{n_3\times n_1}$ is denoted as: $$\mathcal{T} = U \bullet V \bullet W \in \mathbb{R}^{n_1\times n_2\times n_3},$$
with the $(i,j,k)$-th entry of $\mathcal{T}$  being defined as 
```math
\mathcal{T}_{ijk} = U_{ij}V_{jk}W_{ki},
```

where $i=1,2,\cdots n_1;~j=1,2,\cdots,n_2;~k=1,2,\cdots, n_3$. The graphical illustration of the matrix outer product is shown as follows.

![Matrix Outer Product](images/fig-mop.png?raw=true "Matrix Outer Product")
