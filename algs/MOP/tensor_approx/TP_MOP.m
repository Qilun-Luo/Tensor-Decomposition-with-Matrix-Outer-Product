%TP_MOP Tensor Product on Matrix Outer Product
% A = TP_MOP(U, V, W) is product of three factor tensors.
%----------------------------------------------------------------------
% Input:
%   U:  the first factor tensor of size n1 x n2 x r
%   V:  the second factor tensor of size n2 x n3 x r
%   W:  the third factor tensor of size n3 x n1 x r
% Output:
%   A:  the tensor product for U * V * W
%----------------------------------------------------------------------
% Copyright(c) 2021 Qilun Luo 
% All Rights Reserved.

function A = TP_MOP(U, V, W)

% Check
[n1, n2, r] = size(U);
[nn2, n3, r1] = size(V);
[nn3, nn1, r2] = size(W);

if (n1~=nn1) || (n2~=nn2) || (n3~=nn3) || (r~=r1) || (r~=r2)
    error('Dimensions of factor tensors are not compatible')
end

% Product for third-order tensors
U = reshape(U, [1,n1,n2*r]);
V = reshape(pagetranspose(V), [n3,1,n2*r]);
T = pagemtimes(V, U);
T = reshape(permute(T, [3,1,2]), [n2,r,n3*n1]);
W = reshape(permute(W, [3,1,2]), [r,1,n3*n1]);
A = pagemtimes(T, W);
A = permute(reshape(A, [n2,n3,n1]), [3,1,2]);
