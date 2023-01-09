%UFT_TDMOP Update Factor Tensors for Tensor Decompoistion on 
% Matrix Outer Product.
% U = UFT_TDMOP(A, V, W) updates the factor tensor.
%----------------------------------------------------------------
% Input:
%   A:      a third-order tensor of size n1 x n2 x n3
%   V:      the given second factor tensor
%   W:      the given third factor tensor
% Output:
%   U:      the updated first factor tensor
%----------------------------------------------------------------
% Copyright(c) 2021 Qilun Luo 
% All Rights Reserved.

function U = UFT_TDMOP(A, V, W)

% Check
[n1, n2, n3] = size(A);
[nn2, m3, r] = size(V);
[mm3, nn1, rr] = size(W);

if (n1~=nn1) || (n2~=nn2) || (n3~=m3) || (n3~=mm3) || (r~=rr)
    error('Dimensions of input parameters are not compatible')
end

% % Update with Maltlab R2021a or above
% V = reshape(V, [1, n2, n3*r]);
% W = reshape(pagetranspose(W), [n1, 1, n3*r]);
% T = pagemtimes(W, V);
% mT = reshape(T, [n1*n2,n3*r]);
% T = reshape(mT', [n3,r,n1*n2]);
% TT = pagemtimes(T,'transpose', T,'none');
% CT = num2cell(TT, [1,2]);
% CT{1} = sparse(CT{1});
% DT = blkdiag(CT{:});
% clear CT TT mT V W
% A = reshape(A, [n1*n2, n3])';
% A = permute(A,[1,3,2]);
% TtA = pagemtimes(T,'transpose', A,'none');
% VA = TtA(:);
% U = DT\VA;
% U = reshape(U, [r,n1,n2]);
% U = permute(U, [2,3,1]);

% Update with Maltlab R2022a or above
V = reshape(V, [1, n2, n3*r]);
W = reshape(pagetranspose(W), [n1, 1, n3*r]);
tildeT = reshape(permute(pagemtimes(W, V), [3,1,2]), [n3,r,n1*n2]);
tildeA = reshape(permute(A,[3,1,2]), [n3, 1, n1*n2]);
tildeU = pagemldivide(pagemtimes(tildeT, 'ctranspose', tildeT, 'none'), ...
    pagemtimes(tildeT, 'ctranspose', tildeA, 'none'));
U = permute(reshape(tildeU, [r, n1, n2]), [2,3,1]);




