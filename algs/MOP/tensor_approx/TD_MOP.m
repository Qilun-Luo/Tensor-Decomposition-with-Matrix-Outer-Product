%TD_MOP Tensor Decomposition on Matrix Outer Product.
% [X, F, Out] = TD_MOP(A, r, opts) decomposes a third-order tensor A
% of size n1 x n2 x n3 into three factor tensors U, V, W.
%----------------------------------------------------------------
% Input:
%   A:      a third-order tensor of size n1 x n2 x n3
%   r:      a given rank for factor tensors
%   opts:   optional parameters
% Output:
%   X:      the approximated tensor with rank r
%   F:      struct data with factor tensors: U, V, W
%   Out:    other output information
%----------------------------------------------------------------
% Copyright(c) 2021 Qilun Luo 
% All Rights Reserved.

function [X, F, Out] = TD_MOP(A, r, opts)

% Optional parameters
max_iter = 100;
tol = 1e-4;
debug = 0;
if ~exist('opts', 'var')
    opts = [];
end  
if  isfield(opts, 'max_iter');  max_iter = opts.max_iter;   end
if  isfield(opts, 'tol');       tol = opts.tol;             end
if  isfield(opts, 'debug');     debug = opts.debug;         end

% Initialization
if isfield(opts, 'init')
    [U, V, W] = opts.init{:};
else
    [n1, n2, n3] = size(A);
    U = rand(n1, n2, r);
    V = rand(n2, n3, r);
    W = rand(n3, n1, r);
end

normA = norm(A(:));
Out.rse = [];
Out.obj = [];


X = TP_MOP(U, V, W);
rse_old = 0;
normR = norm(A(:)-X(:)); 
rse = normR/normA;
obj = normR^2/2;
Out.rse = [Out.rse; rse];
Out.obj = [Out.obj; obj];

if debug
    fprintf('TD_MOP: \n')
end
for d = 1:max_iter
    rse_chg = abs(rse-rse_old);
    if debug
        fprintf('Iter = %d,\trse = %.4f, \trse_chg=%.4f\n', d, rse, rse_chg);
    end
    if rse_chg < tol
        break;
    end
    % Update U
    U = UFT_TDMOP(A, V, W);
    % Update V
    V = UFT_TDMOP(shiftdim(A, 1), W, U);
    % Update W
    W = UFT_TDMOP(shiftdim(A, 2), U, V);
    
    % Stopping criterion
    X = TP_MOP(U, V, W);
    rse_old = rse;
    normR = norm(A(:)-X(:)); 
    rse = normR/normA;
    obj = normR^2/2;
    Out.rse = [Out.rse; rse];
    Out.obj = [Out.obj; obj];  
end
F.U = U;
F.V = V;
F.W = W;
