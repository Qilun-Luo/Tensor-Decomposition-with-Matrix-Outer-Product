%VBMOP_RPCA Variational Bayesian Inference over Matrix Outer Product on
% RPCA Model with shared Lambda
% [X, S, Out] = VBMOP(A, r, opts)
%----------------------------------------------------------------
% Input:
%   Y:      observe dataset
%   opts:   optional parameters
% Output:
%   X:      low-rank component
%   S:      sparse component
%   Out:    Other output information
%----------------------------------------------------------------
% Copyright(c) 2021 Qilun Luo 
% All Rights Reserved.

function [X, S, Out] = VBMOP(Y, opts)

%% Parameters
max_iter = 100;
tol = 1e-4;
init = [];
debug = 1;
Prune = 0;
it_step = 10;
LMAX_ = 1e4;

[n1, n2, n3] = size(Y);

Y2sum = sum(Y(:).^2);
scale2 = Y2sum/(n1*n2*n3);
scale = sqrt(scale2);

r = min([n1, n2, n3]);

if ~exist('opts', 'var')
    opts = [];
end 
if  isfield(opts, 'max_iter');      max_iter = opts.max_iter;           end
if  isfield(opts, 'tol');           tol = opts.tol;                     end
if  isfield(opts, 'init');          init = opts.init;                   end
if  isfield(opts, 'r');             r = opts.r;                         end
if  isfield(opts, 'a0_lambda');     a0_lambda = opts.a0_lambda;         end
if  isfield(opts, 'b0_lambda');     b0_lambda = opts.b0_lambda;         end
if  isfield(opts, 'a0_gamma');      a0_gamma = opts.a0_gamma;           end
if  isfield(opts, 'b0_gamma');      b0_gamma = opts.b0_gamma;           end
if  isfield(opts, 'a0_tau');        a0_tau = opts.a0_tau;               end
if  isfield(opts, 'b0_tau');        b0_tau = opts.b0_tau;               end
if  isfield(opts, 'Prune');         Prune = opts.Prune;                 end
if  isfield(opts, 'debug');         debug = opts.debug;                 end
if  isfield(opts, 'it_step');       it_step = opts.it_step;             end
if  isfield(opts, 'LMAX_');         LMAX_ = opts.LMAX_;                 end
if  isfield(opts, 'tau')          
    tau = opts.tau;                     
else
    tau = a0_tau/b0_tau;
end
if  isfield(opts, 'gamma')          
    gamma = opts.gamma;                     
else
    gamma = ones(n1, n2, n3)*a0_gamma/b0_gamma;
end
if  isfield(opts, 'lambda')          
    lambda = opts.lambda;                     
else
    lambda = ones(r, 1).*(a0_lambda./b0_lambda);
end

%% Initialization
% Factor Tensors
if isempty(init)
    U = rand(n1 ,n2, r);
    V = rand(n2, n3, r);
    W = rand(n3, n1, r);
else
    [U, V, W] = init{:};
end

% Sparse Component
S = rand(n1, n2, n3)./(a0_gamma./b0_gamma);

%% Pre-defined
X = TP_MOP(U, V, W);

SigV = repmat(scale*eye(r), [1,1,n2,n3]);
v = reshape(permute(V, [3,1,2]), [r,1,n2,n3]);
RV = pagemtimes(v, 'none', v,'ctranspose') + SigV;
SigW = repmat(scale*eye(r), [1,1,n3,n1]);
w = reshape(permute(W, [3,1,2]), [r, 1, n3,n1]);
RW = pagemtimes(w, 'none', w,'ctranspose') + SigW;

%% Iteration
Out.rrse = [];
Out.rse = [];
Out.psnr = [];

for it = 1:max_iter
    % Old records
    X0 = X;
    S0 = S;
    E = Y - S;

    % Update tensor factor U, V, W
    [U, RU] = Update_Factor_Tensor(E, tau, lambda, V, W, RV, RW);
    [V, RV] = Update_Factor_Tensor(shiftdim(E, 1), tau, lambda, W, U, RW, RU);
    [W, RW] = Update_Factor_Tensor(shiftdim(E, 2), tau, lambda, U, V, RU, RV);

    % Compute the low-rank tensor X
    X = TP_MOP(U, V, W);

    % Update sparse tensor S
    SigS = 1./(tau+gamma);
    S = tau.*SigS.*(Y-X);

    % Update prior tau
    D = Y-S-X;
    RU_ = reshape(permute(RU, [3,4,1,2]), [n1,n2,r^2]);
    RV_ = reshape(permute(RV, [3,4,1,2]), [n2,n3,r^2]);
    RW_ = reshape(permute(RW, [3,4,1,2]), [n3,n1,r^2]);
    tau_chg = norm(D(:))^2 + sum(SigS, 'all') +...
        sum(TP_MOP(RU_, RV_, RW_), 'all') - norm(X(:))^2;
    a_tau = a0_tau + (n1*n2*n3)/2;
    b_tau = b0_tau + tau_chg/2;
    tau = a_tau/b_tau;

    % Update prior lambda
    a_lambda = a0_lambda + (n1*n2+n2*n3+n3*n1)/2;
    b_lambda = b0_lambda + (diag(sum(RU,[3,4]))+diag(sum(RV,[3,4]))+diag(sum(RW,[3,4])))/2;
    lambda = a_lambda./b_lambda;

    % Update prior gamma
    a_gamma = a0_gamma + 1/2;
    b_gamma = b0_gamma + (S.*S+SigS)/2;
    gamma = a_gamma./b_gamma;

    % Prune
    if Prune
        ind = find(lambda<LMAX_);
        if length(ind)<r
            U = U(:,:,ind);
            V = V(:,:,ind);
            W = W(:,:,ind);
            X = TP_MOP(U, V, W);
            RU = RU(ind, ind, :, :);
            RV = RV(ind, ind, :, :);
            RW = RW(ind, ind, :, :);
            lambda = lambda(ind);
            if length(a0_lambda)==r
                a0_lambda(ind) = a0_lambda(ind);
            end
            if length(b0_lambda)==r
                b0_lambda(ind) = b0_lambda(ind);
            end
            r = length(ind);
        end
    end

    % Check Convergence
    X_chg = norm(X(:)-X0(:))/norm(X0(:));
    S_chg = norm(S(:)-S0(:))/norm(S0(:));
    rrse = norm(X(:)-X0(:), 'fro')/norm(X0(:), 'fro');

    Out.rrse = [Out.rrse; rrse];
    rse = nan;
    mypsnr = nan;
    if isfield(opts, 'Xtrue')
        Xtrue = opts.Xtrue;
        rse = norm(X(:)-Xtrue(:), 'fro')/norm(Xtrue(:), 'fro');
        mypsnr = psnr(X, Xtrue, max(Xtrue(:)));
        Out.rse = [Out.rse; rse];
        Out.psnr = [Out.psnr; mypsnr];
    end

    if (debug && mod(it, it_step)==0)
        fprintf(['Iter %d: rrse=%.4f, rse = %.4f, psnr=%.4f, r = %d, ' ...
            'X_chg = %.5f, S_chg = %.5f, tau=%.2f.\n'], ...
            it, rrse, rse, mypsnr, r, X_chg, S_chg, tau);
    end

    if max([X_chg, S_chg, rrse])<tol
        break
    end
end

%% Record the model
model.X = X;
model.S = S;
model.U = U;
model.V = V;
model.W = W;
model.r = r;
Out.model = model;

end

function [T, RT] = Update_Factor_Tensor(E, tau, lambda, T1, T2, R1, R2)
%Update the factor tensors
% Input:
%   E:      residual tensor Y - S
%   tau:    prior over observed tensor
%   lambda: prior over factor tensors
%   T1:     the first given tensor
%   T2:     the second given tensor
%   R1:     second moment of the first given tensor
%   R2:     second moment of the second given tensor
% Output:
%   T:      the updated factor tensor
%   RT:     E(TT^T) + Var(T)
%--------------------------------------------------------------------------

[n1, n2, n3] = size(E);
r = length(lambda);
R1_ = reshape(permute(R1, [3,4,1,2]), [n2, 1, n3, r, r]);
R2_ = reshape(permute(R2, [4,3,1,2]), [n1, 1, n3, r, r]);
RS = permute(squeeze(sum(pagemtimes(R2_, 'none', R1_, 'ctranspose'), 3)), [3,4,1,2]);
QT = tau*RS + diag(lambda);
[PU, PS, PV] = pagesvd(QT, 'vector');
PS_inv = 1./(PS+eps);
SigT = pagemtimes(PU.*pagectranspose(PS_inv), 'none', PV, 'ctranspose');
E_ = reshape(permute(E, [3,1,2]), [n3, 1, n1, n2]);
T1_ = reshape(T1, [n2, 1, n3, r]);
T2_ = reshape(pagectranspose(T2), [n1, 1, n3, r]);
Z = permute(pagemtimes(T2_, 'none', T1_, 'ctranspose'), [3,4,1,2]);
t = tau*pagemtimes(pagemtimes(SigT, 'none', Z, 'ctranspose'), E_);
RT = pagemtimes(t, 'none', t, 'ctranspose') + SigT; 
T = squeeze(permute(t, [3,4,1,2]));

end