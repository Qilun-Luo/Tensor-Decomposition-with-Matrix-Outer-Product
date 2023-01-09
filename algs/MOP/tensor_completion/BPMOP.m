%BPMOP Baiyesian Probabilistic Tensor Decomposition on Matrix Outer
% Product with Gamma prior on alpha
% [Us, Vs, Ws, Out] = BPMOP(Tr, R, params, init, opts)
%----------------------------------------------------------------
% Input:
%   Tr:     training set
%   R:      feature dimension   
%   init:   initial factor tensors
%   opts:   optional parameters
% Output:
%   yTe:    Predicted values
%   Out:    Other output information
%----------------------------------------------------------------
% Copyright(c) 2021 Qilun Luo 
% All Rights Reserved.

function [yTe, Out] = BPMOP(Tr, D, init, opts)

% Default params
max_iter = 50;
a0 = 1;
b0 = 1;
nu_0 = D;
omega_0 = eye(D);
mu_0 = 0;
beta_0 = 1;
nS = 10; % number of samples
debug = 1;


if ~exist('opts', 'var')
    opts = [];
end 
if  isfield(opts, 'max_iter');      max_iter = opts.max_iter;           end
if  isfield(opts, 'a0');            a0 = opts.a0;                       end
if  isfield(opts, 'b0');            b0 = opts.b0;                       end
if  isfield(opts, 'nu_0');          nu_0 = opts.nu_0;                   end
if  isfield(opts, 'omega_0');       omega_0 = opts.omega_0;             end
if  isfield(opts, 'mu_0');          mu_0 = opts.mu_0;                   end
if  isfield(opts, 'beta_0');        beta_0 = opts.beta_0;               end
if  isfield(opts, 'nS');            nS = opts.nS;                       end
if  isfield(opts, 'Te');            Te = opts.Te;                       end
if  isfield(opts, 'debug');         debug = opts.debug;                 end

burnN = max_iter - nS;
sz = num2cell(Tr.size);
[n1, n2, n3] = sz{:};
subs = Tr.subs;
vals = Tr.vals;
L = length(vals);

iomega_0 = inv(omega_0);

N1 = n1*n2;
N2 = n2*n3;
N3 = n3*n1;
szU = [n1, n2];
szV = [n2, n3];
szW = [n3, n1];

% Initialization
if isempty(init)
    Uf = rand(N1, D);
    Vf = rand(N2, D);
    Wf = rand(N3, D);
else
    [Uf, Vf, Wf] = init{:};
end

yTr = MOP_Rec_Flat(subs, Uf, Vf, Wf, szU, szV, szW);
rmseTr = my_RMSE(yTr, vals);

yTe = [];
if isfield(opts, 'Te')
    subs_Te = Te.subs;
    vals_Te = Te.vals;
    L_Te = length(vals_Te); 
    yTe = MOP_Rec_Flat(subs_Te, Uf, Vf, Wf, szU, szV, szW);
    rmseTe = my_RMSE(yTe, vals_Te);
end

if debug
    fprintf('Initial: RMSE = %0.4f/%0.4f.\n', rmseTr, rmseTe);
end

sz = [n1, n2, n3];
% Compute the factor index
[sub_Uk, sub_Uval] = factorIndex(subs, vals, sz, 1, 2);
[sub_Vk, sub_Vval] = factorIndex(subs, vals, sz, 2, 3);
[sub_Wk, sub_Wval] = factorIndex(subs, vals, sz, 3, 1);


ysTr = zeros(L,1);
if isfield(opts, 'Te')
    ysTe = zeros(L_Te, 1);
end

Out.rmseTr = [];
Out.rmseTe = [];

tic
for l = 1:max_iter
    if debug
        fprintf('-Iter%d... ', l);
    end


    %% Sample alpha
    a0_ = a0 + L/2;
    b0_ = b0 + sum((yTr-vals).^2)/2;
    alpha = gamrnd(a0_, 1./b0_); 
    
    %% Sample prior factor
    [mu_U, Lambda_U] = sample_prior_factor(Uf, N1, beta_0, nu_0, mu_0, iomega_0);
    [mu_V, Lambda_V] = sample_prior_factor(Vf, N2, beta_0, nu_0, mu_0, iomega_0);
    [mu_W, Lambda_W] = sample_prior_factor(Wf, N3, beta_0, nu_0, mu_0, iomega_0);

    
    %% Sample features 
    parfor ind = 1:N1
        [i, j] = ind2sub(szU, ind);
        Uf(ind, :) = sample_features(alpha, mu_U, Lambda_U, Vf, szV, Wf, szW, sub_Uk{ind}, sub_Uval{ind}, i, j);
    end 
    parfor ind = 1:N2
        [j, k] = ind2sub(szV, ind);
        Vf(ind, :) = sample_features(alpha, mu_V, Lambda_V, Wf, szW, Uf, szU, sub_Vk{ind}, sub_Vval{ind}, j, k);
    end
    parfor ind = 1:N3
        [k, i] = ind2sub(szW, ind);
        Wf(ind, :) = sample_features(alpha, mu_W, Lambda_W, Uf, szU, Vf, szV, sub_Wk{ind}, sub_Wval{ind}, k, i);
    end


    %% Record samples
    yTr = MOP_Rec_Flat(subs, Uf, Vf, Wf, szU, szV, szW);
    if l>burnN
        ysTr = ysTr + yTr;
        rmseTr = my_RMSE(ysTr/(l-burnN), vals);
        Out.rmseTr = [Out.rmseTr; rmseTr];
    else
        rmseTr = my_RMSE(yTr, vals);
    end
    
    rmseTe = nan;
    if isfield(opts, 'Te')
        if l>burnN
            yTe = MOP_Rec_Flat(subs_Te, Uf, Vf, Wf, szU, szV, szW);
            ysTe = ysTe + yTe;
            rmseTe = my_RMSE(ysTe/(l-burnN), vals_Te);
            Out.rmseTe = [Out.rmseTe; rmseTe];
        end
    end
    
    if debug
        fprintf('. alpha = %g. Using %d samples. RMSE=%0.4f/%0.4f. ETA=%0.2f hr\n', ...
            alpha, max(0, l-burnN), rmseTr, rmseTe, (max_iter - l)*toc/(l)/3600);
    end

end

yTr = ysTr/(max_iter-burnN);
Out.yTr = yTr;
if isfield(opts, 'Te')
    yTe = ysTe/(max_iter-burnN);
    Out.yTe = yTe;
end

end

%% sample_prior_factor
function [mu_F, Lambda_F] = sample_prior_factor(Ff, N, beta_0, nu_0, mu_0, iomega_0)
% Input:
%   F:          factor tensor
%   N:          number of factors (e.g. n1 x n2)
%   D:          dimension of features
%   beta_0, nu_0, mu_0, iomega_0: given params
% Output:
%   mu_F:       prior mean vector
%   Lambda_F:   prior precision matrix

    Fm = mean(Ff)';
    beta_0_  = beta_0 + N;
    nu_0_ = nu_0 + N;
    mu_0_ = (beta_0*mu_0 + N*Fm)/beta_0_;
    d_nu_Fm = mu_0 - Fm;
    iomega_0_ = iomega_0 + Ff'*Ff - N*(Fm*Fm') + (beta_0*N)/(beta_0+N)*(d_nu_Fm*d_nu_Fm');
    omega_0_ = inv(iomega_0_);
    Lambda_F = wishrnd((omega_0_ + omega_0_)/2, nu_0_); % prior
    
    mu_F = mvnrndpre(mu_0_, beta_0_*Lambda_F);
%     mu_F = mvnrnd(mu_0_, inv(beta_0_*Lambda_F))';
end

%% sample_features
function y = sample_features(alpha, mu_F, Lambda_F, Vf, szV, Wf, szW, sub_k, sub_val, i, j)
% Input:
%   alpha:      rating precision alpha
%   mu_F:       mean prior
%   Lambda_F:   precision prior
%   Vf:         the first reshaped factor matrix
%   szV:        size of factor V
%   Wf:         the second reshaped factor matrix
%   szW:        size of factor W
%   sub_k:      subscripts for the k-th dim
%   i:          the i-th row of factor tensor
%   j:          the j-th column of factor tensor

% Output:
%   y:          sample feature vector
   
    if isempty(sub_k)
        y = mvnrndpre(mu_F, Lambda_F);
    else
        indV = sub2ind(szV, j*ones(size(sub_k)), sub_k);
        indW = sub2ind(szW, sub_k, i*ones(size(sub_k)));
        Q = Vf(indV, :).*Wf(indW, :);

        Lambda_F_  = Lambda_F + alpha*(Q'*Q);
        mu_F_ = Lambda_F_\(Lambda_F*mu_F + alpha*(Q'*sub_val));
        
        y = mvnrndpre(mu_F_, Lambda_F_);
    end

end





