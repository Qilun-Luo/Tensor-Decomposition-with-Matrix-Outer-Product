%demo for TRPCA on Hyperspectral data

clear
close all

rng('default') % For reproducibility

addpath(genpath('utils/'))
addpath(genpath('algs/'));

data_path = 'data/HSI/';

data_name = {
    'cloth_ms.mat',
    'Urban.mat',
    'Samson.mat',
    'Salinas.mat',
    'PaviaU.mat',
    'Indian_pines.mat',
};

test_list = 3:3;

%% Algs settings
flag_VBMOP = 1; % Proposed

%% Iteration
for i = test_list
    %% Loading data
    load(fullfile(data_path, data_name{i}));
    normalize = max(T(:));
    X = T/normalize;

    [n1,n2,n3] = size(X);
    Xn = X;
    
    rhos = 0.3;
    ind = find(rand(n1*n2*n3,1)<rhos);
    Xn(ind) = rand(length(ind),1);

    %% record 
    alg_name = {};
    alg_result = {};
    alg_out = {};
    alg_rse = {};
    alg_cpu = {};
    alg_ssim = {};
    alg_psnr = {};
    alg_cnt = 1;

    %% -- Sample
    X_dif_sample = Xn - X;
    X_rse_sample = norm(X_dif_sample(:))/norm(X(:));
    X_psnr_sample = psnr(Xn, X, max(X(:)));
    X_ssim_sample = ssim(Xn, X);
    % record
    alg_name{alg_cnt} = 'Sample';
    alg_result{alg_cnt} = Xn;
    alg_out{alg_cnt} = Xn;
    alg_cpu{alg_cnt} = 0;
    alg_rse{alg_cnt} = X_rse_sample;
    alg_ssim{alg_cnt} = X_ssim_sample;
    alg_psnr{alg_cnt} = X_psnr_sample;
    alg_cnt = alg_cnt + 1;

    %% Alg: VBMOP - Proposed
    if flag_VBMOP
        opts = [];
        opts.tol = 1e-4;
        opts.max_iter = 100;

        opts.init = [];
        opts.r = 5;
        opts.a0_lambda = 1e-1;
        opts.b0_lambda = 1e-4;
        opts.a0_gamma = 1e-1;
        opts.b0_gamma = 1e-4;
        opts.a0_tau = 1e-1;
        opts.b0_tau = 1e-4;
        opts.debug = 1;
        opts.Prune = 1;
        opts.it_step = 10;
        opts.LMAX_ = 1e4;

        [~, F, ~] = TD_MOP(Xn, opts.r);
        U = F.U;
        V = F.V;
        W = F.W;
        opts.init = {U, V, W};

        opts.Xtrue = X;

        alg_name{alg_cnt} = 'VBMOP';
        fprintf('Processing method: %12s\n', alg_name{alg_cnt});
        t_VBMOP = tic;

        [X_VBMOP, S_VBMOP, Out_VBMOP] = VBMOP(Xn, opts);

        X_dif_VBMOP = X_VBMOP - X;
        X_rse_VBMOP = norm(X_dif_VBMOP(:))/norm(X(:));
        X_psnr_VBMOP = psnr(abs(X_VBMOP), X);
        X_ssim_VBMOP = ssim(abs(X_VBMOP), X);
        % record
        alg_result{alg_cnt} = X_VBMOP;
        alg_out{alg_cnt} = Out_VBMOP;
        alg_cpu{alg_cnt} = toc(t_VBMOP);
        alg_rse{alg_cnt} = X_rse_VBMOP;
        alg_ssim{alg_cnt} = X_ssim_VBMOP;
        alg_psnr{alg_cnt} = X_psnr_VBMOP;
        alg_cnt = alg_cnt + 1;
    end

    %% Result table
    flag_report = 1;
    fprintf('Test on data: %s\n', data_name{i});

    if flag_report
        fprintf('%12s\t%4s\t%4s\t%4s\t%4s\n',...\
            'Algs', 'CPU', 'RSE', 'PSNR', 'SSIM');
        for j = 1:alg_cnt-1
            fprintf('%12s\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
                alg_name{j}, alg_cpu{j}, alg_rse{j}, alg_psnr{j}, alg_ssim{j});
        end
    end

end




