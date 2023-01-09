%demo for tensor completion problem on Hyperspectral data

clear
close all
rng('default') % For reproducibility

addpath(genpath('utils/'))
addpath(genpath('algs/'));

data_path = 'data/HSI/';
data_sparse_path = 'data/HSI/HSI_sparse/';

data_name = {
    'Urban.mat',
    'Cuprite.mat',
    'Samson.mat',
    'Salinas.mat',
    'PaviaU.mat',
};

test_list = 3:3;

mr = 0.9; % missing rate

flag_BPMOP = 1; % Proposed

%% Running test case
for t = test_list
    %% Loading data
    if ~exist(data_sparse_path, 'dir')
        mkdir(data_sparse_path);
    end
    if ~exist(strcat(data_sparse_path, 'sp_', data_name{t}), 'file')
        load(fullfile(data_path, data_name{t}));
        normalize = max(T(:));
        X = T/normalize;
        [n1,n2,n3] = size(X);
        sz = [n1,n2,n3];

        dataP = numel(X); % population of data
        Known = randsample(dataP, round((1-mr)*dataP));
        [Known,~] = sort(Known);
        TTr = [];
        TTe = [];
        [idx1, idx2, idx3] = ind2sub(sz, Known);
        vals = X(Known);
        subs = [idx1, idx2, idx3];
        TTr.subs = subs;
        TTr.vals = vals;
        TTr.size = sz;

        unKnown = setdiff((1:dataP)', Known);
        [idx1_te, idx2_te, idx3_te] = ind2sub(sz, unKnown);
        vals_te = X(unKnown);
        subs_te = [idx1_te, idx2_te, idx3_te];
        TTe.subs = subs_te;
        TTe.vals = vals_te;
        TTe.size = sz;
        save(strcat(data_sparse_path, 'sp_', data_name{t}),...
            'TTr', 'TTe', 'Known', 'unKnown', 'X', 'sz')
    else
        load(fullfile(strcat(data_sparse_path, 'sp_', data_name{t})));
    end
    
    %% Sampling
    Omega = zeros(sz);
    Omega(Known) = 1;
    Xn = zeros(sz);
    Xn(Omega==1) = X(Omega==1);
    
    %% record 
    alg_name = {};
    alg_result = {};
    alg_out = {};
    alg_rse = {};
    alg_cpu = {};
    alg_ssim = {};
    alg_psnr = {};
    alg_cnt = 1;
    
    %% Alg: Sample
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
    
    %% Alg: BPMOP (Proposed)
    if flag_BPMOP       
        D = 10;
        opts.max_iter = 150;
        opts.a0 = 1;
        opts.b0 = 1;
        opts.nu_0 = D;
        opts.omega_0 = eye(D);
        opts.mu_0 = 0;
        opts.beta_0 = 1;
        opts.nS = 100;
        opts.Te = TTe;
        opts.debug = 1;
        
        [n1, n2, n3] = size(X);
        Uf = randn(n1*n2, D);
        Vf = randn(n2*n3, D)+1;
        Wf = randn(n3*n1, D)+1;
        init = {Uf, Vf, Wf};
        
        t_BPMOP = tic;
        alg_name{alg_cnt} = 'BPMOP';
        fprintf('Processing method: %12s\n', alg_name{alg_cnt});
        
        [y_BPMOP, Out_BPMOP] = BPMOP(TTr, D, init, opts);
        
        Xn_BPMOP = zeros(sz);
        Xn_BPMOP(Known) = X(Known);
        Xn_BPMOP(unKnown) = y_BPMOP;
        X_dif_BPMOP = Xn_BPMOP - X;
        X_rse_BPMOP = norm(X_dif_BPMOP(:))/norm(X(:));
        X_psnr_BPMOP = psnr(Xn_BPMOP, X, max(X(:)));
        X_ssim_BPMOP = ssim(Xn_BPMOP, X);
        
        % record
        alg_result{alg_cnt} = Xn_BPMOP;
        alg_out{alg_cnt} = Out_BPMOP;
        alg_cpu{alg_cnt} = toc(t_BPMOP);
        alg_rse{alg_cnt} = X_rse_BPMOP;
        alg_ssim{alg_cnt} = X_ssim_BPMOP;
        alg_psnr{alg_cnt} = X_psnr_BPMOP;
        alg_cnt = alg_cnt + 1;
    end
    
    fprintf('Dataset:%s\n', data_name{t})
    
    %% Result reporting
    flag_report = 1;
    if flag_report
        fprintf('%12s\t%4s\t%4s\t%4s\t%4s\n',...\
            'Algs', 'CPU', 'RSE', 'PSNR', 'SSIM');
        for j = 1:alg_cnt-1
            fprintf('%12s\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
                alg_name{j}, alg_cpu{j}, alg_rse{j}, alg_psnr{j}, alg_ssim{j});
        end
    end

    
end