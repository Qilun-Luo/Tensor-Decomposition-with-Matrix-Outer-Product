%Demo for alternating least square on synthetic data

%% Preparation
clear
close all
rng('shuffle')

addpath(genpath('utils/'))
addpath(genpath('algs/'));


%% Algs setting
flag_TDMOP = 1;

run_num = 1;
alg_sum_rse = {};
alg_sum_cpu = {};
alg_iter_rse = {};

%% Algs runnings
for rn = 1:run_num
    
    %% Data Generating
    n1 = 140;
    n2 = 200;
    n3 = 200;

    r = 50;
    U = rand(n1, n2, r);
    V = rand(n2, n3, r);
    W = rand(n3, n1, r);
    M = TP_MOP(U, V, W);
  
    %% Records
    alg_name = {};
    alg_result = {};
    alg_rse = {};
    alg_out = {};
    alg_cpu = {};
    alg_cnt = 1;
    
    %% TDMOP - Tensor Decomposition on Matrix Outer Product
    if flag_TDMOP
        r_TDMOP_list = {
            1,
            3,
            5
        };
        for c_mop = 1:length(r_TDMOP_list)
            opts = [];
            opts.tol = 1e-4;
            opts.max_iter = 500;
            opts.debug = 0;

            r_TDMOP = r_TDMOP_list{c_mop};
            fprintf('Processing TDMOP with rank %d\n', r_TDMOP);

            t_TDMOP = tic;
            [X_TDMOP, F_TDMOP, Out_TDMOP] = TD_MOP(M, r_TDMOP, opts);
            time_TDMOP = toc(t_TDMOP);
            X_diff_TDMOP = X_TDMOP - M;
            alg_rse{alg_cnt} = norm(X_diff_TDMOP(:))/norm(M(:));  
            alg_name{alg_cnt} = sprintf('TDMOP-rank-%d', r_TDMOP);
            alg_result{alg_cnt} = X_TDMOP;
            alg_out{alg_cnt} = Out_TDMOP;
            alg_cpu{alg_cnt} = time_TDMOP;
            if rn == 1
                alg_sum_rse{alg_cnt} = alg_rse{alg_cnt};
                alg_sum_cpu{alg_cnt} = alg_cpu{alg_cnt};
            else
                alg_sum_rse{alg_cnt} = alg_sum_rse{alg_cnt} + alg_rse{alg_cnt};
                alg_sum_cpu{alg_cnt} = alg_sum_cpu{alg_cnt} + alg_cpu{alg_cnt};
            end
            alg_iter_rse{alg_cnt}{rn} = alg_out{alg_cnt}.rse;
            alg_cnt = alg_cnt + 1;
        end
    end
end


%% Reports
flag_report = 1;
if flag_report   
    fprintf('%18s\t%8s\t%8s\n', 'Algs', 'avg-CPU', 'avg-RSE');
    for j = 1:alg_cnt-1
        fprintf('%18s\t%.4f\t%.4f\n', ...\
            alg_name{j}, alg_sum_cpu{j}/run_num, alg_sum_rse{j}/run_num);
    end
end







