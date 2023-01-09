%demo for TRPCA on background subtraction

clear
close all

addpath('utils')
addpath(genpath('algs\'));

%% load data
cate_list = {
    'turbulence',
    'shadow',
    'thermal',
    'dynamicBackground',
};
file_list = {
    'turbulence3',
    'peopleInShade',
    'park',
    'boats',
};
range_list = {
    [900, 1000],
    [900, 1000],
    [300, 400],
    [1900, 2000],
};


test_list = 2:2;

res_draw = 1;

%% Algs settings
flag_VBMOP = 1;     % Proposed


root_path = 'data';


for t = test_list
        %% Path setting
        category = cate_list{t};
        idxFrom = range_list{t}(1);
        idxTo = range_list{t}(2);
        file_name = file_list{t};

        data_path = fullfile(root_path, 'dataset2014', 'dataset');
        dataset_name = fullfile(category, file_name, 'input');
        output_path = fullfile(root_path, 'dataset2014', 'results');
        output_folder = fullfile(category, file_name);
        tmp_save_path = fullfile(root_path, 'tmp' ,'quantitative');

        if ~exist(tmp_save_path, 'dir')
            mkdir(tmp_save_path);
        end
        mat_file = strcat(tmp_save_path, file_name, '_', num2str(idxFrom), '_', num2str(idxTo));
        ext_name = 'jpg';
        show_flag = 0;
        if ~exist(strcat(mat_file, '_rgb.mat'), 'file')
            [X, height, width, imageNames] = load_video_for_quantitative(data_path, dataset_name, ext_name, show_flag, idxFrom, idxTo);
            save(strcat(mat_file, '_rgb.mat'), 'X', 'height', 'width', 'imageNames');
        else
            load(strcat(mat_file, '_rgb.mat'));
        end
        [height_width, dims, nframes] = size(X);

        %% Recorder
        alg_name = {};
        alg_result = {};
        alg_out = {};
        alg_cpu = {};
        alg_cnt = 1;

        %% VBMOP (Proposed)
        if flag_VBMOP
            opts = [];
            opts.tol = 1e-4;
            opts.max_iter = 100;
            opts.init = [];
            opts.r = 1;
            opts.a0_lambda = 1e0;
            opts.b0_lambda = 1e0;
            opts.a0_gamma = 1e-1;
            opts.b0_gamma = 1e-5;
            opts.a0_tau = 1e-2;
            opts.b0_tau = 1e-5;
            opts.debug = 1;
            opts.Prune = 1;
            opts.it_step = 10;
            opts.LMAX_ = 1e4;
 
            alg_name{alg_cnt} = 'VBMOP';
            fprintf('Processing method: %12s\n', alg_name{alg_cnt});
            t_VBMOP = tic;
            
            Y = reshape(X, [height, width, dims*nframes]);
            [X_VBMOP, S_VBMOP, Out_VBMOP] = VBMOP(Y, opts);

            X_VBMOP = reshape(X_VBMOP, [height_width, dims, nframes]);
            S_VBMOP = reshape(S_VBMOP, [height_width, dims, nframes]);

            if res_draw
                figure
            end
            for i = 1:nframes
                S_VBMOP_frame = reshape(S_VBMOP(:, :, i), [height, width, dims]);
                Tmask_VBMOP = medfilt2(double(hard_threshold(mean(S_VBMOP_frame,3))),[5 5]);

                if res_draw
                    subplot(1,3,1)
                    imshow(uint8(reshape(X_VBMOP(:, :, i), [height, width, dims])))
                    subplot(1,3,2)  
                    imshow(uint8(S_VBMOP_frame));
                    subplot(1,3,3)
                    imshow(Tmask_VBMOP)
                end

                save_path = fullfile(output_path, output_folder, alg_name{alg_cnt});
                if ~exist(save_path, 'dir')
                    mkdir(save_path);
                end
                imwrite(Tmask_VBMOP, fullfile(save_path, strcat('b', imageNames{i})));
                pause(0.01)
            end

            % record
            alg_result{alg_cnt} = S_VBMOP;
            alg_out{alg_cnt} = Out_VBMOP;
            alg_cpu{alg_cnt} = toc(t_VBMOP);
            alg_cnt = alg_cnt + 1;
        end

        %% Compute quantitative measures
        extension = '.jpg';
        range = [idxFrom, idxTo];

        videoPath = fullfile(data_path, category, file_name);
        binaryFolder = fullfile(output_path, category, file_name);

        fprintf('===================================================\n')
        fprintf('Category: %s\tDateset: %s\n', category, file_name)
        fprintf('Alg_name\tCPU\tRecall\tPrecision\tFMeasure\n')
        for i = 1:alg_cnt-1
            [confusionMatrix, stats] = compute_measures(videoPath, fullfile(binaryFolder, alg_name{i}), range, extension);
            fprintf('%s\t%.2f\t%.4f\t%.4f\t%.4f\t\n', alg_name{i},alg_cpu{i}, stats(1), stats(6), stats(7))
        end

end


