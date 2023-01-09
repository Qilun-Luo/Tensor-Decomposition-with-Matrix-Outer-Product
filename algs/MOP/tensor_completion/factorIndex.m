%factorIndex generates the index for factor tensor with given subs.
% [subU] = factorIndex(subs, n1, n2, k)
%----------------------------------------------------------------
% Input:
%   subs:   given subscripts
%   vals:   the corresponding values
%   sz:     size of the tensor
%   d1:     indicates the d1 dim
%   d2:     indicates the d2 dim
% Output:
%   sub_k:  subscripts for the k-th dim
%   sub_v:  the corresponding subscripts for vals
%----------------------------------------------------------------
% Copyright(c) 2021 Qilun Luo 
% All Rights Reserved.

function [sub_k, sub_v] = factorIndex(subs, vals, sz, d1, d2)

gI = accumarray(subs(:, [d1, d2]), 1:length(subs), [sz(d1), sz(d2)], @(x){x});
k = setdiff(1:3, [d1, d2]);

%% -V1
sub_k = cell(sz(d1), sz(d2));
sub_v = cell(sz(d1), sz(d2));
for i = 1:sz(d1)
    for j = 1:sz(d2)
        sub_k{i,j} = subs(gI{i,j}, k);
        sub_v{i,j} = vals(gI{i,j});
    end
end

%% -V2
% sub_k = cellfun(@(x) subs(x, k), gI, 'UniformOutput', false);
% sub_v = cellfun(@(x) vals(x), gI, 'UniformOutput', false);

end



