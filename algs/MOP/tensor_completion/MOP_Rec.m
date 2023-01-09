%MOP_Rec reconstruct the matrix outer product over the subs
% res = MOP_Rec(subs, U, V, W)
%----------------------------------------------------------------
% Input:
%   subs:   the given subsripts
%   U:      the first factor tensor
%   V:      the second factor tensor
%   W:      the third factor tensor
% Output:
%   y:      predicted values
%----------------------------------------------------------------
% Copyright(c) 2021 Qilun Luo 
% All Rights Reserved.

function y = MOP_Rec(subs, U, V, W)

%% V1
% nz = length(subs);
% y = zeros(nz, 1);
% for k = 1:nz
%     y(k) = sum(U(subs(k,1), subs(k,2), :).*V(subs(k,2), subs(k,3), :).*W(subs(k,3), subs(k,1), :));
% end

%% V2
D = size(U, 3);
idx1 = sub2ind(size(U, [1,2]), subs(:,1),  subs(:,2));
idx2 = sub2ind(size(V, [1,2]), subs(:,2),  subs(:,3));
idx3 = sub2ind(size(W, [1,2]), subs(:,3),  subs(:,1));
U = reshape(U, [], D);
V = reshape(V, [], D);
W = reshape(W, [], D);
y = sum(U(idx1,:).*V(idx2,:).*W(idx3,:), 2);
