%MOP_RecM reconstruct the matrix outer product over the subs with flatten
%tensor
% res = MOP_Rec_Flat(subs, Uf, Vf, Wf)
%----------------------------------------------------------------
% Input:
%   subs:   the given subsripts
%   Uf:     the flatten first factor tensor
%   Vf:     the flatten second factor tensor
%   Wf:     the flatten third factor tensor
%   szU:    size of the front slice of U
%   szV:    size of the front slice of V
%   szW:    size of the front slice of W
% Output:
%   y:      predicted values
%----------------------------------------------------------------
% Copyright(c) 2021 Qilun Luo 
% All Rights Reserved.

function y = MOP_Rec_Flat(subs, Uf, Vf, Wf, szU, szV, szW)

idx1 = sub2ind(szU, subs(:,1),  subs(:,2));
idx2 = sub2ind(szV, subs(:,2),  subs(:,3));
idx3 = sub2ind(szW, subs(:,3),  subs(:,1));
y = sum(Uf(idx1,:).*Vf(idx2,:).*Wf(idx3,:), 2);
