%RMSE computes the residual mean square error.
% r = RMSE(Y, G)
%----------------------------------------------------------------
% Input:
%   Y: predicted value
%   G: Ground truth
% Output:
%   r: residual mean square error
%----------------------------------------------------------------
% Copyright(c) 2021 Qilun Luo 
% All Rights Reserved.

function r = my_RMSE(Y, G)
    E = Y - G;
    r = sqrt(mean(E(:).^2));
end