function [dx, dy] = sopt_mltb_gradient_op(I, weights_dx, weights_dy)
% sopt_mltb_gradient_op - Compute gradient
%
% Compute the gradient of a two dimensional signal.
%
% Inputs:
%
%   - I: Input two dimesional signal.
%
%   - weights_dx: Weights in the x direction.
%
%   - weights_dy: Weights in the y direction.
%
% Outputs:
% 
%   - dx: Gradient in x direction.
% 
%   - dy: Gradient in y direction.

dx = [I(2:end, :)-I(1:end-1, :) ; zeros(1, size(I, 2))];
dy = [I(:, 2:end)-I(:, 1:end-1) , zeros(size(I, 1), 1)];

if nargin>1
    dx = dx .* weights_dx;
    dy = dy .* weights_dy;
end

end
