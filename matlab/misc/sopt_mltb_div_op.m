function I = sopt_mltb_div_op(dx, dy, weights_dx, weights_dy)
% sopt_mltb_div_op - Compute divergence
%
% Compute the divergence (adjoint of the gradient) of a two dimensional
% signal from the horizontal and vertical gradients.
%
% Inputs:
%
%   - dx: Gradient in x.
%
%   - dy: Gradient in y.
%
%   - weights_dx: Weights in the x direction.
%
%   - weights_dy: Weights in the y direction.
%
% Outputs:
%
%   - I: Divergence.

if nargin > 2
    dx = dx .* conj(weights_dx);
    dy = dy .* conj(weights_dy);
end

I = [dx(1, :) ; dx(2:end-1, :)-dx(1:end-2, :) ; -dx(end-1, :)];
I = I + [dy(:, 1) , dy(:, 2:end-1)-dy(:, 1:end-2) , -dy(:, end-1)];

end
