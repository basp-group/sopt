function I = sopt_mltb_div_op_sphere(dx, dy, includeNorthpole, ...
  weights_dx, weights_dy)
% sopt_mltb_div_op_sphere - Compute divergence on sphere
%
% Compute the divergence (adjoint of the gradient) of a two dimensional
% signal on the sphere.  The phi direction (x) is periodic, while the theta
% direction (y) is not.
%
% Inputs:
%
%   - dx: Gradient in x.
%
%   - dy: Gradient in y.
%
%   - includeNorthPole: Flag indicating whether the North pole is included
%       in the sampling grid (1 = North pole included, 0 = North pole not
%       included).
%
%   - weights_dx: Weights in the x (phi) direction.
%
%   - weights_dy: Weights in the y (theta) direction.
%
% Outputs:
%
%   - I: Divergence.

if nargin > 3
    dx = dx .* conj(weights_dx);
    dy = dy .* conj(weights_dy);
end
if(includeNorthpole)
    I = [zeros(1, size(dx,2)) ; dx(2, :); dx(3:end-1, :)-dx(2:end-2, :); -dx(end-1, :)];
    I = I + [dy(:, 1) - dy(:,end) , dy(:, 2:end)-dy(:, 1:end-1)];
else
    I = [dx(1, :) ; dx(2:end-1, :)-dx(1:end-2, :) ; -dx(end-1, :)];
    I = I + [dy(:, 1) - dy(:,end) , dy(:, 2:end)-dy(:, 1:end-1)];
end
end
