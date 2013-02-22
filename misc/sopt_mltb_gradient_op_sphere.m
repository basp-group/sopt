function [dx, dy] = sopt_mltb_gradient_op_sphere(I, includeNorthpole, ...
  weights_dx, weights_dy)
% sopt_mltb_gradient_op_sphere - Compute gradient on sphere
%
% Compute the gradientof a signal on the sphere.  The phi direction (x) is
% periodic, while the theta direction (y) is not.
%
% Inputs:
%
%   - I: Input two dimesional signal on the sphere.
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
%   - dx: Gradient in x (phi) direction.
% 
%   - dy: Gradient in y (theta) direction.

dx = zeros(size(I, 1),size(I, 2));
I_big = zeros(size(I, 1)+2,size(I, 2));
I_big(2:end-1, 1:end) = I;

% Theta direction
if(includeNorthpole)
    dx = [I(2:end, :)-I(1:end-1, :) ; zeros(1, size(I, 2))];
    dx(1,:) = zeros();
else
    dx = [I(2:end, :)-I(1:end-1, :) ; zeros(1, size(I, 2))];
end

% Phi direction
if(includeNorthpole)
    dy = [I(:, 2:end)-I(:, 1:end-1) , I(:, 1)-I(:, end)];
else
    dy = [I(:, 2:end)-I(:, 1:end-1) , I(:, 1)-I(:, end)];
end

if nargin>2
    dx = dx .* weights_dx;
    dy = dy .* weights_dy;
end

end
