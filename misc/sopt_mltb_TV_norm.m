function y = sopt_mltb_TV_norm(u, sphere_flag, ...
    incNP, weights_dx, weights_dy)
% sopt_mltb_TV_norm - Compute TV norm
%
% Compute the TV of an image on the plane or sphere.
%
% Inputs:
%
%   - u: Image to compute TV norm of.
%
%   - sphere_flag: Flag indicating whether to compute the TV norm on the
%       sphere (1 = on sphere, 2 = on plane).
%
%   - includeNP: Flag indicating whether the North pole is included
%       in the sampling grid (1 = North pole included, 0 = North pole not
%       included).
%
%   - weights_dx: Weights in the x (phi) direction.
%
%   - weights_dy: Weights in the y (theta) direction.
%
% Outputs:
%
%   - y: TV norm of image.

if sphere_flag
    if nargin>3 
        [dx, dy] = sopt_mltb_gradient_op_sphere(u, incNP, weights_dx, weights_dy);
    else
        [dx, dy] = sopt_mltb_gradient_op_sphere(u, incNP);
    end
else
    if nargin>3 
        [dx, dy] = sopt_mltb_gradient_op(u, weights_dx, weights_dy);
    else
        [dx, dy] = sopt_mltb_gradient_op(u);
    end
end
temp = sqrt(abs(dx).^2 + abs(dy).^2);
y = sum(temp(:));

end
