function sol = sopt_mltb_prox_TVoA(b, lambda, param)
% sopt_mltb_prox_TVoA - Agumented total variation proximal operator
%
% Compute the TV proximal operator when an additional linear operator A is
% incorporated in the TV norm, i.e. solve
%
%   min_{x} ||y - x||_2^2 + lambda * ||A x||_{TV}
%
% where x is the input vector and the solution z* is returned as sol.  
% The structure param should contain the following fields:
%
%   - max_iter: Maximum number of iterations (default = 200).
%
%   - rel_obj: Minimum relative change of the objective value 
%       (default = 1e-4).  The algorithm stops if
%           | ||x(t)||_TV - ||x(t-1)||_TV | / ||x(t)||_1 < rel_obj,
%       where x(t) is the estimate of the solution at iteration t.
%
%   - verbose: Verbosity level (0 = no log, 1 = summary at convergence, 
%       2 = print main steps; default = 1).
%
%   - A: Forward transform (default = Identity).
%
%   - At: Adjoint of At (default = Identity).
%
%   - nu: Bound on the norm^2 of the operator A, i.e.
%       ||A x||^2 <= nu * ||x||^2 (default = 1)
%
% Reference:
% [1] A. Beck and  M. Teboulle, "Fast gradient-based algorithms for
% constrained Total Variation Image Denoising and Deblurring Problems",
% IEEE Transactions on Image Processing, VOL. 18, NO. 11, 2419-2434,
% November 2009.

% Optional input arguments
if ~isfield(param, 'rel_obj'), param.rel_obj = 1e-4; end
if ~isfield(param, 'verbose'), param.verbose = 1; end
if ~isfield(param, 'max_iter'), param.max_iter = 200; end
if ~isfield(param, 'At'), param.At = @(x) x; end
if ~isfield(param, 'A'), param.A = @(x) x; end
if ~isfield(param, 'nu'), param.nu = 1; end

% Advanced input arguments (not exposed in documentation)
if ~isfield(param, 'weights_dx'), param.weights_dx = 1; end
if ~isfield(param, 'weights_dy'), param.weights_dy = 1; end
if ~isfield(param, 'zero_weights_flag'), param.zero_weights_flag = 1; end
if ~isfield(param, 'identical_weights_flag')
    param.identical_weights_flag = 0; 
end
if ~isfield(param, 'sphere_flag'), param.sphere_flag = 0; end
if ~isfield(param, 'incNP'), param.incNP = 0; end

% Set grad and div operators to planar or spherical case and also 
% include weights or not (depending on parameter flags).
if (param.sphere_flag)
   G = @sopt_mltb_gradient_op_sphere;
   D = @sopt_mltb_div_op_sphere;
else
   G = @sopt_mltb_gradient_op;
   D = @sopt_mltb_div_op;
end

if (~param.identical_weights_flag && param.zero_weights_flag)
    grad = @(x) G(x, param.weights_dx, param.weights_dy);
    div = @(r, s) D(r, s, param.weights_dx, param.weights_dy);
    max_weights = max([abs(param.weights_dx(:)); ...
        abs(param.weights_dy(:))])^2;
else
    grad = @(x) G(x);
    div = @(r, s) D(r, s);
end

% Initializations
[r, s] = grad(param.A(b*0));
pold = r; qold = s;
told = 1; prev_obj = 0;

% Main iterations
if param.verbose > 1
    fprintf('  Proximal TV operator:\n');
end
for iter = 1:param.max_iter
    
    % Current solution
    sol = b - lambda*param.At(div(r, s));
    
    % Objective function value
    obj = .5*norm(b(:)-sol(:), 2) + lambda * ...
        sopt_mltb_TV_norm(param.A(sol), param.weights_dx, param.weights_dy);
    rel_obj = abs(obj-prev_obj)/obj;
    prev_obj = obj;
    
    % Stopping criterion
    if param.verbose>1
        fprintf('   Iter %i, obj = %e, rel_obj = %e\n', ...
            iter, obj, rel_obj);
    end
    if rel_obj < param.rel_obj
        crit_TV = 'TOL_EPS'; break;
    end
    
    % Udpate divergence vectors and project
    [dx, dy] = grad(param.A(sol));
    if (param.identical_weights_flag)
        r = r - 1/(8*lambda*param.nu) * dx;
        s = s - 1/(8*lambda*param.nu) * dy;
        weights = max(param.weights_dx, sqrt(abs(r).^2+abs(s).^2));
        p = r./weights.*param.weights_dx; q = s./weights.*param.weights_dx;
    else
        if (~param.zero_weights_flag)
            r = r - 1/(8*lambda*param.nu) * dx;
            s = s - 1/(8*lambda*param.nu) * dy;
            weights = max(1, sqrt(abs(r./param.weights_dx).^2+...
                abs(s./param.weights_dy).^2));
            p = r./weights; q = s./weights;
        else
            % Weights go into grad and div operators so usual update
            r = r - 1/(8*lambda*param.nu*max_weights) * dx;
            s = s - 1/(8*lambda*param.nu*max_weights) * dy;
            weights = max(1, sqrt(abs(r).^2+abs(s).^2));
            p = r./weights; q = s./weights;
        end
    end
    
    % FISTA update
    t = (1+sqrt(4*told^2))/2;
    r = p + (told-1)/t * (p - pold); pold = p;
    s = q + (told-1)/t * (q - qold); qold = q;
    told = t;
    
end

% Log after the minimization
if ~exist('crit_TV', 'var'), crit_TV = 'MAX_IT'; end
if param.verbose >= 1
    fprintf(['  Prox_TV: obj = %e, rel_obj = %e,' ...
        ' %s, iter = %i\n'], obj, rel_obj, crit_TV, iter);
end

end
