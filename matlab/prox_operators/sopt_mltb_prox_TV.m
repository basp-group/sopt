function sol = sopt_mltb_prox_TV(x, lambda, param)
% sopt_mltb_prox_TV - Total variation proximal operator
%
% Compute the TV proximal operator, i.e. solve
%
%   min_{z} ||x - z||_2^2 + lambda * ||z||_{TV}
%
% where x is the input vector and the solution z* is returned as sol.  
% The structure param should contain the following fields:
%
%   - max_iter: Maximum number of iterations (default = 200).
%
%   - rel_obj: Minimum relative change of the objective value 
%       (default = 1e-4).  The algorithm stops if
%           | ||x(t)||_TV - ||x(t-1)||_TV | / ||x(t)||_TV < rel_obj,
%       where x(t) is the estimate of the solution at iteration t.
%
%   - verbose: Verbosity level (0 = no log, 1 = summary at convergence, 
%       2 = print main steps; default = 1).
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

% Initializations
[r, s] = sopt_mltb_gradient_op(x*0);
pold = r; qold = s;
told = 1; prev_obj = 0;

% Main iterations
if param.verbose > 1
    fprintf('  Proximal TV operator:\n');
end
for iter = 1:param.max_iter
    
    % Current solution
    sol = x - lambda*sopt_mltb_div_op(r, s);
    
    % Objective function value
    obj = .5*norm(x(:)-sol(:), 2)^2 + lambda * sopt_mltb_TV_norm(sol, 0);
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
    [dx, dy] = sopt_mltb_gradient_op(sol);
    r = r - 1/(8*lambda) * dx; s = s - 1/(8*lambda) * dy;
    weights = max(1, sqrt(abs(r).^2+abs(s).^2));
    p = r./weights; q = s./weights;
    
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
