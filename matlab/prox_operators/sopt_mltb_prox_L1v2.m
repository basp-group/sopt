function [sol, norm_l1] = prox_L1(x, lambda, param)
% PROJ_L1 - Proximal operator with L1 norm
%
% sol = prox_L1(x, lambda, param) solves:
%
%   min_{z} 0.5*||x - z||_2^2 + lambda * ||Psit x||_1
%
% The input argument param contains the following fields:
%
%   - Psit: Sparsifying transform (default: Id).
%
%   - Psi: Adjoint of Psit (default: Id).
%
%   - tight: 1 if Psit is a tight frame or 0 if not (default = 1)
%
%   - nu: bound on the norm of the operator A, i.e.
%       ||Psi x||^2 <= nu * ||x||^2 (default: 1)
%
%   - max_iter: max. nb. of iterations (default: 200).
%
%   - rel_obj: minimum relative change of the objective value (default:
%   1e-4)
%       The algorithm stops if
%           | ||x(t)||_1 - ||x(t-1)||_1 | / ||x(t)||_1 < rel_obj,
%       where x(t) is the estimate of the solution at iteration t.
%
%   - verbose: 0 no log, 1 a summary at convergence, 2 print main
%   steps (default: 1)
%
%   - weights: weights for a weighted L1-norm (default = 1)
%
%
%
% References:
% [1] M.J. Fadili and J-L. Starck, "Monotone operator splitting for
% optimization problems in sparse recovery" , IEEE ICIP, Cairo,
% Egypt, 2009.
% [2] Amir Beck and Marc Teboulle, "A Fast Iterative Shrinkage-Thresholding
% Algorithm for Linear Inverse Problems",  SIAM Journal on Imaging Sciences
% 2 (2009), no. 1, 183--202.
%
%
%
% Author: Gilles Puy
% E-mail: gilles.puy@epfl.ch
% Date: Dec. 1, 2010
%

% Optional input arguments
if ~isfield(param, 'verbose'), param.verbose = 1; end
if ~isfield(param, 'Psit'), param.Psi = @(x) x; param.Psit = @(x) x; end
if ~isfield(param, 'tight'), param.tight = 1; end
if ~isfield(param, 'nu'), param.nu = 1; end
if ~isfield(param, 'rel_obj'), param.rel_obj = 1e-4; end
if ~isfield(param, 'max_iter'), param.max_iter = 200; end
if ~isfield(param, 'Psit'), param.Psit = @(x) x; end
if ~isfield(param, 'Psi'), param.Psi = @(x) x; end
if ~isfield(param, 'weights'), param.weights = 1; end
if ~isfield(param, 'pos'), param.pos = 0; end
if ~isfield(param, 'real'), param.real = 0; end

% Useful functions
soft = @(z, T) sign(z).*max(abs(z)-T, 0);

% Projection
if param.tight && ~param.pos && ~param.real % TIGHT FRAME CASE
    
    temp = param.Psit(x);
    sol = x + 1/param.nu * param.Psi(soft(temp, ...
        lambda*param.nu*param.weights)-temp);
    crit_L1 = 'REL_OBJ'; iter_L1 = 1;
    dummy = param.Psit(sol);
    %coef=soft(temp,lambda*param.nu*param.weights);
    norm_l1 = sum(param.weights(:).*abs(dummy(:)));
    
else % NON TIGHT FRAME CASE OR CONSTRAINT INVOLVED
    
    % Initializations
    
    sol = x;
    if param.pos || param.real
        sol = real(sol);
    end
    dummy = param.Psit(sol);
    u_l1 = zeros(size(dummy));
    prev_obj = 0; iter_L1 = 0;
    
    % Soft-thresholding
    % Init
    if param.verbose > 1
        fprintf('  Proximal L1 operator:\n');
    end
    while 1
        
        % L1 norm of the estimate
        norm_l1 = sum(param.weights(:).*abs(dummy(:)));
        
        obj = .5*norm(x(:) - sol(:), 2)^2 + lambda * norm_l1;
        rel_obj = abs(obj-prev_obj)/obj;
        
        % Log
        if param.verbose>1
            fprintf('   Iter %i, prox_fval = %e, rel_fval = %e\n', ...
                iter_L1, obj, rel_obj);
        end
        
        % Stopping criterion
        if (rel_obj < param.rel_obj)
            crit_L1 = 'REL_OB'; break;
        elseif iter_L1 >= param.max_iter
            crit_L1 = 'MAX_IT'; break;
        end
        
        % Soft-thresholding
        res = u_l1*param.nu + dummy;
        dummy = soft(res, lambda*param.nu*param.weights);
        u_l1 = 1/param.nu * (res - dummy);
        sol = x - param.Psi(u_l1);
        
        if param.pos
            sol = real(sol); sol(sol<0) = 0;
        end
        
        if param.real
            sol = real(sol); 
        end
        
        % Update
        prev_obj = obj;
        iter_L1 = iter_L1 + 1;
        dummy = param.Psit(sol);
        
    end
end

% Log after the projection onto the L2-ball
if param.verbose >= 1
    fprintf(['  prox_L1: prox_fval = %e,', ...
        ' %s, iter = %i\n'], norm_l1, crit_L1, iter_L1);
end

end