function sol = sopt_mltb_prox_L1(x, lambda, param)
% sopt_mltb_prox_L1 - Proximal operator with L1 norm
%
% Compute the L1 proximal operator, i.e. solve
%
%   min_{z} 0.5*||x - z||_2^2 + lambda * ||Psit x||_1 ,
%
% where x is the input vector and the solution z* is returned as sol.  
% The structure param should contain the following fields:
%
%   - Psit: Sparsifying transform (default = Identity).
%
%   - Psi: Adjoint of Psit (default = Identity).
%
%   - tight: 1 if Psit is a tight frame or 0 otherwise (default = 1).
%
%   - nu: Bound on the norm^2 of the operator Psi, i.e.
%       ||Psi x||^2 <= nu * ||x||^2 (default = 1).
%
%   - max_iter: Maximum number of iterations (default = 200).
%
%   - rel_obj: Minimum relative change of the objective value 
%       (default = 1e-4).  The algorithm stops if
%           | ||x(t)||_1 - ||x(t-1)||_1 | / ||x(t)||_1 < rel_obj,
%       where x(t) is the estimate of the solution at iteration t.
%
%   - verbose: Verbosity level (0 = no log, 1 = summary at convergence, 
%       2 = print main steps; default = 1).
%
%   - weights: Weights for a weighted L1-norm (default = 1).
%
%   - pos: Positivity flag (1 = positive solution,
%       0 = general complex case; default = 0).
%
% References:
% [1] M.J. Fadili and J-L. Starck, "Monotone operator splitting for
% optimization problems in sparse recovery" , IEEE ICIP, Cairo,
% Egypt, 2009.
% [2] Amir Beck and Marc Teboulle, "A Fast Iterative Shrinkage-Thresholding
% Algorithm for Linear Inverse Problems",  SIAM Journal on Imaging Sciences
% 2 (2009), no. 1, 183--202.

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

% Useful functions
soft = @(z, T) sign(z).*max(abs(z)-T, 0);

% Projection
if param.tight && ~param.pos % TIGHT FRAME CASE
    
    temp = param.Psit(x);
    sol = x + 1/param.nu * param.Psi(soft(temp, ...
        lambda*param.nu*param.weights)-temp);
    crit_L1 = 'REL_OBJ'; iter_L1 = 1;
    dummy = param.Psit(sol);
    norm_l1 = sum(param.weights(:).*abs(dummy(:)));
    
else % NON TIGHT FRAME CASE OR CONSTRAINT INVOLVED
    
    % Initializations
    u_l1 = zeros(size(param.Psit(x)));
    sol = x - param.Psi(u_l1);
    prev_l1 = 0; iter_L1 = 0;
    
    % Soft-thresholding
    % Init
    if param.verbose > 1
        fprintf('  Proximal l1 operator:\n');
    end
    while 1
        
        % L1 norm of the estimate
        dummy = param.Psit(sol);
        
        norm_l1 = .5*norm(x(:) - sol(:), 2)^2 + lambda * ...
            sum(param.weights(:).*abs(dummy(:)));
        rel_l1 = abs(norm_l1-prev_l1)/norm_l1;
        
        % Log
        if param.verbose>1
            fprintf('   Iter %i, ||Psit x||_1 = %e, rel_l1 = %e\n', ...
                iter_L1, norm_l1, rel_l1);
        end
        
        % Stopping criterion
        if (rel_l1 < param.rel_obj)
            crit_L1 = 'REL_OB'; break;
        elseif iter_L1 >= param.max_iter
            crit_L1 = 'MAX_IT'; break;
        end
        
        % Soft-thresholding
        res = u_l1*param.nu + param.Psit(sol);
        dummy = soft(res, lambda*param.nu*param.weights);
        if param.pos
            dummy = real(dummy); dummy(dummy<0) = 0;
        end
        u_l1 = 1/param.nu * (res - dummy);
        sol = x - param.Psi(u_l1);
        
        % Update
        prev_l1 = norm_l1;
        iter_L1 = iter_L1 + 1;
        
    end
end

% Log after the projection onto the L2-ball
if param.verbose >= 1
    fprintf(['  prox_L1: ||Psi x||_1 = %e,', ...
        ' %s, iter = %i\n'], norm_l1, crit_L1, iter_L1);
end

end
