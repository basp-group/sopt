function sol = sopt_mltb_solve_BPDN(y, epsilon, A, At, Psi, Psit, param)
% sopt_mltb_solve_BPDN - Solve BPDN problem
%
% Solve the Basis Pursuit Denoising (BPDN) problem
%
%   min ||Psit x||_1   s.t.  ||y-A x||_2 < epsilon
%
% where y contains the measurements, A is the forward measurement operator 
% and At the associated adjoint operator, Psit is a sparfying transform and
% Psi its adjoint. The structure param should contain the following fields:
%
%   General parameters:
% 
%   - verbose: Verbosity level (0 = no log, 1 = summary at convergence, 
%       2 = print main steps; default = 1).
%
%   - max_iter: Maximum number of iterations (default = 200).
%
%   - rel_obj: Minimum relative change of the objective value 
%       (default = 1e-4).  The algorithm stops if
%           | ||x(t)||_1 - ||x(t-1)||_1 | / ||x(t)||_1 < rel_obj,
%       where x(t) is the estimate of the solution at iteration t.
%
%   - gamma: Convergence speed (weighting of L1 norm when solving for
%       L1 proximal operator) (default = 1e-1).
%
%   - initsol: Initial solution for a warmstart.
% 
%   Projection onto the L2-ball:
%
%   - param.tight_B2: 1 if A is a tight frame or 0 otherwise (default = 1).
% 
%   - nu_B2: Bound on the norm of the operator A, i.e.
%       ||A x||^2 <= nu * ||x||^2 (default = 1).
%
%   - tol_B2: Tolerance for the projection onto the L2 ball. The algorithms
%       stops if
%         epsilon/(1-tol) <= ||y - A z||_2 <= epsilon/(1+tol)
%       (default = 1e-3).
%
%   - max_iter_B2: Maximum number of iterations for the projection onto the
%       L2 ball (default = 200).
%
%   - pos_B2: Positivity flag (1 to impose positivity, 0 otherwise;
%       default = 0).
%
%   - real_B2: Reality flag (1 to impose reality, 0 otherwise;
%       default = 0).
% 
%   Proximal L1 operator:
%
%   - rel_obj_L1: Used as stopping criterion for the proximal L1
%       operator. Minimum relative change of the objective value between
%       two successive estimates.
%
%   - max_iter_L1: Used as stopping criterion for the proximal L1
%       operator. Maximun number of iterations.
% 
%   - param.nu_L1: Bound on the norm^2 of the operator Psi, i.e.
%       ||Psi x||^2 <= nu * ||x||^2 (default = 1).
% 
%   - param.tight_L1: 1 if Psit is a tight frame, 0 otherwise 
%       (default = 1).
% 
%   - param.weights: Weights for a weighted L1-norm defined
%       by sum_i{weights_i.*abs(x_i)} (default = 1). 
%
%   - pos_l1: Positivity flag (1 to impose positivity, 0 otherwise;
%       default = 0).
%
% References:
% [1] P. L. Combettes and J-C. Pesquet, "A Douglas-Rachford Splitting 
% Approach to Nonsmooth Convex Variational Signal Recovery", IEEE Journal
% of Selected Topics in Signal Processing, vol. 1, no. 4, pp. 564-574, 2007.

% Optional input arguments
if ~isfield(param, 'verbose'), param.verbose = 1; end
if ~isfield(param, 'rel_obj'), param.rel_obj = 1e-4; end
if ~isfield(param, 'max_iter'), param.max_iter = 200; end
if ~isfield(param, 'gamma'), param.gamma = 1e-2; end
if ~isfield(param, 'pos_l1'), param.pos_l1 = 0; end

% Input arguments for projection onto the L2 ball
param_B2.A = A; param_B2.At = At;
param_B2.y = y; param_B2.epsilon = epsilon;
param_B2.verbose = param.verbose;
if isfield(param, 'nu_B2'), param_B2.nu = param.nu_B2; end
if isfield(param, 'tol_B2'), param_B2.tol = param.tol_B2; end
if isfield(param, 'tight_B2'), param_B2.tight = param.tight_B2; end
if isfield(param, 'max_iter_B2')
    param_B2.max_iter = param.max_iter_B2;
end
if isfield(param,'pos_B2'), param_B2.pos=param.pos_B2; end
if isfield(param,'real_B2'), param_B2.real=param.real_B2; end

% Input arguments for prox L1
param_L1.Psi = Psi; param_L1.Psit = Psit; param_L1.pos = param.pos_l1;
param_L1.verbose = param.verbose; 
%param_L1.verbose = 2; 
param_L1.rel_obj = param.rel_obj;
if isfield(param, 'nu_L1')
    param_L1.nu = param.nu_L1;
end
if isfield(param, 'tight_L1')
    param_L1.tight = param.tight_L1;
end
if isfield(param, 'max_iter_L1')
    param_L1.max_iter = param.max_iter_L1;
end
if isfield(param, 'rel_obj_L1')
    param_L1.rel_obj = param.rel_obj_L1;
end
if isfield(param, 'weights')
    param_L1.weights = param.weights;
else
    param_L1.weights = 1;
end

% Initialization
if isfield(param,'initsol')
    xhat = param.initsol;
else
    xhat = At(y); 
end

iter = 1; prev_norm = 0;

% Main loop
while 1
    
    if param.verbose >= 1
        fprintf('Iteration %i:\n', iter);
    end
    
    % Projection onto the L2-ball
    [sol, param_B2.u] = sopt_mltb_fast_proj_B2(xhat, param_B2);
    
    % Global stopping criterion
    dummy = Psit(sol);
    curr_norm = sum(param_L1.weights(:).*abs(dummy(:)));    
    rel_norm = abs(curr_norm - prev_norm)/curr_norm;
    if param.verbose >= 1
        fprintf('  ||x||_1 = %e, rel_norm = %e\n', ...
            curr_norm, rel_norm);
    end
    if (rel_norm < param.rel_obj)
        crit_BPDN = 'REL_NORM';
        break;
    elseif iter >= param.max_iter
        crit_BPDN = 'MAX_IT';
        break;
    end
    
    % Proximal L1 operator
    xhat = 2*sol - xhat;
    temp = sopt_mltb_prox_L1(xhat, param.gamma, param_L1);
    xhat = temp + sol - xhat;
    
    % Update variables
    iter = iter + 1;
    prev_norm = curr_norm;
    
end

% Log
if param.verbose >= 1
  
    % L1 norm
    fprintf('\n Solution found:\n');
    fprintf(' Final L1 norm: %e\n', curr_norm);
    
    % Residual
    dummy = A(sol); res = norm(y(:)-dummy(:), 2);
    fprintf(' epsilon = %e, ||y-Ax||_2=%e\n', epsilon, res);
    
    % Stopping criterion
    fprintf(' %i iterations\n', iter);
    fprintf(' Stopping criterion: %s \n\n', crit_BPDN);
    
end

end
