function [xsol, z] = sopt_mltb_admm_bpconw(y, epsilon, A, At, Psi, Psit, w, param)
%
% sol = admm_bpconw(y, epsilon, A, At, Psi, Psit, w, param) solves:
%
%   min ||Psit x||_1   s.t.  ||W*(y-A x)||_2 <= epsilon
%
%
% y contains the measurements. A is the forward measurement operator and
% At the associated adjoint operator. Psit is a sparfying transform and Psi
% its adjoint. w is a vector containing the weights for the L2 norm.
% PARAM a Matlab structure containing the following fields:
%
%   General parameters:
% 
%   - verbose: 0 no log, 1 print main steps, 2 print all steps.
%
%   - max_iter: max. nb. of iterations (default: 200).
%
%   - rel_obj: minimum relative change of the objective value (default:
%   1e-4)
%       The algorithm stops if
%           | ||x(t)||_1 - ||x(t-1)||_1 | / ||x(t)||_1 < rel_obj,
%       where x(t) is the estimate of the solution at iteration t.
%
%   - gamma: control the converge speed (default: 1e-1).
% 
% 
%   Proximal L1 operator:
%
%   - rel_obj_L1: Used as stopping criterion for the proximal L1
%   operator. Min. relative change of the objective value between two
%   successive estimates.
%
%   - max_iter_L1: Used as stopping criterion for the proximal L1
%   operator. Maximun number of iterations.
% 
%   - param.nu_L1: bound on the norm^2 of the operator Psi, i.e.
%       ||Psi x||^2 <= nu * ||x||^2 (default: 1)
% 
%   - param.tight_L1: 1 if Psit is a tight frame or 0 if not (default = 1)
% 
%   - param.weights: weights (default = 1) for a weighted L1-norm defined
%       as sum_i{weights_i.*abs(x_i)}
%           
%
% Author: Rafael Carrillo
% E-mail: rafael.carrillo@epfl.ch
% Date: Nov. 22, 2014
%

% Optional input arguments
if ~isfield(param, 'verbose'), param.verbose = 1; end
if ~isfield(param, 'rel_obj'), param.rel_obj = 1e-4; end
if ~isfield(param, 'max_iter'), param.max_iter = 200; end
if ~isfield(param, 'gamma'), param.gamma = 1e-2; end
if ~isfield(param, 'nu'), param.nu = 1; end

% Input arguments for prox L1
param_L1.Psi = Psi; 
param_L1.Psit = Psit; 
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
else
    param_L1.rel_obj = param.rel_obj;
end
if isfield(param, 'weights')
    param_L1.weights = param.weights;
else
    param_L1.weights = 1;
end
if isfield(param, 'pos_L1')
    param_L1.pos = param.pos_L1;
else
    param_L1.pos = 0;
end
if isfield(param, 'verbose_L1')
    param_L1.verbose = param.verbose_L1;
else
    param_L1.verbose = param.verbose;
end

% Useful functions for the projection
sc = @(z) z*min(epsilon/norm(z(:)), 1); % scaling


%Initializations.

%Initial solution
if isfield(param,'initsol')
    xsol = param.initsol;
else
    xsol = 1/param.nu*At(y); 
end

%Initial dual variables
if isfield(param, 'initz')
    z = param.initz;
else
    z = zeros(size(y));
end

%Initial residual
res = A(xsol) - y;

%Flags initialization
dummy = Psit(xsol);
fval = sum(param_L1.weights(:).*abs(dummy(:))); 
flag = 0;

%Step sizes computation

%Step size primal 
mu = 1.0/param.nu;

%Step size for the dual variables
beta = 0.9;

%Main loop. Sequential.
for t = 1:param.max_iter
    
    if (param.verbose >= 1)
        fprintf('Iter %i\n',t);
    end
    
    %Slack variable update
    s = sc(-w.*(z + res))./w;
    
    %Gradient formation
    r = At(z + res + s);
    
    %Gradient decend
    r = xsol - mu*r;
    
    %Prox L1 norm (global solution)
    prev_fval = fval;
    [xsol, fval] = sopt_mltb_prox_L1(r, param.gamma*mu, param_L1);
    
    %Residual
    res = A(xsol) - y;
    
    %Lagrange multipliers update
    z = z + beta*(res + s);
     
    %Check feasibility
    res1 = norm(res(:).*w);
    
    %Check relative change of objective function      
    rel_fval = abs(fval - prev_fval)/fval;
    
    %Log
    if (param.verbose >= 1)
        fprintf(' L1 norm = %e, rel_fval = %e\n', ...
            fval, rel_fval);
        fprintf(' epsilon = %e, residual = %e\n\n', epsilon, res1);
    end
    
    %Global stopping criteria
    if (rel_fval < param.rel_obj && res1 <= epsilon*1.001)
        flag = 1;
        break;
    end
          
end


%Final log
if (param.verbose > 0)
    if (flag == 1)
        fprintf('Solution found\n');
        fprintf(' Objective function = %e\n', fval);
        fprintf(' Final residual = %e\n', res1);
    else
        fprintf('Maximum number of iterations reached\n');
        fprintf(' Objective function = %e\n', fval);
        fprintf(' Relative variation = %e\n', rel_fval);
        fprintf(' Final residual = %e\n', res1);
        fprintf(' epsilon = %e\n', epsilon);
    end
end

end

