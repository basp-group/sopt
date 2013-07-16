function sol = sopt_mltb_solve_rwBPDN(y, epsilon, A, At, Psi, Psit, ...
  paramT, sigma, tol, maxiter, initsol)
% sopt_mltb_solve_rwBPDN - Solve reweighted BPDN problem
%
% Solve the reweighted L1 minimization function using an homotopy
% continuation method to approximate the L0 norm.  At each iteration the 
% following problem is solved:
%
%   min_x ||W Psit x||_1   s.t.  ||y-A x||_2 < epsilon
%
% where W is a diagonal matrix with diagonal elements given by
%
%   [W]_ii = delta(t)/(delta(t)+|[Psit x(t)]_i|).
%
% The input parameters are defined as follows.
%
%   - y: Input data (measurements).
%
%   - epsilon: Noise bound.
%
%   - A: Forward measurement operator.
%
%   - At: Adjoint measurement operator.
%
%   - Psi: Synthesis sparsity transform.
%
%   - Psit: Analysis sparsity transform.
%
%   - paramT: Structure containing parameters for the L1 solver (see 
%       documentation for sopt_mltb_solve_BPDN).  
%
%   - sigma: Noise standard deviation in the analysis domain.
%
%   - tol: Minimum relative change in the solution.
%       The algorithm stops if 
%           ||x(t)-x(t-1)||_2/||x(t-1)||_2 < tol.
%       where x(t) is the estimate of the solution at iteration t.
%
%   - maxiter: Maximum number of iterations of the reweighted algorithm.
%
%   - initsol: Initial solution for a warmstart.
%
% References:
% [1] R. E. Carrillo, J. D. McEwen, D. Van De Ville, J.-Ph. Thiran, and 
% Y. Wiaux. Sparsity averaging for compressive imaging. IEEE Sig. Proc. 
% Let., in press, 2013.
% [2] R. E. Carrillo, J. D. McEwen, and Y. Wiaux. Sparsity Averaging 
% Reweighted Analysis (SARA): a novel algorithm for radio-interferometric 
% imaging. Mon. Not. Roy. Astron. Soc., 426(2):1223-1234, 2012.

param=paramT;
iter=0;
rel_dist=1;

if nargin<11
    fprintf('RW iteration: %i\n', iter);
    sol = sopt_mltb_solve_BPDN(y, epsilon, A, At, Psi, Psit, param);
else
    sol = initsol;
end

temp=Psit(sol);
delta=std(temp(:));

while (rel_dist>tol && iter<maxiter)
  
    iter=iter+1;
    delta=max(sigma/10,delta);
    fprintf('RW iteration: %i\n', iter);
    fprintf('delta = %e\n', delta);
    
    % Weights
    weights=abs(Psit(sol));
    param.weights=delta./(delta+weights);
    
    % Warm start
    param.initsol=sol;
    param.gamma=1e-1*max(weights(:));
    sol1=sol;
    
    % Weighted L1 problem
    sol = sopt_mltb_solve_BPDN(y, epsilon, A, At, Psi, Psit, param);
    
    % Relative distance
    rel_dist=norm(sol(:)-sol1(:))/norm(sol1(:));
    fprintf('Relative distance = %e\n\n', rel_dist);
    delta = delta/10;
    
end

