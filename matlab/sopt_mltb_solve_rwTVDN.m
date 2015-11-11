function sol = sopt_mltb_solve_rwTVDN(y, epsilon, A, At, paramT, ...
  sigma, tol, maxiter, initsol)
% sopt_mltb_solve_rwTVDN - Solve reweighted TVDN problem
%
% Solve the reweighted TV minimization function using an homotopy
% continuation method to approximate the L0 norm of the magnitude of the 
% gradient.  At each iteration the following problem is solved:
%
%   min_x ||W x||_TV   s.t.  ||y-A x||_2 < epsilon
%
% where W is a diagonal matrix with diagonal elements given by esentially
% the inverse of the graident.
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
%   - paramT: Structure containing parameters for the TV solver (see 
%       documentation for sopt_mltb_solve_TVDN).  
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

Psit = @(x) x; Psi=@(x) x;

param=paramT;
iter=0;
rel_dist=1;

if nargin<9
    fprintf('RW iteration: %i\n', iter);
    sol = sopt_mltb_solve_TVoA_B2(y, epsilon, A, At, Psi, Psit, param);
else
    sol = initsol;
end


delta=std(sol(:));

while (rel_dist>tol && iter<maxiter)
  
    iter=iter+1;
    delta=max(sigma/10,delta);
    fprintf('RW iteration: %i\n', iter);
    fprintf('delta = %e\n', delta);
    
    %Warm start
    param.initsol=sol;
    sol1=sol;
    
    % Weights
    [param.weights_dx_TV param.weights_dy_TV] = sopt_mltb_gradient_op(real(sol));
    param.weights_dx_TV = delta./(abs(param.weights_dx_TV)+delta);
    param.weights_dy_TV = delta./(abs(param.weights_dy_TV)+delta);
    param.identical_weights_flag_TV = 0;
    param.gamma=1e-1*max(abs(sol1(:)));
    
    %Weighted TV problem
    sol = sopt_mltb_solve_TVDNoA(y, epsilon, A, At, Psi, Psit, param);

    %Relative distance
    rel_dist=norm(sol(:)-sol1(:))/norm(sol1(:)); 
    fprintf('relative distance = %e\n\n', rel_dist);
    delta = delta/10;
    
end

