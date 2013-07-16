function [xsol] = sopt_solver_l1(y, xsol, nr, A, At, Psi, Psit, varargin)
% sopt_solver_l1
%
% This function solves the problem
% 
%   $\min_{x} ||W \Psi^\dagger x||_1 s.t. ||y - A x||_2 < \epsilon$,
% 
% by default, where $\Psi \in C^{N_x \times N_r} $ is the sparsifying
% operator, $W \in R_{+}^{N_x}$ is the diagonal weight matrix,
% $A \in C^{N_y \times N_x}$ is the measurement operator,
% $\epsilon$ is a noise tolerance and $y\in C^{N_y}$ is the
% measurement vector.  The solution is denoted $x^\star \in
% C^{N_x}$.
% 
% The solver can be used to solve the analysis problem, as posed, or the
% synthesis problem 
%
%   $\min_{\alpha} ||W \alpha||_1 s.t. ||y - A \Psi \alpha||_2 < \epsilon$,
% 
% by setting the Analysis flag to false.  The solution is then recovered by
% $x^\star = \Psi \alpha$.  However, note that the optimisation parameters
% specified below correspond to the operators of the analysis problem. 
% The synthesis problem follows by tbe mapping $A \rightarrow A\Psi$ and 
% $\Psi^\dagger \rightarrow I$, where $I$ is the identity.
%
% Default usage is given by
%
%   xsol = sopt_solver_l1(y, xsol, nr, A, At, Psi, Psit);
%
% where y is the measurement vector, xsol is the inital solution (set to
% zero if not known), nr is the dimension of the signal in the
% representation domain, and A and Psi are the measurement and sparsifying 
% operators respectively, with adjoints At and Psit, respectively.
% Operators should be passed with string values specifying the Matlab
% functions implementing the operator.
%
% Options consist of parameter type and value pairs.  Valid options
% include:
%
%  'Weights'               = { Weight vector (default=ones(nr,1)) }
%  'Analysis'              = { false  [solve synthesis problem],
%                              true   [solve analysis problem (default)] } 
%
%  'ParamMaxIter'          = { Maximum number of iterations for the global 
%                              L1 problem (default=200) }
%  'ParamGamma'            = { Convergence parameter for the DR algorithm, 
%                              gamma>0 (default=0.1) }
%  'ParamRelObj'           = { Convergence criteria specifying inimum 
%                              relative change of the objective value
%                              (default=5e-4) }
%  'ParamEpsilon'          = { Radius of the L2 ball (default=1) }
%  'ParamRealOut'          = { Flag for real output signal 
%                              (default=false) }
%  'ParamRealMeas'         = { Flag for real measurements (default=false) }
%
%  'ParamL1ProxMaxIter'    = { Maximum number of iterations (default=200) }
%  'ParamL1ProxRelObj'     = { Minimum relative change of the objective 
%                              value (default=1e-3) }
%  'ParamL1ProxNu'         = { Bound on the squared norm of the operator 
%                              Psi (default=1) }
%  'ParamL1ProxTight'      = { Tight frame flag for Psi (default=false) }
%  'ParamL1ProxPositivity' = { Positivity flag (default=false) }
%
%  'ParamL2BallMaxIter'    = { Maximum number of iterations (default=200) }
%  'ParamL2BallTol'        = { (default=1e-3) }
%  'ParamL2BallNu'         = { Bound on the squared norm of the operator A 
%                              (default=1) }
%  'ParamL2BallTight'      = { Tight frame flag for A (default=false) }
%  'ParamL2BallPositivity' = { Positivity flag (default=false) }
%  'ParamL2BallReality'    = { Reality flag (default=false) }


%% Parse arguments.
p = inputParser;

% Set up parser for required arguments.
p.addRequired('y', @isnumeric);
p.addRequired('xsol', @isnumeric);
p.addRequired('nr', @isnumeric);
p.addRequired('A', @ischar);
p.addRequired('At', @ischar);
p.addRequired('Psi', @ischar);
p.addRequired('Psit', @ischar);

% Set up parser for non-optimisation parameter arguments.
p.addParamValue('Weights', 1, @isnumeric);
p.addParamValue('Analysis', true, @islogical);

% Set up parser for general optimisation parameters.
p.addParamValue('ParamMaxIter', 200, @isnumeric);
p.addParamValue('ParamGamma', 0.1, @isnumeric);
p.addParamValue('ParamRelObj', 5e-4, @isnumeric);
p.addParamValue('ParamEpsilon', 1, @isnumeric);
p.addParamValue('ParamRealOut', false, @islogical);
p.addParamValue('ParamRealMeas', false, @islogical);

% Set up parser for parameters for the l1 prox.
p.addParamValue('ParamL1ProxMaxIter', 200, @isnumeric);
p.addParamValue('ParamL1ProxRelObj', 1e-3, @isnumeric);
p.addParamValue('ParamL1ProxNu', 1, @isnumeric);
p.addParamValue('ParamL1ProxTight', false, @islogical);
p.addParamValue('ParamL1ProxPositivity', false, @islogical);

% Set up parser for parameters for the l2 prox.
p.addParamValue('ParamL2BallMaxIter', 200, @isnumeric);
p.addParamValue('ParamL2BallTol', 1e-3, @isnumeric);
p.addParamValue('ParamL2BallNu', 1, @isnumeric);
p.addParamValue('ParamL2BallTight', false, @islogical);
p.addParamValue('ParamL2BallPositivity', false, @islogical);
p.addParamValue('ParamL2BallReality', false, @islogical);

% Parse.
p.parse(y, xsol, nr, A, At, Psi, Psit, varargin{:})
args = p.Results
ny = length(y);
nx = length(xsol);

% Set defaults.
if (length(args.Weights) == 1), args.Weights = ones(nr,1); end

  
%% Call mex function.
xsol = sopt_solver_l1_mex(y, xsol, ny, nx, nr, A, At, Psi, Psit, ...
  args.Weights, args.Analysis, ...
  args.ParamMaxIter, args.ParamGamma, args.ParamRelObj, ...
  args.ParamEpsilon, args.ParamRealOut, args.ParamRealMeas, ...
  args.ParamL1ProxMaxIter, args.ParamL1ProxRelObj, args.ParamL1ProxNu, ...
  args.ParamL1ProxTight, args.ParamL1ProxPositivity, ...
  args.ParamL2BallMaxIter, args.ParamL2BallTol, args.ParamL2BallNu, ...
  args.ParamL2BallTight, args.ParamL2BallPositivity, ...
  args.ParamL2BallReality);


