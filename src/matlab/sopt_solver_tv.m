function [xsol] = sopt_solver_tv(y, xsol, nr, A, At, Psi, Psit, varargin)
% sopt_solver_tv


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
xsol = sopt_solver_tv_mex(y, xsol, ny, nx, nr, A, At, Psi, Psit, ...
  args.Weights, args.Analysis, ...
  args.ParamMaxIter, args.ParamGamma, args.ParamRelObj, ...
  args.ParamEpsilon, args.ParamRealOut, args.ParamRealMeas, ...
  args.ParamL1ProxMaxIter, args.ParamL1ProxRelObj, args.ParamL1ProxNu, ...
  args.ParamL1ProxTight, args.ParamL1ProxPositivity, ...
  args.ParamL2BallMaxIter, args.ParamL2BallTol, args.ParamL2BallNu, ...
  args.ParamL2BallTight, args.ParamL2BallPositivity, ...
  args.ParamL2BallReality);


