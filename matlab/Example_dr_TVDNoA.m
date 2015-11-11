%% Example TVDNoA
% Example to demonstrate TVoA_B2 solver, where an additional operator is
% included in the TV norm of the TVDN problem.


%% Clear workspace
clc;
clear;

%% Define paths
addpath misc/
addpath prox_operators/

%% Parameters
N = 32;
input_snr = 1; % Noise level (on the measurements)
randn('seed', 1);

%% Load an image
im_ref = phantom(N);
%
figure(1);
imagesc(im_ref); axis image; axis off;
colormap gray; title('Original image'); drawnow;

%% Create an artifial operator to test prox_TVoS
S = randn(N^2, N^2)/N^2;
im = reshape(S\im_ref(:), N, N); % S*im is thus sparse in TV

%% Add Gaussian i.i.d. noise to im
y = im;
sigma_noise = 10^(-input_snr/20)*std(im_ref(:));
y = y + randn(size(y))*sigma_noise;

%% Solving modified ROF
% Parameters for TVDN
param.verbose = 2; % Print log or not
param.gamma = 1; % Converge parameter
param.rel_obj = 1e-4; % Stopping criterion for the TVDN problem
param.max_iter = 100; % Max. number of iterations for the TVDN problem
param.max_iter_TV = 100; % Max. nb. of iter. for the sub-problem (proximal TV operator)
param.nu_B2 = 1; % Bound on the norm of the measurement operator (Id here)
param.tight_B2 = 1; % Indicate that A is a tight frame (1) or not (0)
param.nu_TV = norm(S)^2; % Bound on the norm of the operator S
func_S = @(x) reshape(S*x(:), N, N);
func_St = @(x) reshape(S'*x(:), N, N);
% Tolerance on noise
epsilon = sqrt(chi2inv(0.99, numel(im)))*sigma_noise;
% Identity
A = @(x) x;
% Solve
sol = sopt_mltb_dr_TVDNoA(y, epsilon, A, A, func_S, func_St, param);
% Show reconstructed image
figure(2);
imagesc(func_S(sol)); axis image; axis off;
colormap gray; title('Reconstructed image'); drawnow;
