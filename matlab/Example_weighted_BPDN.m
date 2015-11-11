%% Exampled_weighted_L1
% Example to demonstrate use of BPDN solver when incorporating weights
% (performs one re-weighting of previous solution).


%% Clear workspace
clc;
clear;

%% Define paths
addpath misc/
addpath prox_operators/

%% Parameters
N = 64;
input_snr = 30; % Noise level (on the measurements)
randn('seed', 1); rand('seed', 1);

%% Create an image with few spikes
im = zeros(N); ind = randperm(N^2); im(ind(1:100)) = 1;
%
figure(1);
subplot(221), imagesc(im); axis image; axis off;
colormap gray; title('Original image'); drawnow;

%% Create a mask
% Mask
mask = rand(size(im)) < 0.095; ind = find(mask==1);
% Masking matrix (sparse matrix in matlab)
Ma = sparse(1:numel(ind), ind, ones(numel(ind), 1), numel(ind), numel(im));
% Masking operator
A = @(x) Ma*x(:);
At = @(x) reshape(Ma'*x(:), size(im));

%% Reconstruct from a few Fourier measurements

% Composition (Masking o Fourier)
A = @(x) Ma*reshape(fft2(x)/sqrt(numel(im)), numel(x), 1);
At = @(x) ifft2(reshape(Ma'*x(:), size(im))*sqrt(numel(im)));

% Sparsity operator
Psit = @(x) x; Psi = Psit;

% Select 33% of Fourier coefficients
y = A(im);

% Add Gaussian i.i.d. noise
sigma_noise = 10^(-input_snr/20)*std(im(:));
y = y + (randn(size(y)) + 1i*randn(size(y)))*sigma_noise/sqrt(2);

% Display the downsampled image
figure(1);
subplot(222); imagesc(real(At(y))); axis image; axis off;
colormap gray; title('Measured image'); drawnow;

% Tolerance on noise
epsilon = sqrt(chi2inv(0.99, 2*numel(ind))/2)*sigma_noise;

% Parameters for BPDN
param.verbose = 1; % Print log or not
param.gamma = 1e-1; % Converge parameter
param.rel_obj = 1e-4; % Stopping criterion for the TVDN problem
param.max_iter = 300; % Max. number of iterations for the TVDN problem
param.nu_B2 = 1; % Bound on the norm of the operator A
param.tol_B2 = 1e-4; % Tolerance for the projection onto the L2-ball
param.tight_B2 = 1; % Indicate if A is a tight frame (1) or not (0)
param.tight_L1 = 1; % Indicate if Psit is a tight frame (1) or not (0)
param.pos_l1 = 1; %

% Solve BPDN problem (without weights)
sol = sopt_mltb_solve_BPDN(y, epsilon, A, At, Psi, Psit, param);

% Show reconstructed image
figure(1);
subplot(223); imagesc(real(sol)); axis image; axis off;
colormap gray; title(['First estimate - ', ...
    num2str(sopt_mltb_SNR(im, real(sol))), 'dB']); drawnow;

% Refine the estimate
param.weights = 1./(abs(sol)+1e-5);
sol = sopt_mltb_solve_BPDN(y, epsilon, A, At, Psi, Psit, param);

% Show reconstructed image
figure(1);
subplot(224); imagesc(real(sol)); axis image; axis off;
colormap gray; title(['Second estimate - ', ...
    num2str(sopt_mltb_SNR(im, real(sol))), 'dB']); drawnow;
