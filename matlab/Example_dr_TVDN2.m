%% Example_TVDN2
% Two examples to demonstrate use of TVDN solver.  Simple inpainting and
% Fourier measurement examples are included.


%% Clear workspace
clc;
clear;

%% Define paths
addpath misc/
addpath prox_operators/

%% Parameters
input_snr = 30; % Noise level (on the measurements)

%% Load an image
im = im2double(imread('cameraman.tif'));
%
figure(1);
imagesc(im); axis image; axis off;
colormap gray; title('Original image'); drawnow;

%% Create a mask with 33% of 1 (the rest is set to 0)
% Mask
mask = rand(size(im)) < 0.33; ind = find(mask==1);
% Masking matrix (sparse matrix in matlab)
Ma = sparse(1:numel(ind), ind, ones(numel(ind), 1), numel(ind), numel(im));
% Masking operator
A = @(x) Ma*x(:); % Select 33% of the values in x;
At = @(x) reshape(Ma'*x(:), size(im)); % Adjoint operator + reshape image

%% First problem: Inpainting problem
% Select 33% of pixels
y = A(im);
% Add Gaussian i.i.d. noise
sigma_noise = 10^(-input_snr/20)*std(im(:));
y = y + randn(size(y))*sigma_noise;
% Display the downsampled image
figure(2); clf;
subplot(121); imagesc(At(y)); axis image; axis off;
colormap gray; title('Measured image'); drawnow;
% Parameters for TVDN
param.verbose = 1; % Print log or not
param.gamma = 1e-1; % Converge parameter
param.rel_obj = 1e-4; % Stopping criterion for the TVDN problem
param.max_iter = 200; % Max. number of iterations for the TVDN problem
param_TV.max_iter_TV = 100; % Max. nb. of iter. for the sub-problem (proximal TV operator)
param.nu_B2 = 1; % Bound on the norm of the operator A
param.tol_B2 = 1e-4; % Tolerance for the projection onto the L2-ball
param.tight_B2 = 0; % Indicate if A is a tight frame (1) or not (0)
param.max_iter_B2 = 500;
% Tolerance on noise
epsilon = sqrt(chi2inv(0.99, numel(ind)))*sigma_noise;
% Solve TVDN problem
sol = sopt_mltb_dr_TVDN(y, epsilon, A, At, param);
% Show reconstructed image
figure(2);
subplot(122); imagesc(sol); axis image; axis off;
colormap gray; title('Reconstructed image'); drawnow;

%% Second problem: Reconstruct from 33% of Fourier measurements
% Composition (Masking o Fourier)
A = @(x) Ma*reshape(fft2(x)/sqrt(numel(im)), numel(x), 1);
At = @(x) ifft2(reshape(Ma'*x(:), size(im))*sqrt(numel(im)));
% Select 33% of Fourier coefficients
y = A(im);
% Add Gaussian i.i.d. noise
sigma_noise = 10^(-input_snr/20)*std(im(:));
y = y + (randn(size(y)) + 1i*randn(size(y)))*sigma_noise/sqrt(2);
% Display the downsampled image
figure(3); clf;
subplot(121); imagesc(real(At(y))); axis image; axis off;
colormap gray; title('Measured image'); drawnow;
% Tolerance on noise
epsilon = sqrt(chi2inv(0.99, 2*numel(ind))/2)*sigma_noise;
% Solve TVDN problem
sol = sopt_mltb_dr_TVDN(y, epsilon, A, At, param);
% Show reconstructed image
figure(3);
subplot(122); imagesc(real(sol)); axis image; axis off;
colormap gray; title('Reconstructed image'); drawnow;
