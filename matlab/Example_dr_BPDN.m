%% Example_BPDN
% Example to demonstrate use of BPDN solver.  A random Fourier sampling
% measurement operator is considered.  Daubechies 8 wavelets are used for
% the sparsifying operator.


%% Clear workspace

clc
clear;


%% Define paths

addpath misc/
addpath prox_operators/


%% Define parameters

% Coverages (half the plane for Fourier sampling)
p = [0.50];

% Noise level (on the measurements)
input_snr = 30;


%% Read image

% Load image
im = im2double(imread('cameraman.tif'));

% Normalise
im = im/max(max(im));

% Enforce positivity
im(im<0) = 0;


%% Define sparsity operator

dwtmode('per');

% Decomposition level of the wavelet transform.
nlevel = 4;

% Daubechies 8 wavelet operator
[C,S] = wavedec2(im,nlevel,'db8');

Psit = @(x) wavedec2(x, nlevel, 'db8');
Psi = @(x) waverec2(x, S, 'db8');


%% Run simulations

% Random Fourier sampling example
%  Define mask
%  Uniform sampling of the half Fourier plane
mask = rand(size(im)) < p;
mask(:,1:floor(size(im,2)/2))=0;
mask = ifftshift(mask);
mask(1,1)=0;
mask(floor(size(im,1)/2):end,1)=0;

ind = find(mask==1);
Ma = sparse(1:numel(ind), ind, ...
  ones(numel(ind), 1), numel(ind), numel(im));

% Composition (Masking o Fourier)
A = @(x) Ma*reshape(fft2(x)/sqrt(numel(ind)), numel(x), 1);
At = @(x) (ifft2(reshape(Ma'*x(:), size(im))*sqrt(numel(ind))));

% Apply measurement operator
y = A(im);

% Add Gaussian i.i.d. noise
sigma_noise = 10^(-input_snr/20)*std(im(:));
noise = (randn(size(y)) + 1i*randn(size(y)))*sigma_noise/sqrt(2);
y = y + noise;

% Tolerance on noise
epsilon = sqrt(numel(y) + 2*sqrt(numel(y)))*sigma_noise;
epsilon_up = sqrt(numel(y) + 2.1*sqrt(numel(y)))*sigma_noise;
tol_B2 = (epsilon_up/epsilon)-1; % Tolerance for the projection onto the L2-ball

% Solve optimisation problem

% Parameters for BPDN
param.verbose = 1; % Print log or not
param.gamma = 1e-1; % Convergence parameter
param.rel_obj = 5e-4; % Stopping criterion for the L1 problem
param.max_iter = 200; % Max. number of iterations for the L1 problem
param.nu_B2 = 1; % Bound on the norm of the operator A
param.tight_B2 = 0; % Indicate if A is a tight frame (1) or not (0)
param.max_iter_B2 = 200; %Max. number of iterations of the L2-ball projection
param.pos_B2 = 1; %Positivity flag
param.tol_B2 = tol_B2; % Tolerance for the projection onto the L2-ball
param.tight_L1 = 1; % Indicate if Psit is a tight frame (1) or not (0)
%Optional parameters when Psit is not a tight frame:
%param.nu_L1 = 1; % Bound on the norm of the operator Psit
%param.max_iter_L1 = 200; %Max. number of iterations of the L1 prox
%param.rel_obj_L1 = 1e-3; % Tolerance for the prox L1

% Solve BPDN problem with positivity constraint
sol1 = sopt_mltb_dr_BPDN(y, epsilon, A, At, Psi, Psit, param);

% Compute SNR
RSNR1 = 20*log10(norm(im,'fro') ...
  / norm(im-sol1,'fro'));

% Example with only reality constraint
param.pos_B2 = 0; %Positivity flag
param.real_B2 = 1; %Reality flag

% Solve BPDN problem
sol2 = sopt_mltb_dr_BPDN(y, epsilon, A, At, Psi, Psit, param);

% Compute SNR
RSNR2 = 20*log10(norm(im,'fro') ...
  / norm(im-sol2,'fro'));


%% Show results

figure
imagesc(sol1), axis off, axis image, colorbar
title(['Rec. with positivity const. SNR=',num2str(RSNR1), 'dB'])
colormap gray

figure
imagesc(sol2), axis off, axis image, colorbar
title(['Rec. with reality const. SNR=',num2str(RSNR2), 'dB'])
colormap gray
