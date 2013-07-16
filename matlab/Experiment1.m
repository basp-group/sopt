%% Experiment1
% In this experiment we evaluate the performance of SARA for spread 
% spectrum acquisition. We use a 256x256 version of Lena as a test image. 
% Number of measurements is M = 0.2N and input SNR is set to 30 dB. These
% parameters can be changed by modifying the variables p (for the
% undersampling ratio) and input_snr (for the input SNR).


%% Clear workspace

clc
clear;


%% Define paths

addpath misc/
addpath prox_operators/
addpath test_images/



%% Read image

imagename = 'lena_256.tiff';

% Load image
im = im2double(imread(imagename));

% Normalise
im = im/max(max(im));

% Enforce positivity
im(im<0) = 0;

%% Parameters

input_snr = 30; % Noise level (on the measurements)

%Undersampling ratio M/N
p=0.2;


%% Sparsity operators

%Wavelet decomposition depth
nlevel=4;

dwtmode('per');
[C,S]=wavedec2(im,nlevel,'db8'); 
ncoef=length(C);
[C1,S1]=wavedec2(im,nlevel,'db1'); 
ncoef1=length(C1);
[C2,S2]=wavedec2(im,nlevel,'db2'); 
ncoef2=length(C2);
[C3,S3]=wavedec2(im,nlevel,'db3'); 
ncoef3=length(C3);
[C4,S4]=wavedec2(im,nlevel,'db4'); 
ncoef4=length(C4);
[C5,S5]=wavedec2(im,nlevel,'db5'); 
ncoef5=length(C5);
[C6,S6]=wavedec2(im,nlevel,'db6'); 
ncoef6=length(C6);
[C7,S7]=wavedec2(im,nlevel,'db7'); 
ncoef7=length(C7);

%SARA

Psit = @(x) [wavedec2(x,nlevel,'db1')'; wavedec2(x,nlevel,'db2')';wavedec2(x,nlevel,'db3')';...
    wavedec2(x,nlevel,'db4')'; wavedec2(x,nlevel,'db5')'; wavedec2(x,nlevel,'db6')';...
    wavedec2(x,nlevel,'db7')';wavedec2(x,nlevel,'db8')']/sqrt(8); 

Psi = @(x) (waverec2(x(1:ncoef1),S1,'db1')+waverec2(x(ncoef1+1:ncoef1+ncoef2),S2,'db2')+...
    waverec2(x(ncoef1+ncoef2+1:ncoef1+ncoef2+ncoef3),S3,'db3')+...
    waverec2(x(ncoef1+ncoef2+ncoef3+1:ncoef1+ncoef2+ncoef3+ncoef4),S4,'db4')+...
    waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5),S5,'db5')+...
    waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6),S6,'db6')+...
    waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7),S7,'db7')+...
    waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7+ncoef),S,'db8'))/sqrt(8);

%Db8 wavelet basis
Psit2 = @(x) wavedec2(x, nlevel,'db8'); 
Psi2 = @(x) waverec2(x,S,'db8');

%Curvelet
%CurveLab needs to be installed to run Curvelet simulations
realv = 1;
Cv = fdct_usfft(im,realv);
Mod = sopt_mltb_struct2size(Cv);

Psit3 = @(x) sopt_mltb_fwdcurvelet(x,realv); 
Psi3 = @(x) sopt_mltb_adjcurvelet(x,Mod,realv);

%% Spread spectrum operator
% Mask
mask = rand(size(im)) < p; 
ind = find(mask==1);
% Masking matrix (sparse matrix in matlab)
Ma = sparse(1:numel(ind), ind, ones(numel(ind), 1), numel(ind), numel(im));
    
%Spread spectrum sequence
    
ss=rand(size(im));
C=(2*(ss<0.5)-1);

A = @(x) Ma*reshape(fft2(C.*x)/sqrt(numel(ind)), numel(x), 1);
At = @(x) C.*(ifft2(reshape(Ma'*x(:), size(im))*sqrt(numel(ind))));
    
% Sampling
y = A(im);
% Add Gaussian i.i.d. noise
sigma_noise = 10^(-input_snr/20)*std(im(:));
y = y + (randn(size(y)) + 1i*randn(size(y)))*sigma_noise/sqrt(2);
    
    
% Tolerance on noise
epsilon = sqrt(numel(y)+2*sqrt(numel(y)))*sigma_noise;
epsilon_up = sqrt(numel(y)+2.1*sqrt(numel(y)))*sigma_noise;
    
    
% Parameters for BPDN
param.verbose = 1; % Print log or not
param.gamma = 1e-1; % Converge parameter
param.rel_obj = 5e-4; % Stopping criterion for the L1 problem
param.max_iter = 200; % Max. number of iterations for the L1 problem
param.nu_B2 = 1; % Bound on the norm of the operator A
param.tol_B2 = 1-(epsilon/epsilon_up); % Tolerance for the projection onto the L2-ball
param.tight_B2 = 0; % Indicate if A is a tight frame (1) or not (0)
param.pos_B2 = 1; %Positivity constraint: (1) active, (0) not active
param.max_iter_B2=300;
param.tight_L1 = 1; % Indicate if Psit is a tight frame (1) or not (0)
param.nu_L1 = 1;
param.max_iter_L1 = 20;
param.rel_obj_L1 = 1e-2;
    
    
% Solve BPSA problem
    
sol1 = sopt_mltb_solve_BPDN(y, epsilon, A, At, Psi, Psit, param);
    
RSNR1=20*log10(norm(im,'fro')/norm(im-sol1,'fro'));
    
% SARA
% It uses the solution to BPSA as a warm start
maxiter=10;
sigma=sigma_noise*sqrt(numel(y)/(numel(im)*8));
tol=1e-3;
  
sol2 = sopt_mltb_solve_rwBPDN(y, epsilon, A, At, Psi, Psit, param, sigma, tol, maxiter, sol1);

RSNR2=20*log10(norm(im,'fro')/norm(im-sol2,'fro'));

% Solve BPBb8 problem
    
sol3 = sopt_mltb_solve_BPDN(y, epsilon, A, At, Psi2, Psit2, param);
    
RSNR3=20*log10(norm(im,'fro')/norm(im-sol3,'fro'));
    
% RWBPDb8
% It uses the solution to BPDBb8 as a warm start
maxiter=10;
sigma=sigma_noise*sqrt(numel(y)/(numel(im)));
tol=1e-3;
  
sol4 = sopt_mltb_solve_rwBPDN(y, epsilon, A, At, Psi2, Psit2, param, sigma, tol, maxiter, sol3);
      
RSNR4=20*log10(norm(im,'fro')/norm(im-sol4,'fro'));

% Parameters for Curvelet

% Parameters for BPDN
param3.verbose = 1; % Print log or not
param3.gamma = 1e-1; % Converge parameter
param3.rel_obj = 5e-4; % Stopping criterion for the L1 problem
param3.max_iter = 200; % Max. number of iterations for the L1 problem
param3.nu_B2 = 1; % Bound on the norm of the operator A
param3.tol_B2 = 1-(epsilon/epsilon_up); % Tolerance for the projection onto the L2-ball
param3.tight_B2 = 1; % Indicate if A is a tight frame (1) or not (0)
param3.pos_B2 = 1; % Positivity constraint flag. (1) active (0) otherwise
param3.tight_L1 = 1; % Indicate if Psit is a tight frame (1) or not (0)

    


% Solve BP Curvelet problem
    
sol5 = sopt_mltb_solve_BPDN(y, epsilon, A, At, Psi3, Psit3, param3);
    
RSNR5=20*log10(norm(im,'fro')/norm(im-sol5,'fro'));
    
% RW-Curvelet
% It uses the solution to BPDBb8 as a warm start
maxiter=10;
sigma=sigma_noise*sqrt(numel(y)/(numel(im)));
tol=1e-3;
  
sol6 = sopt_mltb_solve_rwBPDN(y, epsilon, A, At, Psi3, Psit3, param3, sigma, tol, maxiter, sol5);
     
RSNR6=20*log10(norm(im,'fro')/norm(im-sol6,'fro'));

    
% Parameters for TVDN
param1.verbose = 1; % Print log or not
param1.gamma = 1e-1; % Converge parameter
param1.rel_obj = 5e-4; % Stopping criterion for the TVDN problem
param1.max_iter = 200; % Max. number of iterations for the TVDN problem
param1.max_iter_TV = 200; % Max. nb. of iter. for the sub-problem (proximal TV operator)
param1.nu_B2 = 1; % Bound on the norm of the operator A
param1.tol_B2 = 1-(epsilon/epsilon_up); % Tolerance for the projection onto the L2-ball
param1.tight_B2 = 0; % Indicate if A is a tight frame (1) or not (0)
param1.max_iter_B2 = 300;
param1.pos_B2 = 1; % Positivity constraint flag. (1) active (0) otherwise
    
% Solve TV problem
    
sol7 = sopt_mltb_solve_TVDN(y, epsilon, A, At, param1);
    
RSNR7=20*log10(norm(im,'fro')/norm(im-sol7,'fro'));
    
% RWTV
% It uses the solution to TV as a warm start
maxiter=10;
sigma=sigma_noise*sqrt(numel(y)/(numel(im)));
tol=1e-3;
  
sol8 = sopt_mltb_solve_rwTVDN(y, epsilon, A, At, param1,sigma, tol, maxiter, sol7);
    
RSNR8=20*log10(norm(im,'fro')/norm(im-sol8,'fro'));


%Show reconstructed images

figure, imagesc(sol1,[0 1]); axis image; axis off; colormap gray;
title(['BPSA, SNR=',num2str(RSNR1), 'dB'])
figure, imagesc(sol2,[0 1]); axis image; axis off; colormap gray;
title(['SARA, SNR=',num2str(RSNR2), 'dB'])

figure, imagesc(sol3,[0 1]); axis image; axis off; colormap gray;
title(['BPDb8, SNR=',num2str(RSNR3), 'dB'])
figure, imagesc(sol4,[0 1]); axis image; axis off; colormap gray;
title(['RW- BPDb8, SNR=',num2str(RSNR4), 'dB'])

figure, imagesc(sol5,[0 1]); axis image; axis off; colormap gray;
title(['Curvelet, SNR=',num2str(RSNR5), 'dB'])
figure, imagesc(sol6,[0 1]); axis image; axis off; colormap gray;
title(['RW-Curvelet, SNR=',num2str(RSNR6), 'dB'])

figure, imagesc(sol7,[0 1]); axis image; axis off; colormap gray;
title(['TV, SNR=',num2str(RSNR7), 'dB'])
figure, imagesc(sol8,[0 1]); axis image; axis off; colormap gray;
title(['RW-TV, SNR=',num2str(RSNR8), 'dB'])











