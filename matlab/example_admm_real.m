clear
addpath test_images/
addpath misc/
addpath prox_operators/

%% Load data
stringname='real_data.vis';
Y = importdata(stringname);

%Measurement vector
y = Y(:,3) + 1i*Y(:,4);

%u-v coverage
u = Y(:,1);
v = Y(:,2);

%Noise standard deviation vector
sigma = Y(:,5);
%Weighting vector
w = 1./sigma;


clear Y

M = length(y);

%% Measurement operator initialization
%Image dimensions
Nx = 2048;
Ny = 2048;

N = Nx*Ny;

%Oversampling factors for nufft
ox = 2;
oy = 2;

%Number of neighbours for nufft
Kx = 8;
Ky = 8;

%Initialize nufft parameters
fprintf('Initializing the NUFFT operator\n\n');
tstart1=tic;
st = nufft_init([v u],[Ny Nx],[Ky Kx],[oy*Ny ox*Nx], [Ny/2 Nx/2]);
tend1=toc(tstart1);
fprintf('Time for the initialization: %e\n\n', tend1);

%Operator functions
A = @(x) nufft(x, st);
At = @(x) nufft_adj(x, st);

%Maximum eigenvalue of operato A^TA
eval = pow_method(A, At, [Ny,Nx], 1e-4, 100, 1);

%Bounn for the L2 residual
epsilon = sqrt(M + 2*sqrt(M));

%Dirty image
dirty = At(y);
dirty1 = 2*real(dirty)/eval;
dirty1(dirty1<0) = 0;

%% Sparsity operator definition

%Wavelets parameters
nlevel=8;
dwtmode('per');

% Sparsity operator for SARA

[C1,S1]=wavedec2(dirty1,nlevel,'db1'); 
ncoef1=length(C1);
[C2,S2]=wavedec2(dirty1,nlevel,'db2'); 
ncoef2=length(C2);
[C3,S3]=wavedec2(dirty1,nlevel,'db3'); 
ncoef3=length(C3);
[C4,S4]=wavedec2(dirty1,nlevel,'db4'); 
ncoef4=length(C4);
[C5,S5]=wavedec2(dirty1,nlevel,'db5'); 
ncoef5=length(C5);
[C6,S6]=wavedec2(dirty1,nlevel,'db6'); 
ncoef6=length(C6);
[C7,S7]=wavedec2(dirty1,nlevel,'db7'); 
ncoef7=length(C7);
[C8,S8]=wavedec2(dirty1,nlevel,'db8'); 
ncoef8=length(C8);

clear C1 C2 C3 C4 C5 C6 C7 C8


Psit = @(x) [wavedec2(x,nlevel,'db1')'; wavedec2(x,nlevel,'db2')';...
    wavedec2(x,nlevel,'db3')';wavedec2(x,nlevel,'db4')';...
    wavedec2(x,nlevel,'db5')'; wavedec2(x,nlevel,'db6')';...
    wavedec2(x,nlevel,'db7')'; wavedec2(x,nlevel,'db8')'; x(:)]/sqrt(9); 
Psi = @(x) (waverec2(x(1:ncoef1),S1,'db1')+...
    waverec2(x(ncoef1+1:ncoef1+ncoef2),S2,'db2')+...
    waverec2(x(2*ncoef1+1:2*ncoef1+ncoef2),S3,'db3')+...
    waverec2(x(3*ncoef1+1:3*ncoef1+ncoef2),S4,'db4')+...
    waverec2(x(4*ncoef1+1:4*ncoef1+ncoef2),S5,'db5')+...
    waverec2(x(5*ncoef1+1:5*ncoef1+ncoef2),S6,'db6')+...
    waverec2(x(6*ncoef1+1:6*ncoef1+ncoef2),S7,'db7')+...
    waverec2(x(7*ncoef1+1:7*ncoef1+ncoef2),S8,'db8')+...
    reshape(x(8*ncoef1+1:8*ncoef1+ncoef2), [Ny Nx]))/sqrt(9);


%Parameters for BPDN
param1.verbose = 1; % Print log or not
param1.gamma = 1e3; % Converge parameter
param1.rel_obj = 1e-6; % Stopping criterion for the L1 problem
param1.max_iter = 20; % Max. number of iterations for the L1 problem
param1.nu = eval; % Bound on the norm of the operator A
param1.tight_L1 = 0; % Indicate if Psit is a tight frame (1) or not (0)
param1.max_iter_L1 = 100;
param1.rel_obj_L1 = 5e-2;
param1.pos_L1 = 1;
param1.nu_L1 = 1;
param1.verbose_L1 = 2;
param1.initsol = dirty1;
%param1.initz = z;
     
%Solve BPDN
tstart = tic;
[sol, z] = sopt_mltb_admm_bpconw(y, epsilon, A, At, Psi, Psit, w, param1);
tend = toc(tstart)


residual = At(y - A(sol));
DR = eval*max(sol(:))/(norm(residual(:))/sqrt(N))

%save('results/real_results_8.mat','sol','residual','tend','DR','param1','nlevel');

soln = sol/max(sol(:));
dirtyn = dirty1/max(dirty1(:));

figure, imagesc(log10(soln(850:1250,800:1200) + 1e-4)), colorbar, axis image
figure, imagesc(log10(dirtyn(850:1250,800:1200) + 1e-2)), colorbar, axis image

figure, imagesc(real(residual(850:1250,800:1200))/eval), colorbar, axis image
