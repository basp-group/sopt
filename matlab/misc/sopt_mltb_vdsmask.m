function mask = sopt_mltb_vdsmask(N,M,p)
% sopt_mltb_vdsmask - Create variable density sampling profile
%
% Creates a binary mask generated from a variable density sampling 
% profile for two dimensional images in the frequency domain.   The mask 
% contains on average p*N*M ones's.
%
% Inputs:
%
%   - N and M: Size of mask.
%
%   - p: Coverage percentage.
%
% Note: The undersampling ratio is passed as 2*p and then the function
% only takes samples in half plane to account for signal reality. The range
% of p is 0 < p <= 0.5.
%
% Outputs:
%
%   - mask: Binary mask with the sampling profile in the frequency domain.

p = 2*p;

if p==1
    mask=ones(N,M);
else
    
    nb_meas=round(p*N*M);
    tol=ceil(p*N*M)-floor(p*N*M);
    d=1;

    [x,y] = meshgrid(linspace(-1, 1, M), linspace(-1, 1, N)); % Cartesian grid
    r = sqrt(x.^2+y.^2); r = r/max(r(:)); % Polar grid

    alpha=-1;
    it=0;
    maxit=20;
    while (alpha<-0.01 || alpha>0.01) && it<maxit
        pdf = (1-r).^d;
        [new_pdf,alpha] = sopt_mltb_modifypdf(pdf, nb_meas);
        if alpha<0
            d=d+0.1;
        else
            d=d-0.1;
        end
        it=it+1;
    end

    mask = zeros(size(new_pdf));
    while sum(mask(:))>nb_meas+tol || sum(mask(:))<nb_meas-tol
        mask = sopt_mltb_genmask(new_pdf);
    end

end

%Samples in half plane to account for signal reality.
M1=floor(M/2);
mask(:,1:M1)=0;
mask = ifftshift(mask);

N1=floor(N/2);
mask(N1+1:N,1)=0;

