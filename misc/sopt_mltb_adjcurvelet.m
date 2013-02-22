function restim = sopt_mltb_adjcurvelet(coef, Mod, real)
% sopt_mltb_adjcurvelet - Adjoint curvelet transform
%
% Compute the adjoint curvelet transform from the curvelet
% coefficient vector.
%
% Inputs:
%
%   - coef: Input curvelet coefficient vector.
%
%   - Mod: Data structure that stores sizes of the curvelet transform 
%       generated from the CurveLab toolbox.
%
%   - real: Flag indicating if the transform is real or complex (1 = real,  
%       0 = complex).
%
% Outputs:
%
%   - restim: Estimated image from the Curvelet coefficients.

CR=cell(size(Mod));
marker=0;

for s=1:length(Mod)
   CR{s} = cell(size(Mod{s}));
   for w=1:length(Mod{s})
     m=Mod{s}{w}(1);
     n=Mod{s}{w}(2);
     CR{s}{w}=reshape(coef(marker+1:marker+(m*n)),m,n);
     marker=marker+m*n;
   end
end
 
% Apply adjoint curvelet transform
restim = ifdct_usfft(CR,real);
