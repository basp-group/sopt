function coef = sopt_mltb_fwdcurvelet(im,real)
% sopt_mltb_fwdcurvelet - Forward curvelet transform
%
% Compute the forward curvelet transform of an image and stores it in
% vector coef.
%
% Inputs:
%
%   - im: Input image.
%
%   - real: Flag indicating if the transform is real or complex (1 = real,  
%       0 = complex).
%
% Outputs:
%
%   - coef: Curvelet coefficients.

C = fdct_usfft(im,real); 

% Compute vector version of curvelets coeffcients
coef = [];

for s=1:length(C)
  for w=1:length(C{s})
    A = C{s}{w};
    coef= [coef; A(:)];
  end
end
