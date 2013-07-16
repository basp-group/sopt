function [new_pdf, alpha] = sopt_mltb_modifypdf(pdf, nb_meas)
% sopt_mltb_modifypdf - Modify PDF of sampling profile
% 
% Checks PDF of the sampling profile and normalizes it. It is used
% in sopt_mltb_vdsmask in the generation of variable density sampling
% profiles.
%
% Inputs:
%
%   - pdf: Sampling profile function.
%
%   - nb_meas: Number of measurements.
%
% Outputs:
%
%   - new_pdf: New sampling profile function.
%
%   - alpha: DC level for the required number of samples.

if sum(pdf(:)<0)>0
    error('PDF contains negative values');
end
pdf = pdf/max(pdf(:));

% Find alpha
alpha_min = -1; alpha_max = 1;
while 1
    alpha = (alpha_min + alpha_max)/2;
    new_pdf = pdf + alpha;
    new_pdf(new_pdf>1) = 1; new_pdf(new_pdf<0) = 0;
    M = round(sum(new_pdf(:)));
    if M > nb_meas
        alpha_max = alpha;
    elseif M < nb_meas
        alpha_min = alpha;
    else
        break;
    end
end

