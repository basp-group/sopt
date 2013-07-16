function mask = sopt_mltb_genmask(pdf, seed)
% sopt_mltb_genmask - Generate mask
%
% Generate a binary mask with variable density sampling. It is used
% in sopt_mltb_vdsmask in the generation of variable density sampling
% profiles.
%
% Inputs:
%
%   - pdf: Sampling profile function.
%
%   - seed: Seed for the random number generator. Optional.
%
% Outputs:
%
%   - mask: Binary mask.

if nargin==2
    rand('seed', seed);
end

mask = rand(size(pdf))<pdf;

