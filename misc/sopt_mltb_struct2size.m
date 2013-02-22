function Mod = sopt_mltb_struct2size(C)
%sopt_mltb_struct2size - Compute curvelet data structure
%
% Compute data structure to store the sizes of the curvelet transform 
% generated from the CurveLab toolbox.
%
% Inputs:
%
%   - C: Curvelet coefficients arranged in the original format.
%
% Outputs:
%
%   - Mod: Data structure with the appropiate sizes.

 Mod=cell(size(C));
 for s=1:length(C)
   Mod{s} = cell(size(C{s}));
   for w=1:length(C{s})
     [m,n]=size(C{s}{w});
     Mod{s}{w}=[m,n];
   end
 end
