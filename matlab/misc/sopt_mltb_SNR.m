function snr = sopt_mltb_SNR(map_init, map_recon)
% SNR - Compute the SNR between two images
% 
% C omputes the SNR between the maps map_init and map_recon.  The SNR is 
% computed by
%    10 * log10( var(MAP_INIT) / var(MAP_INIT-MAP_NOISY) )
%  where var stands for the matlab built-in function that computes the
%  variance.
%
% Inputs:
%
%   - map_init: Initial image.
%
%   - map_recon: Reconstructed image.
%
% Outputs:
%
%   - snr: SNR.

noise = map_init(:)-map_recon(:);
var_init = var(map_init(:));
var_den = var(noise(:));
snr = 10 * log10(var_init/var_den);

end
