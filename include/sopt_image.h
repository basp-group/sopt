
#ifndef SOPT_IMAGE
#define SOPT_IMAGE
#include "sopt_config.h"


int spot_image_tiff_read(double **img, int *nx, int *ny, char *filename, 
			 double scale, double offset);
int spot_image_tiff_write(double *img, int nx, int ny, char *filename, 
			  double scale, double offset);

#endif
