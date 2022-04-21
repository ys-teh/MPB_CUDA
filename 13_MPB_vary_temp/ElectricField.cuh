#ifndef _ELECTRICFIELD_CUH_
#define _ELECTRICFIELD_CUH_

#include <math.h>
#include <stdio.h>
#include <cufft.h>

#include "GlobalConstants.hpp"

//Construct rho tensor.
__global__ 
void compute_rho(   float *dev_possible_states, 
			 		int *dev_current_states,
					float *dev_rho);

//Compute electric field in Fourier space.
__global__ 
void compute_E_field_in_Fourier_space(  float *dev_A11_tensor,
										float *dev_A22_tensor,
										float *dev_A12_tensor,
										cufftComplex *dev_fft_rho,
										cufftComplex *dev_fft_E);

//Compute E field from surface contribution only
__global__
void compute_E_field_surf_contribution(float *dev_rho, 
					float *dev_rho_partial_sum1,
					float *dev_rho_partial_sum2);

// Finish computing surface term
__global__
void finish_compute_E_field_surf_contribution(float *dev_rho_partial_sum1,
                                              float *dev_rho_partial_sum2,
                                              float *dev_E_surf);

//Construct the E field.
__global__ 
void compute_E_field(   float *dev_E_w_self, 
				 		float *dev_E,
				 		float *dev_rho,
						float *dev_E_surf);

//Compute dipole energy.
__global__ 
void compute_dipole_energy( float *dev_rho,
							float *dev_E,
							float *dev_partial_energy);

#endif
