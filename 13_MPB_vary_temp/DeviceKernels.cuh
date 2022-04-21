#ifndef _DEVICEKERNELS_CUH_
#define _DEVICEKERNELS_CUH_

#include <math.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#include "GlobalConstants.hpp"

__global__ 
void random_init (unsigned int seed, curandState_t* states);

__global__ 
void random_casting(curandState_t* states, unsigned int* numbers);

__global__ 
void random_uniform(curandState_t* states, float* numbers);

__global__ 
void find_neighbors(unsigned int* neighbors1,
					unsigned int* neighbors2,
					unsigned int* neighbors3,
					unsigned int* neighbors4);

// Compute energy to update state
__global__ 
void update_states( unsigned int* random_sites, 
					float* random_values,
					int* composition, 
					int* dipole_states, 
					float* possible_states, 
					unsigned int* neighbors1, 
					unsigned int* neighbors2,
					unsigned int* neighbors3, 
					unsigned int* neighbors4,
					float* E,
					float* beta);

//Compute total energy.
__global__ 
void compute_total_energy( int* composition,
						   int* dipole_states,
						   unsigned int* neighbors1, 
						   unsigned int* neighbors2,
						   unsigned int* neighbors3, 
						   unsigned int* neighbors4,
						   float* dev_rho,
						   float* dev_E,
						   float* dev_partial_energy);


#endif
