//==============================================================================
// RANDOM FIELD ISING MODEL SOLVED USING MONTE CARLO (2D)
// Last updated on Sept 19, 2019
// The following functions are checked: compute_rho, 
//                                      compute_E_field_in_Fourier_space,
//                                      compute_E_field,
//                                      create_composition,
//                                      
//==============================================================================

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cufft.h>
#include <time.h>

#include <string> 
#include <curand_kernel.h>

#include "HostFunctions.hpp"
#include "DeviceKernels.cuh"
#include "ElectricField.cuh"

//Header and Macro to handle CUDA errors.
static void HandleError( cudaError_t err,
						 const char *file,
						 int line ) {
	//Creates the error messege.
	if (err != cudaSuccess) {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//============================================================================
// MAIN
//============================================================================

int main (int argc, char* argv[]){

	// Parse parameters
	int percentage_T_sites = atoi(argv[1]);
	int test_number = atoi(argv[2]);

	//CUDA Time 
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	float elapsedTime;

	// Create composition 
	int* composition = (int*)malloc( sizeof(int) * N ); 
	create_composition( percentage_T_sites, composition );

	int* dev_composition;
	HANDLE_ERROR( cudaMalloc((void**)&dev_composition, sizeof(int) * N) );
	HANDLE_ERROR( cudaMemcpy(dev_composition, composition, sizeof(int) * N, cudaMemcpyHostToDevice) );

	save_composition(composition,percentage_T_sites,test_number);

	// Create initial guess of dipole states
	int* dipole_states = (int*)malloc( sizeof(int) * N );
	guess_dipole_states(dipole_states);

	int* dev_dipole_states;
	HANDLE_ERROR( cudaMalloc((void**)&dev_dipole_states, sizeof(int) * N) );
	HANDLE_ERROR( cudaMemcpy(dev_dipole_states, dipole_states, sizeof(int) * N, cudaMemcpyHostToDevice) );

	//Initialize List of possible states on CPU.
	float* possible_states = (float*)malloc( sizeof(float) * 16 );
	create_possible_states( possible_states );

	//Initialize and copy possible_states to GPU.
	float* dev_possible_states; 
	HANDLE_ERROR( cudaMalloc((void**)&dev_possible_states, sizeof(float) * 16) );
	HANDLE_ERROR( cudaMemcpy(dev_possible_states, possible_states, sizeof(float) * 16, cudaMemcpyHostToDevice) );

	// Load A tensors (needed for the computation of E field)
	float* A11_tensor = (float*)malloc( sizeof(float) * N_half );
	float* A22_tensor = (float*)malloc( sizeof(float) * N_half );
	float* A12_tensor = (float*)malloc( sizeof(float) * N_half );

	loadFiles("files_for_gpu/", A11_tensor, A22_tensor, A12_tensor);

	//Define, allocate, and copy A tensors to GPU.
	float* dev_A11_tensor;
	float* dev_A22_tensor;
	float* dev_A12_tensor;

	HANDLE_ERROR( cudaMalloc((void**)&dev_A11_tensor, sizeof(float) * N_half) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_A22_tensor, sizeof(float) * N_half) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_A12_tensor, sizeof(float) * N_half) );

	HANDLE_ERROR( cudaMemcpy(dev_A11_tensor, A11_tensor, N_half * sizeof(float), cudaMemcpyHostToDevice) );	
	HANDLE_ERROR( cudaMemcpy(dev_A22_tensor, A22_tensor, N_half * sizeof(float), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(dev_A12_tensor, A12_tensor, N_half * sizeof(float), cudaMemcpyHostToDevice) );

	//Free unused CPU pointers.
	free(composition);
	free(possible_states);
	free(A11_tensor);
	free(A22_tensor);
	free(A12_tensor);

	//Definition and allocation of rho.
	float* dev_rho;
	HANDLE_ERROR( cudaMalloc((void**)&dev_rho, sizeof(float) * (2*N)) );
	
	//Definition and allocation of weighted Energy states.
	cufftComplex* dev_fft_rho;
	HANDLE_ERROR( cudaMalloc((void**)&dev_fft_rho, sizeof(cufftComplex) * (2*N_half)) );

	//Definition and allocation of FFT Energy states.
	cufftComplex* dev_fft_E;
	HANDLE_ERROR( cudaMalloc((void**)&dev_fft_E, sizeof(cufftComplex) * (2*N_half)) );

	//Definition and allocation of Energy states weights.
	float* dev_E_w_self;
	HANDLE_ERROR( cudaMalloc((void**)&dev_E_w_self, sizeof(float) * (2*N)) );

	//Definition of E field associated with surface term
	float* dev_rho_partial_sum1;
	HANDLE_ERROR( cudaMalloc((void**)&dev_rho_partial_sum1, sizeof(float) * grid) );
	float* rho_partial_sum1 = (float*)malloc( sizeof(float) * grid );
        float* dev_rho_partial_sum2;
        HANDLE_ERROR( cudaMalloc((void**)&dev_rho_partial_sum2, sizeof(float) * grid) );
	float* rho_partial_sum2 = (float*)malloc( sizeof(float) * grid );
	float* dev_E_surf;
	HANDLE_ERROR( cudaMalloc((void**)&dev_E_surf, sizeof(float) * (2)) );

	//Definition and allocation of Energy states.
	float* dev_E;
	HANDLE_ERROR( cudaMalloc((void**)&dev_E, sizeof(float) * (2*N)) );

	// Declaration of fft plan
	cufftResult result;
	int R[2] = {L,L};

	cufftHandle plan_forward;
	result = cufftPlanMany(&plan_forward, 2, R, NULL, 1, N, NULL, 1, N, CUFFT_R2C, 2);
	if ( result != CUFFT_SUCCESS)
		printf("CUFFT error: Plan creation failed");

	cufftHandle plan_inverse;
	result = cufftPlanMany(&plan_inverse, 2, R, NULL, 1, N, NULL, 1, N, CUFFT_C2R, 2);
	if ( result != CUFFT_SUCCESS)
		printf("CUFFT error: Plan creation failed");

	// Initialize random states of curand (random number generator in cuda)
	curandState_t* dev_random_states;
	HANDLE_ERROR( cudaMalloc((void**)&dev_random_states, M * sizeof(curandState_t)) );
	random_init<<<M, 1>>>(time(0), dev_random_states);
	unsigned int* dev_random_sites;
	HANDLE_ERROR( cudaMalloc((void**)&dev_random_sites, M * sizeof(unsigned int)) );

	curandState_t* dev_random_states2;
	HANDLE_ERROR( cudaMalloc((void**)&dev_random_states2, M * sizeof(curandState_t)) );
	random_init<<<M, 1>>>(time(0)*5-1234567, dev_random_states2);
	float* dev_random_values;
	HANDLE_ERROR( cudaMalloc((void**)&dev_random_values, M * sizeof(float)) );

	// Define neighbor list on GPU.
	unsigned int* dev_neighbors1;
	unsigned int* dev_neighbors2;
	unsigned int* dev_neighbors3;
	unsigned int* dev_neighbors4;

	//Allocate memory for neighbor indeces.
	HANDLE_ERROR( cudaMalloc((void**)&dev_neighbors1, N * sizeof(unsigned int)) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_neighbors2, N * sizeof(unsigned int)) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_neighbors3, N * sizeof(unsigned int)) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_neighbors4, N * sizeof(unsigned int)) );

	//Find cell neighbors.
	find_neighbors<<<grid, block>>>(dev_neighbors1,
									dev_neighbors2,
									dev_neighbors3,
									dev_neighbors4);

/*	// Define total energy
	float* dev_partial_energy;
	HANDLE_ERROR( cudaMalloc((void**)&dev_partial_energy, sizeof(float) * grid) );
	float* partial_energy = (float*)malloc( sizeof(float) * grid );
	float W_total;
*/
	// Define and initialize inverse temperature
	float beta = 0.0;
	beta += beta_initial;
	float* dev_beta;
	HANDLE_ERROR( cudaMalloc((void**)&dev_beta, sizeof(float) ) );
	HANDLE_ERROR( cudaMemcpy(dev_beta, &beta, sizeof(float),cudaMemcpyHostToDevice) );
    
/*	// Save files
	char FileName[100];
	sprintf(FileName, "results/total_energy_percent%i_test%i.txt",percentage_T_sites,test_number);
	FILE* ptr = fopen(FileName, "w");
	if (ptr == NULL){
	  printf("Error opening file!\n");
	  exit(1);
	}

	char FileName2[100];
	sprintf(FileName2, "results/tetragonal_dipoles_percent%i_test%i.txt",percentage_T_sites,test_number);
	FILE* ptr2 = fopen(FileName2, "w");
	if (ptr2 == NULL){
	  printf("Error opening file!\n");
	  exit(1);
	}
*/

	//==========================================================================
	// MONTE CARLO ITERATIONS
	//==========================================================================
	cudaEventRecord( start, 0 );

	for (unsigned int iter = 1; iter <= max_iter; iter++){
		// Construct rho
		compute_rho<<<grid,block>>>(dev_possible_states,
									dev_dipole_states,
									dev_rho);

		// Perform FFT of rho
		cufftExecR2C(plan_forward, dev_rho, dev_fft_rho);

		// Compute electric field in frequency space
		compute_E_field_in_Fourier_space<<<grid_half,block>>>(dev_A11_tensor,
															  dev_A22_tensor,
															  dev_A12_tensor,
															  dev_fft_rho,
															  dev_fft_E);

		// Compute electric field in real space
		cufftExecC2R(plan_inverse, dev_fft_E, dev_E_w_self);

		// Subtract away electric field due to self-interaction and add surface term
		compute_E_field_surf_contribution<<<grid,block>>>(dev_rho, 
                                        			dev_rho_partial_sum1,
								dev_rho_partial_sum2);

		// Finish computing the rest of surface term
		finish_compute_E_field_surf_contribution<<<1,grid>>>(dev_rho_partial_sum1,
                                              dev_rho_partial_sum2,
                                              dev_E_surf);

		compute_E_field<<<grid,block>>>(  dev_E_w_self,
											dev_E,
											dev_rho,
											dev_E_surf);

/*		// Compute total energy
		if (iter%(iter_per_sweep*100) == 0){
			compute_total_energy<<<grid,block>>>( dev_composition,
												dev_dipole_states,
												dev_neighbors1, 
												dev_neighbors2,
												dev_neighbors3, 
												dev_neighbors4,
												dev_rho,
												dev_E,
												dev_partial_energy);

			cudaMemcpy(partial_energy, dev_partial_energy, grid * sizeof(float), cudaMemcpyDeviceToHost);
			W_total = 0.0;
			for (int i=0; i<grid; i++)
				W_total += partial_energy[i];
			fprintf(ptr, "%i, %f\n", iter/iter_per_sweep, W_total);
			printf("Sweep %i completed.\n", iter/iter_per_sweep);
		}
*/

		// Randomly select M number of lattice sites and probability values
		random_casting<<<M,1>>>(dev_random_states, dev_random_sites);
		random_uniform<<<M,1>>>(dev_random_states2, dev_random_values);

		// Update state
		update_states<<<M,8>>>(dev_random_sites, dev_random_values,
								dev_composition, dev_dipole_states, dev_possible_states, 
								dev_neighbors1, dev_neighbors2,
								dev_neighbors3, dev_neighbors4, 
								dev_E, dev_beta);

		if (iter%(iter_per_beta_step) == 0){

//			cudaMemcpy(dipole_states, dev_dipole_states, N * sizeof(int), cudaMemcpyDeviceToHost);
//			float percent = compute_percentage_T_dipoles(dipole_states);
//			fprintf(ptr2, "%i, %f\n", iter/iter_per_sweep, percent);
//			save_dipole_states(dipole_states,beta,percentage_T_sites,test_number);
			beta += beta_step;
			cudaMemcpy(dev_beta, &beta, sizeof(float),cudaMemcpyHostToDevice);
		}

	} //End of for-loop

	//Compute time elapsed
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsedTime, start, stop );
	printf("Time taken for %i iterations: %f s\n", max_iter, elapsedTime/(float)1000.0);

	// Copy dipole_states to CPU
	HANDLE_ERROR( cudaMemcpy(dipole_states, dev_dipole_states, N * sizeof(int), cudaMemcpyDeviceToHost) );

	//Save the dipoles states.
	save_dipole_states(dipole_states,
				beta,
				percentage_T_sites,
				test_number);

	// Compute percentage of T dipole states
/*	float percent = compute_percentage_T_dipoles(dipole_states);
	fprintf(ptr2, "%i, %f\n", max_iter/iter_per_sweep, percent);
*/
	// Close file
//	fclose(ptr);
//	fclose(ptr2);

	//Free CPU/GPU memory.
	free(dipole_states);
//	free(partial_energy);
	free(rho_partial_sum1);
	free(rho_partial_sum2);
	cufftDestroy(plan_forward);
	cufftDestroy(plan_inverse);
	cudaFree(dev_A11_tensor);
	cudaFree(dev_A22_tensor);
	cudaFree(dev_A12_tensor);
	cudaFree(dev_rho);
	cudaFree(dev_fft_rho);
	cudaFree(dev_fft_E);
	cudaFree(dev_E_w_self);
	cudaFree(dev_rho_partial_sum1);
	cudaFree(dev_rho_partial_sum2);
	cudaFree(dev_E_surf);
	cudaFree(dev_E);
	cudaFree(dev_dipole_states);
	cudaFree(dev_possible_states);
	cudaFree(dev_composition);
	cudaFree(dev_neighbors1);
	cudaFree(dev_neighbors2);
	cudaFree(dev_neighbors3);
	cudaFree(dev_neighbors4);
	cudaFree(dev_random_sites);
	cudaFree(dev_random_values);
	cudaFree(dev_random_states);
	cudaFree(dev_random_states2);
	cudaFree(dev_beta);
//	cudaFree(dev_partial_energy);	
}
