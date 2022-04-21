#include "ElectricField.cuh"

//Construct rho tensor.
__global__ 
void compute_rho(float *dev_possible_states, 
				 int *dev_current_states,
				 float *dev_rho){

	//Global index for each Thread.
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//#pragma unroll 3
	while (tid < N){
    	int current_state = dev_current_states[tid];
		dev_rho[tid] = dev_possible_states[current_state*2];
		dev_rho[tid+N] = dev_possible_states[current_state*2 + 1];

		//Increase the threads.
		tid += blockDim.x * gridDim.x;
	}
}

//Compute electric field in Fourier space
__global__ 
void compute_E_field_in_Fourier_space(  float *dev_A11_tensor,
										float *dev_A22_tensor,
										float *dev_A12_tensor,
										cufftComplex *dev_fft_rho,
										cufftComplex *dev_fft_E){

	//Global index for each Thread.
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    //#pragma unroll 3
	while (tid < N_half){
		int tid1 = tid + N_half;
		dev_fft_E[tid].x = dev_A11_tensor[tid] * dev_fft_rho[tid].x + 
						   dev_A12_tensor[tid] * dev_fft_rho[tid1].x;

		dev_fft_E[tid].y = dev_A11_tensor[tid] * dev_fft_rho[tid].y + 
						   dev_A12_tensor[tid] * dev_fft_rho[tid1].y;

		dev_fft_E[tid1].x = dev_A12_tensor[tid] * dev_fft_rho[tid].x + 
							dev_A22_tensor[tid] * dev_fft_rho[tid1].x;

		dev_fft_E[tid1].y = dev_A12_tensor[tid] * dev_fft_rho[tid].y + 
							dev_A22_tensor[tid] * dev_fft_rho[tid1].y;

		//Increase the threads.
		tid += blockDim.x * gridDim.x;
	}
}

//Compute E field from surface contribution only
__global__
void compute_E_field_surf_contribution(float *dev_rho, float *dev_rho_partial_sum1, float *dev_rho_partial_sum2){

	//Shared memory
	__shared__ float rho_tmp_sum1[block];
	__shared__ float rho_tmp_sum2[block];

	//Indices
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int id = threadIdx.x;

	//Initialize
	rho_tmp_sum1[id] = 0.0;
	rho_tmp_sum2[id] = 0.0;
	if(tid<N){
		rho_tmp_sum1[id] = dev_rho[tid];
		rho_tmp_sum2[id] = dev_rho[tid+N];
	}
	__syncthreads();

	//Parallel block reduction
	int i = blockDim.x/2;
	while(i!=0){
		if(id<i){
			rho_tmp_sum1[id] += rho_tmp_sum1[id+i];
			rho_tmp_sum2[id] += rho_tmp_sum2[id+i];
		}
		__syncthreads();
		i /= 2;
	}

	//Thread 0 records the result
	if(id==0){
		dev_rho_partial_sum1[blockIdx.x] = rho_tmp_sum1[0];
		dev_rho_partial_sum2[blockIdx.x] = rho_tmp_sum2[0];
	}
}

// Finish computing surface term
__global__
void finish_compute_E_field_surf_contribution(float *dev_rho_partial_sum1,
                                              float *dev_rho_partial_sum2,
                                              float *dev_E_surf){

        //Shared memory
        __shared__ float tmp1[block];
        __shared__ float tmp2[block];

        //Indices
        int id = threadIdx.x;

        //Initialize
        tmp1[id] = 0.0;
        tmp2[id] = 0.0;
        if(id<grid){
                tmp1[id] = dev_rho_partial_sum1[id];
                tmp2[id] = dev_rho_partial_sum2[id];
        }
        __syncthreads();

        //Parallel block reduction
        int i = blockDim.x/2;
        while(i!=0){
                if(id<i){
                        tmp1[id] += tmp1[id+i];
                        tmp2[id] += tmp2[id+i];
                }
                __syncthreads();
                i /= 2;
        }

        //Thread 0 records the result
        if(id==0){
                float coeff = -2.0*PI/(float)N;
                dev_E_surf[0] = coeff*tmp1[0];
                dev_E_surf[1] = coeff*tmp2[0];
        }
}


//Construct the E field.
__global__ 
void compute_E_field(float *dev_E_w_self, 
				 	float *dev_E,
				 	float *dev_rho,
					float *dev_E_surf){

	//Global index for each Thread.
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//Auxiliar variables.
//	float sigma = 1.0/sqrt(2.0)/4.5;
//	float coeff = 1.0/(sigma*sigma);
    float coeff = 40.5;

    //#pragma unroll 3
	while (tid < N){
		dev_E[tid] = dev_E_w_self[tid] + coeff*dev_rho[tid] + dev_E_surf[0];
		dev_E[tid+N] = dev_E_w_self[tid+N] + coeff*dev_rho[tid+N] + dev_E_surf[1];

		//Increase the threads.
		tid += blockDim.x * gridDim.x;
	}
}

//Construct the dipole energy.
__global__ 
void compute_dipole_energy( float *dev_rho,
							float *dev_E,
							float *dev_partial_energy){

	//Load data into shared memory.
	__shared__ float energy[block];

	//Global index for each Thread:
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int energyIndex = threadIdx.x;

	float temp = 0.0;
	if (tid < N){
		temp =  - 0.5 * dev_rho[tid] * dev_E[tid]
				- 0.5 * dev_rho[tid+N] * dev_E[tid+N];
	}
	energy[energyIndex] = temp;
	__syncthreads();

	//Parallel Block Reduction.
	int i = blockDim.x/2;
	while (i != 0){
		if (energyIndex < i)
			energy[energyIndex] += energy[energyIndex + i];
		__syncthreads();
		i /= 2;
	}

	//The thread 0 writes the final result.
	if (energyIndex == 0)
		dev_partial_energy[blockIdx.x] = energy[0];
}

