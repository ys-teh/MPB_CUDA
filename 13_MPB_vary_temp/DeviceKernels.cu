#include "DeviceKernels.cuh"

//Initialize curand
__global__ 
void random_init (unsigned int seed, curandState_t* states){
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init ( seed, tid, 0, &states[tid] );
}

//Generate random number between 0 and N-1
__global__ 
void random_casting(curandState_t* states, unsigned int* numbers){
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	numbers[tid] = curand_uniform(&states[tid]) * N;
	if (numbers[tid] == N)
		numbers[tid] = N-1;
}

//Generate random number between 0 and 1
__global__ 
void random_uniform(curandState_t* states, float* numbers){
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	numbers[tid] = curand_uniform(&states[tid]);
}

//Find neighbors
__global__ 
void find_neighbors(unsigned int* neighbors1,
					unsigned int* neighbors2,
					unsigned int* neighbors3,
					unsigned int* neighbors4){

	//Global index for each Thread.
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < N){
		int x = tid/L;
		int y = tid - x*L;

		// Neighbor 1
		neighbors1[tid] = ((x+1)%L)*L + y;
		// Neighbor 2
		neighbors2[tid] = ((x-1+L)%L)*L + y;
		// Neighbor 3
		neighbors3[tid] = x*L + (y+1)%L;
		// Neighbor 4
		neighbors4[tid] = x*L + (y-1+L)%L;

		//Increase the threads.
		tid += blockDim.x * gridDim.x;
	}

}

//__device__ float dev_beta;

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
					float* beta) {

  __shared__ float W_total[8];
  __shared__ float cdf[8];
  __shared__ int check_state[8];
  unsigned int tid = threadIdx.x;
  unsigned int site = random_sites[blockIdx.x];
  int dp;
  float W_local, W_exchange, W_dipole, temp;

  // Local energy
  if (composition[site]==1 && tid<4 || composition[site]==0 && tid>=4){
    W_local = H_CONST;
  }
  else {
    W_local = 0.0;
  }

  // Exchange energy
  dp = dipole_states[neighbors1[site]];
  W_exchange = possible_states[dp*2]*possible_states[tid*2] + 
               possible_states[dp*2+1]*possible_states[tid*2+1];
  dp = dipole_states[neighbors2[site]];
  W_exchange += possible_states[dp*2]*possible_states[tid*2] + 
                possible_states[dp*2+1]*possible_states[tid*2+1];
  dp = dipole_states[neighbors3[site]];
  W_exchange += possible_states[dp*2]*possible_states[tid*2] + 
                possible_states[dp*2+1]*possible_states[tid*2+1];
  dp = dipole_states[neighbors4[site]];
  W_exchange += possible_states[dp*2]*possible_states[tid*2] + 
                possible_states[dp*2+1]*possible_states[tid*2+1];

  W_exchange *= -ALPHA_CONST;

  // Dipole energy
  W_dipole = -E[site]*possible_states[tid*2] 
             -E[site+N]*possible_states[tid*2+1];
  W_dipole *= D_CONST;

  // Total energy associated with the selected lattice site
  W_total[tid] = W_local + W_exchange + W_dipole;
  __syncthreads();

  // Use the smallest energy value as reference value
  temp = W_total[0];
  for (int i = 1; i < 8; i++){
    if (temp > W_total[i])
      temp = W_total[i];
  }
  
  // Energy difference
  cdf[tid] = exp(-(beta[0])*(W_total[tid]-temp));
  __syncthreads();

  if (tid == 0){
    for (unsigned int i = 1; i < 8; i++){
      cdf[i] += cdf[i-1];
    }  
  }
  __syncthreads();
  cdf[tid] /= cdf[7];
  if (cdf[tid] < random_values[blockIdx.x])
    check_state[tid] = 1;
  else
    check_state[tid] = 0;
  __syncthreads();

  // Update state
  if (tid == 0){
    for (unsigned int i = 1; i < 8; i++){
      check_state[0] += check_state[i];
    }
    dipole_states[site] = check_state[0];

  }

}

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
						   float* dev_partial_energy){

	//Load data into shared memory.
	__shared__ float energy[block];

	//Global index for each Thread:
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int energyIndex = threadIdx.x;

	float temp = 0.0;
	while (tid < N){
		// Define energy types, nearest neighbors
		float W_local, W_exchange, W_dipole;
		unsigned int nn;

		// Local energy
		if (composition[tid] == 1 && dipole_states[tid] < 4 || 
			composition[tid] == 0 && dipole_states[tid] >= 4){
			W_local = H_CONST;
		}
		else {
		W_local = 0.0;
		}

		// Exchange energy
		nn = neighbors1[tid];
		W_exchange = dev_rho[nn] * dev_rho[tid] + 
						dev_rho[nn+N] * dev_rho[tid+N];
		nn = neighbors2[tid];
		W_exchange += dev_rho[nn] * dev_rho[tid] + 
						dev_rho[nn+N] * dev_rho[tid+N];
		nn = neighbors3[tid];
		W_exchange += dev_rho[nn] * dev_rho[tid] + 
						dev_rho[nn+N] * dev_rho[tid+N];
		nn = neighbors4[tid];
		W_exchange += dev_rho[nn] * dev_rho[tid] + 
						dev_rho[nn+N] * dev_rho[tid+N];

		W_exchange *= -0.5*ALPHA_CONST;

		// Dipole energy
		W_dipole =  - 0.5 * dev_rho[tid] * dev_E[tid]
					- 0.5 * dev_rho[tid+N] * dev_E[tid+N];
		W_dipole *= D_CONST;
		
		// Total energy associated with each lattice site
		temp += W_local + W_exchange + W_dipole;
        tid += blockDim.x * gridDim.x;

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
