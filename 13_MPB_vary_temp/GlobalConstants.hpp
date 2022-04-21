#ifndef _GLOBALCONSTANT_HPP_
#define _GLOBALCONSTANT_HPP_

#define PI 3.14159265358979323846
#define H_CONST 1.0 // normalized by a factor of 1/(2*4*pi*epsilon0)
#define ALPHA_CONST 1.0 // normalized by a factor of 1/(2*4*pi*epsilon0)
#define D_CONST 1.0
#define SIGMA 0.157134840 // SIGMA = 1/sqrt(2)/4.5

const int L = 256;
const int N = L * L;
const int N_half = L * (L/2 + 1);
const unsigned int M = pow(2,floor(log10(N)/log10(2)/2));	// Number of sites changed per iteration
const unsigned int max_iter = 101*N/32*(N/M); //30000*(N/M); ////N*(N/M);
const unsigned int iter_per_sweep = N/M;
const unsigned int iter_per_beta_step = N/32*iter_per_sweep; //N*iter_per_sweep;
const float beta_initial = 0.0;
const float beta_step = 0.05;

//Definition of grid and blocks.
const int block = 1024; // Number of threads in a block
const int grid  = (N + block - 1) / block;
const int grid_2  = (2*N + block - 1) / block;
const int grid_half = (N_half + block - 1) / block;


#endif
