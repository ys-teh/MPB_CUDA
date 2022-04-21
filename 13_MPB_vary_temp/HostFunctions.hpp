#ifndef _HOSTFUNCTIONS_CPP_
#define _HOSTFUNCTIONS_CPP_

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h> 

#include <cstring>
#include <sstream>
#include <fstream>
#include <iostream>

#include "GlobalConstants.hpp"

// Create tensor of composition (0=rhombohedral, 1=tetragonal)
void create_composition( int percentage_T_sites, 
						 int* composition);

//Print composition to file
void save_composition(  int* composition,
						int percentage_T_sites,
						int test_number);

// Create possible states 
// First 8 states are rhombohedral, last 6 states are tetragonal.
void create_possible_states( float* possible_states);

//Initial random guess of dipole states.
void guess_dipole_states(int* dipole_states);

//Print dipole states to file.
void save_dipole_states(int* dipole_states,
						float beta,
						int percentage_T_sites,
						int test_number);

//Load the file data into tensors.
void loadFiles( std::string Path, 
				float* A_tensor, 
				int i, 
				int j);

//Load the file data into tensors.
void loadFiles( std::string Path, 
				float* A11_tensor, 
				float* A22_tensor, 
				float* A12_tensor);

// Load A tensors
void load_A_tensors(float* A11_tensor, 
				float* A22_tensor, 
				float* A12_tensor);

// Compute percentage of tetragonal dipole states
float compute_percentage_T_dipoles(int* dipole_states);

#endif
