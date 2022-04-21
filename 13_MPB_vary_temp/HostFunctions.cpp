#include "HostFunctions.hpp"

// Create tensor of composition (0=rhombohedral, 1=tetragonal)
// Input: fraction of T sites
void create_composition( int percentage_T_sites, int* composition){

	int temp;
	unsigned int r;
	int number_T_sites = (percentage_T_sites*N)/100;
	for (unsigned int i = 0; i < number_T_sites; i++){
		composition[i] = 1;
	}
	for (unsigned int i = number_T_sites; i < N; i++){
		composition[i] = 0;
	}

	//Perform random permutation using Fisher-Yates shuffle algorithm
	srand(time(0));
	for (unsigned int i = N-1; i > 0; i--){
		r = rand()%(i+1); // random number s.t. 0 <= j <= i
		temp = composition[i];
		composition[i] = composition[r];
		composition[r] = temp;
	}

/*	//Checker board
        unsigned int j;
        for (unsigned int i = 0; i < N; i++){
                j = i/L;
                if ((j%2==0 && i%2==0) || (j%2==1 && i%2==1))
                        composition[i]=1;
                else
                        composition[i]=0;
        }
*/

/*	// Half 1 half 0
        for (unsigned int i = 0; i < N; i++){
                if (i/(N/2)==0)
                        composition[i]=1;
                else
                        composition[i]=0;
        }
*/
}

// Create possible states 
// First 8 states are rhombohedral, last 6 states are tetragonal.
void create_possible_states( float* possible_states){

	float a = 0.7071; //TO CHANGE

	possible_states[0] = a;               possible_states[1] = a;  
	possible_states[2] = -a;              possible_states[3] = a;  
	possible_states[4] = a;               possible_states[5] = -a;  
	possible_states[6] = -a;              possible_states[7] = -a;  
	possible_states[8] = 1.0;             possible_states[9] = 0.0;  
	possible_states[10] = -1.0;           possible_states[11] = 0.0;  
	possible_states[12] = 0.0;            possible_states[13] = 1.0;  
	possible_states[14] = 0.0;            possible_states[15] = -1.0;  
}

//Initial random guess of dipole states.
void guess_dipole_states(int* dipole_states){
	srand((time(0)-98765)*2+5952);
	for (unsigned int i = 0; i < N; i++){
		dipole_states[i] = rand()%8;
	}
}

//Print composition to file
void save_composition(  int* composition,
						int percentage_T_sites,
						int test_number){

	char FileName[100];
	sprintf(FileName, "results/composition_percent%i_test%i.txt",percentage_T_sites,test_number);
	FILE* ptr = fopen(FileName, "w");
	if (ptr == NULL){
	  printf("Error opening file!\n");
	  exit(1);
	}

	for (unsigned int i = 0; i < N; i++){
	  if ((i+1)%L == 0)
		fprintf(ptr, "%i\n", composition[i]);
	  else
		fprintf(ptr, "%i ", composition[i]);
	}
	fclose(ptr);
}

//Print dipole states to file.
void save_dipole_states(int* dipole_states,
						float beta,
						int percentage_T_sites,
						int test_number){

	char FileName[100];
	sprintf(FileName, "results/dipole_states_percent%i_test%i.txt",percentage_T_sites,test_number);
	FILE* ptr = fopen(FileName, "w");
	if (ptr == NULL){
	  printf("Error opening file!\n");
	  exit(1);
	}

	for (unsigned int i = 0; i < N; i++){
	  if ((i+1)%L == 0)
		fprintf(ptr, "%i\n", dipole_states[i]);
	  else
		fprintf(ptr, "%i ", dipole_states[i]);
	}
	fclose(ptr);
}

//Load the file data into tensors.
void loadFiles( std::string Path, 
				float* A_tensor, 
				int i, 
				int j){

	//Convert process number into string.
	std::stringstream nP;
	std::stringstream ni;
	std::stringstream nj;

	nP << L;
	ni << i;
	nj << j;

	//Generates the file name.
	std::string File;
	File = Path + "A" + ni.str() + nj.str() + "_tensor_" + nP.str() + "_" + nP.str() + "_half.dat";

	//Open the file and writes data.
	std::ifstream Input(File.c_str());
	for (int i=0; i<N; i++)
		Input >> A_tensor[i];
}

//Load the file data into tensors.
void loadFiles( std::string Path, 
				float* A11_tensor, 
				float* A22_tensor, 
				float* A12_tensor){

	//Load the Tensors.
	loadFiles(Path, A11_tensor, 1, 1);
	loadFiles(Path, A22_tensor, 2, 2);
	loadFiles(Path, A12_tensor, 1, 2);
}

// Compute percentage of tetragonal dipole states
float compute_percentage_T_dipoles(int* dipole_states){
	float percent = 0.0;
	for (unsigned int i = 0; i < N; i++){
		if (dipole_states[i] >= 4)
			percent += 1.0;
	}
	percent = percent * 100.0 / (float)N;
	return percent;
}

