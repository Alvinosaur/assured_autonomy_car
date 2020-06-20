#include "mex.h"


/* Input Arguments */
#define	MATRIX_IN      prhs[0]
#define	VEC_IN         prhs[1]
#define IS_TRANSPOSED  prhs[2]

/* Output Arguments */
#define	VEC_OUT        plhs[0]

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
{ 
    
    /* Check for proper number of arguments */    
    if (nrhs != 3) { 
	    mexErrMsgIdAndTxt( "MATLAB:planner:invalidNumInputs",
                "Three input arguments required."); 
        return;
    } else if (nlhs != 1) {
	    mexErrMsgIdAndTxt( "MATLAB:planner:maxlhs",
                "One output argument required."); 
        return;
    } 
        
    /* get the dimensions and values of M x N matrix */
    /* Matlab uses column-major order, but row-major if input is transposed */
    int M = mxGetM(MATRIX_IN);
    int N = mxGetN(MATRIX_IN);
    double* some_matrix = mxGetPr(MATRIX_IN);
    double* arr = malloc(sizeof(double)*M*N);
    printf("Read in matrix of %d x %d:\n", M, N);
    int is_transposed = *(int*)mxGetPr(IS_TRANSPOSED);
    printf("Is Transposed: %d\n", is_transposed);
    int idx = 0;
    for (int c = 0; c < M; c++) {
        for (int r = 0; r < N; r++){
            arr[idx] = some_matrix[r*M + c];
            printf("%.2f, ", some_matrix[r*M + c]);
            idx++;
        }
        printf("\n");
    }
    
    printf("Converted:\n");
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            printf("%.2f, ", arr[r*N + c]);
        }
        printf("\n");
    }
    
    /* get the dimensions and values of 1 x 2 vector*/     
    int Mvec = mxGetM(VEC_IN);
    int Nvec = mxGetN(VEC_IN);
    if(Mvec != 1 || Nvec != 2){
	    mexErrMsgIdAndTxt( "MATLAB:planner:invalidrobotpose",
                "robotpose vector should be 1 by 2.");    
        return;
    }
    double* some_vec = mxGetPr(VEC_IN);
    int some_vec_0 = (int)some_vec[0];
    int some_vec_1 = (int)some_vec[1];
    printf("Input vec: [%d, %d]\n", some_vec_0, some_vec_1);
        
    /* Create a 1 x 2 vector as returned output value */ 
    M = 3;
    N = 2; 
    VEC_OUT = mxCreateNumericMatrix( (mwSize)M, (mwSize)N, mxINT8_CLASS, mxREAL); 
    char* output_ptr = (char*)  mxGetPr(VEC_OUT);
    char* actual = malloc(sizeof(char)*M*N);
    // output "trajectory" of 2-vectors
    // 0, 1
    // 2, 3
    // 4, 5
    char val = 0;
    for (int ti = 0; ti < M; ti++) {
        actual[ti*N] = val++;
        actual[ti*N + 1] = val++;
    }
    for (int ti = 0; ti < M; ti++) {
        output_ptr[0*M + ti] = actual[ti*N];
        output_ptr[1*M + ti] = actual[ti*N + 1];
    }
    return;
}