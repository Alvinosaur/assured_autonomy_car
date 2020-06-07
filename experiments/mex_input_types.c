#include "mex.h"


/* Input Arguments */
#define	MATRIX_IN      prhs[0]
#define	VEC_IN         prhs[1]

/* Output Arguments */
#define	VEC_OUT        plhs[0]

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
{ 
    
    /* Check for proper number of arguments */    
    if (nrhs != 2) { 
	    mexErrMsgIdAndTxt( "MATLAB:planner:invalidNumInputs",
                "Three input arguments required."); 
        return;
    } else if (nlhs != 1) {
	    mexErrMsgIdAndTxt( "MATLAB:planner:maxlhs",
                "One output argument required."); 
        return;
    } 
        
    /* get the dimensions and values of M x N matrix */
    /* Matlab uses column-major order */
    int M = mxGetM(MATRIX_IN);
    int N = mxGetN(MATRIX_IN);
    double* some_matrix = mxGetPr(MATRIX_IN);
    printf("Read in matrix of %d x %d:\n", M, N);
    for (int c = 0; c < N; c++) {
        for (int r = 0; r < M; r++){
            printf("%.3f, ", some_matrix[r*N + c]);
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
    VEC_OUT = mxCreateNumericMatrix( (mwSize)1, (mwSize)2, mxINT8_CLASS, mxREAL); 
    char* output_ptr = (char*)  mxGetPr(VEC_OUT);
    output_ptr[0] = 5;
    output_ptr[1] = 3;
    
    return;
}