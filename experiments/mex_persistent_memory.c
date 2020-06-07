#include <stdlib.h>
#include "mex.h"

static double *myarray = NULL;
double *pr;
void exitFcn()
{
    if (myarray != NULL)
        mxFree(myarray);
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 | mxIsChar(prhs[0]))
        mexErrMsgTxt("Must have one non-string input");
    if (myarray == NULL)
    {
        /* since myarray is initialized to NULL, we know
       this is the first call of the MEX-function 
   after it was loaded.  Therefore, we should
   set up myarray and the exit function. */
        /* Allocate array. Use mexMackMemoryPersistent to make the allocated memory persistent in subsequent calls*/
        printf("First call to MEX-file\n");
        myarray = calloc(1, sizeof(double));
        mexMakeMemoryPersistent(myarray);
        mexAtExit(exitFcn);
    }
    printf("Old string was '%f'.\n", myarray[0]);
    pr = mxGetPr(prhs[0]);
    printf("New string is '%f'.\n", pr[0]);
    printf("\n");
    memcpy((char *)myarray, (char *)mxGetPr(prhs[0]), sizeof(double));
}