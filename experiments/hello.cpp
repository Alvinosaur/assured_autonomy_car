#include <stdio.h>
#include "mex.h"
#include "include/hello_header.h"

int main()
{
    printf("hello\n");
    return 0;
}

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
{ 
    printf("hello\n");
    int x = say_hello();
    printf("%d\n", x);
}