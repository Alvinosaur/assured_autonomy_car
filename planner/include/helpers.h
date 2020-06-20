#ifndef __HELPERS_H_
#define __HELPERS_H_

#include <string>
#include <cmath>

enum MainResultType
{
    INVALID_MAIN_RESULT = -1,

    MAIN_RESULT_SUCCESS = 0,
    MAIN_RESULT_FAILURE = 1,
    MAIN_RESULT_INSUFFICIENT_ARGS = 2,
    MAIN_RESULT_INCORRECT_OPTIONS = 3,
    MAIN_RESULT_UNSUPPORTED_ENV = 4,

    NUM_MAIN_RESULTS
};

template <class T>
void col_to_rowmajor(T *dst, const T *src, int M, int N);

template <class T>
void row_to_colmajor(T *dst, const T *src, int M, int N);

bool almost_equal(double a, double b, double atol);

/*******************************************************************************
 * CheckIsNavigating
 * @brief Returns whether the -s option is being used.
 *
 * @param numOptions The number of options passed through the command line
 * @param argv The command-line arguments
 * @return whether the -s option was passed in on the cmd line
 *******************************************************************************/
bool CheckIsNavigating(int numOptions, char **argv);

/*******************************************************************************
 * CheckSearchDirection -
 * @brief Returns the search direction being used
 *
 * @param numOptions The number of options passed through the command line
 * @param argv The command-line arguments
 * @return A string representing the search direction; "backward" by default
 ******************************************************************************/
std::string CheckSearchDirection(int numOptions, char **argv);

/*******************************************************************************
 * PrintUsage - Prints the proper usage of the sbpl test executable.
 *
 * @param argv The command-line arguments; used to determine the name of the
 *             test executable.
 *******************************************************************************/
void PrintUsage(char *argv[]);

/*******************************************************************************
 * PrintHelp - Prints a help prompt to the command line when the -h option is
 *             used.
 *
 * @param argv The command line arguments; used to determine the name of the
 *             test executable
 *******************************************************************************/
void PrintHelp(char **argv);

#endif