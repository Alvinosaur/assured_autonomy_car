#include "include/helpers.h"

template <class T>
void col_to_rowmajor(T *dst, const T *src, int M, int N)
{
    int idx = 0;
    for (int c = 0; c < M; c++)
    {
        for (int r = 0; r < N; r++)
        {
            dst[idx++] = src[r * M + c];
        }
    }
}

template <class T>
void row_to_colmajor(T *dst, const T *src, int M, int N)
{
    for (int r = 0; r < M; r++)
    {
        for (int c = 0; c < N; c++)
        {
            dst[c * M + r] = src[r * N + c];
        }
    }
}

bool almost_equal(double a, double b, double atol = 1e-5)
{
    return fabs(a - b) < atol;
}

bool CheckIsNavigating(int numOptions, char **argv)
{
    for (int i = 1; i < numOptions + 1; i++)
    {
        if (strcmp(argv[i], "-s") == 0)
        {
            return true;
        }
    }
    return false;
}

std::string CheckSearchDirection(int numOptions, char **argv)
{
    int optionLength = strlen("--search-dir=");
    for (int i = 1; i < numOptions + 1; i++)
    {
        if (strncmp("--search-dir=", argv[i], optionLength) == 0)
        {
            std::string s(&argv[i][optionLength]);
            return s;
        }
    }
    return std::string("backward");
}

void PrintUsage(char *argv[])
{
    printf("USAGE: %s [-s] [--env=<env_t>] [--planner=<planner_t>] [--search-dir=<search_t>] <cfg file> [mot prims]\n",
           argv[0]);
    printf("See '%s -h' for help.\n", argv[0]);
}

void PrintHelp(char **argv)
{
    printf("\n");
    printf("Search-Based Planning Library\n");
    printf("\n");
    printf("    %s -h\n", argv[0]);
    printf("    %s [-s] [--env=<env_t>] [--planner=<planner_t>] [--search-dir=<search_t>] <env cfg> [mot prim]\n",
           argv[0]);
    printf("\n");
    printf("[-s]                      (optional) Find a solution for an example navigation\n");
    printf("                          scenario where the robot only identifies obstacles as\n");
    printf("                          it approaches them.\n");
    printf("[--env=<env_t>]           (optional) Select an environment type to choose what\n");
    printf("                          example to run. The default is \"xytheta\".\n");
    printf("<env_t>                   One of 2d, xytheta, xythetamlev, robarm.\n");
    printf("[--planner=<planner_t>]   (optional) Select a planner to use for the example.\n");
    printf("                          The default is \"arastar\".\n");
    printf("<planner_t>               One of arastar, adstar, rstar, anastar.\n");
    printf("[--search-dir=<search_t>] (optional) Select the type of search to run. The default\n");
    printf("                          is \"backwards\".\n");
    printf("<search_t>                One of backward, forward.\n");
    printf("<env cfg>                 Config file representing the environment configuration.\n");
    printf("                          See sbpl/env_examples/ for examples.\n");
    printf("[mot prim]                (optional) Motion primitives file for x,y,theta lattice\n");
    printf("                          planning. See sbpl/matlab/mprim/ for examples.\n");
    printf("                          NOTE: resolution of motion primtives should match that\n");
    printf("                              of the config file.\n");
    printf("                          NOTE: optional use of these for x,y,theta planning is\n");
    printf("                              deprecated.\n");
    printf("\n");
}