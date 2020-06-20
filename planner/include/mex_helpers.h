#ifndef __MEX_HELPERS_H_
#define __MEX_HELPERS_H_
#include "mex.h"

void parse_full_mex(mxArray *prhs[],
                    double rob_dims[2],
                    double start_pose[3],
                    double end_pose[3],
                    double **map, int map_dims[2], double &cellsize_m)
{
    int M, N;
    double *arr;
    // parse robot dimensions
    mxArray *robot_dims_input = prhs[0];
    M = mxGetM(robot_dims_input);
    N = mxGetN(robot_dims_input);
    assert((M == 1 && N == 2) || (N == 1 && M == 2));
    arr = mxGetPr(robot_dims_input);
    memcpy(rob_dims, arr, sizeof(double) * 2);

    // parse start pose
    mxArray *start_pose_input = prhs[1];
    M = mxGetM(start_pose_input);
    N = mxGetN(start_pose_input);
    assert((M == 1 && N == 3) || (N == 1 && M == 3));
    arr = mxGetPr(start_pose_input);
    memcpy(start_pose, arr, sizeof(double) * 3);

    // parse end pose
    mxArray *end_pose_input = prhs[2];
    M = mxGetM(end_pose_input);
    N = mxGetN(end_pose_input);
    assert((M == 1 && N == 3) || (N == 1 && M == 3));
    arr = mxGetPr(end_pose_input);
    memcpy(end_pose, arr, sizeof(double) * 3);

    // parse map
    mxArray *map_input = prhs[3];
    M = mxGetM(end_pose_input);
    N = mxGetN(end_pose_input);
    map_dims[0] = M;
    map_dims[1] = N;
    arr = mxGetPr(map_input);
    *map = (double *)malloc(sizeof(double) * M * N);
    col_to_rowmajor<double>(map, arr, M, N);

    // parse cell size
    mxArray *cellsize_input = prhs[4];
    M = mxGetM(cellsize_input);
    N = mxGetN(cellsize_input);
    assert(M == 1 && N == 1);
    cellsize_input = *mxGetPr(cellsize_input);
}

void parse_update_mex(mxArray *prhs[],
                      double **update_window,
                      int window_dims[2],
                      double rob_pose[3])
{
    int M, N;
    double *arr;
    // parse robot dimensions
    mxArray *update_window_input = prhs[0];
    M = mxGetM(update_window_input);
    N = mxGetN(update_window_input);
    window_dims[0] = M;
    window_dims[1] = N;
    arr = mxGetPr(robot_dims_input);
    *update_window = (double *)malloc(sizeof(double) * M * N);
    col_to_rowmajor<double>(update_window, arr, M, N);

    // parse new robot pose
    mxArray *pose_input = prhs[2];
    M = mxGetM(pose_input);
    N = mxGetN(pose_input);
    assert((M == 1 && N == 3) || (N == 1 && M == 3));
    arr = mxGetPr(pose_input);
    memcpy(rob_pose, arr, sizeof(double) * 3);
}

#endif