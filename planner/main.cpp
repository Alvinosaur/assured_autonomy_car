/*
 * Copyright (c) 2008, Maxim Likhachev
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Mellon University nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <assert.h>
#include "mex.h"

using namespace std;

#include "include/mex_helpers.h"
#include "include/helpers.h"
#include "include/planner.h"

static Planner *planner = nullptr;
const int des_traj_len = 10;

void exitFcn()
{
    if (planner != nullptr)
    {
        mxFree(planner);
        planner = nullptr;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (planner == NULL)
    {
        assert(nrhs == 5);
        double rob_dims[2];
        double start_pose[3];
        double end_pose[3];
        double *map = nullptr;
        int map_dims[2];
        double cellsize_m;
        parse_full_mex(prhs, rob_dims, start_pos,
                       end_pose, &map, map_dims, cellsize_m);
        char *MotPrimFile = "/Users/Alvin/Documents/Code/safety_reachability_AV_research/assured_autonomy_car/sbpl/matlab/mprim/non_uniform_res01_rad73_err01.mprim";
        bool forwardSearch = true;

        planner = new Planner(map_dims[0], map_dims[1],
                              map,
                              rob_dims[0], rob_dims[1],
                              start_pose[0], start_pose[1], start_pose[2],
                              end_pose[0], end_pose[1], end_pose[2],
                              cellsize_m,
                              MotPrimFile,
                              forwardSearch);

        /* Use mexMackMemoryPersistent to make the allocated memory persistent in subsequent calls*/
        mexMakeMemoryPersistent(planner);
        mexAtExit(exitFcn);
    }
    else
    {
        assert(nrhs == 2);
        // update map and actual robot pose
        double rob_pose[3];
        double *update_window = nullptr;
        int window_dims[2];
        parse_update_mex(prhs, &update_window, window_dims, rob_pose);
        planner->update(update_window, window_dims, rob_pose);
    }

    // out is N x 3 trajectory to follow
    double *temp_output_traj = nullptr;
    int actual_traj_len;
    planner->plan(temp_output_traj, des_traj_len, actual_traj_len);

    // store in col-major format to be used in matlab
    plhs[0] = mxCreateNumericMatrix((mwSize)actual_traj_len,
                                    (mwSize)3, mxDOUBLE_CLASS, mxREAL);
    double *output_trajectory = (double *)mxGetPr(plhs[0]);
    row_to_colmajor<double>(output_trajectory, temp_output_traj,
                            actual_traj_len, 3);
    free(temp_output_traj);
}
