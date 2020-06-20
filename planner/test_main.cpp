#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <assert.h>

using namespace std;

#include "include/helpers.h"
#include "include/planner.h"

void test_flat_map()
{
    int rob_dims[2] = {1, 1};
    double start_pose[3] = {0, 0, 0};
    double end_pose[3] = {9, 9, 0};
    int maxx = 10, maxy = 10;
    double dx = 0.1, dy = 0.1;
    int map_dims[2] = {(int)(maxx / dx),
                       (int)(maxy / dy)};
    // flat map  of all zero values, robot should just drive straight
    double *map = (double *)calloc(map_dims[0] * map_dims[1], sizeof(double));
    double cellsize_m = 0.1;
    char const *MotPrimFile = "/Users/Alvin/Documents/Code/safety_reachability_AV_research/assured_autonomy_car/wmrde/MATLAB/non_uniform_res01_rad3_err005.mprim";
    bool forwardSearch = true;
    Planner *planner = new Planner(map_dims[0], map_dims[1],
                                   map,
                                   rob_dims[0], rob_dims[1],
                                   start_pose[0], start_pose[1], start_pose[2],
                                   end_pose[0], end_pose[1], end_pose[2],
                                   cellsize_m,
                                   MotPrimFile,
                                   forwardSearch);

    // planner->update(update_window, window_dims, rob_pose);
    // out is N x 3 trajectory to follow
    double *temp_output_traj = nullptr;
    int actual_traj_len;
    int des_traj_len = 10;
    planner->plan(temp_output_traj, des_traj_len, actual_traj_len);
    cout << "Acutal length: " << actual_traj_len << endl;
    for (int i = 0; i < actual_traj_len; i++)
    {
        cout << "x,y,theta: " << temp_output_traj[i * 3] << ",";
        cout << temp_output_traj[i * 3 + 1] << ",";
        cout << temp_output_traj[i * 3 + 2] << endl;
    }
    free(temp_output_traj);
}

int main()
{
    // printf("HELLO!\n");
    test_flat_map();
}