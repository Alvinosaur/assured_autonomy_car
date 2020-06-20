#ifndef __PLANNER_H_
#define __PLANNER_H_

#include <math.h>
#include <assert.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

#include "environment_navxythetalat.h"
#include "adplanner.h"
#include "helpers.h"

struct Planner
{
        double allocated_time_secs_foreachplan = 10.0; // in seconds
        double initialEpsilon = 3.0;                   // Dstar Lite, iteratively decrease epsilon
        bool bsearchuntilfirstsolution = false;
        bool bforwardsearch = false;

        double goaltol_x = 0.1, goaltol_y = 0.1, goaltol_theta = 5 * M_PI / 180.0;

        bool bPrint = false;
        bool bPrintMap = false;

        EnvironmentNAVXYTHETALAT environment_navxythetalat;
        ADPlanner *internal_planner = nullptr;

        vector<nav2dcell_t> changedcellsV;
        vector<int> solution_stateIDs_V;
        vector<sbpl_xy_theta_pt_t> solution_xythetaPath;

        Planner(int width, int height,
                double *map_xyz,
                double rob_width, double rob_length,
                double startx, double starty, double starttheta,
                double goalx, double goaly, double goaltheta,
                double cellsize_m,
                char const *motPrimFilename,
                bool forwardSearch);

        void update(double *updated_window, int minx,
                    int maxx, int miny, int maxy);

        double get_max_mpriv_dist();

        void plan(double *&output_trajectory, int desired_traj_len,
                  int &actual_traj_len);
};

#endif