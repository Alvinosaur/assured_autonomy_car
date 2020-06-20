#include "include/planner.h"
#include "include/utils.h"
#include "include/mdpconfig.h"

/** 
 * width and height are size of discretized grid, not actual C-space,
 * so input map_xyz should have size of width * height
 * 
 * 
 * 
*/
Planner::Planner(int width, int height,
                 double *map_xyz,
                 double rob_width, double rob_length,
                 double startx, double starty, double starttheta,
                 double goalx, double goaly, double goaltheta,
                 double cellsize_m,
                 char const *motPrimFilename,
                 bool forwardSearch)
{
    // initialize true map from the environment file without perimeter or motion primitives
    environment_navxythetalat.InitializeEnv(
        width,
        height,
        map_xyz,
        rob_width, rob_length,
        startx, starty, starttheta,
        goalx, goaly, goaltheta,
        goaltol_x, goaltol_y, goaltol_theta,
        cellsize_m,
        motPrimFilename);

    // set start and goal states
    int startstateid = environment_navxythetalat.SetStart(startx, starty, starttheta);
    int goalstateid = environment_navxythetalat.SetGoal(goalx, goaly, goaltheta);

    // initialize MDP info
    MDPConfig MDPCfg(startstateid, goalstateid);

    bforwardsearch = forwardSearch;
    internal_planner = new ADPlanner(&environment_navxythetalat, bforwardsearch);

    // set the start and goal states for the planner and other search variables
    if (internal_planner->set_start(MDPCfg.startstateid) == 0)
    {
        throw SBPL_Exception("ERROR: failed to set start state");
    }
    if (internal_planner->set_goal(MDPCfg.goalstateid) == 0)
    {
        throw SBPL_Exception("ERROR: failed to set goal state");
    }
    internal_planner->set_initialsolution_eps(initialEpsilon);
    internal_planner->set_search_mode(bsearchuntilfirstsolution);
}

double Planner::get_max_mpriv_dist()
{
    double maxMotPrimLengthSquared = 0.0;
    double maxMotPrimLength = 0.0;
    const EnvNAVXYTHETALATConfig_t *cfg =
        environment_navxythetalat.GetEnvNavConfig();
    for (int i = 0; i < (int)cfg->mprimV.size(); i++)
    {
        const SBPL_xytheta_mprimitive &mprim = cfg->mprimV.at(i);
        int dx = mprim.endcell.x;
        int dy = mprim.endcell.y;
        if (dx * dx + dy * dy > maxMotPrimLengthSquared)
        {
            maxMotPrimLengthSquared = dx * dx + dy * dy;
        }
    }
    maxMotPrimLength = sqrt(maxMotPrimLengthSquared);
    return maxMotPrimLength;
}

void Planner::update(double *updated_window, int minx,
                     int maxx, int miny, int maxy)
{
    nav2dcell_t nav2dcell;
    vector<int> preds_of_changededgesIDV;

    // simulate sensing the cells
    int window_idx = 0;
    for (int y = miny; y < maxy; y++)
    {
        for (int x = minx; x < maxx; x++)
        {
            double trueZ = updated_window[window_idx++];
            // update the cell if we haven't seen it before
            if (!almost_equal(
                    environment_navxythetalat.GetZ(x, y), trueZ,
                    1e-2))
            {
                environment_navxythetalat.UpdateZ(x, y, trueZ);
                // store the changed cells
                nav2dcell.x = x;
                nav2dcell.y = y;
                changedcellsV.push_back(nav2dcell);
            }
        }
    }
    // get the affected states
    environment_navxythetalat.GetPredsofChangedEdges(&changedcellsV, &preds_of_changededgesIDV);
    // let know the incremental planner about them
    //use by AD* planner (incremental)
    internal_planner->update_preds_of_changededges(&preds_of_changededgesIDV);
}

void Planner::plan(double *&output_trajectory, int desired_traj_len, int &actual_traj_len)
{
    bool mapChanged = changedcellsV.size() > 0;
    bool no_plan_exists = (solution_stateIDs_V.size() == 0);

    std::vector<int> goal_vec = environment_navxythetalat.GetGoal();
    int goalx_c = goal_vec[0];
    int goaly_c = goal_vec[1];
    int goaltheta_c = goal_vec[2];

    std::vector<int> start_vec = environment_navxythetalat.GetStart();
    int startx_c = start_vec[0];
    int starty_c = start_vec[1];
    int starttheta_c = start_vec[2];

    if (mapChanged || no_plan_exists)
    {
        // plan a path
        bool bPlanExists = false;
        solution_stateIDs_V.clear();
        solution_xythetaPath.clear();
        // printf("new planning...\n");
        bPlanExists = (internal_planner->replan(allocated_time_secs_foreachplan, &solution_stateIDs_V) == 1);
        // printf("done with the solution of size=%d and sol. eps=%f\n", (unsigned int)solution_stateIDs_V.size(),
        //        internal_planner->get_solution_eps());
        // environment_navxythetalat.PrintTimeStat(stdout);

        environment_navxythetalat.ConvertStateIDPathintoXYThetaPath(&solution_stateIDs_V, &solution_xythetaPath);
        if (!bPlanExists)
        {
            throw SBPL_Exception("ERROR: Plan not found!!");
        }
    }
    // else if no need to replan, just use existing plan
    actual_traj_len = min(desired_traj_len,
                          static_cast<int>(solution_xythetaPath.size()));
    output_trajectory = (double *)malloc(sizeof(double) * actual_traj_len * 3);

    /*
    x1,y1,th1
    x2,y2,th2
    ...
    */
    for (int i = 0; i < actual_traj_len; i++)
    {
        sbpl_xy_theta_pt_t pt = solution_xythetaPath[i];
        output_trajectory[i * 3] = pt.x;
        output_trajectory[i * 3 + 1] = pt.y;
        output_trajectory[i * 3 + 2] = pt.theta;
    }
    solution_xythetaPath.erase(solution_xythetaPath.begin(),
                               solution_xythetaPath.begin() + actual_traj_len);
}