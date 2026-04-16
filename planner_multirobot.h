/*=================================================================
 * planner_multirobot.h
 * Multi-Robot Waste Collection - 16350 Project
 *=================================================================*/

#pragma once
#include <vector>
#include <utility>

struct Trash {
    int x, y;
    int zone;
    int claimed_by;   // robot id, or -1 if unclaimed
};

struct Robot {
    int x, y;
    int target_idx;
    std::vector<std::pair<int,int>> path;
    int collected;
};

/*
 * strategy: 0 = greedy nearest
 *           1 = priority/saturation weighted (A* target selection)
 *           2 = unified graph search (TBD, falls back to 1)
 */
void planner_multirobot(
    int*    map,
    int     collision_thresh,
    int     x_size,
    int     y_size,
    Robot*  robots,
    int     num_robots,
    Trash*  trash_arr,
    int     num_trash,
    double* zone_rates,
    int     curr_time,
    int     strategy,
    double  alpha,
    double  beta,
    double  gamma_param,
    int*    action_ptr
);