/*=================================================================
 *
 * planner_multirobot.cpp
 * Multi-Robot Waste Collection - 16350 Project
 * Akshat Kumar, Hayden Lee
 *
 * World state mirrors proposal:
 *   S = {M, Z, T, R, t, t_total}
 *   Map M  : 2D grid, cell value = zone_id (0=free,1-3=zone) or -1=obstacle
 *   Zones Z: each zone has a trash generation rate
 *   Trash T: {x, y, zone_id, claimed_by_robot_id}
 *   Robots R: {x, y, target_trash_idx, path}
 *
 *=================================================================*/

#include "planner_multirobot.h"
#include <math.h>
#include <queue>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <limits>

#define GETMAPINDEX(X, Y, XSIZE, YSIZE) ((Y-1)*XSIZE + (X-1))

#if !defined(MAX)
#define MAX(A, B) ((A) > (B) ? (A) : (B))
#endif
#if !defined(MIN)
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#endif

// 8-connected grid (NO wait action — trash is stationary, waiting is never optimal)
#define NUMOFDIRS 8
static const int dX[NUMOFDIRS] = {-1, -1, -1,  0,  0,  1, 1, 1};
static const int dY[NUMOFDIRS] = {-1,  0,  1, -1,  1, -1, 0, 1};

// ---------------------------------------------------------------------------
// A* internal state (Trash and Robot structs live in planner_multirobot.h)
// ---------------------------------------------------------------------------

struct AStarState {
    int x, y;
    int g;
    double f;
    int parent_x, parent_y;

    AStarState() : x(0), y(0), g(0), f(0), parent_x(-1), parent_y(-1) {}
    AStarState(int x_, int y_, int g_, double f_)
        : x(x_), y(y_), g(g_), f(f_), parent_x(-1), parent_y(-1) {}
};

struct CompareState {
    bool operator()(const AStarState& a, const AStarState& b) {
        return a.f > b.f;
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline double euclidean(int x1, int y1, int x2, int y2) {
    return sqrt((double)(x1-x2)*(x1-x2) + (double)(y1-y2)*(y1-y2));
}

static inline int manhattan(int x1, int y1, int x2, int y2) {
    return abs(x1-x2) + abs(y1-y2);
}

static inline long long stateKey(int x, int y, int xsize) {
    return (long long)x * xsize + y;
}

static bool cellFree(int* map, int collision_thresh, int x, int y, int xsize, int ysize) {
    if (x < 1 || x > xsize || y < 1 || y > ysize) return false;
    return map[GETMAPINDEX(x, y, xsize, ysize)] < collision_thresh;
}

// ---------------------------------------------------------------------------
// A* from (sx,sy) to (gx,gy) — returns path excluding start, including goal
// ---------------------------------------------------------------------------

static std::vector<std::pair<int,int>> astar(
    int* map, int collision_thresh, int xsize, int ysize,
    int sx, int sy, int gx, int gy)
{
    if (!cellFree(map, collision_thresh, gx, gy, xsize, ysize))
        return {};

    std::priority_queue<AStarState, std::vector<AStarState>, CompareState> open;
    std::unordered_map<long long, int> visited;
    std::unordered_map<long long, AStarState> state_map;

    AStarState start(sx, sy, 0, euclidean(sx, sy, gx, gy));
    long long start_key = stateKey(sx, sy, xsize);
    open.push(start);
    visited[start_key] = 0;
    state_map[start_key] = start;

    AStarState goal_state;
    bool found = false;
    int iters = 0;

    while (!open.empty() && iters++ < 100000) {
        AStarState cur = open.top(); open.pop();

        if (cur.x == gx && cur.y == gy) {
            goal_state = cur;
            found = true;
            break;
        }

        long long cur_key = stateKey(cur.x, cur.y, xsize);
        if (visited.count(cur_key) && visited[cur_key] < cur.g) continue;

        for (int dir = 0; dir < NUMOFDIRS; dir++) {
            int nx = cur.x + dX[dir];
            int ny = cur.y + dY[dir];
            if (!cellFree(map, collision_thresh, nx, ny, xsize, ysize)) continue;

            int ng = cur.g + map[GETMAPINDEX(nx, ny, xsize, ysize)];
            long long nk = stateKey(nx, ny, xsize);

            if (visited.count(nk) && visited[nk] <= ng) continue;

            double nf = ng + euclidean(nx, ny, gx, gy);
            AStarState ns(nx, ny, ng, nf);
            ns.parent_x = cur.x;
            ns.parent_y = cur.y;

            visited[nk] = ng;
            state_map[nk] = ns;
            open.push(ns);
        }
    }

    if (!found) return {};

    // Reconstruct path
    std::vector<std::pair<int,int>> path;
    AStarState cur = goal_state;
    while (cur.parent_x != -1) {
        path.push_back({cur.x, cur.y});
        long long pk = stateKey(cur.parent_x, cur.parent_y, xsize);
        cur = state_map[pk];
    }
    std::reverse(path.begin(), path.end());
    return path;
}

// ---------------------------------------------------------------------------
// Target selection strategies
// ---------------------------------------------------------------------------

/*
 * STRATEGY 1: Greedy nearest — pick unclaimed trash with min distance.
 */
static int selectTarget_Greedy(
    const Robot& robot,
    const std::vector<Trash>& trash,
    int robot_id)
{
    int best = -1;
    double best_dist = std::numeric_limits<double>::max();

    for (int i = 0; i < (int)trash.size(); i++) {
        if (trash[i].claimed_by != -1 && trash[i].claimed_by != robot_id) continue;
        double d = euclidean(robot.x, robot.y, trash[i].x, trash[i].y);
        if (d < best_dist) { best_dist = d; best = i; }
    }
    return best;
}

/*
 * STRATEGY 2: Priority/Saturation weighted (A* target selection).
 *
 * From proposal cost function:
 *   score(T_i, robot_i, t, S) = alpha * p(z_I, t)
 *                              + beta  / (path_cost + 1)
 *                              + gamma * lambda_{z_I}
 *              divided by saturation sigma_i(z_I, t)
 *
 * where sigma_i(z,t) = 1 + |{j != i | t_j != null && Z(t_j) == z}|
 *   (how many OTHER robots are already heading to that zone)
 *
 * Here we use a simplified version:
 *   score = (alpha * zone_priority + beta / (dist+1) + gamma * zone_rate)
 *           / saturation
 *
 * Hyperparameters alpha, beta, gamma passed in from caller.
 */
static int selectTarget_Priority(
    const Robot& robot,
    const std::vector<Trash>& trash,
    const std::vector<Robot>& robots,
    const double* zone_rates,   // zone_rates[zone_id-1] = generation rate
    int robot_id,
    double alpha, double beta, double gamma_)
{
    int best = -1;
    double best_score = -std::numeric_limits<double>::max();

    for (int i = 0; i < (int)trash.size(); i++) {
        if (trash[i].claimed_by != -1 && trash[i].claimed_by != robot_id) continue;

        int z = trash[i].zone;
        double p_z = (double)z;                    // zone priority = zone id (1,2,3)
        double lambda_z = zone_rates[z - 1];       // generation rate
        double dist = euclidean(robot.x, robot.y, trash[i].x, trash[i].y);

        // Saturation: count other robots targeting this zone
        int sigma = 1;
        for (int j = 0; j < (int)robots.size(); j++) {
            if (j == robot_id) continue;
            if (robots[j].target_idx != -1 &&
                trash[robots[j].target_idx].zone == z) sigma++;
        }

        double score = (alpha * p_z + beta / (dist + 1.0) + gamma_ * lambda_z)
                       / (double)sigma;

        if (score > best_score) { best_score = score; best = i; }
    }
    return best;
}

/*
 * STRATEGY 3: Unified graph search (placeholder — full implementation TBD).
 * Add a pseudo-goal node with cost = 1/value and run A* over (x,y,target) space.
 * For now falls back to Strategy 2.
 */
static int selectTarget_Unified(
    const Robot& robot,
    const std::vector<Trash>& trash,
    const std::vector<Robot>& robots,
    const double* zone_rates,
    int robot_id,
    double alpha, double beta, double gamma_)
{
    // TODO: implement unified graph search over (x, y, target_id) state space
    // with pseudo-goal node addition (cost = 1/value as described in proposal)
    return selectTarget_Priority(robot, trash, robots, zone_rates, robot_id,
                                  alpha, beta, gamma_);
}

// ---------------------------------------------------------------------------
// Main planner — called once per tick, updates all robots
// ---------------------------------------------------------------------------

/*
 * planner_multirobot()
 *
 * Inputs:
 *   map              : flattened 2D costmap (cell value = traversal cost)
 *   collision_thresh : cells >= this value are obstacles
 *   x_size, y_size   : grid dimensions
 *   robots           : array of num_robots Robot structs (modified in place)
 *   num_robots       : number of robots
 *   trash            : array of Trash structs (modified in place — claims set)
 *   num_trash        : number of active trash items
 *   zone_rates       : zone generation rates (length = num_zones)
 *   curr_time        : current simulation tick
 *   strategy         : 0 = greedy, 1 = priority/saturation, 2 = unified
 *   alpha,beta,gamma : hyperparameters for strategy 1 and 2
 *
 * Outputs:
 *   action_ptr       : flat array [r0x, r0y, r1x, r1y, ...] of next positions
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
    int*    action_ptr)
{
    std::vector<Trash>  trash(trash_arr, trash_arr + num_trash);
    std::vector<Robot>  robot_vec(robots, robots + num_robots);

    for (int i = 0; i < num_robots; i++) {
        Robot& r = robot_vec[i];

        // If target is gone (collected by another robot), clear it
        if (r.target_idx >= 0 && r.target_idx >= num_trash) {
            r.target_idx = -1;
            r.path.clear();
        }
        if (r.target_idx >= 0 && trash[r.target_idx].claimed_by != i) {
            r.target_idx = -1;
            r.path.clear();
        }

        // Select a new target if needed
        if (r.target_idx == -1) {
            int t_idx = -1;
            if (strategy == 0)
                t_idx = selectTarget_Greedy(r, trash, i);
            else if (strategy == 1)
                t_idx = selectTarget_Priority(r, trash, robot_vec, zone_rates,
                                              i, alpha, beta, gamma_param);
            else
                t_idx = selectTarget_Unified(r, trash, robot_vec, zone_rates,
                                             i, alpha, beta, gamma_param);

            if (t_idx >= 0) {
                // Unclaim previous holder if any
                if (trash[t_idx].claimed_by != -1)
                    robot_vec[trash[t_idx].claimed_by].target_idx = -1;
                trash[t_idx].claimed_by = i;
                r.target_idx = t_idx;
                r.path = astar(map, collision_thresh, x_size, y_size,
                                r.x, r.y, trash[t_idx].x, trash[t_idx].y);
            }
        }

        // Replan if path is empty but target exists
        if (r.target_idx >= 0 && r.path.empty()) {
            r.path = astar(map, collision_thresh, x_size, y_size,
                            r.x, r.y,
                            trash[r.target_idx].x, trash[r.target_idx].y);
        }

        // Take one step along path
        int next_x = r.x, next_y = r.y;
        if (!r.path.empty()) {
            next_x = r.path[0].first;
            next_y = r.path[0].second;
            r.path.erase(r.path.begin());
        }

        action_ptr[i * 2]     = next_x;
        action_ptr[i * 2 + 1] = next_y;

        // Write back
        r.x = next_x;
        r.y = next_y;
        robots[i] = r;
    }

    // Write back trash claim state
    for (int i = 0; i < num_trash; i++)
        trash_arr[i] = trash[i];
}