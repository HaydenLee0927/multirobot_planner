/*=================================================================
 *
 * runtest.cpp - Multi-Robot Waste Collection
 * 16350 Project - Akshat Kumar, Hayden Lee
 *
 * Usage:
 *   ./runtest <map_file> [strategy] [alpha] [beta] [gamma] [t_max]
 *
 *   strategy: 0 = greedy nearest         (default)
 *             1 = priority/saturation
 *             2 = unified graph search
 *
 * Outputs:
 *   robot_trajectory_0.txt, robot_trajectory_1.txt, ...
 *   Each line: t,x,y,collected
 *
 *=================================================================*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cassert>

#include "planner_multirobot.h"

int main(int argc, char *argv[])
{
    // ---------------------------------------------------------------
    // READ PROBLEM
    // ---------------------------------------------------------------
    if (argc < 2)
    {
        std::cout << "Usage: ./runtest <map_file> [strategy=0] "
                     "[alpha=1.0] [beta=5.0] [gamma=2.0] [t_max=200]"
                  << std::endl;
        return -1;
    }

    int    strategy = argc > 2 ? std::stoi(argv[2]) : 0;
    double alpha    = argc > 3 ? std::stod(argv[3]) : 1.0;
    double beta     = argc > 4 ? std::stod(argv[4]) : 5.0;
    double gamma_p  = argc > 5 ? std::stod(argv[5]) : 2.0;
    int    t_max    = argc > 6 ? std::stoi(argv[6]) : 200;

    std::cout << "Reading problem definition from: " << argv[1] << std::endl;

    std::ifstream myfile;
    myfile.open(argv[1]);
    if (!myfile.is_open())
    {
        std::cout << "Failed to open the file." << std::endl;
        return -1;
    }

    char letter;
    std::string line;

    // --- N: map size ---
    myfile >> letter;
    if (letter != 'N') { std::cout << "error parsing file at N" << std::endl; return -1; }
    int x_size, y_size;
    myfile >> x_size >> letter >> y_size;
    std::cout << "map size: " << x_size << "x" << y_size << std::endl;

    // --- C: collision threshold ---
    myfile >> letter;
    if (letter != 'C') { std::cout << "error parsing file at C" << std::endl; return -1; }
    int collision_thresh;
    myfile >> collision_thresh;
    std::cout << "collision threshold: " << collision_thresh << std::endl;

    // --- ROBOTS ---
    do { std::getline(myfile, line); } while (line != "ROBOTS");
    int num_robots;
    myfile >> num_robots;
    std::cout << "num robots: " << num_robots << std::endl;

    std::vector<Robot> robots(num_robots);
    for (int i = 0; i < num_robots; i++)
    {
        std::getline(myfile, line);
        while (line.empty()) std::getline(myfile, line);
        std::stringstream ss(line);
        ss >> robots[i].x >> letter >> robots[i].y;
        robots[i].target_idx = -1;
        robots[i].collected  = 0;
        std::cout << "robot " << i << " start: "
                  << robots[i].x << "," << robots[i].y << std::endl;
    }

    // --- ZONES ---
    do { std::getline(myfile, line); } while (line != "ZONES");
    int num_zones;
    myfile >> num_zones;
    std::cout << "num zones: " << num_zones << std::endl;

    struct Zone { int id, x1, y1, x2, y2; double rate; };
    std::vector<Zone>   zones(num_zones);
    std::vector<double> zone_rates(num_zones);
    for (int i = 0; i < num_zones; i++)
    {
        std::getline(myfile, line);
        while (line.empty()) std::getline(myfile, line);
        std::stringstream ss(line);
        char c;
        ss >> zones[i].id >> c >> zones[i].x1 >> c >> zones[i].y1
           >> c >> zones[i].x2 >> c >> zones[i].y2 >> c >> zones[i].rate;
        zone_rates[i] = zones[i].rate;
        std::cout << "zone " << zones[i].id
                  << " rate=" << zones[i].rate << std::endl;
    }

    // --- TRASH ---
    do { std::getline(myfile, line); } while (line != "TRASH");
    int num_initial_trash;
    myfile >> num_initial_trash;
    std::cout << "initial trash: " << num_initial_trash << std::endl;

    std::vector<Trash> trash(num_initial_trash);
    for (int i = 0; i < num_initial_trash; i++)
    {
        std::getline(myfile, line);
        while (line.empty()) std::getline(myfile, line);
        std::stringstream ss(line);
        char c;
        ss >> trash[i].x >> c >> trash[i].y >> c >> trash[i].zone;
        trash[i].claimed_by = -1;
    }

    // --- M: costmap ---
    do { std::getline(myfile, line); } while (line != "M");

    int* map = new int[x_size * y_size];
    for (int i = 0; i < x_size; i++)
    {
        std::getline(myfile, line);
        std::stringstream ss(line);
        for (int j = 0; j < y_size; j++)
        {
            double value;
            ss >> value;
            map[j * x_size + i] = (int)value;
            if (j != y_size - 1) ss.ignore();
        }
    }
    myfile.close();

    std::cout << "\nRunning planner (strategy=" << strategy
              << " alpha=" << alpha
              << " beta=" << beta
              << " gamma=" << gamma_p
              << " t_max=" << t_max << ")" << std::endl;

    // ---------------------------------------------------------------
    // Open trajectory output files (one per robot, matches visualizer)
    // ---------------------------------------------------------------
    std::vector<std::ofstream> traj_files(num_robots);
    for (int i = 0; i < num_robots; i++)
    {
        std::string fname = "robot_trajectory_" + std::to_string(i) + ".txt";
        traj_files[i].open(fname);
        if (!traj_files[i].is_open())
        {
            std::cerr << "Failed to open " << fname << std::endl;
            return -1;
        }
        // t=0 starting position
        traj_files[i] << 0 << ","
                      << robots[i].x << ","
                      << robots[i].y << ",0" << std::endl;
    }

    // ---------------------------------------------------------------
    // CONTROL LOOP
    // ---------------------------------------------------------------
    int  curr_time   = 0;
    int  total_collected = 0;
    int* action_ptr  = new int[num_robots * 2];

    srand(42);  // reproducible trash spawning

    while (curr_time < t_max)
    {
        auto start = std::chrono::high_resolution_clock::now();

        curr_time++;

        // Spawn new trash per zone rate
        for (const auto& z : zones)
        {
            double r = (double)rand() / RAND_MAX;
            if (r < z.rate)
            {
                // find a free cell in this zone
                for (int attempt = 0; attempt < 50; attempt++)
                {
                    int tx = z.x1 + rand() % (z.x2 - z.x1 + 1);
                    int ty = z.y1 + rand() % (z.y2 - z.y1 + 1);
                    if (map[ty * x_size + tx] >= collision_thresh) continue;
                    bool occupied = false;
                    for (const auto& t : trash)
                        if (t.x == tx && t.y == ty) { occupied = true; break; }
                    if (!occupied)
                    {
                        trash.push_back({tx, ty, z.id, -1});
                        break;
                    }
                }
            }
        }

        int num_trash = (int)trash.size();

        if (num_trash == 0)
        {
            // nothing to collect — robots stay put
            for (int i = 0; i < num_robots; i++)
            {
                action_ptr[i*2]   = robots[i].x;
                action_ptr[i*2+1] = robots[i].y;
            }
        }
        else
        {
            planner_multirobot(
                map, collision_thresh, x_size, y_size,
                robots.data(), num_robots,
                trash.data(),  num_trash,
                zone_rates.data(),
                curr_time, strategy,
                alpha, beta, gamma_p,
                action_ptr
            );
        }

        // Apply actions + validate (mirrors HW1 checks)
        for (int i = 0; i < num_robots; i++)
        {
            int nx = action_ptr[i*2];
            int ny = action_ptr[i*2+1];

            if (nx < 1 || nx > x_size || ny < 1 || ny > y_size)
            {
                std::cout << "ERROR robot " << i
                          << ": out-of-map position commanded" << std::endl;
                return -1;
            }
            if (map[(ny-1)*x_size + nx-1] >= collision_thresh)
            {
                std::cout << "ERROR robot " << i
                          << ": planned action leads to collision" << std::endl;
                return -1;
            }
            if (abs(robots[i].x - nx) > 1 || abs(robots[i].y - ny) > 1)
            {
                std::cout << "ERROR robot " << i
                          << ": invalid action — must move on 8-connected grid" << std::endl;
                return -1;
            }

            robots[i].x = nx;
            robots[i].y = ny;

            // Collect trash at this cell
            for (int j = (int)trash.size()-1; j >= 0; j--)
            {
                if (trash[j].x == robots[i].x && trash[j].y == robots[i].y)
                {
                    robots[i].collected++;
                    total_collected++;
                    if (robots[i].target_idx == j)
                        robots[i].target_idx = -1;
                    // shift target indices for all robots
                    for (int k = 0; k < num_robots; k++)
                        if (robots[k].target_idx > j)
                            robots[k].target_idx--;
                    trash.erase(trash.begin() + j);
                }
            }

            traj_files[i] << curr_time << ","
                          << robots[i].x << ","
                          << robots[i].y << ","
                          << robots[i].collected << std::endl;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

        // Progress print every 50 ticks (mirrors HW1 per-step prints)
        if (curr_time % 50 == 0)
        {
            std::cout << "t=" << curr_time
                      << " | trash active=" << trash.size()
                      << " | total collected=" << total_collected
                      << " | planner time=" << ms << "ms" << std::endl;
        }
    }

    // ---------------------------------------------------------------
    // RESULT (mirrors HW1 output block)
    // ---------------------------------------------------------------
    std::cout << "\nRESULT" << std::endl;
    std::cout << "t_max = " << t_max << std::endl;
    std::cout << "trash remaining = " << trash.size() << std::endl;
    std::cout << "total collected = " << total_collected << std::endl;
    for (int i = 0; i < num_robots; i++)
        std::cout << "robot " << i << " collected = "
                  << robots[i].collected << std::endl;

    for (int i = 0; i < num_robots; i++)
        traj_files[i].close();

    delete[] map;
    delete[] action_ptr;

    return 0;
}