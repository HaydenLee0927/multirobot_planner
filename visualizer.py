"""
visualizer.py - Multi-Robot Waste Collection
16350 Project - Akshat Kumar, Hayden Lee

Changes from HW1 visualizer:
  - parse_mapfile: new sections ROBOTS, ZONES, TRASH replace single R and T
  - parse_robot_trajectory_file: reads N robot trajectories (one file per robot,
    or a single multi-robot file with robot_id column)
  - draw logic: color-coded zones as background, trash as scatter, N robot trails
  - No target trajectory (trash is stationary); trash disappears when collected
  - Score overlay: collected count per robot + total shown in title
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import sys
import os


# ---------------------------------------------------------------------------
# Zone color scheme — matches proposal's color-coded areas
# ---------------------------------------------------------------------------
ZONE_COLORS = {
    0: '#F5F5F0',   # free / no zone (off-white)
    1: '#D0E8FF',   # zone 1 - low rate  (light blue)
    2: '#D4EDDA',   # zone 2 - med rate  (light green)
    3: '#FFE8C0',   # zone 3 - high rate (light amber)
}

ROBOT_COLORS = ['#3B82F6', '#16A34A', '#DC2626', '#7C3AED', '#D97706']
ROBOT_LABELS = ['Robot 0', 'Robot 1', 'Robot 2', 'Robot 3', 'Robot 4']


# ---------------------------------------------------------------------------
# Map file parser
# ---------------------------------------------------------------------------
def parse_mapfile(filename):
    """
    New map format:

    N
    x_size,y_size
    C
    collision_thresh
    ROBOTS
    num_robots
    r0x,r0y
    r1x,r1y
    ...
    ZONES
    num_zones
    zone_id,x1,y1,x2,y2,rate   (bounding box, inclusive)
    ...
    TRASH
    num_trash
    x,y,zone_id
    ...
    M
    row0: val,val,...
    row1: ...
    """
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    idx = 0
    assert lines[idx] == 'N';           idx += 1
    x_size, y_size = map(int, lines[idx].split(',')); idx += 1

    assert lines[idx] == 'C';           idx += 1
    collision_thresh = int(lines[idx]); idx += 1

    assert lines[idx] == 'ROBOTS';      idx += 1
    num_robots = int(lines[idx]);       idx += 1
    robot_starts = []
    for _ in range(num_robots):
        rx, ry = map(int, lines[idx].split(','))
        robot_starts.append((rx, ry)); idx += 1

    assert lines[idx] == 'ZONES';       idx += 1
    num_zones = int(lines[idx]);        idx += 1
    zones = []
    for _ in range(num_zones):
        parts = lines[idx].split(',')
        zones.append({
            'id':   int(parts[0]),
            'x1':   int(parts[1]), 'y1': int(parts[2]),
            'x2':   int(parts[3]), 'y2': int(parts[4]),
            'rate': float(parts[5])
        }); idx += 1

    assert lines[idx] == 'TRASH';       idx += 1
    num_trash = int(lines[idx]);        idx += 1
    initial_trash = []
    for _ in range(num_trash):
        parts = lines[idx].split(',')
        initial_trash.append({'x': int(parts[0]), 'y': int(parts[1]),
                               'zone': int(parts[2])}); idx += 1

    assert lines[idx] == 'M';           idx += 1
    costmap = []
    for i in range(idx, len(lines)):
        row = list(map(float, lines[i].split(',')))
        costmap.append(row)

    costmap = np.asarray(costmap)   # shape: (y_size, x_size) — row=y, col=x

    # Build zone map (0 = no zone)
    zone_map = np.zeros((y_size, x_size), dtype=int)
    for z in zones:
        zone_map[z['y1']:z['y2']+1, z['x1']:z['x2']+1] = z['id']

    return (x_size, y_size, collision_thresh,
            robot_starts, zones, initial_trash, costmap, zone_map)


# ---------------------------------------------------------------------------
# Trajectory file parser
# ---------------------------------------------------------------------------
def parse_trajectories(num_robots, traj_dir='.'):
    """
    Reads one file per robot: robot_trajectory_0.txt, robot_trajectory_1.txt, ...
    Each line: t,x,y[,collected]
    Returns list of lists of dicts.

    Falls back to robot_trajectory.txt with format: t,robot_id,x,y
    """
    trajectories = []

    # Try per-robot files first
    per_robot_files = [os.path.join(traj_dir, f'robot_trajectory_{i}.txt')
                       for i in range(num_robots)]
    if all(os.path.exists(f) for f in per_robot_files):
        for fpath in per_robot_files:
            traj = []
            with open(fpath) as f:
                for line in f:
                    parts = list(map(int, line.strip().split(',')))
                    entry = {'t': parts[0], 'x': parts[1], 'y': parts[2],
                             'collected': parts[3] if len(parts) > 3 else 0}
                    traj.append(entry)
            trajectories.append(traj)
        return trajectories

    # Fall back to combined file: t,robot_id,x,y[,collected]
    combined = os.path.join(traj_dir, 'robot_trajectory.txt')
    if os.path.exists(combined):
        per_robot = [[] for _ in range(num_robots)]
        with open(combined) as f:
            for line in f:
                parts = list(map(int, line.strip().split(',')))
                t, rid, x, y = parts[0], parts[1], parts[2], parts[3]
                collected = parts[4] if len(parts) > 4 else 0
                per_robot[rid].append({'t': t, 'x': x, 'y': y,
                                       'collected': collected})
        return per_robot

    raise FileNotFoundError(
        "No trajectory files found. Expected robot_trajectory_0.txt ... "
        "or robot_trajectory.txt"
    )


# ---------------------------------------------------------------------------
# Build background image from zone map + costmap
# ---------------------------------------------------------------------------
def build_background(zone_map, costmap, collision_thresh, x_size, y_size):
    """
    Returns an RGBA image:
      - Zone colors for free cells
      - Dark gray for obstacles (costmap >= collision_thresh)
    """
    rgba = np.zeros((y_size, x_size, 4), dtype=float)

    zone_rgb = {
        0: (0.96, 0.96, 0.94, 1.0),
        1: (0.82, 0.91, 1.00, 1.0),
        2: (0.83, 0.93, 0.85, 1.0),
        3: (1.00, 0.91, 0.75, 1.0),
    }

    for y in range(y_size):
        for x in range(x_size):
            if costmap[y, x] >= collision_thresh:
                rgba[y, x] = (0.45, 0.45, 0.45, 1.0)  # obstacle
            else:
                z = zone_map[y, x]
                rgba[y, x] = zone_rgb.get(z, zone_rgb[0])

    return rgba


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SPEEDUP = 5

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualizer.py <map_file> [trajectory_dir]")
        sys.exit(1)

    map_file = sys.argv[1]
    traj_dir = sys.argv[2] if len(sys.argv) > 2 else '.'

    (x_size, y_size, collision_thresh,
     robot_starts, zones, initial_trash,
     costmap, zone_map) = parse_mapfile(map_file)

    num_robots = len(robot_starts)

    try:
        trajectories = parse_trajectories(num_robots, traj_dir)
        has_trajectories = True
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Showing static map only.")
        has_trajectories = False

    # --- Build figure ---
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_facecolor('#1A1A2E')

    bg = build_background(zone_map, costmap, collision_thresh, x_size, y_size)
    ax.imshow(bg, origin='upper', extent=[-0.5, x_size-0.5, y_size-0.5, -0.5])

    # Grid lines (subtle)
    ax.set_xticks(np.arange(-0.5, x_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, y_size, 1), minor=True)
    ax.grid(which='minor', color='#00000015', linewidth=0.3)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    # --- Static: robot start markers ---
    for i, (rx, ry) in enumerate(robot_starts):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        ax.plot(rx, ry, 's', color=color, markersize=10,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)

    # --- Static: initial trash ---
    trash_scatter = ax.scatter(
        [t['x'] for t in initial_trash],
        [t['y'] for t in initial_trash],
        c='#EF4444', s=55, zorder=4, marker='o',
        edgecolors='white', linewidths=0.8, label='trash'
    )

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor='#D0E8FF', edgecolor='gray',
                       label='Zone 1 (rate=0.02)'),
        mpatches.Patch(facecolor='#D4EDDA', edgecolor='gray',
                       label='Zone 2 (rate=0.06)'),
        mpatches.Patch(facecolor='#FFE8C0', edgecolor='gray',
                       label='Zone 3 (rate=0.12)'),
        mpatches.Patch(facecolor='#737373', edgecolor='gray',
                       label='Obstacle'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#EF4444',
                   markersize=8, label='Trash'),
    ]
    for i in range(num_robots):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        legend_elements.append(
            plt.Line2D([0],[0], marker='s', color='w',
                       markerfacecolor=color, markersize=8,
                       label=f'Robot {i}')
        )
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=8, framealpha=0.85,
              facecolor='white', edgecolor='#cccccc')

    ax.set_title('Multi-Robot Waste Collection', color='white',
                 fontsize=13, pad=10)

    if not has_trajectories:
        ax.set_title('Multi-Robot Waste Collection — Static Map',
                     color='white', fontsize=13, pad=10)
        plt.tight_layout()
        plt.savefig('map_preview.png', dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.show()
        sys.exit(0)

    # -------------------------------------------------------------------
    # Animation (trajectory playback)
    # -------------------------------------------------------------------
    max_frames = max(len(t) for t in trajectories)

    # Robot trail lines + current position markers
    trail_lines = []
    robot_markers = []
    for i in range(num_robots):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        line, = ax.plot([], [], '-', color=color, lw=1.5,
                        alpha=0.6, zorder=3)
        marker, = ax.plot([], [], 'o', color=color, markersize=11,
                          markeredgecolor='white', markeredgewidth=1.5, zorder=6)
        trail_lines.append(line)
        robot_markers.append(marker)

    score_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes,
        color='white', fontsize=10, va='top',
        bbox=dict(facecolor='#00000066', edgecolor='none', pad=4)
    )

    def init():
        for line in trail_lines: line.set_data([], [])
        for marker in robot_markers: marker.set_data([], [])
        score_text.set_text('')
        return trail_lines + robot_markers + [score_text]

    def update(frame):
        frame_t = frame * SPEEDUP
        totals = []

        for i, traj in enumerate(trajectories):
            # Slice up to current frame
            sub = [p for p in traj if p['t'] <= frame_t]
            if not sub:
                trail_lines[i].set_data([], [])
                robot_markers[i].set_data([], [])
                totals.append(0)
                continue

            xs = [p['x'] for p in sub]
            ys = [p['y'] for p in sub]
            trail_lines[i].set_data(xs, ys)
            robot_markers[i].set_data([xs[-1]], [ys[-1]])
            totals.append(sub[-1].get('collected', 0))

        score_lines = [f'R{i}: {c}' for i, c in enumerate(totals)]
        score_lines.append(f'Total: {sum(totals)}')
        score_lines.append(f't = {frame_t}')
        score_text.set_text('\n'.join(score_lines))

        return trail_lines + robot_markers + [score_text]

    ani = FuncAnimation(
        fig, update,
        frames=(max_frames - 1) // SPEEDUP,
        init_func=init,
        blit=False,
        interval=50
    )

    plt.tight_layout()
    plt.show()
    ani.save("simulation.gif", writer='pillow', fps=20)
    print("Saved simulation.gif")