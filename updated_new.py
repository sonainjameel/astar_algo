#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from create_plan_msgs.srv import CreatePlan
from nav2_simple_commander.robot_navigator import BasicNavigator
import numpy as np
import heapq


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")
        # Create BasicNavigator instance to access the global costmap
        self.basic_navigator = BasicNavigator()

        # Create a service for handling path requests
        self.srv = self.create_service(CreatePlan, 'create_plan', self.create_plan_cb)

    def create_plan_cb(self, request, response):
        """
        Callback for the create_plan service.
        """
        # Retrieve the global costmap
        global_costmap = self.basic_navigator.getGlobalCostmap()

        if global_costmap is None:
            self.get_logger().warn("Global costmap is not available!")
            return response

        # Log costmap info for debugging
        self.get_logger().info(f"Global costmap shape: {global_costmap.shape}")

        # Use the costmap for path planning
        goal_pose = request.goal
        start_pose = request.start
        time_now = self.get_clock().now().to_msg()

        response.path = create_astar_plan(
            start_pose, goal_pose, time_now, global_costmap, resolution=0.05  # Example resolution
        )
        return response


def create_astar_plan(start, goal, time_now, global_costmap, resolution):
    """
    Creates a path using the A* algorithm with the provided global costmap.
    """
    path = Path()
    path.header.frame_id = goal.header.frame_id
    path.header.stamp = time_now

    start_node = world_to_grid(start.pose.position.x, start.pose.position.y, global_costmap.shape, resolution)
    goal_node = world_to_grid(goal.pose.position.x, goal.pose.position.y, global_costmap.shape, resolution)

    if start_node is None or goal_node is None:
        raise ValueError("Start or Goal pose is outside the grid bounds!")

    # Filter the costmap: Treat -1 (unknown) and 100 (obstacle) as non-traversable
    traversable_grid = np.where((global_costmap == 0), 0, 1)

    # Perform A* path planning
    astar_path = astar(traversable_grid, start_node, goal_node)

    for node in astar_path:
        pose = PoseStamped()
        world_x, world_y = grid_to_world(node, global_costmap.shape, resolution)
        pose.pose.position.x = world_x
        pose.pose.position.y = world_y
        pose.header.stamp = time_now
        pose.header.frame_id = goal.header.frame_id
        path.poses.append(pose)

    return path


def astar(grid, start, goal):
    """
    A* algorithm implementation to find the shortest path in a grid.
    """
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        neighbors = get_neighbors(current, grid)
        for neighbor in neighbors:
            tentative_g_score = g_score[current] + 1  # Assuming uniform cost grid
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if neighbor not in [n[1] for n in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # No path found


def get_neighbors(node, grid):
    """
    Get valid neighbors for a grid cell.
    """
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    for d in directions:
        neighbor = (node[0] + d[0], node[1] + d[1])
        if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and grid[neighbor] == 0:
            neighbors.append(neighbor)
    return neighbors


def reconstruct_path(came_from, current):
    """
    Reconstruct the path from the goal to the start.
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def world_to_grid(x, y, grid_shape, resolution):
    """
    Convert world coordinates to grid indices.
    """
    grid_x = int(x / resolution)
    grid_y = int(y / resolution)
    if 0 <= grid_x < grid_shape[1] and 0 <= grid_y < grid_shape[0]:
        return grid_y, grid_x
    return None


def grid_to_world(node, grid_shape, resolution):
    """
    Convert grid indices to world coordinates.
    """
    world_x = node[1] * resolution
    world_y = node[0] * resolution
    return world_x, world_y


def main(args=None):
    rclpy.init(args=args)
    path_planner_node = PathPlannerNode()

    try:
        rclpy.spin(path_planner_node)
    except KeyboardInterrupt:
        pass

    path_planner_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
