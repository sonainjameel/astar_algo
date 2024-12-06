#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from create_plan_msgs.srv import CreatePlan
from nav2_simple_commander.robot_navigator import BasicNavigator
import numpy as np
from heapq import heappush, heappop


class AStar:
    def __init__(self, start, goal, wheelRPM, clearance, costmap):
        self.start = start
        self.goal = goal
        self.wheelRPM = wheelRPM
        self.clearance = clearance
        self.costmap = costmap
        self.costmap_resolution = costmap.info.resolution
        self.costmap_origin = (costmap.info.origin.position.x, costmap.info.origin.position.y)
        self.xLength = costmap.info.width * self.costmap_resolution
        self.yLength = costmap.info.height * self.costmap_resolution
        self.distance = {}
        self.path = {}
        self.costToCome = {}
        self.costToGo = {}
        self.goalThreshold = 15
        self.frequency = 100

    def IsValid(self, currX, currY):
        """Check if a point is within the bounds of the costmap."""
        return (
            currX >= self.costmap_origin[0]
            and currX < self.costmap_origin[0] + self.xLength
            and currY >= self.costmap_origin[1]
            and currY < self.costmap_origin[1] + self.yLength
        )

    def IsObstacle(self, x, y):
        """Check if a point is an obstacle using the costmap."""
        x_idx = int((x - self.costmap_origin[0]) / self.costmap_resolution)
        y_idx = int((y - self.costmap_origin[1]) / self.costmap_resolution)

        if x_idx < 0 or x_idx >= self.costmap.info.width or y_idx < 0 or y_idx >= self.costmap.info.height:
            return True  # Out of bounds is treated as an obstacle

        cost = self.costmap.data[y_idx * self.costmap.info.width + x_idx]
        return cost > 50  # Cells with cost > 50 are treated as obstacles

    def euc_heuristic(self, x, y, weight=3.0):
        return weight * ((self.goal[0] - x) ** 2 + (self.goal[1] - y) ** 2) ** 0.5

    def GetNewPositionOfRobot(self, node, leftRPM, rightRPM):
        leftAngularVelocity = leftRPM * 2 * np.pi / 60.0
        rightAngularVelocity = rightRPM * 2 * np.pi / 60.0
        x, y, theta = node
        w = (0.5 * (rightAngularVelocity - leftAngularVelocity))
        for _ in range(self.frequency):
            dvx = 0.5 * (leftAngularVelocity + rightAngularVelocity) * np.cos(theta)
            dvy = 0.5 * (leftAngularVelocity + rightAngularVelocity) * np.sin(theta)
            x += dvx / self.frequency
            y += dvy / self.frequency
            theta += w / self.frequency
            if not self.IsValid(x, y) or self.IsObstacle(x, y):
                return (x, y, theta, float('inf'), False)
        return (x, y, theta, ((x - node[0]) ** 2 + (y - node[1]) ** 2) ** 0.5, True)

    def search(self):
        exploredStates = []
        queue = []
        self.costToCome[self.start] = 0
        self.costToGo[self.start] = self.euc_heuristic(self.start[0], self.start[1])
        self.distance[self.start] = self.costToCome[self.start] + self.costToGo[self.start]
        heappush(queue, (self.distance[self.start], self.start))
        while queue:
            _, current = heappop(queue)
            if (current[0] - self.goal[0]) ** 2 + (current[1] - self.goal[1]) ** 2 <= self.goalThreshold ** 2:
                backtrack = []
                while current != self.start:
                    backtrack.append(current)
                    current = self.path[current][0]
                backtrack.append(self.start)
                return exploredStates, backtrack[::-1]
            for leftRPM, rightRPM in [(0, self.wheelRPM[0]), (self.wheelRPM[0], 0), (self.wheelRPM[1], self.wheelRPM[1])]:
                _, newX, newY, newTheta, cost, flag = self.GetNewPositionOfRobot(current, leftRPM, rightRPM)
                if flag:
                    newCostToCome = self.costToCome[current] + cost
                    newCostToGo = self.euc_heuristic(newX, newY)
                    newDistance = newCostToCome + newCostToGo
                    newState = (newX, newY, newTheta)
                    if self.distance.get(newState, float('inf')) > newDistance:
                        self.distance[newState] = newDistance
                        self.costToCome[newState] = newCostToCome
                        self.costToGo[newState] = newCostToGo
                        self.path[newState] = (current, cost)
                        heappush(queue, (newDistance, newState))
            exploredStates.append(current)
        return exploredStates, []

class PathPlannerNode(Node):

    def __init__(self):
        super().__init__("path_planner_node")
        self.srv = self.create_service(CreatePlan, 'create_plan', self.create_plan_cb)

    def create_plan_cb(self, request, response):
        goal_pose = request.goal
        start_pose = request.start
        time_now = self.get_clock().now().to_msg()
        global_costmap = self.get_global_costmap()
        response.path = self.create_astar_plan(start_pose, goal_pose, time_now, global_costmap)
        return response

    def get_global_costmap(self):
        """Retrieve the global costmap."""
        navigator = BasicNavigator()
        costmap = navigator.getGlobalCostmap()
        if costmap is None:
            self.get_logger().error("Failed to retrieve the global costmap.")
        return costmap

    def create_astar_plan(self, start, goal, time_now, costmap):
        path = Path()
        path.header.frame_id = goal.header.frame_id
        path.header.stamp = time_now

        start_coords = (start.pose.position.x, start.pose.position.y, 0.0)
        goal_coords = (goal.pose.position.x, goal.pose.position.y)
        astar = AStar(start_coords, goal_coords, (100, 50), clearance=50, costmap=costmap)

        if not astar.IsValid(start_coords[0], start_coords[1]) or astar.IsObstacle(start_coords[0], start_coords[1]):
            self.get_logger().error("Start point is invalid or inside an obstacle.")
            return path
        if not astar.IsValid(goal_coords[0], goal_coords[1]) or astar.IsObstacle(goal_coords[0], goal_coords[1]):
            self.get_logger().error("Goal point is invalid or inside an obstacle.")
            return path

        explored_states, backtrack_states = astar.search()
        if not backtrack_states:
            self.get_logger().error("No valid path found by A*.")
            return path

        for state in backtrack_states:
            pose = PoseStamped()
            pose.pose.position.x = state[0]
            pose.pose.position.y = state[1]
            pose.header.stamp = time_now
            pose.header.frame_id = goal.header.frame_id
            path.poses.append(pose)

        return path

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
