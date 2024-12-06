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
    def __init__(self, start, goal, wheelRPM, clearance):
        self.start = start
        self.goal = goal
        self.xLength = 500
        self.yLength = 500
        self.wheelRPM = wheelRPM
        self.clearance = min(clearance + 10, 25)
        self.radius = 20.0
        self.wheelDistance = 34.0
        self.wheelRadius = 3.8
        self.distance = {}
        self.path = {}
        self.costToCome = {}
        self.costToGo = {}
        self.hashMap = {}
        self.goalThreshold = 15
        self.frequency = 100

    def IsValid(self, currX, currY):
        return (
            currX >= -self.xLength + self.radius + self.clearance
            and currX <= self.xLength - self.radius - self.clearance
            and currY >= -self.yLength + self.radius + self.clearance
            and currY <= self.yLength - self.radius - self.clearance
        )

    def IsObstacle(self, x, y):
        r = self.clearance + self.radius
        obstacles = [
            ((200, 300), 100 + r),
            ((200, -300), 100 + r),
            ((-200, 300), 100 + r),
            ((0, 0), 100 + r),
        ]
        for (ox, oy), radius in obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 <= radius ** 2:
                return True
        return False

    def euc_heuristic(self, x, y, weight=3.0):
        return weight * ((self.goal[0] - x) ** 2 + (self.goal[1] - y) ** 2) ** 0.5

    def GetNewPositionOfRobot(self, node, leftRPM, rightRPM):
        leftAngularVelocity = leftRPM * 2 * np.pi / 60.0
        rightAngularVelocity = rightRPM * 2 * np.pi / 60.0
        x, y, theta = node
        w = (self.wheelRadius / self.wheelDistance) * (rightAngularVelocity - leftAngularVelocity)
        for _ in range(self.frequency):
            dvx = self.wheelRadius * 0.5 * (leftAngularVelocity + rightAngularVelocity) * np.cos(theta)
            dvy = self.wheelRadius * 0.5 * (leftAngularVelocity + rightAngularVelocity) * np.sin(theta)
            x += dvx / self.frequency
            y += dvy / self.frequency
            theta += w / self.frequency
            if not self.IsValid(x, y) or self.IsObstacle(x, y):
                return (x, y, theta, float('inf'), False)
        return (x, y, theta, ((x - node[0]) ** 2 + (y - node[1]) ** 2) ** 0.5, True)

    def ActionMoveRobot(self, node, leftRPM, rightRPM):
        newX, newY, newTheta, cost, flag = self.GetNewPositionOfRobot(node, leftRPM, rightRPM)
        return (flag, newX, newY, newTheta, cost)

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
                flag, newX, newY, newTheta, cost = self.ActionMoveRobot(current, leftRPM, rightRPM)
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
        response.path = self.create_astar_plan(start_pose, goal_pose, time_now)
        return response

    def create_astar_plan(self, start, goal, time_now):
        path = Path()
        path.header.frame_id = goal.header.frame_id
        path.header.stamp = time_now
        start_coords = (start.pose.position.x * 100.0, start.pose.position.y * 100.0, 0.0)
        goal_coords = (goal.pose.position.x * 100.0, goal.pose.position.y * 100.0)
        astar = AStar(start_coords, goal_coords, (100, 50), 20)
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
            pose.pose.position.x = state[0] / 100.0
            pose.pose.position.y = state[1] / 100.0
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
