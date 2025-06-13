from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random



# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    heuristic_value = float('-inf')
    # Check the closest package for the robot
    credit_factor = max(1, robot.credit)
    if robot.package is None:
        closest_package_distance = float('inf')
        for package in env.packages:
            if package.on_board:
                my_distance = manhattan_distance(robot.position, package.position)
                other_robot_distance = manhattan_distance(other_robot.position, package.position)
                if my_distance < other_robot_distance:
                    closest_package_distance = min(closest_package_distance, my_distance)
        heuristic_value = (1 / (closest_package_distance + 1)) * credit_factor * 3
    else:
        distance_to_drop_off = manhattan_distance(robot.position, robot.package.destination)
        package_reward = manhattan_distance(robot.package.position, robot.package.destination) * 2
        heuristic_value = package_reward / (distance_to_drop_off + 1) * credit_factor


    return heuristic_value

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)