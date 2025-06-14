from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)

    # We'll build the heuristic step by step
    heuristic_value = 0

    credit = robot.credit
    battery = robot.battery
    credit_factor = max(1, credit) * 100
    holding_package_factor = 10 if robot.package is not None else 0

    if robot.package is None:
        # Not holding a package: go get one
        closest_package_distance = float('inf')
        for package in env.packages:
            if package.on_board:
                my_distance = manhattan_distance(robot.position, package.position)
                other_distance = manhattan_distance(other_robot.position, package.position)
                # Prefer packages we can reach firstס
                if my_distance < other_distance:
                    closest_package_distance = min(closest_package_distance, my_distance)
        if closest_package_distance < float('inf'):
            heuristic_value += (1 / (closest_package_distance + 1))# weight can be tuned
    else:
        # Holding a package: go to drop-off
        distance_to_drop_off = manhattan_distance(robot.position, robot.package.destination)
        package_value = manhattan_distance(robot.package.position, robot.package.destination) * 2

        # Encourage going toward drop-off
        heuristic_value += (package_value / (distance_to_drop_off + 1)) * 5  # reward for progress

        # Strong bonus if standing on drop-off location and able to drop

    return heuristic_value + credit_factor + holding_package_factor

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)



class AgentMinimax(Agent):
    def __init__(self):
        super().__init__()
    def run_step(self, env, agent_id, time_limit):
        start_time = time.time()
        best_score = float('-inf')
        best_action = None
        legal_actions = env.get_legal_operators(agent_id)

        for action in legal_actions:
            if time.time() - start_time > time_limit:
                break
            cloned_env = env.clone()
            cloned_env.apply_operator(agent_id, action)
            score = self.min_value(cloned_env, 1 - agent_id, start_time, time_limit)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def max_value(self, env, agent_id, start_time, time_limit):
        if env.done():
            return self.utility(env, self.index)
        if time.time() - start_time > time_limit:
            return self.heuristic(env, agent_id)

        v = float('-inf')
        for action in env.get_legal_operators(agent_id):
            if time.time() - start_time > time_limit:
                break
            cloned_env = env.clone()
            cloned_env.apply_operator(agent_id, action)
            v = max(v, self.min_value(cloned_env, 1 - agent_id, start_time, time_limit))
        return v

    def min_value(self, env, agent_id, start_time, time_limit):
        if env.done():
            return self.utility(env, self.index)
        if time.time() - start_time > time_limit:
            return self.heuristic(env, 1 - agent_id)

        v = float('inf')
        for action in env.get_legal_operators(agent_id):
            if time.time() - start_time > time_limit:
                break
            cloned_env = env.clone()
            cloned_env.apply_operator(agent_id, action)
            v = min(v, self.max_value(cloned_env, 1 - agent_id, start_time, time_limit))
        return v

    def heuristic(self, env, agent_id):
        # Use the smart heuristic defined earlier
        return smart_heuristic(env, agent_id)

    def utility(self, env, max_agent_id):
        my_credit = env.get_robot(max_agent_id).credit
        opp_credit = env.get_robot(1 - max_agent_id).credit
        if my_credit > opp_credit:
            return float('inf')  # win
        elif my_credit < opp_credit:
            return float('-inf')  # loss
        else:
            return 0  # draw


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