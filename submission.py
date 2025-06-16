from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
from math import log


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    position = robot.position
    credits = robot.credit
    battery = robot.battery

    CREDIT_FACTOR = 10
    PICKUP_FACTOR = 1
    BATTERY_FACTOR = 4
    DIST_TO_CLOSEST_PACK_FACTOR = 1
    DIST_TO_DESTINATION_FACTOR = 1
    END_FACTOR = 1/2
    RANDOM_FACTOR = random.random() * 0.1

    closest_pickup_distance = float('inf')
    for p in env.packages:
        if p.on_board and manhattan_distance(position,p.position) < closest_pickup_distance:
            closest_pickup_distance = manhattan_distance(position,p.position)

    ADJUSTED_BATTERY = battery * BATTERY_FACTOR
    ADJUSTED_CREDITS = credits * CREDIT_FACTOR

    if battery == 0: # Robot avoids ending the game, unless it has a lot of points
        return ADJUSTED_CREDITS * END_FACTOR + RANDOM_FACTOR

    if robot.package:
        # Robot holds a package
        destination_distance = manhattan_distance(position, robot.package.destination)
        return ADJUSTED_CREDITS + ADJUSTED_BATTERY + DIST_TO_DESTINATION_FACTOR / (destination_distance + 1) + RANDOM_FACTOR

    else:
        # Robot doesn't hold a package
        return ADJUSTED_CREDITS + ADJUSTED_BATTERY + DIST_TO_CLOSEST_PACK_FACTOR/(closest_pickup_distance+1) + RANDOM_FACTOR




class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def __init__(self):
        self.max_agent_id = None
        self.min_agent_id = None

    def time_out(self, start_time, time_limit):
        return (time.time() - start_time) > (time_limit - 0.01)

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        self.max_agent_id = agent_id
        self.min_agent_id = 1 if self.max_agent_id == 0 else 0

        best_action_overall = None
        depth = 1

        while not self.time_out(start_time, time_limit):
            best_action_at_depth = None
            best_score = float('-inf')
            legal_actions = env.get_legal_operators(agent_id)

            for action in legal_actions:
                if self.time_out(start_time, time_limit):
                    return best_action_overall or best_action_at_depth
                cloned_env = env.clone()
                cloned_env.apply_operator(agent_id, action)
                score = self.min_value(cloned_env, depth - 1, start_time, time_limit)

                if score > best_score:
                    best_score = score
                    best_action_at_depth = action

            if not self.time_out(start_time, time_limit):
                best_action_overall = best_action_at_depth
                depth += 1  # safely increase depth for next round

        return best_action_overall

    def max_value(self, env, depth, start_time, time_limit):
        if env.done() or depth == 0 or self.time_out(start_time, time_limit):
            return self.heuristic(env, self.max_agent_id)

        v = float('-inf')
        for action in env.get_legal_operators(self.max_agent_id):
            cloned_env = env.clone()
            cloned_env.apply_operator(self.max_agent_id, action)
            v = max(v, self.min_value(cloned_env, depth - 1, start_time, time_limit))
        return v

    def min_value(self, env, depth, start_time, time_limit):
        if env.done() or depth == 0 or self.time_out(start_time, time_limit):
            return self.heuristic(env, self.max_agent_id)

        v = float('inf')
        for action in env.get_legal_operators(self.min_agent_id):
            cloned_env = env.clone()
            cloned_env.apply_operator(self.min_agent_id, action)
            v = min(v, self.max_value(cloned_env, depth - 1, start_time, time_limit))
        return v

    def heuristic(self, env, agent_id):
        return smart_heuristic(env, agent_id)


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
