from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    opponent = env.get_robot((robot_id + 1) % 2)

    credit = robot.credit
    battery = robot.battery
    holding_package = robot.package is not None

    position = robot.position

    # === Parameters ==
    LOW_BATTERY_THRESHOLD = 8
    w_credit = 100
    w_battery = 1
    w_holding_bonus = 300
    w_target_inv_distance = 10
    w_charge_inv_distance = 1000

    # === Holding package: plan to drop off ===
    if holding_package:
        drop_pos = robot.package.destination
        distance_to_target = manhattan_distance(position, drop_pos)
    else:
        # Plan to pick up package we can reach first
        best_distance = float('inf')
        my_min_dist = float('inf')
        for package in env.packages:
            if package.on_board:
                my_dist = manhattan_distance(position, package.position)
                opp_dist = manhattan_distance(opponent.position, package.position)
                my_min_dist = min(my_min_dist, my_dist)
                if my_dist < opp_dist:
                    best_distance = min(best_distance, my_dist)
        # in case haven't found any reachable package prefer to go to the middle
        distance_to_target = best_distance if best_distance != float('inf') else my_min_dist

    # === Charging need ===
    charge_inv_distance = 0
    if battery < LOW_BATTERY_THRESHOLD:
        min_dist_to_charge = float('inf')
        for charge_station in env.charge_stations:
            dist = manhattan_distance(position, charge_station.position)
            min_dist_to_charge = min(min_dist_to_charge, dist)
        charge_inv_distance = 1 / (min_dist_to_charge + 1)

    # === Combine ===
    heuristic = (
            w_credit * credit
            + w_battery * battery
            + (w_holding_bonus if holding_package else 1) * w_target_inv_distance * (1 / (pow(distance_to_target, 0.1) + 1))
            + w_charge_inv_distance * charge_inv_distance
    )

    return heuristic


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def __init__(self, ):
        self.max_agent_id = None
        self.min_agent_id = None

    def time_out(self, start_time, time_limit):
        # leave 0.01 s to return from recusive calls
        return time.time() - start_time > time_limit - 0.01

    def run_step(self, env, max_agent_id, time_limit):
        start_time = time.time()
        self.max_agent_id = max_agent_id
        self.min_agent_id = 1 if self.max_agent_id == 0 else 0

        best_score = float('-inf')
        best_action = None
        legal_actions = env.get_legal_operators(self.max_agent_id)
        for action in legal_actions:
            if self.time_out(start_time, time_limit):
                return best_action
            cloned_env = env.clone()
            cloned_env.apply_operator(self.max_agent_id, action)
            score = self.min_value(cloned_env, start_time, time_limit)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def max_value(self, env, start_time, time_limit):
        if env.done():
            return self.utility(env, )
        if time.time() - start_time > time_limit:
            return self.heuristic(env, self.max_agent_id)

        maximum_score = float('-inf')
        for action in env.get_legal_operators(self.max_agent_id):
            if self.time_out(start_time, time_limit):
                return self.heuristic(env, self.max_agent_id)
            cloned_env = env.clone()
            cloned_env.apply_operator(self.max_agent_id, action)

            score = self.min_value(cloned_env, start_time, time_limit)
            if score > maximum_score:
                maximum_score = score
        return maximum_score

    def min_value(self, env, start_time, time_limit):
        if env.done():
            return self.utility(env)
        if time.time() - start_time > time_limit:
            return (-1) * self.heuristic(env, self.min_agent_id)

        minimum_score = float('inf')
        for action in env.get_legal_operators(self.min_agent_id):
            if self.time_out(start_time, time_limit):
                return self.heuristic(env, self.min_agent_id)
            cloned_env = env.clone()
            cloned_env.apply_operator(self.min_agent_id, action)
            score = self.max_value(cloned_env, start_time, time_limit)
            if score < minimum_score:
                minimum_score = score
        return minimum_score

    def heuristic(self, env, agent_id):
        # Use the smart heuristic defined earlier
        return smart_heuristic(env, agent_id)

    def utility(self, env):
        maximizer_credit = env.get_robot(self.max_agent_id).credit
        minimizer_credit = env.get_robot(self.min_agent_id).credit
        if maximizer_credit > minimizer_credit:
            return float('inf')
        elif minimizer_credit > maximizer_credit:
            return float('-inf')
        else:
            return 0


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
