from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
from math import log


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    opponent = env.get_robot((robot_id + 1) % 2)

    credit = robot.credit
    battery = robot.battery
    holding_package = robot.package is not None

    position = robot.position
    LOW_BATTERY_THRESHOLD = 5  # threshold for low battery
    distance_to_drop_off = 0
    distance_to_pick_up = 0
    inv_distance_to_drop_off = 0
    inv_distance_to_pick_up = 0
    # === Holding package: plan to drop off ===
    if holding_package:
        drop_pos = robot.package.destination
        distance_to_drop_off = manhattan_distance(position, drop_pos)
        inv_distance_to_drop_off = 1 / pow(distance_to_drop_off + 1, 0.5) if distance_to_drop_off != float('inf') else 0
    else:
        # Plan to pick up package we can reach first
        best_distance = float('inf')
        my_min_dist = float('inf')
        for package in env.packages:
            if package.on_board:
                my_dist = manhattan_distance(position, package.position)
                my_min_dist = min(my_min_dist, my_dist)
        # in case haven't found any reachable package prefer to go to the middle
        inv_distance_to_pick_up = 1 / (my_min_dist + 1)  if my_min_dist != float('inf') else 0

    # === Charging need ===
    charge_inv_distance = 0
    needs_charging = battery < LOW_BATTERY_THRESHOLD
    if needs_charging:
        min_dist_to_charge = float('inf')
        for charge_station in env.charge_stations:
            dist = manhattan_distance(position, charge_station.position)
            min_dist_to_charge = min(min_dist_to_charge, dist)
        charge_inv_distance = 1 / (min_dist_to_charge + 1) if min_dist_to_charge != float('inf') else 0

    if holding_package and not needs_charging:
        # If holding package, prefer to drop it off
        return inv_distance_to_drop_off
    if not holding_package and not needs_charging:
            # If no distance to pick up, prefer to drop off if possible
        return credit * 10 + inv_distance_to_pick_up
    if needs_charging:
        # If low on battery, prefer to charge
        return charge_inv_distance * battery



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
