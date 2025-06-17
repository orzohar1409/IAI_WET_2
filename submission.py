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
        #self.depths = [] # DEBUG

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
                    #self.depths.append(depth) # DEBUG
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
        #self.depths.append(depth) # DEBUG
        #print(f'Depths:{self.depths}, of length {len(self.depths)}') # DEBUG
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
    def __init__(self):
        self.max_agent_id = None
        self.min_agent_id = None
        #self.depths = [] # DEBUG

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
                    #self.depths.append(depth) # DEBUG
                    return best_action_overall or best_action_at_depth
                cloned_env = env.clone()
                cloned_env.apply_operator(agent_id, action)

                # score = self.min_value(cloned_env, depth - 1, start_time, time_limit)  # OLD
                score = self.min_value(cloned_env, depth - 1, start_time, time_limit,
                                       alpha=float('-inf'), beta=float('inf'))  # ADDED alpha-beta

                if score > best_score:
                    best_score = score
                    best_action_at_depth = action

            if not self.time_out(start_time, time_limit):
                best_action_overall = best_action_at_depth
                depth += 1
        #self.depths.append(depth) # DEBUG
        #print(f'Depths:{self.depths}, of length {len(self.depths)}') # DEBUG
        return best_action_overall
# Depths:[10, 10, 9, 10, 10, 10, 10, 10, 9, 10, 9, 9, 9, 9, 9, 9, 9, 10, 13, 15, 26, 24, 22, 23, 21, 19, 17, 16, 16, 16, 16, 17, 16, 17, 17, 18, 17, 17, 19, 20, 19, 20, 23, 26, 27, 25, 23, 21, 19, 20, 18, 17, 16, 16, 16, 15, 15, 16, 15, 16, 16, 16, 17, 17, 18, 17, 17, 16, 15, 16, 15, 18, 18, 21, 19, 23, 21, 21263], of length 78
# Depths:[12, 12, 11, 12, 13, 11, 11, 11, 10, 11, 10, 10, 10, 10, 10, 10, 10, 11, 18, 54, 413], of length 21
    def max_value(self, env, depth, start_time, time_limit, alpha, beta):  # ADDED alpha, beta
        if env.done() or depth == 0 or self.time_out(start_time, time_limit):
            return self.heuristic(env, self.max_agent_id)

        v = float('-inf')
        for action in env.get_legal_operators(self.max_agent_id):
            cloned_env = env.clone()
            cloned_env.apply_operator(self.max_agent_id, action)

            # v = max(v, self.min_value(cloned_env, depth - 1, start_time, time_limit))  # OLD
            v = max(v, self.min_value(cloned_env, depth - 1, start_time, time_limit, alpha, beta))  # ADDED

            if v >= beta:  # ADDED
                return v  # ADDED
            alpha = max(alpha, v)  # ADDED

        return v

    def min_value(self, env, depth, start_time, time_limit, alpha, beta):  # ADDED alpha, beta
        if env.done() or depth == 0 or self.time_out(start_time, time_limit):
            return self.heuristic(env, self.max_agent_id)

        v = float('inf')
        for action in env.get_legal_operators(self.min_agent_id):
            cloned_env = env.clone()
            cloned_env.apply_operator(self.min_agent_id, action)

            # v = min(v, self.max_value(cloned_env, depth - 1, start_time, time_limit))  # OLD
            v = min(v, self.max_value(cloned_env, depth - 1, start_time, time_limit, alpha, beta))  # ADDED

            if v <= alpha:  # ADDED
                return v  # ADDED
            beta = min(beta, v)  # ADDED

        return v

    def heuristic(self, env, agent_id):
        return smart_heuristic(env, agent_id)


class AgentExpectimax(Agent):
    def __init__(self):
        self.max_agent_id = None
        self.min_agent_id = None
        self.max_depth = 2 # initial depth
        self.start_time = None
        self.time_limit = None
    def time_out(self):
        return (time.time() - self.start_time) > (self.time_limit - 0.01)

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit
        self.max_agent_id = agent_id
        self.min_agent_id = 1 if agent_id == 0 else 0  # Assuming two agents, 0 and 1
        best_action = None

        depth = 2
        while True:
            temp_best_action = None
            temp_best_score = float('-inf')
            for action in env.get_legal_operators(agent_id):
                if self.time_out():
                    return best_action
                cloned_env = env.clone()
                cloned_env.apply_operator(agent_id, action)
                score = self.expecti_value(cloned_env, depth - 1)
                if score > temp_best_score:
                    temp_best_score = score
                    temp_best_action = action
            best_action = temp_best_action
            depth += 1

    def expecti_value(self, env, depth):
        if env.done() or depth == 0 or self.time_out():
            return self.heuristic(env, self.max_agent_id)

        expected_value = 0
        actions = env.get_legal_operators(self.min_agent_id)
        weights = []
        for action in actions:
            if action == "move east" or action == "pick up":
                weights.append(2)
            else:
                weights.append(1)

        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        for action, prob in zip(actions, probabilities):
            cloned_env = env.clone()
            cloned_env.apply_operator(self.min_agent_id, action)
            expected_value += prob * self.max_value(cloned_env, depth - 1)

        return expected_value

    def max_value(self, env, depth):
        if env.done() or depth == 0 or self.time_out():
            return self.heuristic(env, self.max_agent_id)

        max_score = float('-inf')
        for action in env.get_legal_operators(self.max_agent_id):
            cloned_env = env.clone()
            cloned_env.apply_operator(self.max_agent_id, action)
            score = self.expecti_value(cloned_env, depth - 1)
            max_score = max(max_score, score)
        return max_score

    def heuristic(self, env, agent_id):
        return smart_heuristic(env, agent_id)
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
