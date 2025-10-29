from __future__ import annotations
import math
import random
import time
from copy import copy
from math import log
from typing import List, Tuple, Iterable
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from expert_pool.join_order_expert.models.mcts_net import ValueNet


class TreeNode:
    def __init__(self, state: PlanState, parent: "TreeNode | None"):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0.0
        self.children: dict[int, TreeNode] = {}


class PlanState:
    """
    Holds incremental join-order construction state.
    """

    def __init__(
            self,
            max_hint_num: int,
            total_num_tables: int,
            num_tables_in_query: int,
            query_vec: Iterable[float],
            all_joins: List[Tuple[int, int]],
            joins_with_predicate: List[Tuple[int, int]],
            nodes: object,  # kept for parity; not used here
    ):
        self.tableNumber = total_num_tables
        self.numberOfTables = num_tables_in_query
        self.queryEncode = query_vec
        self.nodes = nodes

        # tensors on chosen device
        self.inputState1 = torch.tensor(query_vec, dtype=torch.float32)

        # rolling construction buffers
        self.currentStep = 0
        self.order_list = np.zeros(max_hint_num, dtype=int)

        # join lists normalized so (a,b) with a<b
        self.joins: List[Tuple[int, int]] = [(min(a, b), max(a, b)) for a, b in all_joins]
        self.joins_with_predicate: List[Tuple[int, int]] = [
            (min(a, b), max(a, b)) for a, b in joins_with_predicate
        ]

        # adjacency for expansion
        self.join_matrix = {}
        for u, v in all_joins:
            self.join_matrix.setdefault(u, set()).add(v)
            self.join_matrix.setdefault(v, set()).add(u)

        self.possibleActions: set[int] = set()

    def get_possible_actions(self) -> set[int]:
        # small memoization: reuse after step>1
        if self.possibleActions and self.currentStep > 1:
            return self.possibleActions

        possible = set()
        if self.currentStep == 0:
            for a, _ in self.joins_with_predicate:
                possible.add(a)
        elif self.currentStep == 1:
            first = self.order_list[0]
            for a, b in self.joins_with_predicate:
                if a == first:
                    possible.add(b)
        else:
            chosen = set(self.order_list[: self.currentStep])
            for a, b in self.joins:
                if a in chosen and b not in chosen:
                    possible.add(b)
                elif b in chosen and a not in chosen:
                    possible.add(a)

        self.possibleActions = possible
        return possible

    def take_action(self, action: int) -> "PlanState":
        newState = copy(self)
        newState.order_list = copy(self.order_list)
        newState.possibleActions = set(self.possibleActions)

        newState.order_list[newState.currentStep] = action
        newState.currentStep = self.currentStep + 1

        # update possible actions
        if action in newState.possibleActions:
            newState.possibleActions.remove(action)
        for p in newState.join_matrix.get(action, ()):
            if p not in newState.order_list[: newState.currentStep]:
                newState.possibleActions.add(p)

        return newState

    def isTerminal(self) -> bool:
        return self.currentStep == self.numberOfTables


class MCTS:
    def __init__(self, iteration_limit: int, exploration_c: float = 1 / math.sqrt(16)):
        if iteration_limit is None or iteration_limit < 1:
            raise ValueError("iteration_limit must be >= 1")
        self.searchLimit = iteration_limit
        self.explorationConstant = exploration_c
        self.root: TreeNode | None = None

        # timers (for diagnostics)
        self.total_rollout_time = 0.0
        self.total_pred_time = 0.0

    def search(self, initialState: PlanState, rollout_fn, prediction_net: ValueNet):
        self.root = TreeNode(initialState, None)
        for _ in range(self.searchLimit):
            self.execute_round(rollout_fn, prediction_net)

    def continue_search(self, rollout_fn, prediction_net: ValueNet):
        for _ in range(self.searchLimit):
            self.execute_round(rollout_fn, prediction_net)

    def execute_round(self, rollout_fn, prediction_net: ValueNet):
        node = self.select_node(self.root)
        t0 = time.time()
        node, reward, t_pred = rollout_fn(node, prediction_net)
        self.total_rollout_time += time.time() - t0
        self.total_pred_time += t_pred
        self.backpropagate(node, reward)

    def select_node(self, node: TreeNode) -> TreeNode:
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.get_best_child(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node: TreeNode) -> TreeNode:
        for action in node.state.get_possible_actions():
            if action not in node.children:
                newNode = TreeNode(node.state.take_action(action), node)
                node.children[action] = newNode
                node.isFullyExpanded = len(node.state.get_possible_actions()) == len(node.children)
                return newNode
        # Should not happen
        return node

    def backpropagate(self, node: TreeNode, reward: float):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def get_best_child(self, node: TreeNode, c: float) -> TreeNode:
        bestValue = float("-inf")
        best: list[TreeNode] = []
        for child in node.children.values():
            uct = (child.totalReward / max(1, child.numVisits)) + c * math.sqrt(
                2.0 * math.log(max(1, node.numVisits)) / max(1, child.numVisits)
            )
            if uct > bestValue:
                bestValue = uct
                best = [child]
            elif uct == bestValue:
                best.append(child)
        return random.choice(best)


class MCTSHinterSearch:
    def __init__(
            self,
            device: torch.device,
            max_alias_num: int,
            hidden_size: int,
            mcts_input_size: int,
            offset: float,
            max_time_out: float,
            mcts_v: float,
            searchFactor: int,
            try_hint_num: int,
            max_hint_num: int):

        # unpack config into attributes
        self.device = device
        self.max_alias_num = max_alias_num
        self.hidden_size = hidden_size
        self.mcts_input_size = mcts_input_size
        self.offset = offset
        self.max_time_out = max_time_out
        self.mcts_v = mcts_v
        self.searchFactor = searchFactor
        self.try_hint_num = try_hint_num
        self.max_hint_num = max_hint_num

        self.Utility: list[Tuple[np.ndarray, float]] = []
        self.total_cnt = 0

        # model
        self.prediction_net = ValueNet(
            max_hint_num=self.max_hint_num,
            in_dim=self.mcts_input_size,
            n_words=self.max_alias_num,
            hidden_size=self.hidden_size
        ).to(self.device)

        # init weights
        for _, param in self.prediction_net.named_parameters():
            if param.ndim == 2:
                init.xavier_normal_(param)
            else:
                init.uniform_(param)

        self.optimizer = optim.Adam(self.prediction_net.parameters(), lr=3e-4, betas=(0.9, 0.999))

    # ----- value transforms -----
    def _flog(self, x: float) -> float:
        num = int((log((x + self.offset) / self.max_time_out) / log(self.mcts_v)))
        den = int((log(self.offset / self.max_time_out) / log(self.mcts_v)))
        den = den if den != 0 else 1
        return num / den

    def _eflog(self, z: float) -> float:
        scale = int((log(self.offset / self.max_time_out) / log(self.mcts_v)))
        return math.e ** (z * scale * log(self.mcts_v)) * self.max_time_out

    # ----- prediction helpers -----
    def _predict_value(self, inputState1: torch.Tensor, inputState2: torch.Tensor) -> Tuple[float, float]:
        t0 = time.time()
        with torch.no_grad():
            pred = self.prediction_net(inputState1, inputState2)
        dt = time.time() - t0
        pred_np = pred.detach().cpu().numpy()[0][0] / 10.0
        return float(pred_np), dt

    # ----- rollout -----
    def rollout(self, node: TreeNode, prediction_net: ValueNet):
        while not node.isTerminal:
            actions = node.state.get_possible_actions()
            if not actions:
                raise RuntimeError("Non-terminal state has no possible actions")
            action = random.choice(list(actions))
            newNode = TreeNode(node.state.take_action(action), node)
            node.children[action] = newNode
            if len(node.state.get_possible_actions()) == len(node.children):
                node.isFullyExpanded = True
            node = newNode

        # terminal: evaluate
        inputState1 = node.state.inputState1
        order_list = torch.tensor(node.state.order_list, dtype=torch.long, device=self.device)
        reward, model_dt = self._predict_value(inputState1, order_list)
        return node, reward, model_dt

    def find_hints(
            self,
            total_num_tables: int,
            num_tables_in_query: int,
            query_vec: Iterable[float],
            all_joins: List[Tuple[int, int]],
            joins_with_predicate: List[Tuple[int, int]],
            nodes,
            depth: int = 2,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Returns top-N (order_list, utility) pairs.
        """
        random.seed(113)
        self.total_cnt += 1

        init_state = PlanState(
            max_hint_num=self.max_hint_num,
            total_num_tables=total_num_tables,
            num_tables_in_query=num_tables_in_query,
            query_vec=query_vec,
            all_joins=all_joins,
            joins_with_predicate=joins_with_predicate,
            nodes=nodes,
        )

        iters = max(1, int(len(init_state.get_possible_actions()) * self.searchFactor))
        mcts = MCTS(iteration_limit=iters)

        self.Utility = []
        mcts.search(initialState=init_state, rollout_fn=self.rollout, prediction_net=self.prediction_net)
        self._collect_depth(mcts.root, depth)
        benefit_top = sorted(self.Utility, key=lambda x: x[1], reverse=True)
        return benefit_top[: self.try_hint_num]

    def _collect_depth(self, node: TreeNode, depth: int):
        if node.state.currentStep == depth:
            nodeValue = (node.totalReward / max(1, node.numVisits))
            self.Utility.append((node.state.order_list.copy(), self._eflog(nodeValue)))
            return
        for child in node.children.values():
            self._collect_depth(child, depth)

    # ----- training -----
    def _train_step(self, sql_feature: torch.Tensor, order_features: torch.Tensor, target: torch.Tensor) -> float:
        pred = self.prediction_net(sql_feature, order_features)
        loss_value = F.smooth_l1_loss(pred, target)
        self.optimizer.zero_grad()
        loss_value.backward()
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.clamp_(-5.0, 5.0)
        self.optimizer.step()
        return float(loss_value.item())

    def train_on_sample(self, tree_feature: torch.Tensor, sql_feature: torch.Tensor,
                        target_value_ms: float, alias_set: List[int]):
        def plan_to_alias_list(tree_feature):
            def recursive(tree_feature):
                if isinstance(tree_feature[1], tuple):
                    alias_list0 = recursive(tree_feature=tree_feature[1])
                    alias_list1 = recursive(tree_feature=tree_feature[2])
                    if len(alias_list1) == 1:
                        return alias_list0 + alias_list1
                    if len(alias_list0) == 1:
                        return alias_list1 + alias_list0
                    return []
                else:
                    return [tree_feature[1].item()]

            return recursive(tree_feature=tree_feature)

        aliases = plan_to_alias_list(tree_feature)
        if len(aliases) != len(alias_set):
            return None

        if aliases[0] > aliases[1]:
            aliases[0], aliases[1] = aliases[1], aliases[0]

        padded = aliases + [0] * (self.max_hint_num - len(aliases))
        order_features = torch.tensor(padded, dtype=torch.long, device=self.device)
        label = torch.tensor([(self._flog(target_value_ms)) * 10.0], device=self.device, dtype=torch.float32)

        loss_value = self._train_step(sql_feature, order_features, label)
        return loss_value, order_features, label

    def train_on_batch(self, samples: List):
        if not samples:
            return None
        sql_features = torch.stack([s.sql_feature for s in samples]).to(self.device)
        order_features = torch.stack([s.order_feature for s in samples]).to(self.device)
        labels = torch.stack([s.label for s in samples], dim=0).reshape(-1, 1).to(self.device)
        self._train_step(sql_features, order_features, labels)
        return
