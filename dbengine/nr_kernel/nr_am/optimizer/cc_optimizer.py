import json
import os
import sys
import time
import warnings

import nevergrad as ng
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import parse, run

warnings.simplefilter(action="ignore", category=FutureWarning)


class CCLearner(object):
    def setup(self, k, v):
        assert k in ["rank", "access", "timeout"]
        self.setting[k] = v
        self.set_bounds()

    def update_setting(self, setting):
        # accept only keys we care about
        self.setting = {
            "access": bool(setting.get("access", False)),
            "rank": bool(setting.get("rank", False)),
            "timeout": bool(setting.get("timeout", False)),
        }
        self.set_bounds()

    def set_bounds(self):
        check_length = 0

        # Access policy parameters (binary 0/1 per state)
        access_parameters = []
        if self.setting["access"]:
            access_parameters.extend([ng.p.Choice([0, 1], repetitions=self.max_state)])
            check_length += self.max_state

        # Priority policy parameters ([0,1] per state)
        rank_parameters = []
        if self.setting["rank"]:
            rank_parameters.extend(
                [
                    ng.p.Scalar(lower=0, upper=1).set_mutation(sigma=0.2, exponent=None)
                    for _ in range(self.max_state)
                ]
            )
            check_length += self.max_state

        # Timeout policy parameters (global knobs + per-state timeouts)
        timeout_parameters = []
        if self.setting["timeout"]:
            timeout_parameters.extend(
                [
                    ng.p.Scalar(lower=1, upper=100000, init=100_000)
                    for _ in range(self.max_state)
                ]
            )
            check_length += self.max_state

        self.bounds = ng.p.Instrumentation(
            access=ng.p.Tuple(*access_parameters),
            rank=ng.p.Tuple(*rank_parameters),
            timeout=ng.p.Tuple(*timeout_parameters),
        )
        self.check_encoder_length = check_length

    def __init__(
        self,
        base_command,
        name,
        log_dir,
        starting_points,
        max_state,
        seed,
        log_rate=1,
        _runtime=1,
        setting=None,
    ):
        if setting is None:
            setting = {"rank": False, "access": False, "timeout": False}

        self.max_state = max_state
        self.seed = seed
        self.current_iter = 0
        self.patience = 0

        self.setting = {"rank": False, "access": False, "timeout": False}
        self.update_setting(setting)

        self.best_seen_performance = 0
        self.log_rate = log_rate
        self.start_time = time.time()
        self.db_runtime = _runtime
        self.training_stage = 0
        self.best_policy = None
        self.evaluated_history = []
        self.no_update_count = 0
        self.base_command = base_command
        self.name = name
        self.log_dir = log_dir
        self.starting_points = starting_points

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def load_initial_policy_from_file(self, files):
        self.starting_points = []
        for p_dir in files:
            policy = Policy(_from=self, load_file=p_dir)
            self.starting_points.append(policy)

    def save_model(self, policy, name):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        base_path = os.path.join(os.getcwd(), self.log_dir)
        recent_path = os.path.join(base_path, f"{name}_new.txt")
        last_path = os.path.join(base_path, f"{name}_old.txt")
        if os.path.exists(last_path):
            os.remove(last_path)
        if os.path.exists(recent_path):
            os.rename(recent_path, last_path)
        policy.save_to_path(recent_path)

    def evaluate_policy(self, policy):
        base_dir = "./optimizer/bo-steps/"
        os.makedirs(base_dir, exist_ok=True)
        recent_path = os.path.join(base_dir, f"step_{self.current_iter}.txt")
        if os.path.exists(recent_path):
            os.remove(recent_path)
        policy.save_to_path(recent_path)
        self.current_iter += 1

        command = self.base_command
        command.append(f"--policy {recent_path}")
        sys.stdout.flush()
        # print("running = ", " ".join(command))
        run_results = parse(run(" ".join(command), die_after=180))
        if run_results[0] == 0:
            print("panic: the running has been blocked for more than 10s")
        command.pop()
        current_score = run_results[0]
        policy.score = current_score
        self.evaluated_history.append(policy)

        if current_score > self.best_seen_performance:
            self.no_update_count = 0
            print(
                "Optimizer %s found better cc policy in iteration %d, spent time %f: %d TPS"
                % (
                    self.name,
                    self.current_iter - 1,
                    time.time() - self.start_time,
                    current_score,
                )
            )
            self.best_seen_performance = current_score
            self.best_policy = policy
            self.save_model(policy, f"bo{self.current_iter}")
        else:
            self.no_update_count += 1

        if self.current_iter % self.log_rate == 0:
            self.writer.add_scalar(
                "best-seen", self.best_seen_performance, self.current_iter
            )
            self.writer.add_scalar("current-seen", current_score, self.current_iter)
            self.writer.flush()
        return current_score

    def close(self):
        self.writer.close()


class Policy(object):
    def __init__(
        self,
        _access=None,
        _rank=None,
        _timeout=None,
        _from=None,
        load_file=None,
        encoded=None,
    ):
        self.score = -1
        self.learner = _from
        self.max_state = self.learner.max_state

        if load_file is not None:
            with open(load_file, "r") as f:
                self.read_from_file(f)
        elif encoded is not None:
            self.decode(encoded)
        else:
            # interactive-mode policies
            self.access = np.array(
                _access if _access is not None else np.zeros(self.max_state, dtype=int)
            )
            self.rank = np.array(
                _rank if _rank is not None else np.zeros(self.max_state, dtype=float)
            )
            self.timeout_policy = np.array(
                _timeout if _timeout is not None else np.full(self.max_state, 100000.0)
            )

        self._hash = hash(self.__str__())

    def float_correction(self):
        pass

    def write_to_file(self, f_out):
        """Emit one line per state:
        idx detect_all priority timeout
        - idx: 0..max_state-1
        - detect_all: 0/1  (from self.access)
        - priority: float  (from self.rank)
        - timeout:  uint   (from self.timeout_policy; use same unit as engine expects)
        """
        self.float_correction()
        for i in range(self.max_state):
            detect = int(self.access[i])
            prio = float(self.rank[i])
            tout = int(self.timeout_policy[i])
            f_out.write(f"{i} {detect} {prio:.6f} {tout}\n")

    def read_from_file(self, file):
        """Read either 4-field lines (idx detect prio timeout) or 3-field lines
        (detect prio timeout), filling sequentially from idx 0 when index is omitted.
        Lines starting with '#' or blank are ignored.
        """
        n = self.learner.max_state

        # Defaults if file has fewer than n entries
        access = np.zeros(n, dtype=int)
        rank = np.zeros(n, dtype=float)
        tout = np.full(n, 100000, dtype=int)  # pick your default ms if needed

        cursor = 0
        for raw in file:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            assert len(parts) == 4, f"Expected 4 fields, got {len(parts)}: {line}"
            idx = int(parts[0])
            detect = int(parts[1])
            prio = float(parts[2])
            to = int(float(parts[3]))
            if 0 <= idx < n:
                access[idx] = 1 if detect != 0 else 0
                rank[idx] = prio
                tout[idx] = to

        self.access = access
        self.rank = rank
        self.timeout_policy = tout

    def save_to_path(self, path):
        with open(path, "w") as f:
            self.write_to_file(f)

    def encode(self):
        """Encode only access/rank/timeout into the parameter dictionary."""
        if self.learner.best_policy is None:
            self.learner.best_policy = self

        access_params = ()
        rank_params = ()
        timeout_params = ()

        if self.learner.setting["access"]:
            access_params = (tuple(self.access),)
        if self.learner.setting["rank"]:
            rank_params = tuple(self.rank)
        if self.learner.setting["timeout"]:
            timeout_params = tuple(
                np.concatenate((self.extra_policies, self.timeout_policy))
            )

        return {
            "access": access_params,
            "rank": rank_params,
            "timeout": timeout_params,
        }

    def decode(self, param_dict):
        """Decode access/rank/timeout; provide sane defaults if no best_policy yet."""
        bp = self.learner.best_policy

        # ACCESS
        if self.learner.setting["access"]:
            access_values = param_dict.get("access", None)[0]
            assert access_values is not None, "Expected access_values"
            self.access = np.array(access_values, dtype=int)
        else:
            self.access = (
                bp.access.copy()
                if (bp is not None and hasattr(bp, "access"))
                else np.zeros(self.max_state, dtype=int)
            )

        # RANK
        if self.learner.setting["rank"]:
            rank_values = param_dict.get("rank", None)
            assert rank_values is not None, "Expected rank_values"
            self.rank = np.array(rank_values, dtype=float)
        else:
            self.rank = (
                bp.rank.copy()
                if (bp is not None and hasattr(bp, "rank"))
                else np.zeros(self.max_state, dtype=float)
            )

        # TIMEOUT
        if self.learner.setting["timeout"]:
            timeout_values = param_dict.get("timeout", None)
            assert timeout_values is not None, "Expected timeout_values"
            self.timeout_policy = np.array(
                timeout_values[-self.max_state :], dtype=float
            )
            self.extra_policies = np.array(timeout_values[: -self.max_state], dtype=int)
        else:
            if bp is not None and hasattr(bp, "timeout_policy"):
                self.timeout_policy = bp.timeout_policy.copy()
                self.extra_policies = bp.extra_policies.copy()
            else:
                self.timeout_policy = np.full(self.max_state, 100000.0, dtype=float)
                self.extra_policies = np.array(
                    [32] + [2 for _ in range(6 * N_TXN_TYPE)], dtype=int
                )

    def hash(self):
        return hash(json.dumps(self.encode(), sort_keys=True, default=convert_np))


def convert_np(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
