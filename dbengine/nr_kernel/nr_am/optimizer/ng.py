import time
import json
import nevergrad as ng
from cc_optimizer import CCLearner, Policy

TIME_LIMIT = 3 * 60 * 60  # 3 hours

# Single, explicit optimizer (Gaussian Process BO with mild regularization)
BO = ng.optimizers.ParametrizedBO(gp_parameters={"alpha": 1e-2}).set_name("BO")

# Training pipeline: learn access + rank + timeout together (you can split into stages if you prefer)
training_pipeline = [
    {"rank": True, "access": True, "timeout": True, "patient": 100000, "learner": BO},
]


def benchmark_optimizer(learner: CCLearner, optimizer_class, neval: int = 100):
    """Run NG optimizer against learner; compatible with Policy.encode()/decode() that
    only exposes access/rank/timeout."""
    assert optimizer_class is not None, "Optimizer class must be provided"

    def evaluate(*args, **kwargs):
        # kwargs is the encoded dict (access/rank/timeout) from nevergrad ask()
        policy = Policy(encoded=kwargs, _from=learner)
        # learner.evaluate_policy returns a score to maximize -> NG minimizes, so return negative
        return -learner.evaluate_policy(policy)

    optimizer = optimizer_class(parametrization=learner.bounds, budget=neval)

    # Seed with prior evaluations if any (warm-start)
    seen = set()
    for p in getattr(learner, "evaluated_history", []):
        h = p.hash()
        if h in seen:
            continue
        seen.add(h)
        candidate = optimizer.parametrization.spawn_child(new_value=((), p.encode()))
        optimizer.tell(candidate, -p.score)  # NG minimizes; our score is maximize

    # Seed with starting points if provided
    if learner.training_stage == 0 and learner.starting_points:
        for p in learner.starting_points:
            candidate = optimizer.parametrization.spawn_child(new_value=((), p.encode()))
            optimizer.tell(candidate, evaluate(*candidate.args, **candidate.kwargs))
        learner.starting_points = None

    # Main loop
    for _ in range(neval):
        candidate = optimizer.ask()
        score = evaluate(*candidate.args, **candidate.kwargs)
        optimizer.tell(candidate, score)

        # Early exits: patience or wall clock
        if learner.no_update_count > learner.patience:
            print(learner.no_update_count)
            print(learner.patience)
            print("case 1")
            break
        if time.time() - learner.start_time > TIME_LIMIT:
            break

    return learner.best_seen_performance, time.time() - learner.start_time


def training(command, fin_log_dir, state_size, start_policy=None, neval=1000):
    results = []
    learner = CCLearner(command, "FlexiCC learner", fin_log_dir, None, state_size, 13)

    if start_policy is not None:
        learner.load_initial_policy_from_file(start_policy)

    learner.training_stage = 0
    duration = 0
    len_pipe = len(training_pipeline)

    while duration < TIME_LIMIT and learner.training_stage < len_pipe:
        setup = training_pipeline[learner.training_stage % len_pipe]
        learner.update_setting(setup)
        learner.patience = setup["patient"]
        optimizer_class = setup["learner"]

        best_score, duration = benchmark_optimizer(learner, optimizer_class, neval)
        name = optimizer_class.name

        results.append(
            {
                "stage": learner.training_stage,
                "optimizer": name,
                "best_score": best_score,
                "duration": duration,
            }
        )

        learner.training_stage += 1
        learner.no_update_count = 0
        if learner.best_policy is not None:
            print(
                "Starting a new training stage {round}, at iteration {iter} got value {best}!".format(
                    round=learner.training_stage, iter=learner.current_iter, best=learner.best_policy.score
                )
            )
        print(f"Completed {name} in {duration:.2f} seconds, best score: {best_score:.4f}")

    learner.close()
    return results
