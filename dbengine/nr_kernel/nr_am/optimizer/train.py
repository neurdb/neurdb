#!/usr/bin/env python3
import argparse
import glob
import os
import shlex
import numpy as np

import utils as utils
from ng import training

MAX_STATE_ALLOWABLE = 1024


def _build_command(args) -> [str]:
    parts = [
        "python3", "./benchmark/ycsb.py",
        f'--dsn {shlex.quote(args.dsn)}',
        f'--table {shlex.quote(args.table)}',
        f'--threads {int(args.nworkers)}',
        f'--duration {int(args.duration)}',
        f'--warmup {int(args.warmup)}',
        f'--txn-length {int(args.txn_length)}',
        f'--n-keys {int(args.nkeys)}',
        f'--payload-bytes {int(args.payload_bytes)}',
        f'--workload {shlex.quote(args.workload)}',
        f'--zipf-theta {float(args.zipf_theta)}',
        f'--isolation {shlex.quote(args.isolation)}',
        # # IMPORTANT: policy placeholder for training to substitute
        # f'--policy {{policy}}',
    ]

    # Pass through any free-form extra runner args the user wants
    if args.extra_args:
        parts.append(str(args.extra_args).strip())

    # Backwards-compat: also pass bench-opts blob verbatim if provided
    if args.bench_opts:
        parts.append(str(args.bench_opts).strip())

    # Silence runner output if requested (the runner already suppresses make output)
    if args.quiet:
        parts.append('>/dev/null 2>&1')

    return parts


def main(args):
    np.random.seed(args.seed)

    cfg = utils.setup(args)

    # `training` is expected to orchestrate search:
    # it should do: command_template.format(policy=<path>) when running
    return training(
        _build_command(args),
        cfg.get('log_directory'),
        args.state_space,
        args.pickup_policy
    )


def evaluate():
    """
    Entry point that prepares CLI, validates state size, then runs main().
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Optimizer wrapper that calls the asyncpg YCSB-like runner with policy substitution."
    )

    # Generic experiment settings
    parser.add_argument('--base-log-dir', type=str, default='./optimizer/bo-all',
                        help='model save location')
    parser.add_argument('--base-kid-dir', type=str, default='./optimizer/bo',
                        help='kid policy save location')
    parser.add_argument('--expr-name', type=str, default='bo',
                        help='experiment name')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')

    # Workload/runner linkage
    parser.add_argument('--workload-type', type=str, default='ycsb',
                        choices=['tpcc', 'tpce', 'ycsb'],
                        help='(kept for compatibility; runner uses --workload)')
    parser.add_argument('--nworkers', type=int, default=8, help='number of database workers')
    parser.add_argument('--pickup-policy', type=str, nargs='*',
                        default=glob.glob('./optimizer/samples/*'),
                        help='initial policy/policies to start with')
    parser.add_argument('--scale-factor', type=int, default=1,
                        help='scale factor (if used by your training loop)')
    parser.add_argument('--state-space', type=int, default=32,
                        help='state space for policy searching')

    # Legacy free-form
    parser.add_argument('--bench-opts', type=str, default='',
                        help='extra runner flags blob (passed verbatim)')

    # ---- Runner specific (mapped to your asyncpg script) ----
    parser.add_argument('--dsn', type=str, default=os.environ.get("PG_DSN", "postgres://127.0.0.1/neurdb"),
                        help='NeurDB DSN')
    parser.add_argument('--table', type=str, default='accounts', help='table name')
    parser.add_argument('--duration', type=int, default=5, help='benchmark duration seconds')
    parser.add_argument('--warmup', type=int, default=2, help='warmup seconds')
    parser.add_argument('--txn-length', type=int, default=10, help='ops per transaction')
    parser.add_argument('--nkeys', type=int, default=10000, help='number of keys')
    parser.add_argument('--payload-bytes', type=int, default=0, help='payload size per row')
    parser.add_argument('--workload', type=str, default='A', choices=['A', 'B', 'C', 'F', 'custom'],
                        help='YCSB mix for the runner')
    parser.add_argument('--zipf-theta', type=float, default=0.8, help='Zipf skew')
    parser.add_argument('--isolation', type=str, default='SERIALIZABLE',
                        choices=['READ COMMITTED', 'REPEATABLE READ', 'SERIALIZABLE'],
                        help='transaction isolation level for the runner')
    parser.add_argument('--extra-args', type=str, default='',
                        help='additional flags appended to the runner command')
    parser.add_argument('--quiet', action='store_true',
                        help='silence runner stdout/stderr')

    args = parser.parse_args()
    return main(args)


if __name__ == '__main__':
    evaluate()
