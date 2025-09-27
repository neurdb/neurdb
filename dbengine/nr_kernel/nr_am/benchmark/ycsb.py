#!/usr/bin/env python3
import asyncio, asyncpg, random, time, argparse, math, os, statistics
from typing import Tuple
import subprocess, sys

# -------- Zipf sampler (bounded) --------
class ZipfSampler:
    # Generate integers in [1..n] with skew theta in [0,1.2] typical
    def __init__(self, n: int, theta: float):
        assert n >= 1
        self.n = n
        self.theta = max(0.0, theta)
        if self.theta == 0.0:
            self.cdf = None
            return
        # Precompute harmonic-like normalizer
        # P(k) ~ 1 / k^theta
        weights = [1.0 / (k ** self.theta) for k in range(1, n + 1)]
        s = sum(weights)
        acc = 0.0
        self.cdf = []
        for w in weights:
            acc += w / s
            self.cdf.append(acc)

    def sample(self) -> int:
        if self.theta == 0.0:
            return random.randint(1, self.n)
        r = random.random()
        # binary search in CDF
        lo, hi = 0, self.n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cdf[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        return lo + 1

# -------- Workload mixes --------
WORKLOADS = {
    # (read_ratio, write_ratio, rmw_ratio) — ratios within an operation slot
    # YCSB A: 50/50 read/update
    "A": (0.5, 0.5, 0.0),
    # YCSB B: 95/5 read/update
    "B": (0.95, 0.05, 0.0),
    # YCSB C: 100% read
    "C": (1.0, 0.0, 0.0),
    # YCSB F: 0% reads, 100% read-modify-write (in one row op)
    "F": (0.0, 0.0, 1.0),
}

def pick_op(mix: Tuple[float,float,float]) -> str:
    r = random.random()
    if r < mix[0]: return "READ"
    r -= mix[0]
    if r < mix[1]: return "UPDATE"
    return "RMW"   # read-modify-write


# -------- Data prepopulation --------
async def prepopulate(pool, table, n_keys, payload_bytes):
    pad = "x" * max(1, payload_bytes)
    batch = 1000
    async with pool.acquire() as conn:
        existing = await conn.fetchval(f"SELECT count(*) FROM {table}")
        if existing >= n_keys:
            return
        # print(f"[loader] inserting keys {existing+1}..{n_keys} via INSERT batches")
        sql = f"INSERT INTO {table}(id,balance,pad) VALUES($1,$2,$3)"
        for base in range(existing+1, n_keys+1, batch):
            upper = min(base+batch-1, n_keys)
            vals = [(i, 1000, pad) for i in range(base, upper+1)]
            # print(f"[loader] inserting {len(vals)} rows id={base}..{upper}")
            async with conn.transaction():
                await conn.executemany(sql, vals)


# -------- Worker --------
async def run_txn(conn, sampler, args, stats, table, mix):
    # one logical transaction worth of work
    async with conn.transaction(isolation=args.isolation.lower()):
        await conn.execute("SET enable_seqscan = off")
        await conn.execute("SET enable_bitmapscan = off")
        for _ in range(args.txn_length):
            op = pick_op(mix)
            k = sampler.sample()
            if op == "READ":
                await conn.fetchval(f"SELECT balance FROM {table} WHERE id = {k}")
                stats["reads"] += 1
            elif op == "UPDATE":
                data = random.randint(-10000, 10000)
                await conn.execute(f"UPDATE {table} SET balance = {data} WHERE id = {k}")
                stats["writes"] += 1
            else:  # RMW
                val = await conn.fetchval(f"SELECT balance FROM {table} WHERE id = {k}")
                if val is None:
                    raise RuntimeError(f"RMW failed: id={k} not found")
                delta = random.randint(-10, 10)
                await conn.execute(f"UPDATE {table} SET balance = {val + delta} WHERE id = {k}")
                stats["rmw"] += 1


RETRYABLE_SQLSTATES = {"40001", 
                       "40P01", 
                       "55P03"}  # serialization, deadlock, lock timeout (optional)
MAX_RETRIES = 5

async def worker(pool, sampler, args, stats, stop_event):
    table = args.table
    mix = WORKLOADS.get(args.workload.upper(), (args.read_ratio, 1.0 - args.read_ratio, 0.0))

    while not stop_event.is_set():
        t0 = time.perf_counter_ns()
        try:
            async with pool.acquire() as conn:
                attempt = 0
                while True:
                    try:
                        await run_txn(conn, sampler, args, stats, table, mix)
                        stats["commits"] += 1
                        stats["ops"] += args.txn_length
                        break
                    except Exception as e:
                        stats["aborts"] += 1
                        stats["ops"] += args.txn_length
                        sqlstate = getattr(e, "sqlstate", None)
                        # print("[worker] transaction aborted:", e, file=sys.stderr)
                        if sqlstate in RETRYABLE_SQLSTATES and attempt < MAX_RETRIES:
                            attempt += 1
                            backoff = min(0.01 * (2 ** (attempt - 1)), 1.0)
                            await asyncio.sleep(backoff)
                            continue
                        # non-retryable or retries exhausted: stop retrying
                        raise
        except Exception as e:
            # Already counted as abort above; just log
            print(f"[worker] transaction aborted: {e}")
        finally:
            stats["latencies_ns"].append(time.perf_counter_ns() - t0)


# -------- DB setup --------
async def setup_db(pool, table, policy):
    if policy is None:
        sql = f"""
        DROP TABLE IF EXISTS {table};

        CREATE TABLE {table}(
        id      BIGINT PRIMARY KEY,
        balance BIGINT NOT NULL,
        pad     TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS accounts_id_inc ON accounts USING btree (id);
        """
    else:
        sql = f"""
        DROP TABLE IF EXISTS {table};
        DROP EXTENSION IF EXISTS nram CASCADE;
        DROP ACCESS METHOD IF EXISTS nram;
        DROP FUNCTION IF EXISTS nram_tableam_handler(internal);
        CREATE EXTENSION IF NOT EXISTS nram;

        CREATE TABLE {table}(
        id      BIGINT PRIMARY KEY,
        balance BIGINT NOT NULL,
        pad     TEXT NOT NULL DEFAULT ''
        ) USING nram;
        CREATE INDEX IF NOT EXISTS accounts_id_inc ON accounts USING btree (id);
        """
    
    async with pool.acquire() as conn:
        await conn.execute(sql)


async def main():
    p = argparse.ArgumentParser(description="YCSB-like benchmark for neurdb (NRAM table).")
    p.add_argument("--dsn", default=os.environ.get("PG_DSN","postgres://127.0.0.1/neurdb"),
                   help="NeurDB DSN (e.g. postgres://user:pass@host:5432/db)")
    p.add_argument("--table", default="accounts", help="Target table name")
    p.add_argument("--n-keys", type=int, default=10000, help="Number of keys")
    p.add_argument("--payload-bytes", type=int, default=0, help="Size of text padding per row")
    p.add_argument("--threads", type=int, default=4, help="Concurrent workers")
    p.add_argument("--duration", type=int, default=5, help="Benchmark duration seconds")
    p.add_argument("--warmup", type=int, default=2, help="Warmup seconds")
    p.add_argument("--txn-length", type=int, default=10, help="Ops per transaction")
    p.add_argument("--workload", default="A", choices=list(WORKLOADS.keys())+["custom"],
                   help="YCSB mix: A/B/C/F or custom")
    p.add_argument("--read-ratio", type=float, default=0.5,
                   help="Used only if workload=custom (0..1)")
    p.add_argument("--zipf-theta", type=float, default=0.8, help="Zipf skew (0=uniform)")
    p.add_argument("--isolation", default="SERIALIZABLE",
                   choices=["READ COMMITTED","REPEATABLE READ","SERIALIZABLE"])
    p.add_argument("--preload", action="store_false", help="Prepopulate table to n-keys")
    p.add_argument("--policy", default=None, help="Call SELECT nram_load_policy('<name>')")    # No policy for postgres.
    args = p.parse_args()
    
    if args.policy not in [None, "2pl", "occ"]:
        args.policy = os.path.abspath(args.policy)
    
    sampler = ZipfSampler(args.n_keys, args.zipf_theta)
    
    # print("[setup] running `make setup` to cleanup the database …")
    rc = subprocess.call(["make", "setup"],
                         stdout=subprocess.DEVNULL)
    if rc != 0:
        print(f"[setup] `make setup` failed with exit code {rc}, aborting.")
        return
    
    # print(f"[setup] connecting to database {args.dsn} with {args.threads} threads …")
    async def init_conn(conn, iso_sql):
        await conn.execute(f"SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL {iso_sql}")
    iso_sql = args.isolation.upper()
    pool = await asyncpg.create_pool(
        dsn=args.dsn, min_size=args.threads, max_size=args.threads,
    )

    # print(f"[setup] preparing table {args.table} and loading policy {args.policy}")
    await setup_db(pool, args.table, args.policy)
    async with pool.acquire() as c:
        # await c.execute("SET client_min_messages TO WARNING")
        if args.policy is not None:
            await c.execute("SELECT nram_load_policy($1)", args.policy)

    if args.preload:
        await prepopulate(pool, args.table, args.n_keys, args.payload_bytes)

    # Metrics
    manager = {
        "ops": 0,
        "reads": 0,
        "writes": 0,
        "rmw": 0,
        "aborts": 0,
        "commits": 0,
        "latencies_ns": []
    }

    stop = asyncio.Event()

    # Warmup
    if args.warmup > 0:
        print(f"[warmup] {args.warmup}s …")
        warm_stop = asyncio.Event()
        tasks = [asyncio.create_task(worker(pool, sampler, args, manager, warm_stop)) 
                 for _ in range(args.threads)]
        await asyncio.sleep(args.warmup)
        warm_stop.set()
        print("[warmup] stopping workers …")
        await asyncio.gather(*tasks)
        
        # reset counters post-warmup
        for k in list(manager.keys()):
            manager[k] = [] if k == "latencies_ns" else 0

    # Timed run
    print(f"[run] threads={args.threads} duration={args.duration}s workload={args.workload} iso={args.isolation} policy={args.policy or '(unchanged)'}")
    tasks = [asyncio.create_task(worker(pool, sampler, args, manager, stop)) for _ in range(args.threads)]
    t0 = time.perf_counter()
    await asyncio.sleep(args.duration)
    print("[run] stopping workers …")
    stop.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    t1 = time.perf_counter()
    elapsed = t1 - t0

    await pool.close()

    # Report
    ops = manager["ops"]
    r = manager["reads"]; w = manager["writes"]; rmw = manager["rmw"]
    commits = manager["commits"]; aborts = manager["aborts"]
    thr = commits / elapsed

    lats = sorted(manager["latencies_ns"])
    def pct(p):
        if not lats: return float("nan")
        idx = max(0, min(len(lats)-1, int(math.ceil(p/100.0*len(lats))-1)))
        return lats[idx] / 1e6  # ms

    print("\n=== YCSB-like Summary ===")
    print(f"Elapsed        : {elapsed:.3f}s")
    print(f"Throughput     : {thr:.0f} tps")
    print(f"Commits/Aborts : {commits:,} / {aborts:,}")
    print(f"Reads/Writes/RMW: {r:,} / {w:,} / {rmw:,}")
    if lats:
        print("Txn latency ms : p50={:.2f}  p90={:.2f}  p95={:.2f}  p99={:.2f}  max={:.2f}".format(
            pct(50), pct(90), pct(95), pct(99), (lats[-1]/1e6)
        ))
    print("=========================")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
