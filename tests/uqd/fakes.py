"""Synthetic stand-ins for the MuJoCo jobs, for testing the UQD engine.

Each fake gene has a true mean fitness and descriptor; an episode with reset
seed s returns those values plus noise drawn from a generator seeded by
(gene, s), so the same seed always reproduces the same episode, as the real
pipeline does on CPU.
"""

import numpy as np

from src.uqd.seeds import derive_int

PROTOCOL = {"env_id": "Fake-v0", "max_steps": 100, "deterministic_policy": True,
            "reward_spec": None, "descriptor": "fake", "protocol_version": 1}
DESCRIPTORS = [("foot_contact_0", 0.0, 1.0), ("foot_contact_1", 0.0, 1.0)]


class FakeWorld:
    def __init__(self, truths, fitness_noise=10.0, descriptor_noise=0.02, nonfinite_genes=(),
                 build_fail_genes=(), infra_fail_plan=None):
        self.truths = truths                    # gene -> (fitness, (d0, d1))
        self.fitness_noise = fitness_noise
        self.descriptor_noise = descriptor_noise
        self.nonfinite_genes = set(nonfinite_genes)
        self.build_fail_genes = set(build_fail_genes)
        # request/solution id -> number of initial calls that fail as infra
        self.infra_fail_plan = dict(infra_fail_plan or {})
        self.rollout_calls = []
        self.train_calls = []

    def episode(self, gene, seed):
        f, d = self.truths[gene]
        rng = np.random.default_rng(derive_int(gene, int(seed)))
        dd = np.clip(np.array(d) + rng.normal(0, self.descriptor_noise, 2), 0, 1)
        return {"seed": int(seed), "return": float(f + rng.normal(0, self.fitness_noise)),
                "length": 100, "terminated": False, "truncated": True,
                "nonfinite": gene in self.nonfinite_genes,
                "descriptor": {"foot_contact_0": float(dd[0]), "foot_contact_1": float(dd[1])}}

    def _consume_infra(self, key):
        if self.infra_fail_plan.get(key, 0) > 0:
            self.infra_fail_plan[key] -= 1
            return True
        return False


class FakeExecutor:
    def __init__(self, world):
        self.world = world

    def run_rollouts(self, requests, protocol, label=""):
        out = []
        for req in requests:
            self.world.rollout_calls.append((req["solution_id"], tuple(req["seeds"])))
            base = {"request_id": req["request_id"], "solution_id": req["solution_id"],
                    "seeds": req["seeds"], "start_index": req.get("start_index", 0)}
            if self.world._consume_infra(req["solution_id"]):
                out.append({**base, "status": "infra_failure", "failure_kind": "job_lost"})
                continue
            gene = req["solution_id"].split("@")[0].split("#")[0]
            eps = [self.world.episode(gene, s) for s in req["seeds"]]
            out.append({**base, "status": "ok", "episodes": eps, "env_steps": 100 * len(eps),
                        "wall_seconds": 0.01})
        return out

    def run_train(self, spec, timesteps, env_id, eval_max_steps, model_dir, stats_dir):
        self.world.train_calls.append((spec.gene_id, spec.train_seed, spec.eval_seed, spec.eval_episodes))
        if self.world._consume_infra(spec.solution_id):
            return None, "job_lost"
        if spec.gene_id in self.world.build_fail_genes:
            return {"status": "failed", "failure_kind": "gene_build_error", "train_time_sec": 1.0}, None
        seeds = [spec.eval_seed + k for k in range(spec.eval_episodes)]
        eps = [self.world.episode(spec.gene_id, s) for s in seeds]
        return {
            "status": "ok", "timesteps": timesteps, "eval_seeds": seeds, "episodes": eps,
            "eval_env_steps": 100 * len(eps), "train_wall_sec": 2.0, "eval_wall_sec": 0.1,
            "model_path": f"/fake/{spec.gene_id}_seed{spec.train_seed}.zip",
            "checkpoint_sha256": f"sha-{spec.gene_id}-{spec.train_seed}", "param_count": 123,
            "protocol": PROTOCOL,
            "mean_reward": float(np.mean([e["return"] for e in eps])),
        }, None
