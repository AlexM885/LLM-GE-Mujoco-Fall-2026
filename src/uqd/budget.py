"""Cost ledger: every unit of compute a run spends, by category.

Each entry records what was measured and leaves unmeasured quantities as None
rather than guessing. A job lost to the scheduler usually has no measured cost
at all; it is still recorded, so the report can say how many jobs have unknown
cost instead of silently under-counting.

Categories
----------
initial_train     TD3 training of a new gene (env steps = training timesteps)
initial_eval      episodes run by the training job right after training
reeval            extra episodes of an archived checkpoint (rollout sampling)
retrain_train     additional training replicates (sample_unit = retrain)
retrain_eval      episodes summarising those replicates
aria              ARIA perturbation rollouts (post-hoc refinement)
final_eval        held-out final-evaluation episodes
final_retrain     held-out retraining in the final evaluation
"""

from dataclasses import asdict, dataclass
from typing import Optional

CATEGORIES = (
    "initial_train",
    "initial_eval",
    "reeval",
    "retrain_train",
    "retrain_eval",
    "aria",
    "final_eval",
    "final_retrain",
)
#: Categories whose env steps count toward a search budget (final evaluation
#: and post-hoc refinement are reported separately).
SEARCH_CATEGORIES = ("initial_train", "initial_eval", "reeval", "retrain_train", "retrain_eval")


@dataclass
class CostEntry:
    category: str
    solution_id: Optional[str]
    generation: int
    env_steps: Optional[int]          # measured simulator steps, None if unknown
    episodes: Optional[int]
    wall_seconds: Optional[float]     # measured inside the job, None if unknown
    status: str                       # ok | candidate_failure | infra_failure
    note: str = ""

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown cost category {self.category!r}")


class CostLedger:
    def __init__(self):
        self.entries = []

    def record(self, category, solution_id=None, generation=0, env_steps=None,
               episodes=None, wall_seconds=None, status="ok", note=""):
        entry = CostEntry(category, solution_id, int(generation),
                          None if env_steps is None else int(env_steps),
                          None if episodes is None else int(episodes),
                          None if wall_seconds is None else float(wall_seconds),
                          status, note)
        self.entries.append(entry)
        return entry

    def search_env_steps(self):
        """Measured env steps spent by the search (training + evaluation)."""
        return sum(e.env_steps for e in self.entries
                   if e.category in SEARCH_CATEGORIES and e.env_steps is not None)

    def summary(self):
        out = {}
        for cat in CATEGORIES:
            rows = [e for e in self.entries if e.category == cat]
            if not rows:
                continue
            out[cat] = {
                "jobs_or_requests": len(rows),
                "env_steps_measured": sum(e.env_steps for e in rows if e.env_steps is not None),
                "episodes_measured": sum(e.episodes for e in rows if e.episodes is not None),
                "wall_seconds_measured": sum(e.wall_seconds for e in rows if e.wall_seconds is not None),
                "entries_with_unknown_env_steps": sum(1 for e in rows if e.env_steps is None),
                "entries_with_unknown_wall_time": sum(1 for e in rows if e.wall_seconds is None),
                "by_status": {s: sum(1 for e in rows if e.status == s)
                              for s in sorted({e.status for e in rows})},
            }
        out["search_env_steps_measured"] = self.search_env_steps()
        return out

    def to_list(self):
        return [asdict(e) for e in self.entries]

    @classmethod
    def from_list(cls, rows):
        ledger = cls()
        ledger.entries = [CostEntry(**row) for row in rows]
        return ledger
