"""Samples, outcomes and estimates for uncertain-QD bookkeeping.

Two kinds of repeated measurement are kept apart on purpose:

RolloutStats
    Episodes of one exact checkpoint (identified by the SHA-256 of its saved
    weights) under one evaluation protocol. Each episode is one sample. These
    statistics describe how reliably *that trained policy* reproduces its
    fitness and descriptor across start states.

GeneStats
    Independent training runs of one gene (one per training seed). Each run
    contributes exactly one sample: the mean over its own episodes. Episodes
    from the same run are never treated as independent training samples.
    These statistics describe how reliably *the gene* trains.

Aggregation rules (search-time estimates)
-----------------------------------------
fitness estimate       arithmetic mean of the valid samples in the window
                       (UQD reference implementation: nanmean, "Average")
descriptor estimate    per-dimension mean of the same samples
fitness_sd             sample standard deviation (ddof=1) of the samples:
                       the spread of individual outcomes. None when n < 2.
fitness_sem            fitness_sd / sqrt(n): the standard error of the
                       estimated mean, assuming independent samples. None
                       when n < 2. It shrinks with n; fitness_sd does not.
window                 only the most recent ``window`` valid samples enter an
                       estimate; older samples stay in the record.

Outcome classes
---------------
ok                   a normal episode, including one where the robot falls.
                     Low returns are data, never failures.
candidate_failure    caused by the candidate itself: the gene does not build,
                     training diverges to non-finite values, the policy emits
                     non-finite actions, or the job ran past its time limit
                     after starting. A candidate failure disqualifies the
                     solution; its other samples are not used to rank it, so
                     dropping the bad episode can never inflate its estimate.
infra_failure        caused by the platform: submission refused, job lost or
                     cancelled before it ran, missing artifacts, unreadable
                     results, environment-spec mismatch, out-of-memory or
                     filesystem errors. No sample is recorded; the request is
                     retried with the same seeds up to infra.max_retries.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np

OK = "ok"
CANDIDATE_FAILURE = "candidate_failure"
INFRA_FAILURE = "infra_failure"

CANDIDATE_FAILURE_KINDS = {
    "gene_build_error",
    "nonfinite_training",
    "nonfinite_rollout",
    "candidate_timeout",
    "unclassified_runtime_error",
}
INFRA_FAILURE_KINDS = {
    "submission_failed",
    "job_lost",
    "scheduler_timeout",
    "missing_artifact",
    "malformed_result",
    "fitness_spec_mismatch",
    "resource_error",
}


def outcome_class(kind):
    """Map a failure kind to CANDIDATE_FAILURE or INFRA_FAILURE."""
    if kind in CANDIDATE_FAILURE_KINDS:
        return CANDIDATE_FAILURE
    if kind in INFRA_FAILURE_KINDS:
        return INFRA_FAILURE
    raise ValueError(f"unknown failure kind {kind!r}")


def protocol_id(protocol):
    """Stable short id for an evaluation protocol dict."""
    text = json.dumps(protocol, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RolloutSample:
    seed: int
    index: int                     # position in the solution's rollout stream
    status: str                    # OK or CANDIDATE_FAILURE
    fitness: Optional[float] = None
    descriptor: Optional[Tuple[float, ...]] = None
    length: Optional[int] = None
    terminated: Optional[bool] = None
    source: str = "initial"        # initial | reeval | final | aria
    generation: int = 0
    kind: Optional[str] = None     # failure kind when status != OK

    def __post_init__(self):
        if self.status not in (OK, CANDIDATE_FAILURE):
            raise ValueError("rollout samples are either ok or candidate failures; "
                             "infrastructure failures produce no sample")
        if self.status == OK:
            if self.fitness is None or not math.isfinite(self.fitness):
                raise ValueError("an ok sample needs a finite fitness")
            if self.descriptor is None or not all(math.isfinite(x) for x in self.descriptor):
                raise ValueError("an ok sample needs a finite descriptor")
            self.descriptor = tuple(float(x) for x in self.descriptor)


@dataclass
class Estimate:
    n: int
    fitness_mean: float
    fitness_sd: Optional[float]
    fitness_sem: Optional[float]
    fitness_median: float
    descriptor_mean: Tuple[float, ...]
    descriptor_var: Optional[Tuple[float, ...]]
    descriptor_median: Tuple[float, ...]

    def to_dict(self):
        return asdict(self)


def estimate_from_values(fitnesses, descriptors):
    """Estimate from parallel lists of fitness values and descriptor tuples."""
    f = np.asarray(fitnesses, dtype=float)
    d = np.asarray(descriptors, dtype=float)
    n = int(f.shape[0])
    if n == 0:
        raise ValueError("cannot estimate from zero samples")
    if d.ndim == 1:
        d = d.reshape(n, -1)
    sd = float(np.std(f, ddof=1)) if n >= 2 else None
    return Estimate(
        n=n,
        fitness_mean=float(np.mean(f)),
        fitness_sd=sd,
        fitness_sem=(sd / math.sqrt(n)) if sd is not None else None,
        fitness_median=float(np.median(f)),
        descriptor_mean=tuple(float(x) for x in np.mean(d, axis=0)),
        descriptor_var=tuple(float(x) for x in np.var(d, axis=0, ddof=1)) if n >= 2 else None,
        descriptor_median=tuple(float(x) for x in np.median(d, axis=0)),
    )


@dataclass
class RolloutStats:
    """Every rollout sample of one checkpoint under one protocol."""
    checkpoint_id: str
    protocol_id: str
    samples: List[RolloutSample] = field(default_factory=list)

    def add(self, sample):
        if any(s.seed == sample.seed for s in self.samples):
            raise ValueError(f"seed {sample.seed} was already used for checkpoint {self.checkpoint_id}")
        self.samples.append(sample)

    @property
    def next_index(self):
        return 1 + max((s.index for s in self.samples), default=-1)

    @property
    def has_candidate_failure(self):
        return any(s.status == CANDIDATE_FAILURE for s in self.samples)

    def valid_samples(self, window=None):
        ok = [s for s in self.samples if s.status == OK]
        if window is not None:
            ok = ok[-window:]
        return ok

    def count(self, window=None):
        return len(self.valid_samples(window))

    def estimate(self, window=None):
        ok = self.valid_samples(window)
        if not ok:
            return None
        return estimate_from_values([s.fitness for s in ok], [s.descriptor for s in ok])

    def seeds(self):
        return [s.seed for s in self.samples]


@dataclass
class ReplicateSummary:
    """One training run of a gene, reduced to its own rollout estimate."""
    train_seed: int
    checkpoint_id: str
    model_path: str
    rollouts: RolloutStats
    train_timesteps: int

    def summary(self):
        return self.rollouts.estimate()


@dataclass
class GeneStats:
    """Training-seed replicates of one gene; one sample per training run."""
    gene_id: str
    replicates: List[ReplicateSummary] = field(default_factory=list)

    def add(self, replicate):
        if any(r.train_seed == replicate.train_seed for r in self.replicates):
            raise ValueError(f"training seed {replicate.train_seed} already used for gene {self.gene_id}")
        self.replicates.append(replicate)

    @property
    def has_candidate_failure(self):
        return any(r.rollouts.has_candidate_failure for r in self.replicates)

    def replicate_estimates(self):
        return [r.summary() for r in self.replicates if r.summary() is not None]

    def estimate(self):
        """Across-replicate estimate: each training run counts once."""
        per_run = self.replicate_estimates()
        if not per_run:
            return None
        return estimate_from_values([e.fitness_mean for e in per_run],
                                    [e.descriptor_mean for e in per_run])

    def within_run_fitness_sd(self):
        """Mean of the per-run episode SDs (rollout noise, reported separately)."""
        sds = [e.fitness_sd for e in self.replicate_estimates() if e.fitness_sd is not None]
        return float(np.mean(sds)) if sds else None
