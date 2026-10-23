# Top 5 XQC genes from `xqc_mome_run1` (for seed testing)

The five best-scoring genes from the week-long LLM-GE run on HalfCheetah-v4
(Oct 14–21, 70 generations, 631 trained genes). Each was trained for 100k steps
with seeds 0, 1 and 2; the score is the average of the three seeds.

| Rank | File | Gen | Score | Per-seed scores | Std |
|---|---|---|---|---|---|
| 1 | `network_xXx9vYw72TA0NVroeEgO8wPgbwN.py` | 64 | 10,052 | 10,457 / 9,895 / 9,805 | 288 |
| 2 | `network_xXxdsYVPVJrLNc9qFkWFBsMtJVP.py` | 9 | 9,957 | 10,720 / 8,203 / 10,947 | 1,244 |
| 3 | `network_xXxgVhkpL62RSeM2Z5wtIORpGcH.py` | 48 | 9,813 | 10,266 / 8,866 / 10,307 | 670 |
| 4 | `network_xXxCDP6NlWZOXNPsebCM1XiBcpk.py` | 62 | 9,785 | 10,545 / 9,092 / 9,718 | 595 |
| 5 | `network_xXxKf3RrD9DZmtBXKiO1uTeV1lp.py` | 42 | 9,751 | 10,978 / 9,725 / 8,549 | 992 |

For comparison, XQC's original settings (`sota/XQC/llmge/network.py`) score
6,970 on the same budget and seeds.

**What the files are.** XQC genes, not PPO networks: each returns two plain
dicts, network sizes (`get_architecture_kwargs`) and training settings such as
learning rates (`get_xqc_kwargs`). They only run with the XQC trainer in `sota/XQC`.

**How to train one** (branch `shazeb/halfcheetah-xqc`, XQC environment from
`sota/XQC/pace/install_env.sh`):

```bash
cp network_<gene_id>.py ../../../../sota/XQC/llmge/models/
cd sota/XQC && source pace/env.sh
python train_xqc_gene.py -network models.network_<gene_id> -timesteps 100000 -num_seeds 3
```

- `-num_seeds N` trains seeds 0..N-1 together in one process; `-seed S` changes the first seed.
- `-timesteps` is the training budget (evolution used 100k; XQC's paper uses 1M).
- The score goes to `llmge/results/<gene_id>_results.csv`, and per-seed scores to
  `llmge/stats/<gene_id>_stats.json` (`per_seed_return`).
- Run it as a GPU job on PACE (`-p ice-gpu`). In the run, 100k steps × 3 seeds took
  about 26 min for ranks 1–3 and 85–89 min for ranks 4–5 (larger networks).

**Note.** Ranks 2–5 differ by about 200 points while their seeds vary by 600–1,200,
so they are within noise of each other. Only rank 1 has a tight spread, which is why
testing more seeds is worthwhile.
