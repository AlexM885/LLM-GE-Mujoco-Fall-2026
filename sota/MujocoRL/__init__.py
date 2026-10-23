"""MuJoCo RL domain for LLM-Guided Evolution (HalfCheetah + PPO).

Only the MESB modules are imported as a package (``sota.MujocoRL.mesb_archive``
etc.). Evaluation scripts such as ``train_gene.py`` are run directly as
scripts, so this file intentionally imports nothing heavy.
"""
