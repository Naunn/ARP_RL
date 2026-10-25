"""Experiment entrypoints: each script here answers a different research question, but all of
them build their agents and problem instances through `src.instances` and `src.experiments.
experiment_setup` so results stay comparable across scripts.

- `iter_training`: trains DQN + Double DQN once (no ablation variants, no disruptions) on a
  fixed instance -- the simplest baseline pipeline.
- `solving_schedule_experiment`: samples a fresh problem instance every iteration and trains the
  active agent variants on each one, for measuring generalization across instances.
- `disruption_training_experiment`: trains the active agent variants on one instance, then
  repeatedly disrupts it and retrains, for measuring disruption-recovery resilience.
"""
