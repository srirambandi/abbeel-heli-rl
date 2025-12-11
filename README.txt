Reinforcement Learning on Pieter Abbeel's helicopter as a Gymnasium Environment
==============================================================================

This project implements the Pieter Abbeel and Stanford's(http://heli.stanford.edu/) helicopter as a Gymnasium environment and
evaluates four learning algorithms:

• Behavior Cloning (BC)  
• DAgger  
• Proximal Policy Optimization (PPO)  (soon)
• Soft Actor-Critic (SAC)  (soon)

The environment, dataset generation, algorithm implementations, and comparative
evaluation tools are included in this repository.


Directory structure
-------------------
abbeel_heli/         Helicopter dynamics, environment, PID expert, utilities
datasets/            Expert demonstration dataset
models/              Saved models for BC, DAgger, PPO, SAC
plots/               Evaluation plots
bc.py                Behavior Cloning training/eval
dagger.py            DAgger training/eval
compare_algos.py     Runs large-scale evaluation across all algos


Setup
-----
Requires Python 3.10+.


How to run
----------

1. Generate expert dataset:
    python abbeel_heli/datasets/collect_expert.py

2. Train Behavior Cloning:
    python bc.py train

3. Train DAgger:
    python dagger.py train

4. Evaluate each algorithm:
    python bc.py eval
    python dagger.py eval


Author
------
Sri Ram Bandi (sbandi@umass.edu)
