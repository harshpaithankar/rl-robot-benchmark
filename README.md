# RL Robot Benchmark

Benchmarking reinforcement learning algorithms (PPO, SAC, TD3) for robot control:
reward design, sample efficiency, and robustness.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Run a random-policy baseline:

```powershell
python agents/random_agent.py --env Pendulum-v1 --episodes 5
```

## Roadmap

- [x] Project setup and random baseline
- [ ] First trained agent (PPO / SAC) on Pendulum
- [ ] Custom mobile-robot navigation environment
- [ ] Multi-seed algorithm comparison
- [ ] Reward shaping and domain randomization ablations
- [ ] Statistical analysis and report
