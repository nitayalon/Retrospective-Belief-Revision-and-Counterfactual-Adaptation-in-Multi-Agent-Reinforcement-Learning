# HIM+HER Project — Copilot Instructions

## Environment
- WSL Ubuntu, Python 3.12.3
- Project root: ~/Retrospective-Belief-Revision-and-Counterfactual-Adaptation-in-Multi-Agent-Reinforcement-Learning
- Venv: ~/Retrospective-Belief-Revision-and-Counterfactual-Adaptation-in-Multi-Agent-Reinforcement-Learning/venv — always activate with `source venv/bin/activate`
- All commands are WSL bash, never PowerShell

## Architecture
- Full specification is in ARCHITECTURE.md at the project root
- Read the relevant section of ARCHITECTURE.md before implementing any module
- JAX/NumPy boundary rules are in Section 5 — hard constraints, never violate
- Open design questions are in Section 15

## Hard rules
- Never call jax.jit on any function that touches env.step()
- Never store JAX arrays in the replay buffer
- Never pass current_model_id as a traced JAX value into update_step
- Compute model_embed via encode_model() before calling update_step, never inside it
- Always run `pytest tests/ -v 2>&1 | cat` after every change
- Always confirm 51 tests pass before proceeding to the next task
- Always paste raw terminal output — never summarize or replace with bullet points
- Always use `2>&1 | cat` when running training scripts

## Package notes
- Import MPE from mpe2 not pettingzoo.mpe (deprecated)
- pygame 2.6.1 installed
- JAX is CPU-only for now
- orbax-checkpoint 0.11.x — use `import orbax.checkpoint as ocp`
- hydra-core 1.3.2 installed

## Current state
- 51/51 tests passing
- Complete modules: HIM, HER, networks, TrainState, replay buffer,
  predator-prey env wrapper, VanillaAgent baseline
- Immediate next task: run smoke test and verify non-zero rewards

## Copilot behavior
- Before writing any code, state which section of ARCHITECTURE.md you read
- After every file edit, run the full test suite and paste raw output
- If a test fails, fix it before touching anything else
- Never mark tests as skip or xfail without explicit instruction