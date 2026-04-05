import sys
sys.path.insert(0, '.')

from him_her.envs.predator_prey import PredatorPreyEnv

print("Creating environment...")
env = PredatorPreyEnv()

print("Resetting...")
obs, info = env.reset()

print(f"Reset OK: obs.shape={obs.shape}, switch_point={info['switch_point']}")

print("Stepping...")
for i in range(5):
    next_obs, reward, terminated, truncated, info = env.step(0)
    print(f"Step {i}: reward={reward}, terminated={terminated}, truncated={truncated}")
    if terminated or truncated:
        break

print("SUCCESS!")
