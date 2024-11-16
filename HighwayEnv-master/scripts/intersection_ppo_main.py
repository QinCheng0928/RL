import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env  # noqa: F401
import os
current_directory = os.getcwd()
print(current_directory)
# ============================================
#    tensorboard --logdir=intersection_ppo/
#            http://localhost:6006/
# ============================================
# ==================================
#        Main script
# ==================================

def train():
    n_cpu = 4
    batch_size = 128
    env = make_vec_env("intersection-v1", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    # env = gym.make('intersection-v0')
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=200,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.99,
        verbose=2,
        tensorboard_log="intersection_ppo/",
        seed=2000
    )
    # Train the agent
    model.learn(total_timesteps=int(2e4))
    # Save the agent
    model.save("intersection_ppo/model")    

def evaluate():
    model = PPO.load(current_directory + "\intersection_ppo\model")
    # env = gym.make("intersection-v0", render_mode="rgb_array")
    env = gym.make("intersection-v1", render_mode="human")
    for i in range(50):
        obs, info = env.reset()
        done = truncated = False
        rewards = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            rewards += reward
            env.render()
            # 获取并打印受控车辆的速度和动作
            if hasattr(env.unwrapped, "controlled_vehicles") and env.unwrapped.controlled_vehicles:
                controlled_vehicle_speed = env.unwrapped.controlled_vehicles[0].speed
                print("Controlled Vehicle Speed:", controlled_vehicle_speed)
                print("Selected Action:", action)
        print(str(i) + "th: reward = " + str(rewards))

if __name__ == "__main__":
    istrain = False
    if istrain:
        print("Training...")
        train()
    else:
        print("evaluating...")
        evaluate()
