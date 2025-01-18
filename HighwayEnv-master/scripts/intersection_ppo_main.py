import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
import highway_env
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    batch_size = 8
    env = make_vec_env("intersection-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    # env = gym.make('intersection-v0')
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        n_steps= batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.99,
        verbose=2,
        tensorboard_log="intersection_ppo_random/",
        seed=2000,
        device='cuda'  # 指定使用 GPU
    )
    # 检查使用的设备
    print("Device used:", model.policy.device)
    # Train the agent
    model.learn(total_timesteps=int(2e5))
    # Save the agent
    model.save("intersection_ppo_random/model")    

def evaluate():
    model = PPO.load(current_directory + "\intersection_ppo_random\model")
    env = gym.make("intersection-v0", render_mode="human")
    
    # 计数器，记录奖励小于0的次数
    negative_rewards_count = 0
    total_episodes = 50  # 评估的总回合数
    
    for i in range(total_episodes):
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
                # print("Controlled Vehicle Speed:", controlled_vehicle_speed)
                # print("Selected Action:", action)
        
        print(f"{i}th: reward = {rewards}")
        
        # 如果奖励小于0，则增加计数器
        if rewards < 0:
            negative_rewards_count += 1
    
    # 计算奖励小于0的比例
    negative_ratio = negative_rewards_count / total_episodes
    print(f"撞车比例: {negative_ratio:.2f}")

if __name__ == "__main__":
    istrain = True
    if istrain:
        print("Training...")
        train()
    else:
        print("evaluating...")
        evaluate()

