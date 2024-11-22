import gymnasium as gym
import sys
import os
from stable_baselines3 import PPO

# 确保项目路径正确
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import highway_env


def load_model(model_path):
    """
    加载模型文件，如果文件不存在则返回 None。
    """
    if os.path.exists(model_path):
        print(f"正在加载模型：{model_path}")
        return PPO.load(model_path)
    else:
        print(f"未找到模型文件：{model_path}，将切换到随机执行模式。")
        return None


def configure_env():
    """
    创建并配置环境。
    """
    env = gym.make("intersection-v0", render_mode="human")
    env.configure({
        "screen_width": 600,  # 屏幕宽度
        "screen_height": 400,  # 屏幕高度
        "show_trajectories": True,  # 显示车辆轨迹
        "scaling": 1.5  # 放大比例
    })
    return env


def select_execution_mode(model):
    """
    选择执行模式：
    - 如果模型存在，用户可选择按策略执行或随机执行。
    - 如果模型不存在，则默认随机执行。
    """
    if model:
        use_policy = input("请选择执行模式 (p: 按策略执行, r: 随机执行): ").strip().lower() == "p"
    else:
        use_policy = False
        print("已切换到随机执行模式，因为无法加载模型。")
    return use_policy


def execute_env(env, model, use_policy):
    """
    开始运行环境，根据用户选择的模式执行动作：
    - 按策略执行或随机执行。
    """
    obs = env.reset()  # 初始化环境

    while True:
        user_input = input(">> (输入 'q' 退出，回车继续执行下一步): ").strip()
        if user_input.lower() == "q":  # 按 Q 退出程序
            print("程序退出")
            break
        else:
            # 根据模式选择动作
            if use_policy and model:
                action, _states = model.predict(obs)  # 按策略选择动作
            else:
                action = env.action_space.sample()  # 随机选择动作

            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            env.render()  # 渲染当前环境
            print(f"动作: {action}, 奖励: {reward}")

            # 如果环境结束，重置环境
            if done:
                print("环境已结束，重新开始")
                obs = env.reset()


def main():
    """
    主函数，加载模型、配置环境并执行。
    """
    # 设置模型路径
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_directory, "intersection_ppo", "model")

    # 加载模型
    model = load_model(model_path)

    # 创建并配置环境
    env = configure_env()

    try:
        # 选择执行模式
        use_policy = select_execution_mode(model)

        # 执行环境
        execute_env(env, model, use_policy)
    finally:
        # 关闭环境
        env.close()
        print("环境已关闭")


if __name__ == "__main__":
    main()
