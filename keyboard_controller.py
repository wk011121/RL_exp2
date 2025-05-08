import pygame
import numpy as np
import cv2
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import ActionTuple

# Constants for Unity environment configuration
TEAM_NAME = "ControlEP?team=0"
AGENT_ID = 0

class KeyboardController:
    def __init__(self, env_path: str, port: int = 2000):
        # 初始化Unity通道
        self.engine_config_channel = EngineConfigurationChannel()
        self.engine_config_channel.set_configuration_parameters(
            width=600, height=400, time_scale=1.0
        )
        self.env_param_channel = EnvironmentParametersChannel()

        # 初始化环境
        self.env = UnityEnvironment(
            file_name=env_path, 
            base_port=port,
            side_channels=[self.env_param_channel, self.engine_config_channel],
            no_graphics=False
        )
        self.env.reset()
        
        # 获取初始观测和动作空间信息
        decision_steps, _ = self.env.get_steps(TEAM_NAME)
        
        self.reward = 0.0
        self.done = False
        self.current_position = np.zeros(3)
        self.distance = 0.0

        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Unity环境键盘控制")
        self.font = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()

        # 给予操作说明
        print("键盘控制说明:")
        print("↑: 前进")
        print("↓: 后退")
        print("←: 左转")
        print("→: 右转")
        print("空格: 重置环境")
        print("ESC: 退出")
        
    def get_action(self):
        """根据键盘输入生成动作"""
        # 默认动作: [vy, vx, vw, arm_angle, gripper]
        action = np.zeros(5, dtype=np.float32)
        keys = pygame.key.get_pressed()
        
        # 移动控制
        if keys[pygame.K_UP]:
            action[1] = 5.0  # 前进
        if keys[pygame.K_DOWN]:
            action[1] = -5.0  # 后退
        if keys[pygame.K_LEFT]:
            action[2] = -1.5  # 左转
        if keys[pygame.K_RIGHT]:
            action[2] = 1.5  # 右转
            
        # 固定手臂角度和夹爪控制
        action[3] = 10.0  # 手臂角度
        action[4] = 1.0   # 夹爪控制
        
        return ActionTuple(np.array([action]))

    def display_info(self):
        """在屏幕上显示信息"""
        self.screen.fill((0, 0, 0))
        
        # 显示奖励
        reward_text = self.font.render(f'Reward: {self.reward:.4f}', True, (255, 255, 255))
        self.screen.blit(reward_text, (10, 10))
        
        # 显示位置信息
        pos_text = self.font.render(f'Position: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f})', 
                                    True, (255, 255, 255))
        self.screen.blit(pos_text, (10, 50))
        
        # 显示距离信息
        dist_text = self.font.render(f'Distance: {self.distance:.4f}', True, (255, 255, 255))
        self.screen.blit(dist_text, (10, 90))
        
        # 显示控制说明
        instructions = [
            "Introduction:",
            "up: forward",
            "down: backward",
            "left: left turn",
            "right: right turn",
            "Space: reset",
            "ESC: exit", 
        ]
        
        for i, text in enumerate(instructions):
            instr_text = self.font.render(text, True, (200, 200, 200))
            self.screen.blit(instr_text, (10, 200 + i*30))
            
        pygame.display.flip()

    def reset_env(self):
        """重置环境"""
        self.env.reset()
        decision_steps, _ = self.env.get_steps(TEAM_NAME)
        self.reward = 0.0
        
        if len(decision_steps) > 0:
            # 更新位置信息
            self.current_position = decision_steps.obs[1][AGENT_ID][4:7]
            self.calculate_distance(decision_steps)
            
        return decision_steps

    def calculate_distance(self, results):
        """计算智能体到矿物的距离"""
        if len(results) > 0:
            robot_pos = results.obs[1][AGENT_ID][4:7]
            # 计算到原点的距离作为近似
            self.distance = np.sqrt(robot_pos[0] ** 2 + robot_pos[2] ** 2)
            return self.distance
        return 0.0

    def process_observation(self, results):
        """处理观测"""
        if len(results) > 0:
            # 更新位置信息
            self.current_position = results.obs[1][AGENT_ID][4:7]
            self.calculate_distance(results)
            # 如果想显示图像，可以取消下面的注释
            image = np.array(results.obs[0][AGENT_ID] * 255, dtype=np.uint8)
             # 将图像转换为RGB格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Observation', image)
            cv2.waitKey(1)

    def run(self):
        """运行主循环"""
        decision_steps = self.reset_env()
        
        while not self.done:
            self.clock.tick(30)  # 限制帧率
            
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.done = True
                    elif event.key == pygame.K_SPACE:
                        decision_steps = self.reset_env()
            
            # 获取动作并执行
            action = self.get_action()
            self.env.set_actions(TEAM_NAME, action)
            self.env.step()
            
            # 获取结果
            decision_steps, terminal_steps = self.env.get_steps(TEAM_NAME)
            
            # 处理结果
            if len(terminal_steps) > 0:
                self.reward = terminal_steps.reward[AGENT_ID]
                self.process_observation(terminal_steps)
                # 检查是否应该重置环境（达到终止状态）
                decision_steps = self.reset_env()
            elif len(decision_steps) > 0:
                self.reward = decision_steps.reward[AGENT_ID]
                print('-----------------------------------')
                print(f"Reward: {self.reward:.4f}")
                self.process_observation(decision_steps)
            
            # 显示信息
            self.display_info()
            
        # 清理资源
        self.env.close()
        pygame.quit()

def main():
    env_path = "/home/wk/workspace/python_ws/RL_EXP2/RL_Navigation/EpMineEnv-main/envs/SingleAgent/MineField/drl.x86_64"
    controller = KeyboardController(env_path=env_path, port=3000)
    controller.run()

if __name__ == "__main__":
    main()