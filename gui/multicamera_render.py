from robosuite.utils import OpenCVRenderer
from robosuite import load_controller_config
import sys
import argparse
import numpy as np
import cv2
import robosuite as suite
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout,  QGridLayout, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
from PyQt5.QtWidgets import QStyleFactory
np.set_printoptions(precision=5, suppress=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="MaholoLaboratory_eefR_Move2Pipette")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--horizon", type=int, default=10000)
    args = parser.parse_args()

class Updater(QObject):
    update_signal_front = pyqtSignal(np.ndarray)
    update_signal_bird = pyqtSignal(np.ndarray)
    update_signal_side = pyqtSignal(np.ndarray)
    update_signal_agent = pyqtSignal(np.ndarray)
    update_info_signal = pyqtSignal()

    def __init__(self, env):
        super().__init__()
        self.env = env
        
    def update(self):

        obs, reward, done, _ = self.env.step(self.action)
        
        # 使用 Robosuite 渲染并发射四个摄像头的图像
        for camera_name in ["front", "bird", "side", "agent"]:
            image = self.env.sim.render(camera_name=f"{camera_name}view", width=2560, height=1344)
            image = np.flip(image, axis=0)
            getattr(self, f"update_signal_{camera_name}").emit(image)
        self.update_info_signal.emit()


class MainWindow(QMainWindow):
    def __init__(self, env):
        super().__init__()
        self.env = env

        # 初始化 QLabel 控件
        self.image_label_front = QLabel(self)
        self.image_label_bird = QLabel(self)
        self.image_label_side = QLabel(self)
        self.image_label_agent = QLabel(self)

        self.initUI()
        self.action_index = 0
        self.setupActionTimer()

        # 创建 ImageUpdater 实例并分配给 self.updater
        self.updater = Updater(env)
        self.updater.update_signal_front.connect(self.updateImageFront)
        self.updater.update_signal_bird.connect(self.updateImageBird)
        self.updater.update_signal_side.connect(self.updateImageSide)
        self.updater.update_signal_agent.connect(self.updateImageAgent)
        self.updater.update_info_signal.connect(self.updateEnvInfo)  # 连接信号

    def initUI(self):
        self.setWindowTitle('Robosuite Environment')
        self.setGeometry(400, 0, 1056, 800)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        # 主布局
        main_layout = QVBoxLayout()

        # 网格布局用于显示图像
        grid_layout = QGridLayout()
        grid_layout.setSpacing(4)  # 移除控件之间的间距
        grid_layout.setContentsMargins(4, 4, 4, 4)  # 移除布局边缘的边距

        # 初始化 QLabel 控件并设置大小策略
        self.image_label_front = QLabel(self)
        self.image_label_front.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label_bird = QLabel(self)
        self.image_label_bird.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label_side = QLabel(self)
        self.image_label_side.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label_agent = QLabel(self)
        self.image_label_agent.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        grid_layout.addWidget(self.image_label_front, 0, 0)
        grid_layout.addWidget(self.image_label_bird, 0, 1)
        grid_layout.addWidget(self.image_label_side, 1, 0)
        grid_layout.addWidget(self.image_label_agent, 1, 1)

        # 初始化 QLabel 控件并设置大小策略
        self.initImageLabel('frontview', 0, 0, grid_layout)
        self.initImageLabel('birdview', 0, 1, grid_layout)
        self.initImageLabel('sideview', 1, 0, grid_layout)
        self.initImageLabel('agentview', 1, 1, grid_layout)

        # 将网格布局添加到主布局
        main_layout.addLayout(grid_layout)

        # 信息标签
        self.info_label = QLabel(self)
        self.info_label.setWordWrap(True)
        self.info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        main_layout.addWidget(self.info_label)

        # 设置中心控件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def initImageLabel(self, camera_name, row, column, grid_layout):
        label = QLabel(self)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # 文字对齐方式
        label.setStyleSheet("QLabel { color : white; }")  # 文字颜色

        # 创建一个子 QLabel 来显示相机名称
        name_label = QLabel(camera_name, label)
        name_label.setStyleSheet("background-color: rgba(0, 0, 0, 128); padding: 2px;")  # 半透明背景

        # 将 QLabel 添加到网格布局
        grid_layout.addWidget(label, row, column)

        # 将子 QLabel 添加到类属性中，以便可以更新图像
        setattr(self, f'image_label_{camera_name}', label)


    def updateImageFront(self, image):
        self.updateImage(self.image_label_front, image)

    def updateImageBird(self, image):
        self.updateImage(self.image_label_bird, image)

    def updateImageSide(self, image):
        self.updateImage(self.image_label_side, image)

    def updateImageAgent(self, image):
        self.updateImage(self.image_label_agent, image)


    def updateImage(self, label, image):
        if label is None:
            return  # 如果传入的 label 是 None，直接返回不执行更新

        # 确保 image.data 是字节类型
        image_data = image.data
        if isinstance(image_data, memoryview):
            image_data = image_data.tobytes()

        # 创建 QImage 对象
        qimage = QImage(image_data, image.shape[1], image.shape[0], QImage.Format_RGB888)

        # 将 QImage 转换为 QPixmap 并设置到相应的 QLabel 上
        pixmap = QPixmap.fromImage(qimage).scaled(self.image_label_front.width(), self.image_label_front.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)


    def updateEnvInfo(self):
        joint_positions = self.env.robots[0].sim.data.qpos
        info_text = f"""
        🔱 ROUND {self.action_index}
        ✤✤ action ✤✤ : {self.action}
        ✤✤ Robot ✤✤
        body         : {joint_positions[0]}
        left_arm     : {joint_positions[1:8]}
        right_arm    : {joint_positions[10:17]}
        left_gripper : {joint_positions[8:10]}
        right_gripper: {joint_positions[17:19]}
        ✤✤ Object ✤✤
        pipette004_pos : {joint_positions[19:22]}, pipette004_quat: {joint_positions[22:26]}
        tube008_pos    : {joint_positions[26:29]}, tube008_quat   : {joint_positions[29:33]}
        """
        self.info_label.setText(info_text)

    def updateAction(self):
        # self.action = np.random.uniform(-1, 1, size=self.env.action_dim)
        self.action = np.zeros(14)
        self.updater.action = self.action
        self.updater.update()
        self.updateEnvInfo()
        self.action_index += 1

    def setupActionTimer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateAction)
        self.timer.start(20)

    def keyPressEvent(self, event):
        delta = 1
        arm_delta = 7
        char = event.text()  # 获取按键对应的字符
        print(f"KeyPressEvent Triggered Pressed: {char}")

        # 使用 char 变量来判断按下的键
        if char == "w":
            self.action[0+arm_delta] = -delta
        elif char == "s":
            self.action[0+arm_delta] = delta
        elif char == "s":
            self.action[0+arm_delta] = delta
        elif char == "a":
            self.action[1+arm_delta] = -delta
        elif char == "d":
            self.action[1+arm_delta] = delta
        elif char == "q":
            self.action[2+arm_delta] = delta
        elif char == "e":
            self.action[2+arm_delta] = -delta

        elif char == "j":
            self.action[3+arm_delta] = delta/2
        elif char == "l":
            self.action[3+arm_delta] = -delta/2
        elif char == "k":
            self.action[4+arm_delta] = delta/2
        elif char == "i":
            self.action[4+arm_delta] = -delta/2
        elif char == "o":
            self.action[5+arm_delta] = delta/2
        elif char == "u":
            self.action[5+arm_delta] = -delta/2
        elif char == "1":
            self.action[6+arm_delta] = delta
        elif char == "0":
            self.action[6+arm_delta] = -delta

        elif char == "W":
            self.action[0] = -delta
        elif char == "S":
            self.action[0] = delta
        elif char == "A":
            self.action[1] = -delta
        elif char == "D":
            self.action[1] = delta
        elif char == "Q":
            self.action[2] = delta
        elif char == "E":
            self.action[2] = -delta

        elif char == "J":
            self.action[3] = delta/2
        elif char == "L":
            self.action[3] = -delta/2
        elif char == "K":
            self.action[4] = delta/2
        elif char == "I":
            self.action[4] = -delta/2
        elif char == "O":
            self.action[5] = delta/2
        elif char == "U":
            self.action[5] = -delta/2
        elif char == "!":
            self.action[6] = delta
        elif char == "~":
            self.action[6] = -delta

        # 立即应用动作
        self.updater.action = self.action
        self.updater.update()

    def keyReleaseEvent(self, event):
        # 释放按键时重置动作
        self.action[:] = 0

        # 立即停止动作
        self.updater.action = self.action
        self.updater.update()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller_config = load_controller_config(default_controller=args.controller)
    env = suite.make(
        args.environment,
        args.robots,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        control_freq=50,
        horizon=args.horizon,
    )

    window = MainWindow(env)
    window.show()
    sys.exit(app.exec_())
