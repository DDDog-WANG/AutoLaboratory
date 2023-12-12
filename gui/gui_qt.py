import argparse
import sys
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils import OpenCVRenderer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QGridLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
from PyQt5.QtWidgets import QStyleFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="MaholoLaboratory_eefR_Move2Pipette")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--controller", type=str, default="JOINT_POSITION")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--video_name", type=str, default="my_video")
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)
    args = parser.parse_args()



class MainWindow(QMainWindow):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.actions = np.zeros(17)  # 初始化动作数组
        self.initUI()

        self.updater = ImageUpdater(env, self.actions)  # 将动作数组传递给ImageUpdater
        self.updater.update_signal.connect(self.updateImage)  # 连接信号

        # 设置定时器，定期调用updater的update方法
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updater.update)
        self.timer.start(30)  # 每50毫秒更新一次

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('Robosuite Environment')
        self.setGeometry(100, 100, 2560, 1960)
        self.setStyleSheet("QMainWindow { background-color: #333; color: #DDD; }")
        self.setStyle(QStyleFactory.create('Fusion'))  # 设置应用风格为Fusion

        self.image_label = QLabel(self)

        main_layout = QHBoxLayout()
        control_layout = QGridLayout()  # 使用网格布局来组织标签和按钮

        # 为标签和按钮设置样式
        label_style = "QLabel { font-size: 14px; }"
        button_style = """
            QPushButton { 
                font-size: 14px; 
                color: #DDD; 
                background-color: #555;
                border: none;
                padding: 5px;
                border-radius: 2px;
                min-width: 40px;
            }
            QPushButton:hover { background-color: #666; }
            QPushButton:pressed { background-color: #777; }
        """
        
        self.value_labels = []  # 用于显示当前值的标签列表
        for i in range(17):
            # 创建表示轴名称的标签
            axis_names = ["Body", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "Reef", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "Leef"]
            axis_label = QLabel(axis_names[i])

            # 创建显示当前值的标签
            value_label = QLabel('0.0')
            self.value_labels.append(value_label)

            # 创建上箭头按钮，连接到相应的槽函数
            up_button = QPushButton('↑')
            up_button.clicked.connect(lambda _, index=i: self.change_value(index, 0.1))

            # 创建下箭头按钮，连接到相应的槽函数
            down_button = QPushButton('↓')
            down_button.clicked.connect(lambda _, index=i: self.change_value(index, -0.1))

            # 创建重置按钮，连接到相应的槽函数
            reset_button = QPushButton('Reset')
            reset_button.clicked.connect(lambda _, index=i: self.reset_value(index))

            # 将标签和按钮添加到网格布局
            control_layout.addWidget(axis_label, i, 0)
            control_layout.addWidget(value_label, i, 1)
            control_layout.addWidget(up_button, i, 2)
            control_layout.addWidget(down_button, i, 3)
            control_layout.addWidget(reset_button, i, 4)

            axis_label.setStyleSheet(label_style)
            value_label.setStyleSheet(label_style)
            up_button.setStyleSheet(button_style)
            down_button.setStyleSheet(button_style)
            reset_button.setStyleSheet(button_style)

            
        # 首先将图像显示部分添加到主布局中
        main_layout.addWidget(self.image_label, 4)  # 图像显示占主要空间

        # 然后将控制按钮布局添加到主布局的右侧
        main_layout.addLayout(control_layout, 1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def change_value(self, index, delta):
        # 更新action数组的值，并确保其在-1和1之间
        self.actions[index] = np.clip(self.actions[index] + delta, -1, 1)
        # 更新显示标签
        self.value_labels[index].setText(f'{self.actions[index]:.1f}')
        # 调用更新函数，发送更新后的动作到环境
        self.updater.update()

    def reset_value(self, index):
        # 将指定动作重置为0
        self.actions[index] = 0.0
        # 更新显示标签
        self.value_labels[index].setText('0.0')
        # 调用更新函数，发送更新后的动作到环境
        self.updater.update()

    def updateImage(self, image):
        if self.image_label is None:
            return  # 如果self.image_label被删除，直接返回不执行更新

        # 确保 image.data 是字节类型
        image_data = image.data
        if isinstance(image_data, memoryview):
            image_data = image_data.tobytes()

        # 创建 QImage 对象
        qimage = QImage(image_data, image.shape[1], image.shape[0], QImage.Format_RGB888)

        # 将 QImage 转换为 QPixmap 并设置到 QLabel 上
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

class ImageUpdater(QObject):
    update_signal = pyqtSignal(np.ndarray)

    def __init__(self, env, actions):
        super().__init__()
        self.env = env
        self.actions = actions
        self.renderer = OpenCVRenderer(env.sim)
        
    def update(self):
        # 执行动作并获取观测值
        obs, reward, done, _ = self.env.step(self.actions)
        
        # 使用 OpenCV 渲染图像
        image = self.env.sim.render(camera_name=self.renderer.camera_name, height=1200, width=1920)[..., ::-1]  # 适当调整尺寸
        image = np.flip(image, axis=0)
        
        # 发射信号
        self.update_signal.emit(image)

if __name__ == "__main__":
    controller_config = load_controller_config(default_controller=args.controller)
    env = suite.make(
        args.environment,
        args.robots,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        control_freq=30,
        render_camera=args.camera,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        horizon=args.horizon,
    )

    app = QApplication(sys.argv)
    ex = MainWindow(env)
    ex.show()
    sys.exit(app.exec_())