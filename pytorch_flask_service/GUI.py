from PyQt5.QtWidgets import (QWidget, QGridLayout, QApplication,
                             QPushButton, QLabel, QFileDialog)
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt
import sys
from PIL import Image
import argparse
from predict import predict


class MedicalReportGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.initUI()

    def initUI(self):
        # 主窗口设置
        self.setWindowTitle('医学影像报告生成系统')
        self.setWindowIcon(QIcon("cat.png"))
        self.resize(1200, 800)  # 更大的窗口尺寸

        # 主布局
        main_layout = QGridLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 左侧面板 - 图片区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x1:1, y1:1,
                    stop:0 #1a2980, stop:1 #26d0ce);
                border-radius: 10px;
                border: 2px solid #2a5c84;
                padding: 10px;
            }
        """)
        self.image_label.setMinimumSize(550, 550)

        # 按钮区域
        self.select_btn = QPushButton("选择医学影像")
        self.run_btn = QPushButton("生成诊断报告")

        # 按钮样式
        btn_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x1:1, y1:0,
                    stop:0 #1a2980, stop:1 #26d0ce);
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                min-width: 150px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x1:1, y1:0,
                    stop:0 #26d0ce, stop:1 #1a2980);
            }
            QPushButton:pressed {
                background: #1a2980;
            }
        """
        self.select_btn.setStyleSheet(btn_style)
        self.run_btn.setStyleSheet(btn_style)

        # 右侧面板 - 结果区域
        self.result_title = QLabel("AI诊断报告")
        self.result_title.setStyleSheet("""
            QLabel {
                color: #1a2980;
                font-size: 24px;
                font-weight: bold;
                padding-bottom: 10px;
                border-bottom: 2px solid #26d0ce;
            }
        """)
        self.result_title.setAlignment(Qt.AlignCenter)

        # 文本区域样式
        text_style = """
            QLabel {
                background-color: rgba(255, 255, 255, 0.8);
                color: #333;
                font-size: 14px;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #d3e0f1;
                min-height: 100px;
            }
        """

        # Findings区域
        self.findings_label = QLabel("影像发现(Findings):")
        self.findings_label.setStyleSheet("font-size: 16px; color: #1a2980; font-weight: bold;")

        self.findings_content = QLabel("请先选择医学影像并点击生成按钮...")
        self.findings_content.setStyleSheet(text_style)
        self.findings_content.setWordWrap(True)
        self.findings_content.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        # Impressions区域
        self.impressions_label = QLabel("诊断印象(Impressions):")
        self.impressions_label.setStyleSheet("font-size: 16px; color: #1a2980; font-weight: bold;")

        self.impressions_content = QLabel("诊断结果将显示在这里...")
        self.impressions_content.setStyleSheet(text_style)
        self.impressions_content.setWordWrap(True)
        self.impressions_content.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        # 布局组织
        main_layout.addWidget(self.image_label, 0, 0, 2, 1)
        main_layout.addWidget(self.select_btn, 0, 1)
        main_layout.addWidget(self.run_btn, 1, 1)
        main_layout.addWidget(self.result_title, 0, 2, 1, 2)
        main_layout.addWidget(self.findings_label, 1, 2)
        main_layout.addWidget(self.findings_content, 2, 2)
        main_layout.addWidget(self.impressions_label, 1, 3)
        main_layout.addWidget(self.impressions_content, 2, 3)

        # 设置列宽比例
        main_layout.setColumnStretch(0, 5)  # 图片列
        main_layout.setColumnStretch(1, 1)  # 按钮列
        main_layout.setColumnStretch(2, 2)  # Findings列
        main_layout.setColumnStretch(3, 2)  # Impressions列

        # 设置行高比例
        main_layout.setRowStretch(0, 1)  # 标题行
        main_layout.setRowStretch(1, 1)  # 标签行
        main_layout.setRowStretch(2, 3)  # 内容行

        # 连接信号槽
        self.select_btn.clicked.connect(self.open_image)
        self.run_btn.clicked.connect(self.generate_report)

        # 设置字体
        font = QFont()
        font.setFamily("Microsoft YaHei")
        self.setFont(font)

    def open_image(self):
        img_path, _ = QFileDialog.getOpenFileName(
            self, "选择医学影像", "",
            "医学影像 (*.jpg *.png *.dcm);;所有文件 (*)"
        )

        if img_path:
            self.image_path = img_path
            pixmap = QPixmap(img_path).scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
            self.findings_content.setText("已加载影像，请点击生成按钮...")
            self.impressions_content.setText("等待生成诊断结果...")

    def generate_report(self):
        if not self.image_path:
            self.show_error("请先选择医学影像!")
            return

        try:
            # 设置参数
            args = self.get_arguments()

            # 加载图像
            img = Image.open(self.image_path).convert('RGB')

            # 生成报告
            imp, fnp = predict(img, args)

            # 显示结果
            self.impressions_content.setText(imp)
            self.findings_content.setText(fnp)

        except Exception as e:
            self.show_error(f"生成报告时出错: {str(e)}")

    def get_arguments(self):
        """创建并返回参数对象"""
        parser = argparse.ArgumentParser()

        # 路径参数
        parser.add_argument('--model_path', type=str, default='./')
        parser.add_argument('--vocab_path', type=str, default='IUdata_vocab_0threshold.pkl')
        parser.add_argument('--image_dir', type=str, default='IUdata/NLMCXR_Frontal')
        parser.add_argument('--eval_json_dir', type=str, default='IUdata/IUdata_test.json')

        # 模型参数
        parser.add_argument('--eval_batch_size', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--max_impression_len', type=int, default=15)
        parser.add_argument('--max_single_sen_len', type=int, default=15)
        parser.add_argument('--max_sen_num', type=int, default=7)
        parser.add_argument('--single_punc', type=bool, default=True)
        parser.add_argument('--imp_fin_only', type=bool, default=False)

        # 图像处理参数
        parser.add_argument('--resize_size', type=int, default=256)
        parser.add_argument('--crop_size', type=int, default=224)
        parser.add_argument('--embed_size', type=int, default=512)
        parser.add_argument('--hidden_size', type=int, default=512)

        return parser.parse_args([])  # 传递空列表避免命令行参数

    def show_error(self, message):
        """显示错误消息"""
        self.findings_content.setText(f"错误: {message}")
        self.impressions_content.setText("请检查设置并重试")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    window = MedicalReportGenerator()
    window.show()

    sys.exit(app.exec_())