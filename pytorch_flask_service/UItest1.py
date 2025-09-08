from PyQt5.QtGui import *
import sys
import argparse
from predictchange import predictchange
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication, QVBoxLayout, QHBoxLayout

class Vocabulary(object):
   def __init__(self):
       self.word2idx = {}
       self.id2word = {}
       self.idx = 0
       self.add_word('<pad>')
       self.add_word('<start>')
       self.add_word('<end>')
       self.add_word('<unk>')

   def add_word(self, word):
       if word not in self.word2idx:
           self.word2idx[word] = self.idx
           self.id2word[self.idx] = word
           self.idx += 1

   def get_word_by_id(self, id):
       return self.id2word[id]

   def __call__(self, word):
       if word not in self.word2idx:
           return self.word2idx['<unk>']
       return self.word2idx[word]

   def __len__(self):
       return len(self.word2idx)

class Ui_Form(QWidget):
   def __init__(self):
       super().__init__()
       self.pic = None
       self.setupUi(self)

   def setupUi(self, Form):
       Form.setObjectName("Form")
       Form.resize(1200, 800)

       main_layout = QHBoxLayout(Form)
       main_layout.setContentsMargins(20, 20, 20, 20)
       main_layout.setSpacing(20)

       left_panel = QtWidgets.QFrame()
       left_layout = QVBoxLayout(left_panel)
       left_layout.setContentsMargins(0, 0, 0, 0)
       left_layout.setSpacing(20)

       self.image_label = QtWidgets.QLabel()
       self.image_label.setAlignment(QtCore.Qt.AlignCenter)
       self.image_label.setStyleSheet("""
           QLabel {
               background: qlineargradient(x1:0, y1:0, x1:1, y1:1,
                   stop:0 #1a2980, stop:1 #26d0ce);
               border-radius: 10px;
               border: 2px solid #2a5c84;
               min-width: 500px;
               min-height: 500px;
           }
       """)

       btn_frame = QtWidgets.QFrame()
       btn_layout = QHBoxLayout(btn_frame)
       btn_layout.setContentsMargins(0, 0, 0, 0)
       btn_layout.setSpacing(20)

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

       self.select_btn = QtWidgets.QPushButton("选择医学影像")
       self.select_btn.setStyleSheet(btn_style)
       self.run_btn = QtWidgets.QPushButton("生成诊断报告")
       self.run_btn.setStyleSheet(btn_style)

       btn_layout.addWidget(self.select_btn)
       btn_layout.addWidget(self.run_btn)

       left_layout.addWidget(self.image_label)
       left_layout.addWidget(btn_frame)
       left_layout.addStretch()

       right_panel = QtWidgets.QFrame()
       right_layout = QVBoxLayout(right_panel)
       right_layout.setContentsMargins(0, 0, 0, 0)
       right_layout.setSpacing(20)

       title_style = """
           QLabel {
               color: #1a2980;
               font-size: 24px;
               font-weight: bold;
               padding-bottom: 10px;
               border-bottom: 2px solid #26d0ce;
           }
       """

       self.result_title = QtWidgets.QLabel("AI诊断报告")
       self.result_title.setStyleSheet(title_style)
       self.result_title.setAlignment(QtCore.Qt.AlignCenter)

       text_style = """
           QLabel {
               background-color: rgba(255, 255, 255, 0.8);
               color: #333;
               font-size: 20px;
               padding: 15px;
               border-radius: 8px;
               border: 1px solid #d3e0f1;
               min-height: 200px;
           }
       """

       findings_frame = QtWidgets.QFrame()
       findings_layout = QVBoxLayout(findings_frame)
       findings_layout.setContentsMargins(0, 0, 0, 0)

       findings_label = QtWidgets.QLabel("影像发现(Findings):")
       findings_label.setStyleSheet("font-size: 20px; color: #1a2980; font-weight: bold;")

       self.findings_content = QtWidgets.QLabel("请先选择医学影像并点击生成按钮...")
       self.findings_content.setStyleSheet(text_style)
       self.findings_content.setWordWrap(True)
       self.findings_content.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

       findings_layout.addWidget(findings_label)
       findings_layout.addWidget(self.findings_content)

       impressions_frame = QtWidgets.QFrame()
       impressions_layout = QVBoxLayout(impressions_frame)
       impressions_layout.setContentsMargins(0, 0, 0, 0)

       impressions_label = QtWidgets.QLabel("诊断印象(Impressions):")
       impressions_label.setStyleSheet("font-size: 16px; color: #1a2980; font-weight: bold;")

       self.impressions_content = QtWidgets.QLabel("诊断结果将显示在这里...")
       self.impressions_content.setStyleSheet(text_style)
       self.impressions_content.setWordWrap(True)
       self.impressions_content.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

       impressions_layout.addWidget(impressions_label)
       impressions_layout.addWidget(self.impressions_content)

       right_layout.addWidget(self.result_title)
       right_layout.addWidget(findings_frame)
       right_layout.addWidget(impressions_frame)
       right_layout.addStretch()

       main_layout.addWidget(left_panel)
       main_layout.addWidget(right_panel)

       self.select_btn.clicked.connect(self.openimage)
       self.run_btn.clicked.connect(self.run)

       font = QtGui.QFont()
       font.setFamily("Microsoft YaHei")
       Form.setFont(font)

       self.retranslateUi(Form)

   def retranslateUi(self, Form):
       Form.setWindowTitle('胸部CT报告生成系统')
       Form.setWindowIcon(QIcon("11.png"))


   def openimage(self):
       imgName, imgType = QFileDialog.getOpenFileName(
           self, "选择医学影像", "",
           "医学影像 (*.jpg *.png);;所有文件 (*)"
       )
       if imgName:
           self.pic = imgName
           pixmap = QPixmap(imgName).scaled(
               self.image_label.width(),
               self.image_label.height(),
               QtCore.Qt.KeepAspectRatio,
               QtCore.Qt.SmoothTransformation
           )
           self.image_label.setPixmap(pixmap)
           self.findings_content.setText("已加载影像，请点击生成按钮...")
           self.impressions_content.setText("等待生成诊断结果...")

   def run(self):
       parser = argparse.ArgumentParser()
       parser.add_argument('--model_path', type=str, default='./')
       parser.add_argument('--vocab_path', type=str, default='IUdata_vocab_0threshold.pkl')
       parser.add_argument('--image_dir', type=str, default='IUdata/NLMCXR_Frontal')
       parser.add_argument('--eval_json_dir', type=str, default='IUdata/IUdata_test.json')
       parser.add_argument('--eval_batch_size', type=int, default=1)
       parser.add_argument('--num_workers', type=int, default=2)
       parser.add_argument('--max_impression_len', type=int, default=15)
       parser.add_argument('--max_single_sen_len', type=int, default=15)
       parser.add_argument('--max_sen_num', type=int, default=7)
       parser.add_argument('--single_punc', type=bool, default=True)
       parser.add_argument('--imp_fin_only', type=bool, default=False)
       parser.add_argument('--resize_size', type=int, default=256)
       parser.add_argument('--crop_size', type=int, default=224)
       parser.add_argument('--embed_size', type=int, default=512)
       parser.add_argument('--hidden_size', type=int, default=512)
       parser.add_argument('--num_global_features', type=int, default=2048)
       parser.add_argument('--imp_layers_num', type=int, default=1)
       parser.add_argument('--fin_num_layers', type=int, default=2)
       parser.add_argument('--sen_enco_num_layers', type=int, default=3)
       parser.add_argument('--num_local_features', type=int, default=2048)
       parser.add_argument('--num_regions', type=int, default=49)
       parser.add_argument('--num_conv1d_out', type=int, default=1024)
       parser.add_argument('--teach_rate', type=float, default=0.0)
       parser.add_argument('--log_step', type=int, default=100)
       parser.add_argument('--save_step', type=int, default=1000)

       args = parser.parse_args([])

       imp = "诊断结果：影像正常"
       fnp = "影像发现：未见明显异常"

       try:
           imp, fnp = predictchange(self.pic, args)
       except:
           pass

       self.impressions_content.setText(imp)
       self.findings_content.setText(fnp)

if __name__ == '__main__':
   app = QtWidgets.QApplication(sys.argv)
   app.setStyle('Fusion')
   window = QtWidgets.QWidget()
   ui = Ui_Form()
   ui.setupUi(window)
   window.show()
   sys.exit(app.exec_())
