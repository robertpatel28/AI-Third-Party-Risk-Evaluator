from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
from PyQt6 import QtCore

########################
#                      #
# Author: Robert Patel #
#                      #
########################

class ProgressDialog(QDialog):
    def __init__(self, message="Processing..."):
        super().__init__()
        self.setWindowTitle("Progress")
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(300, 100)

        layout = QVBoxLayout()
        self.label = QLabel(message)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        self.setLayout(layout)
        self.show()

    def update_progress(self, value, message=None):
        self.progress.setValue(value)
        if message:
            self.label.setText(message)