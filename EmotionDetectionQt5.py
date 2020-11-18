import cv2
import sys
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QRect
from PyQt5.QtGui import QImage, QPixmap
from simplesaad_model import SimpleSaad


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    f = None
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    frontalcatface_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
    altface_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    detection_mode = 0
    model = SimpleSaad()

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                if self.detection_mode == 1:
                    frame = self.default_processing(frame)
                elif self.detection_mode > 1:
                    frame = self.processing(frame, self.detection_mode)
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

    def default_processing(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            # apply model to get emotion label
            # label = model(image)
            label = "n/a"
            cv2.putText(frame, f"{label}", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, .7, (0, 0, 255))
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        return frame

    def processing(self, frame, mode):
        if mode == 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
            return frame
        elif mode == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.frontalcatface_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
            return frame
        elif mode == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.altface_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
            return frame
        elif mode == 5:
            # apply SimpleSaad pre-trained Keras model
            return self.model.evaluate_face(frame)
        else:
            print("mode not available")
            return frame

    def activate_detection(self,mode):
        self.detection_mode = mode

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.mode = 0
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1800, 1200)
        self.label = QLabel(self)
        self.label.move(280, 120)
        self.label.resize(640, 480)
        self.pushButton = QPushButton(self)
        self.pushButton.setGeometry(QRect(538, 40, 180, 25))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Activate Face Detection")
        self.pushButton.clicked.connect(self.activate_detection)
        self.th = Thread(self)
        self.th.changePixmap.connect(self.setImage)
        self.th.start()
        self.show()

    def activate_detection(self):
        if self.mode == 0:
            self.th.activate_detection(5)
            self.mode = 1
            self.pushButton.setText("Deactivate Face Detection")
        else:
            self.th.activate_detection(0)
            self.pushButton.setText("Activate Face Detection")
            self.mode = 0

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())