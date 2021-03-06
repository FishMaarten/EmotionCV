import dlib
import cv2

from fastai.vision import *

path = '/home/becode/Documents/Use Cases/Bagaar/'
learn = load_learner(path, 'model.pkl')

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame, (x1, y1), (x2,y2), (200, 0, 0), 4)

        img_cp = gray[y1:y2, x1:x2].copy()

        prediction, idx, probability = learn.predict(Image(pil2tensor(img_cp, np.float32).div_(225)))

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        cv2.putText(frame, str(prediction), (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255),
                    2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break


cv2.destroyAllWindows()
