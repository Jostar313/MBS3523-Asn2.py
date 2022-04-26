import cv2
from keras.models import load_model
import numpy as np
import time

facedetect = cv2.CascadeClassifier('C:\\Users\long\PycharmProjects\pythonProject\haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX

model = load_model('keras_model.h5')

t_old = 0
t_new = 0

def get_className(classNo):
    if classNo == 0:
        return "HYL"
    elif classNo == 1:
        return "Teemo"
    elif classNo == 2:
        return "Yip"


while True:
    ret, frame = cam.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        crop_img = frame[y:y + h, x:x + h]
        img = cv2.resize(crop_img, (224, 224))
        img = img.reshape(1, 224, 224, 3)
        prediction = model.predict(img)
        classIndex = np.argmax(prediction)
        probabilityValue = np.amax(prediction)

        if classIndex == 0 or 1 or 2:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 255, 0), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (80, 255, 0), -2)
            cv2.putText(frame, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str(round(probabilityValue * 100, 2)) + "%", (10, 110), font, 1.5, (255, 0, 0), 2,
                    cv2.LINE_AA)

    t_new = time.time()
    fps = 1 / (t_new - t_old)
    t_old = t_new
    cv2.putText(frame, 'FPS = ' + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Face Recognition Result", frame)
    if cv2.waitKey(1) & 0xff == 27:
        break

cam.release()
cv2.destroyAllWindows()
