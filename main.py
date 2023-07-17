import cv2
import tensorflow as tf
import numpy as np
from youtube import youtube
from spotify import spotifyApi


def main():

    class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    model = tf.keras.models.load_model("./custom_model.h5")
    faceDetect = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()

        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)

        emotion_label = ""

        for x, y, w, h in faces:
            sub_face_img = gray[y: y + h, x: x + w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            emotion_label = class_names[label]
            print(emotion_label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, class_names[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            break

        cv2.imshow("Frame", frame)
        cv2.imwrite("emotion_frame.jpg", frame)
        cv2.waitKey(5000)
        youtube(emotion_label)
        spotifyApi(emotion_label)
        break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Execute the main function
    main()
