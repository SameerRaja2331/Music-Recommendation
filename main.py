import cv2
import tensorflow as tf
import numpy as np
from youtube import youtube
from spotify import spotifyApi


def main():

    # Here we are defining the class names, loading the pre-trained cnn model we developed and
    # classifier that is used to detect faces.
    class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    model = tf.keras.models.load_model("./custom_model.h5")
    faceDetect = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    video = cv2.VideoCapture(0)

    while True:

        # Reads a frame from the video capture device using the read() method of the video object. 
        ret, frame = video.read()

        # This checks if the frame was not successfully read
        if not ret:
            continue

        # This converts the captured frame from the BGR color format to grayscale using the cvtColor() function of OpenCV.     
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Using the pre-trained face detection classifier (faceDetect) to detect faces in the grayscale frame.
        #detectMultiScale() function returns a list of bounding boxes (x, y, width, height) around the detected faces.
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)

        emotion_label = ""

        # iterates over the detected faces and performs the following operations
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

        # imshow displays the frame    
        cv2.imshow("Frame", frame)

        #once the emotion is detected after that we are storing the image copy of that frame.
        cv2.imwrite("emotion_frame.jpg", frame)

        # This pauses the execution for 5000 milliseconds (5 seconds). During this time, the frame is displayed,
        # allowing the user to view the results.
        cv2.waitKey(5000)

        # Function call for searching and playing a YouTube video based on the emotion detected.
        youtube(emotion_label)

        # Function call to suggest users spotify playlist that is related to their emotion detected.
        spotifyApi(emotion_label)
        break

    # This releases the video capture device, freeing up system resources.    
    video.release()

    #This closes all OpenCV windows that were opened during the program execution.
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Execute the main function
    main()
