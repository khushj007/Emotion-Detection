import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load your pre-trained model (ensure the path and format are correct)
model = tf.keras.models.load_model('models\with_augmentation\CNN_with_aug.keras')  # Update path or format if needed

# Emotion labelsclear
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the face classifier
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract the face from the frame
        face = gray[y:y + h, x:x + w]
        
        # Resize the face image to the size expected by the model
        #face = cv2.resize(face, (224, 224))
        face = cv2.resize(face, (48, 48))
        
        # Convert grayscale to RGB
        #face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        # Normalize the face image

        
        face = face.astype("float32") / 255.0
        
        
        # Convert face to array and expand dimensions to match the model input
        face = img_to_array(face)
        
        face = np.expand_dims(face, axis=0)
       

        # Predict emotion
        prediction = model.predict(face)[0]
        emotion = emotion_labels[np.argmax(prediction)]

        # Display the emotion on the frame
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Show the frame with the detected faces and emotions
    cv2.imshow("Emotion Detector", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
