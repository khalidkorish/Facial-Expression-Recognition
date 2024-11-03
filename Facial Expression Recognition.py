from PIL import Image
import numpy as np
from keras.utils import img_to_array
from keras.models import load_model
from keras.models import model_from_json
import os
import cv2
import face_detection as f


#importiong ohoto using camera
answer=input("Do you want use camera? (y/n): ")
if answer=="y":
    cap = cv2.VideoCapture(0) # Open a connection to the camera (0 is usually the default camera)

    if not cap.isOpened():# Check if the camera is opened successfully
        print("Error: Could not open camera.")
        exit()

    ret, frame = cap.read() # Capture a single frame


    if not ret:  # Check if the frame was captured successfully
        print("Error: Could not read frame.")
        exit()

    cap.release() # Release the camera

    # Save the captured frame to an image file
    cv2.imwrite('captured_image.jpg', frame)
    path=('captured_image.jpg')

elif answer=="n":
    path=input("Enter photo path: ")

file_out="recognition"                     # store value of face detection in directorey called recognition
f.detect_faces(path,file_out)              # call fun face detection

# open img using PIL

img = Image.open('recognition/face_1.jpg')
img=img.convert('L')                            # convert to gray level
img=img.resize((48,48))                           # resize image
img = img_to_array(img) 
img_pixels = np.expand_dims(img, axis=0)
img_pixels /= 255.0                                #normalization


#load model and prediction
json_file = open(os.path.join(os.path.dirname(__file__), 'shared/modelcomplete.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()



model = model_from_json(loaded_model_json)
model.load_weights(os.path.join(os.path.dirname(__file__), 'shared/modelcomplete.h5')) 
    # Load weights and them to model
predictions = model.predict(img_pixels)
max_index = int(np.argmax(predictions))
acc = np.max(predictions)
acc =round(acc,2) 
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
prediction = emotions[max_index]


print("Class: ", prediction)
print("acc: ", acc)


(Image.open('recognition/face_1.jpg')).show()