import cv2
import os


def detect_faces(input_image_path, output_folder_path):
    # Load the image from file
    image = cv2.imread(input_image_path)

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Loop through detected faces and save each face separately
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y+h, x:x+w]
        output_path = os.path.join(output_folder_path, f"face_{i + 1}.jpg")
        cv2.imwrite(output_path, face)
        print(f"Face {i + 1} saved at {output_path}")
    print("ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")
# Example usage
# input_image_path = ""
# output_folder_path = "recognition"

# detect_faces(input_image_path, output_folder_path)
