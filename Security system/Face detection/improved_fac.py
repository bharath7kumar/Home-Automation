import face_recognition
import cv2
import numpy as np
import os
# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

iterator = 0
known_faces = sorted(os.listdir('known_faces'))
face_image = []
face_image_encoding = []

for image in known_faces:
    face_image.append(cv2.imread("known_faces/"+image).astype(np.uint8))
    face_image_encoding.append(face_recognition.face_encodings(face_image)[0] )

# Load a sample picture and learn how to recognize it.
#obama_image = cv2.imread("emma.jpeg").astype(np.uint8)
#obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
#biden_image = cv2.imread("me.jpg").astype(np.uint8)
#biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names

known_face_encodings = face_image_encoding

known_face_names = [image.split('.')[0] for image in known_faces]

#print(known_face_encodings)
#print(known_face_names)

#known_face_encodings = [
#    obama_face_encoding,
#    biden_face_encoding
#]
#known_face_names = [
#    "Emma",
#    "Bharath"
#]
frame_skip = True
k = 0
i = 0
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    if frame_skip:
    # Find all the faces and face enqcodings in the frame of video
       face_locations = face_recognition.face_locations(rgb_frame)
       face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
       for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

         name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
         if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            cv2.imwrite("unknown_faces/image_"+str(i)+".jpg", frame)
            i = i + 1
         print(name)
      

    # Display the resulting image
    cv2.imshow('Video', frame)
    k=k+1
    if k%50==0:
        frame_skip= True
    else:
        frame_skip= False
        

    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
