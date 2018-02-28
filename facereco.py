# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 20:14:00 2018

@author: Raj Shah
"""
from PIL import Image,ImageDraw
import cv2
import face_recognition

#Loading the image
image=face_recognition.load_image_file("C:/Users/Raj Shah/Pictures/Saved Pictures/IMG_0106.JPG")
face_locations=face_recognition.face_locations(image)  #Locating Images on the face
#print(face_locations[0][0])
top, right, bottom, left = face_locations[0]
crop_img=image[top:bottom,left:right]
cv2.imshow('sample',crop_img)

face_landmarks_list=face_recognition.face_landmarks(image)
for face_landmarks in face_landmarks_list:
    pil_img=Image.fromarray(image)
    d=ImageDraw.Draw(pil_img,'RGBA')
    # Make the eyebrows into a nightmare
    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

    # Gloss the lips
    d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
    d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

    # Sparkle the eyes
    d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
    d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

    # Apply some eyeliner
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

    pil_img.show()
    
video_mode=True
if video_mode:
    video_capture = cv2.VideoCapture(0)
    image_encoding = face_recognition.face_encodings(image)[0]
    
    temp_image = face_recognition.load_image_file("C:/Users/Raj Shah/Pictures/Saved Pictures/sample_s.jpg")
    temp_image_encoding=face_recognition.face_encodings(temp_image)[0]
    
    hemi_image = face_recognition.load_image_file("C:/Users/Raj Shah/Pictures/Saved Pictures/sample_h.jpg")
    hemi_image_encoding=face_recognition.face_encodings(hemi_image)[0]
    known_face_encodings=[image_encoding,temp_image_encoding,hemi_image_encoding]
    
    known_face_names=["Raj","Sample_S","Sample_H"]
    
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    
    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame
        
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(rgb_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(rgb_frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', rgb_frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
cv2.waitKey(0)
cv2.destroyAllWindows()