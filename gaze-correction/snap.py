import itertools
from operator import gt
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
mp_face_mesh = mp.solutions.face_mesh
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                         min_detection_confidence=0.5)
face_mesh_videos = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, 
                                         min_detection_confidence=0.5,min_tracking_confidence=0.3)
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize the VideoCapture object to read from the webcam.
def detectFacialLandmarks(image, face_mesh, display = True):
    '''
    This function performs facial landmarks detection on an image.
    
    '''
    # Perform the facial landmarks detection on the image, after converting it into RGB format.
    results = face_mesh.process(image[:,:,::-1])
    
    # Create a copy of the input image to draw facial landmarks.
    output_image = image[:,:,::-1].copy()
    
    # Check if facial landmarks in the image are found.
    if results.multi_face_landmarks:
 
        # Iterate over the found faces.
        for face_landmarks in results.multi_face_landmarks:
 
            # Draw the facial landmarks on the output image with the face mesh tesselation
            # connections using default face mesh tesselation style.
            mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None, 
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
 
            # Draw the facial landmarks on the output image with the face mesh contours
            # connections using default face mesh contours style.
            mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None, 
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
 
    # Check if the original input image and the output image are specified to be displayed.
    if not display:
        # Return the output image in BGR format and results of facial landmarks detection.
        return np.ascontiguousarray(output_image[:,:,::-1], dtype=np.uint8), results   

def isOpen(image, face_mesh_results, face_part, threshold=5, display=True):
    '''
    This function checks whether the an eye or mouth of the person(s) is open, 
    utilizing its facial landmarks.
    '''
    image_height, image_width, _ = image.shape
    
   
    # Create a dictionary to store the isOpen status of the face part of all the detected faces.
    status={}
    # Check if the face part is right eye.    
    if face_part == 'RIGHT EYE':
        INDEXES = mp_face_mesh.FACEMESH_RIGHT_EYE 
    elif face_part == 'LEFT EYE' :
        INDEXES = mp_face_mesh.FACEMESH_LEFT_EYE   
    #     # # Specify the location to write the is right eye open status.
    #     # loc = (image_width-300, 30)
        
    #     # Initialize a increment that will be added to the status writing location, 
    #     # so that the statuses of two faces donot overlap.
    #     # increment=30
    
    # # Otherwise return nothing.
    else:
        return
    
    # Iterate over the found faces.
    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
        
         # Get the height of the face part.
        _, height, _ = getSize(image, face_landmarks, INDEXES)
        
         # Get the height of the whole face.
        _, face_height, _ = getSize(image, face_landmarks, mp_face_mesh.FACEMESH_FACE_OVAL)
        
        # Check if the face part is open.
        if (height/face_height)*100 > threshold:
            
            # Set status of the face part to open.
            status[face_no] = 'OPEN'
            
            # Set color which will be used to write the status to green.
            color=(0,255,0)
        
        # Otherwise.
        else:
            # Set status of the face part to close.
            status[face_no] = 'CLOSE'
        # Return the output image and the isOpen statuses of the face part of each detected face.
        return status

def getSize(image, face_landmarks, INDEXES):
    '''
    This function calculate the height and width of a face part utilizing its landmarks.
    '''
    image_height, image_width, _ = image.shape
    
    # Convert the indexes of the landmarks of the face part into a list.
    INDEXES_LIST = list(itertools.chain(*INDEXES))
    landmarks = []
    for INDEX in INDEXES_LIST:
        landmarks.append([int(face_landmarks.landmark[INDEX].x * image_width),
                               int(face_landmarks.landmark[INDEX].y * image_height)])
    _, _, width, height = cv2.boundingRect(np.array(landmarks))
    landmarks = np.array(landmarks)
    return width, height, landmarks

def overlay(image, filter_img, face_landmarks, face_part, INDEXES, display=True):
    '''
    This function will overlay a filter image over a face part of a person in the image/frame.
    '''
    annotated_image = image.copy()
    try:
        filter_img_height, filter_img_width, _  = filter_img.shape
        _, face_part_height, landmarks = getSize(image, face_landmarks, INDEXES)
        required_height = int(face_part_height*2.5)
        resized_filter_img = cv2.resize(filter_img, (int(filter_img_width*
                                                         (required_height/filter_img_height)),
                                                     required_height))
        
        filter_img_height, filter_img_width, _  = resized_filter_img.shape
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)
 
        center = landmarks.mean(axis=0).astype("int")
        # Calculate the location where the eye filter image will be placed.  
        location = (int(center[0]-filter_img_width/2), int(center[1]-filter_img_height/2))
 
        # Retrieve the region of interest from the image where the filter image will be placed.
        ROI = image[location[1]: location[1] + filter_img_height,
                    location[0]: location[0] + filter_img_width]
 
        # Perform Bitwise-AND operation. This will set the pixel values of the region where,
        # filter image will be placed to zero.
        resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask)
 
        # Add the resultant image and the resized filter image.
        # This will update the pixel values of the resultant image at the indexes where 
        # pixel values are zero, to the pixel values of the filter image.
        resultant_image = cv2.add(resultant_image, resized_filter_img)
 
        # Update the image's region of interest with resultant image.
        annotated_image[location[1]: location[1] + filter_img_height,
                        location[0]: location[0] + filter_img_width] = resultant_image
            
    # Catch and handle the error(s).
    except Exception as e:
        pass
    
    # Check if the annotated image is specified to be displayed.
    if display:
        plt.figure(figsize=[10,10])
        plt.imshow(annotated_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    else:
        return annotated_image

camera_video = cv2.VideoCapture(0)
 
lefteye_animation = cv2.VideoCapture('gaze-correction/ezgif-4-1d0457a163.mp4')
lefteye_frame_counter = 0
# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        continue
        
    _, lefteye_frame = lefteye_animation.read()
    # Increment the  video frame counter.
    lefteye_frame_counter += 1
    
    # Check if the current frame is the last frame of the eye video.
    if lefteye_frame_counter == lefteye_animation.get(cv2.CAP_PROP_FRAME_COUNT):     
        # Set the current frame position to first frame to restart the video.
        lefteye_animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
        lefteye_frame_counter = 0
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    # Perform Face landmarks detection.
    _, face_mesh_results = detectFacialLandmarks(frame, face_mesh_videos, display=False)
    
    if face_mesh_results.multi_face_landmarks:
        
        # Get the left eye isOpen status of the person in the frame.
        left_eye_status = isOpen(frame, face_mesh_results, 'LEFT EYE', 
                                        threshold=4.5 , display=False)
        
        # Get the right eye isOpen status of the person in the frame.
        right_eye_status = isOpen(frame, face_mesh_results, 'RIGHT EYE', 
                                         threshold=4.5, display=False)
        
        # Iterate over the found faces.
        for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            if right_eye_status[face_num] == 'OPEN':
                # Overlay the right eye image on the frame at the appropriate location.
                frame = overlay(frame, lefteye_frame, face_landmarks,
                                'RIGHT EYE', mp_face_mesh.FACEMESH_RIGHT_EYE, display=False)
            
            if left_eye_status[face_num] == 'OPEN':
                frame = overlay(frame, lefteye_frame, face_landmarks, 
                                'LEFT_EYE', mp_face_mesh.FACEMESH_LEFT_EYE, display=False)

    # Display the frame.
    cv2.imshow('Face Filter', frame)
    if cv2.waitKey(10) == 27:
        break
             
camera_video.release()
cv2.destroyAllWindows()