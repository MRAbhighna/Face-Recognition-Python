import cv2
import os
import numpy as np
import json

# --- Configuration ---
# Path to the pre-trained Haar Cascade XML file for face detection
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' # type: ignore
# Directory to store training data (create this folder and subfolders for each person)
training_data_dir = 'training-data' 
# The desired size for face images during training
face_size = (100, 100)
# File name for the saved face recognition model
model_file_name = 'face_recognizer_model.yml' # Using YML for saving
# File name for the saved names map (to associate labels with names)
names_map_file_name = 'names_map.json'

# --- 1. Face Detection Setup ---
face_detector = cv2.CascadeClassifier(haar_cascade_path)
if face_detector.empty():
    print(f"Error: Could not load Haar Cascade classifier from {haar_cascade_path}")
    exit()

# --- 2. Camera Selection Menu ---
def select_camera():
    """
    Presents a menu to the user to select a camera by its number (index).
    Includes input validation to ensure a valid integer is entered.
    """
    while True:
        try:
            camera_number_str = input("Enter the camera number (e.g., 0 for default, 1 for external, etc.): ")
            camera_number = int(camera_number_str)
            
            if camera_number >= 0:
                print(f"Using camera with index: {camera_number}")
                return camera_number
            else:
                print("Camera number cannot be negative. Please enter a non-negative integer.")
        except ValueError:
            print("Invalid input. Please enter a valid integer for the camera number.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

# --- 3. Training Data Preparation ---
def get_images_and_labels(data_dir):
    """
    Collects face images and their corresponding labels from the training data directory.
    """
    faces = []
    labels = []
    names = {}
    current_id = 0

    # Assign labels (IDs) to each person based on directory names
    # Ensure directory exists before listing
    if not os.path.exists(data_dir):
        print(f"Error: Training data directory '{data_dir}' not found.")
        return [], [], {}

    for dir_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, dir_name)
        if os.path.isdir(person_path): # Only process directories
            names[dir_name] = current_id
            current_id += 1

    # Process images for training
    for name in names:
        person_dir = os.path.join(data_dir, name)
        for image_name in os.listdir(person_dir):
            if image_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")): # Case-insensitive check
                image_path = os.path.join(person_dir, image_name)
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warning: Could not read image {image_path}. Skipping.")
                    continue

                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                detected_faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in detected_faces:
                    face_roi = gray_img[y:y+h, x:x+w]
                    resized_face = cv2.resize(face_roi, face_size, interpolation=cv2.INTER_AREA)
                    faces.append(resized_face)
                    labels.append(names[name])
                    
    cv2.destroyAllWindows() # Close any potential windows opened during debugging
    return faces, labels, names

# --- 4. Train or Load the Face Recognizer (LBPH) ---
face_recognizer = cv2.face.LBPHFaceRecognizer_create() # type: ignore
names_map = {}

if os.path.exists(model_file_name) and os.path.exists(names_map_file_name):
    print(f"Loading existing model from {model_file_name} and {names_map_file_name}...")
    try:
        face_recognizer.read(model_file_name)
        with open(names_map_file_name, 'r') as f:
            names_map = json.load(f)
        print("Model and names map loaded successfully.")
    except Exception as e:
        print(f"Error loading model or names map: {e}. Retraining...")
        # If loading fails, proceed to train
        faces, labels, names_map = get_images_and_labels(training_data_dir)
        if not faces:
            print("No faces found for training. Please add images to the 'training-data' directory.")
            exit()
        print("Training the face recognizer...")
        face_recognizer.train(faces, np.array(labels))
        face_recognizer.write(model_file_name) # Save the new model
        with open(names_map_file_name, 'w') as f:
            json.dump(names_map, f) # Save the new names map
        print("Training complete and model saved.")
else:
    print("No existing model found. Training new model...")
    faces, labels, names_map = get_images_and_labels(training_data_dir)
    
    if not faces:
        print("No faces found for training. Please add images to the 'training-data' directory.")
        exit()

    print("Training the face recognizer...")
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write(model_file_name) # Save the trained model
    with open(names_map_file_name, 'w') as f:
        json.dump(names_map, f) # Save the names map
    print("Training complete and model saved.")

# --- 5. Real-time Face Recognition ---

# Get the selected camera index from the user
selected_camera_index = select_camera()

video_capture = cv2.VideoCapture(selected_camera_index)

if not video_capture.isOpened():
    print(f"Error: Could not open camera {selected_camera_index}.")
    exit()

# Reverse the names_map to get names from labels for display
reverse_names_map = {str(label): name for name, label in names_map.items()} # Ensure label is string if stored as such

print("\nStarting real-time face recognition. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in detected_faces:
        face_roi = gray_frame[y:y+h, x:x+w]
        
        # Resize face for consistent input to the recognizer
        resized_face = cv2.resize(face_roi, face_size, interpolation=cv2.INTER_AREA)

        label, confidence = face_recognizer.predict(resized_face)
        # Note: The 'label' returned by predict is an integer.
        # Ensure the key in reverse_names_map matches the type (integer or string)
        name = reverse_names_map.get(str(label), "Unknown") # Use str(label) to match string keys from JSON
        
        # You can adjust this confidence threshold based on your training data and desired accuracy
        # Lower confidence means higher similarity, but may lead to misclassifications for unknowns.
        # Higher confidence means lower similarity, potentially labeling more known faces as "Unknown".
        # Experiment with this value to find the best balance.

        if confidence > 100:
            confindence = 0
            text = "Unknown"
            color = (255, 255, 255)
        elif confidence > 65:  # Adjust this threshold as needed
            text = f"{name} ({int(confidence)}%)"
            color = (0, 255, 0) # Green for known faces
        else:
            text = "Unknown"
            color = (255, 255, 255) # White for unknown faces

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
