import os
import cv2
import face_recognition
import numpy as np
from db_handler import create_table, insert_face

# Ensure table exists
create_table()

# Path to your photos folder
PHOTOS_DIR = "photos"

# Loop through all images in photos folder
for filename in os.listdir(PHOTOS_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(PHOTOS_DIR, filename)
        print(f"Processing {filename}...")

        # Load image
        image = face_recognition.load_image_file(path)

        # Detect face encodings
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            embedding = encodings[0].astype(np.float32)  # store as float32
            # Use filename (without extension) as student name
            name = os.path.splitext(filename)[0]
            # Insert into DB
            insert_face(name, embedding)
            print(f"✅ Inserted {name} into database.")
        else:
            print(f"⚠️ No face found in {filename}, skipping.")
