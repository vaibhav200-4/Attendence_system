import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime
from db_handler import create_table, insert_face, get_all_faces, update_deduction, mark_absent, get_payroll

# ----------------- Setup -----------------
# Ensure DB table exists
create_table()

# Load known faces from database
faces = get_all_faces()
known_face_names = [f[0] for f in faces]
known_face_encodings = [f[1] for f in faces]

# Create attendance folder
os.makedirs("attendance_records", exist_ok=True)
current_date = datetime.now().strftime("%Y-%m-%d")
csv_filename = os.path.join("attendance_records", f"{current_date}.csv")

# Track already marked students
marked_students = set()

# Open CSV file
f = open(csv_filename, 'a', newline='')
lnwriter = csv.writer(f)

# If new file, add header
if os.stat(csv_filename).st_size == 0:
    lnwriter.writerow(["Name", "Time"])

# Open webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'a' to add a new student face, 'q' or ESC to quit.")

# Function: check if student already marked today
def already_marked_today(name, csv_filename):
    if not os.path.exists(csv_filename):
        return False
    with open(csv_filename, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row and row[0] == name:
                return True
    return False

# ----------------- Loop -----------------
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Show live video
    cv2.imshow("Face Attendance System", frame)

    # Resize frame for faster recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces in frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        name = "Unknown"
        if len(known_face_encodings) > 0:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)

        # ✅ Mark attendance only if not already in today's CSV
        # Inside your face recognition loop:
        if name != "Unknown" and not already_marked_today(name, csv_filename):
            current_time = datetime.now().strftime("%H:%M:%S")
            lnwriter.writerow([name, current_time])
            f.flush()
            print(f"{name} marked present at {current_time}")

            # Payroll Logic
            check_in_time = datetime.strptime(current_time, "%H:%M:%S").time()
            office_start = datetime.strptime("10:00:00", "%H:%M:%S").time()
            cutoff_time = datetime.strptime("11:00:00", "%H:%M:%S").time()

            if check_in_time > office_start and check_in_time < cutoff_time:
                update_deduction(name, 50)  # late → 50 deduction
                print(f"⚠ {name} was late! ₹50 deducted.")
            elif check_in_time >= cutoff_time:
                mark_absent(name)  # absent → 1 day salary deducted
                print(f"❌ {name} is Absent! 1 day salary deducted.")

            # Show updated payroll
            payroll_info = get_payroll(name)
            print(f"Payroll Updated: {payroll_info}")

    # ----------- Handle Keys ------------
    key = cv2.waitKey(1)

    # Add new student
    if key == ord('a'):
        if len(face_encodings) == 1:
            new_name = input("Enter student name: ").strip()
            if new_name:
                insert_face(new_name, face_encodings[0])
                known_face_names.append(new_name)
                known_face_encodings.append(face_encodings[0])
                print(f"✅ New student '{new_name}' added to database.")
            else:
                print("⚠️ No name entered, skipping.")
        else:
            print("⚠️ Please ensure only one face is visible to add.")

    # Quit program
    if key == ord('q') or key == 27:  # 'q' or ESC
        print("Exiting...")
        break

# ----------------- Cleanup -----------------
video_capture.release()
f.close()
cv2.destroyAllWindows()
print("Attendance session ended.")