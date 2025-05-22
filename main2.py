import face_recognition as fr
import cv2
import numpy as np
import os

# === Load training images ===
train_path = "./train/"
known_names = []
known_name_encodings = []

print("Loading training images...")
for filename in os.listdir(train_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(train_path, filename)
        image = fr.load_image_file(image_path)
        encodings = fr.face_encodings(image)
        if encodings:
            known_name_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0].capitalize()
            known_names.append(name)
        else:
            print(f"⚠️ No face found in training image: {filename}")

print(f"✅ Loaded {len(known_names)} known faces.\n")

# === Ask user to pick a test image ===
test_path = "./test/"
test_images = [f for f in os.listdir(test_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if not test_images:
    print("❌ No test images found!")
    exit()

print("Available test images:")
for i, img in enumerate(test_images):
    print(f"{i+1}. {img}")

choice = int(input("Enter the number of the image to test: ")) - 1
if choice < 0 or choice >= len(test_images):
    print("❌ Invalid choice.")
    exit()

test_image_path = os.path.join(test_path, test_images[choice])
print(f"\nUsing test image: {test_images[choice]}\n")

# === Load and process the test image ===
image = cv2.imread(test_image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

face_locations = fr.face_locations(rgb_image)
face_encodings = fr.face_encodings(rgb_image, face_locations)

recognized_names = []

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = fr.compare_faces(known_name_encodings, face_encoding)
    name = "Unknown"

    face_distances = fr.face_distance(known_name_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    
    if matches[best_match_index]:
        name = known_names[best_match_index]
        recognized_names.append(name)

    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# === Print recognized names ===
if recognized_names:
    print("✅ Recognized people in the image:")
    for person in set(recognized_names):
        print(f" - {person}")
else:
    print("⚠️ No known faces were recognized.")

# === Show the result ===
cv2.imshow("Attendance Result", image)
cv2.imwrite("./output.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
input("Press Enter to exit...")
