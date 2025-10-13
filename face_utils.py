import os
import torch
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from database import FaceDatabase
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self, data_dir="custom_faces", embeddings_path='face_embeddings.npz'):
        print("🚀 Initializing FaceNet Recognition System...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🧠 Running on device: {self.device}")

        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40, device=self.device)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        self.data_dir = data_dir
        self.embeddings_path = embeddings_path
        self.known_embeddings = {}
        self.database = FaceDatabase()
        self.recent_saves = {}

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        self.build_and_load_embeddings()
        print("✅ Face Recognition System Ready!")

    def build_and_load_embeddings(self):
        print("🔄 Building and loading face embeddings...")
        embeddings = {}
        for person in os.listdir(self.data_dir):
            person_dir = os.path.join(self.data_dir, person)
            if not os.path.isdir(person_dir): continue
            person_embeddings = []
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    face_tensor = self.mtcnn(img)
                    if face_tensor is not None:
                        with torch.no_grad():
                            emb = self.model(face_tensor.unsqueeze(0).to(self.device))
                        person_embeddings.append(emb.cpu().numpy().flatten())
                except Exception as e:
                    print(f"⚠️ Could not process image {img_path}: {e}")
            if person_embeddings:
                embeddings[person] = np.mean(person_embeddings, axis=0)
                print(f"✅ Generated embedding for: {person}")
        np.savez(self.embeddings_path, **embeddings)
        self.known_embeddings = embeddings
        print(f"💾 Saved embeddings to {self.embeddings_path}")

    def _is_recently_saved(self, name):
        """Checks if a person/unknown was saved to the DB in the last 30 seconds."""
        if name in self.recent_saves:
            if (datetime.now() - self.recent_saves[name]).total_seconds() < 30:
                return True
        return False

    def recognize_faces(self, frame, threshold=0.7):
        """Detects, recognizes faces, and saves all detections (including Unknown)."""
        face_info = []
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = self.mtcnn.detect(img)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]: continue

                face_pil = img.crop((x1, y1, x2, y2))
                face_tensor = self.mtcnn(face_pil)
                
                name, confidence, best_score = "Unknown", 0.0, 0.0

                if face_tensor is not None:
                    with torch.no_grad():
                        emb = self.model(face_tensor.unsqueeze(0).to(self.device))
                    current_embedding = emb.cpu().numpy().flatten()

                    best_match = "Unknown"
                    for person, stored_emb in self.known_embeddings.items():
                        sim = cosine_similarity([current_embedding], [stored_emb])[0][0]
                        if sim > best_score:
                            best_score, best_match = sim, person
                    
                    if best_score >= threshold:
                        name = best_match
                        confidence = best_score * 100

                # --- THIS IS THE FIX FOR UNKNOWN COUNT ---
                # Always record the best score, even for unknowns, and save to DB
                db_confidence = confidence if name != "Unknown" else best_score * 100
                
                # Save detection to database if it's a new event (known or unknown)
                if not self._is_recently_saved(name):
                    self.database.save_detection(name, f"{db_confidence:.1f}%", "FaceNet")
                    self.recent_saves[name] = datetime.now()

                # Prepare display label
                display_label = f"{name} ({db_confidence:.1f}%)"
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                # Draw on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                face_info.append({
                    "name": name,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "confidence": f"{db_confidence:.1f}%"
                })

        return frame, face_info

    def add_new_face(self, image, name):
        """Saves a new face image and rebuilds the embeddings."""
        person_folder = os.path.join(self.data_dir, name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        existing_images = [f for f in os.listdir(person_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        next_number = len(existing_images) + 1
        filename = f"{next_number:03d}.jpg"
        filepath = os.path.join(person_folder, filename)

        success = cv2.imwrite(filepath, image)
        if success:
            print(f"💾 Saved new face: {filepath}")
            self.build_and_load_embeddings()
            return True
        return False

    def get_known_people(self):
        """Gets a list of people with registered faces."""
        people = []
        if not os.path.exists(self.data_dir): return people
        for person_name in os.listdir(self.data_dir):
            person_path = os.path.join(self.data_dir, person_name)
            if os.path.isdir(person_path):
                image_count = len([f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                if image_count > 0:
                    people.append({'name': person_name, 'image_count': image_count})
        return sorted(people, key=lambda x: x['name'])