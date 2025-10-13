import os
import torch
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- INITIAL SETUP -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ------------------- FUNCTION DEFINITIONS -------------------

def get_face_embedding_from_pil(img):
    """Get embedding from a PIL image (used for live camera frames)."""
    face = mtcnn(img)
    if face is None:
        return None
    with torch.no_grad():
        emb = model(face.unsqueeze(0).to(device))
    return emb.cpu().numpy().flatten()

def build_embeddings(data_dir, save_path='face_embeddings.npz'):
    """
    Create and save average embeddings for each person in the dataset.
    Folder structure:
    ├── dataset/
        ├── Aswin/
        │   ├── 1.jpg
        │   ├── 2.jpg
        ├── John/
        │   ├── a.jpg
        │   ├── b.jpg
    """
    embeddings = {}
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue
        person_embeddings = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            face = mtcnn(img)
            if face is not None:
                with torch.no_grad():
                    emb = model(face.unsqueeze(0).to(device))
                person_embeddings.append(emb.cpu().numpy().flatten())
        if person_embeddings:
            embeddings[person] = np.mean(person_embeddings, axis=0)

    np.savez(save_path, **embeddings)
    print(f"✅ Saved embeddings to {save_path}")

def recognize_from_frame(frame, data, threshold=0.6):
    """Recognize face in a single frame."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    emb = get_face_embedding_from_pil(img)
    if emb is None:
        return "No face"

    best_match = None
    best_score = 0
    for person, stored_emb in data.items():
        sim = cosine_similarity([emb], [stored_emb])[0][0]
        if sim > best_score:
            best_score = sim
            best_match = person

    if best_score >= threshold:
        return f"{best_match} ({best_score:.2f})"
    else:
        return f"Unknown ({best_score:.2f})"

# ------------------- LIVE CAMERA MODE -------------------
def recognize_live(embedding_path='face_embeddings.npz'):
    """Run real-time face recognition using webcam."""
    data = np.load(embedding_path)

    cap = cv2.VideoCapture(0)
    print("🎥 Starting camera... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        boxes, _ = mtcnn.detect(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                label = recognize_from_frame(face_crop, data)

                # Draw rectangle and label
                color = (0, 255, 0) if "Unknown" not in label else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Face Recognition (Press q to exit)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------- MAIN -------------------
if __name__ == "__main__":
    dataset_dir = "custom_faces"  # path to your dataset folder
    if not os.path.exists("face_embeddings.npz"):
        build_embeddings(dataset_dir)

    recognize_live("face_embeddings.npz")
