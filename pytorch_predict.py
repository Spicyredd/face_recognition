import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import pickle
import numpy as np
from pytorch_train import FaceNet, cosine_similarity

# ... (Your other code, including the `FaceNet` class, `cosine_similarity` function, etc.)

def recognize_face(model, image_path, known_embeddings, known_labels):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Use the same size as during training
        transforms.ToTensor(),
    ])

    image = Image.open(image_path)
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        embedding = model(input_tensor).cpu().numpy()

    # Calculate cosine similarity with each known embedding
    similarities = []
    for known_embedding in known_embeddings:
        similarity = cosine_similarity(embedding, known_embedding.T)
        similarities.append(similarity)

    if similarities:
        max_similarity = np.max(similarities)
        predicted_label = known_labels[np.argmax(similarities)]
        return predicted_label, max_similarity
    else:
        return None, 0.0

# Load the model
model = FaceNet(embedding_size=128)  # Use the same embedding size as during training
model.load_state_dict(torch.load('/content/facenet_model.pth'))
model.eval()  # Set to evaluation mode

# Load the known embeddings
with open('/content/known_face_embeddings.pkl', 'rb') as f:
    known_face_embeddings = pickle.load(f)

# Extract known embeddings and labels
known_embeddings = [embedding for embedding, _ in known_face_embeddings]
known_labels = [label for _, label in known_face_embeddings]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ... (Load the model and embeddings as shown above)

image_path = '/content/training/shankar/shankar0.jpeg'
predicted_label, max_similarity = recognize_face(model, image_path, known_embeddings, known_labels)

if predicted_label is not None:
    print(f'Predicted label: {predicted_label}, Similarity: {max_similarity:.4f}')
else:
    print("No matching face found.")