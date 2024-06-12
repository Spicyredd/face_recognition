import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import timm
from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN(
    image_size=224,
    margin = 20,
    min_face_size = 20,
    keep_all=True,
    select_largest=True
)

def extract_face(image_path):
    image = cv2.imread(image_path)
    boxes, probs = mtcnn.detect(image)
    if boxes is not None:
        boxes = [[int(box[-1])]+[int(x) for x in box[:-1]] for box in boxes]
        top, right, bottom, left = boxes[0]
        cropped_image = image[bottom:top, right:left]
        return cropped_image

#defining FaceDataset
class FaceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Iterate through subfolders (people)
        for person_dir in os.listdir(root):
            person_path = os.path.join(root, person_dir)
            if os.path.isdir(person_path):
                for file in os.listdir(person_path):
                    if file.endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(person_path, file))
                        self.labels.append(person_dir)  # Use subfolder name as label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        face_image = extract_face(image_path)
        
        if self.transform:
            face_image = self.transform(face_image)
        
        return face_image, self.labels[idx]  # Return image and label

# Define the training process
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        # optimizer.zero_grad()
        # outputs = model(inputs)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        # running_loss += loss.item()
        running_loss = 3

    return running_loss / len(train_loader)

# Define the evaluation process
def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

# Define the function for calculating the cosine similarity between two face embeddings
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

if __name__ == "__main__":
    # Define the face recognition function
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # Define the face recognition function (using InceptionResnetV1 for embedding extraction)
    def recognize_face(model, image, known_embeddings, known_labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = resnet(input_tensor).cpu().numpy()  # Use InceptionResnetV1 for embedding extraction
            embedding = embedding.flatten()

        similarities = [cosine_similarity(embedding, known_embedding) for known_embedding in known_embeddings]

        if similarities:
            max_similarity = np.max(similarities)
            predicted_label = known_labels[np.argmax(similarities)]
        else:
            print("No matching embeddings found")
            max_similarity = 0.0
            predicted_label = None

        return predicted_label, max_similarity

    # Set up the training data and dataloaders
    train_data = datasets.ImageFolder(root='faces_training', transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]))
    val_data = datasets.ImageFolder(root='faces_validation', transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]))


    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # Set up the model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    resnet = resnet.to(device)  # Move the model to the device
    model = InceptionResnetV1(pretrained='vggface2').to(device)  # Define the model (you can use InceptionResnetV1 here)
    optimizer = optim.Adam(model.parameters())  # Define the optimizer
    criterion = nn.CrossEntropyLoss()  # Define the loss function

    known_face_embeddings = []
    name_mapping = {}

    # Train the model
    epochs = 10
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_accuracy = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {val_accuracy:.4f}')
        for i, (face_image, label) in enumerate(train_data):

            # Load and preprocess the image
            face_image = face_image.to(device)  # Move image to device

            face_image = face_image.unsqueeze(0)  # Add batch dimension

            # Get the face embedding
            with torch.no_grad():
                output = model(face_image).cpu().numpy()  # Use InceptionResnetV1 for embedding extraction
                output = output.flatten()

            name = train_data.classes[label]

            # Store the embedding and its label
            known_face_embeddings.append((output, name))

    import pickle
    with open('known_face_embeddings.pkl', 'wb') as f:
        pickle.dump(known_face_embeddings, f)

    # ... (Your existing code for video processing using MTCNN and recognize_face)