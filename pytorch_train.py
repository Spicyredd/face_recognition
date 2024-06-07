import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os

# Define the FaceNet model architecture
class FaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Replace the last fully connected layer with a new one for embedding
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_size)

    def forward(self, x):
        x = self.resnet(x)
        return x

# Define the training process
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

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

# Define the face recognition function
def recognize_face(model, image_path, known_embeddings, known_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    embedding = None
    with torch.no_grad():
        embedding = model(input_batch).cpu().numpy()

    similarities = [cosine_similarity(embedding, known_embedding)
                    for known_embedding in known_embeddings]

    # Check if similarities is not empty before finding the max
    if similarities:
        max_similarity = np.max(similarities)
        predicted_label = known_labels[np.argmax(similarities)]
    else:
        print("No matching embeddings found")
        max_similarity = 0.3
        predicted_label = 0.3

    return predicted_label, max_similarity

# Set up the training data and dataloaders
# train_data = datasets.ImageFolder(root='/content/training', transform=transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor()
# ]))
# val_data = datasets.ImageFolder(root='/content/training', transform=transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor()
# ]))

# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# # Set up the model, optimizer, and loss function
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FaceNet().to(device)
# optimizer = optim.Adam(model.parameters())
# criterion = nn.CrossEntropyLoss()

# known_face_embeddings = []
# name_mapping = {}

# # Train the model
# epochs = 10
# for epoch in range(epochs):
#     train_loss = train(model, train_loader, optimizer, criterion, device)
#     val_accuracy = evaluate(model, val_loader, device)
#     print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {val_accuracy:.4f}')
#     for i, (face_image, label) in enumerate(train_data):

#         # Load and preprocess the image
#         face_image = face_image.to(device)  # Move image to device

#         face_image = face_image.unsqueeze(0)  # Add batch dimension

#         # Get the face embedding
#         with torch.no_grad():
#             output = model(face_image)
#             face_embedding = output.cpu().numpy()

#         name = train_data.classes[label]
#         name_mapping[label] = name

#         # Store the embedding and its label
#         known_face_embeddings.append((face_embedding, name_mapping))

# # Save the trained model
# torch.save(model.state_dict(), 'facenet_model.pth')

# import pickle
# with open('/content/known_face_embeddings.pkl', 'wb') as f:
#     pickle.dump(known_face_embeddings, f)