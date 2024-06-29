import torch
from torchvision import datasets, transforms
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle

# Define the function for calculating the cosine similarity between two face embeddings
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

if __name__ == "__main__":
    # Set up the training data and dataloaders
    train_data = datasets.ImageFolder(root='faces_cropped_training', transform=transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),  # Data augmentation: Randomly flip horizontally
        transforms.RandomRotation(10),       # Data augmentation: Rotate up to 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Data augmentation: Color jitter
        transforms.ToTensor()
    ]))

    # val_data = datasets.ImageFolder(root='faces_validation', transform=transforms.Compose([
    #     transforms.Resize((160, 160)),
    #     transforms.ToTensor()
    # ]))

    # Set up the model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    resnet = resnet.to(device)  # Move the model to the device
    
    known_face_embeddings = []

    name_mapping = {}

    # Train the model (or load embeddings if already trained)
    epochs = 10
    if not known_face_embeddings:
        print('process running')
        for i, (face_image, label) in enumerate(train_data):
            face_image = face_image.to(device)
            face_image = face_image.unsqueeze(0)
            with torch.no_grad():
                output = resnet(face_image).cpu().numpy()
                output = output.flatten()

            name = train_data.classes[label]
            known_face_embeddings.append((output, name))

        # Save trained embeddings
        with open('known_face_embeddings_big.pkl', 'wb') as f:
            pickle.dump(known_face_embeddings, f)