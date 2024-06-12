import torch
from torchvision import datasets, transforms
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle

# Define the function for calculating the cosine similarity between two face embeddings
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Define the face recognition function
def recognize_face(model, image, known_embeddings, known_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(input_tensor).cpu().numpy()
        embedding = embedding.flatten()

    similarities = [cosine_similarity(embedding, known_embedding) for known_embedding, _ in known_embeddings]

    if similarities:
        max_similarity = np.max(similarities)
        predicted_label = known_labels[np.argmax(similarities)]
    else:
        print("No matching embeddings found")
        max_similarity = 0.0
        predicted_label = None

    return predicted_label, max_similarity

if __name__ == "__main__":
    # Set up the training data and dataloaders
    train_data = datasets.ImageFolder(root='faces_training', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Data augmentation: Randomly flip horizontally
        transforms.RandomRotation(10),       # Data augmentation: Rotate up to 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Data augmentation: Color jitter
        transforms.ToTensor()
    ]))

    val_data = datasets.ImageFolder(root='faces_validation', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]))

    # Set up the model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    resnet = resnet.to(device)  # Move the model to the device

    # Load known embeddings from a file (if they exist)
    # try:
    #     with open('known_face_embeddings.pkl', 'rb') as f:
    #         known_face_embeddings = pickle.load(f)
    # except FileNotFoundError:
    known_face_embeddings = []

    name_mapping = {}

    # Train the model (or load embeddings if already trained)
    epochs = 10
    if not known_face_embeddings:
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}')
            for i, (face_image, label) in enumerate(train_data):
                face_image = face_image.to(device)
                face_image = face_image.unsqueeze(0)
                with torch.no_grad():
                    output = resnet(face_image).cpu().numpy()
                    output = output.flatten()

                name = train_data.classes[label]
                known_face_embeddings.append((output, name))

        # Save trained embeddings
        with open('known_face_embeddings.pkl', 'wb') as f:
            pickle.dump(known_face_embeddings, f)

    # Extract labels from the known embeddings
    # known_labels = [name for _, name in known_face_embeddings]

    # ... (Your existing code for video processing using MTCNN and recognize_face)

    # Example: Process a single image for demonstration
    # test_image = transforms.ToTensor()(datasets.ImageFolder(root='faces_test').imgs[0][0])
    # predicted_label, max_similarity = recognize_face(resnet, test_image, known_face_embeddings, known_labels)
    # print(f"Predicted Label: {predicted_label}, Similarity: {max_similarity}") 