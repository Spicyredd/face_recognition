import torch
from torchvision import datasets, transforms
import numpy as np
from facenet_pytorch import InceptionResnetV1
import pickle
from torch.utils.data import DataLoader
def calculate_accuracy(predictions, labels, class_names, threshold=0.8):
    """Calculates accuracy using cosine similarity and a threshold.

    Args:
        predictions: A list of face embeddings.
        labels: A list of corresponding labels.
        class_names: A list of class names for the dataset.
        threshold: The similarity threshold for a match.

    Returns:
        Accuracy as a float.
    """
    correct_predictions = 0
    total_predictions = 0

    for i in range(len(predictions)):
        for j in range(len(predictions)):
            if i != j:  # Compare different embeddings
                similarity = cosine_similarity(predictions[i], predictions[j])
                if similarity >= threshold:
                    predicted_label = class_names[labels[j]]
                else:
                    predicted_label = "Unknown"

                if predicted_label == class_names[labels[i]]:
                    correct_predictions += 1
                total_predictions += 1

    return correct_predictions / total_predictions if total_predictions else 0.0

# Define the function for calculating the cosine similarity between two face embeddings
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Define the face recognition function
def recognize_face(model, image, known_embeddings, known_labels, threshold=0.6):
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
        if max_similarity >= threshold:  
            predicted_label = known_labels[np.argmax(similarities)]
        else:
            predicted_label = "Unknown"
    else:
        print("No matching embeddings found")
        max_similarity = 0.0
        predicted_label = "Unknown"

    return predicted_label, max_similarity

if __name__ == "__main__":
    # Set up the training data and dataloaders
    train_data = datasets.ImageFolder(root='faces_training', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ]))

    val_data = datasets.ImageFolder(root='faces_validation', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]))

    # Set up the model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    resnet = resnet.to(device)  

    known_face_embeddings = []

    name_mapping = {}

    # Train the model (or load embeddings if already trained)
    epochs = 15 
    if not known_face_embeddings:
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)  # Use the validation set
        optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-5)
        criterion = torch.nn.CosineSimilarity(dim=1)

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}')
            resnet.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                embeddings = resnet(images)

                # Calculate loss for each pair of embeddings
                loss_values = []
                for j in range(embeddings.shape[0]):
                    for k in range(j + 1, embeddings.shape[0]):
                        if labels[j] == labels[k]:
                            loss = 1 - criterion(embeddings[j].unsqueeze(0), embeddings[k].unsqueeze(0))
                            loss_values.append(loss)

                # Calculate the average loss over all pairs 
                if loss_values:
                    loss = torch.mean(torch.stack(loss_values)) 
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

            # Calculate training accuracy (for the entire epoch)
            train_predictions = []
            train_labels = []
            resnet.eval()
            with torch.no_grad():
                for images, labels in train_loader:
                    images = images.to(device)
                    embeddings = resnet(images).cpu().numpy()
                    for j, embedding in enumerate(embeddings):
                        train_predictions.append(embedding.flatten())
                        train_labels.append(labels[j].item())

            train_accuracy = calculate_accuracy(train_predictions, train_labels, train_data.classes, threshold=0.8)
            train_losses.append(running_loss / len(train_loader))
            train_accuracies.append(train_accuracy)
            print(f'Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}')

            # Calculate validation accuracy (for the entire epoch)
            val_predictions = []
            val_labels = []
            resnet.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    embeddings = resnet(images).cpu().numpy()
                    for j, embedding in enumerate(embeddings):
                        val_predictions.append(embedding.flatten())
                        val_labels.append(labels[j].item())

            val_accuracy = calculate_accuracy(val_predictions, val_labels, val_data.classes, threshold=0.8)
            val_losses.append(running_loss / len(val_loader))
            val_accuracies.append(val_accuracy)
            print(f'Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}')

        # Calculate and save known embeddings after training
        resnet.eval()
        with torch.no_grad():
            known_face_embeddings = []
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                embeddings = resnet(images).cpu().numpy()
                for j, embedding in enumerate(embeddings):
                    name = train_data.classes[labels[j].item()]
                    known_face_embeddings.append((embedding.flatten(), name))

            # Save trained embeddings
            with open('known_face_embeddings.pkl', 'wb') as f:
                pickle.dump(known_face_embeddings, f)

    known_labels = [name for _, name in known_face_embeddings]

    # ... (Your existing code for video processing using MTCNN and recognize_face)

