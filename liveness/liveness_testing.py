import timm
import torch

EPOCHS = 500
PATIENCE = 3
BATCH_SIZE = 16
EPOCH_LEN = len(str(EPOCHS))
torch.manual_seed(39)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

device = torch.device(device)

model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
model.head = torch.nn.Linear(model.head.in_features, 2)
model = model.to(device)
model.load_state_dict(
    torch.load(
        "weights/vit_teacher.pth"
    )
)
model.eval()

with torch.no_grad():
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        tp += (pred.eq(1) & target.eq(1).view_as(pred)).sum().item()
        tn += (pred.eq(0) & target.eq(0).view_as(pred)).sum().item()
        fp += (pred.eq(1) & target.eq(0).view_as(pred)).sum().item()
        fn += (pred.eq(0) & target.eq(1).view_as(pred)).sum().item()

        correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    far = fp / (fp + tn)
    frr = fn / (fn + tp)

    recall = tp / (tp + fn)

    hter = (far + frr ) / 2

    print(f"test acc: {accuracy * 100}%")
    print(f"recall: {recall * 100}%")
    print(f"far: {far * 100}%")
    print(f"frr: {frr * 100}%")
    print(f"hter: {hter * 100}%")