import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from dataset import HandwrittenChineseDataset
from model import CNN, ResNet, BasicBlock
import argparse

def test(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Transformations
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Dataset
    val_dataset = HandwrittenChineseDataset(root_dir=args.dataset_path, transform=transform_test, train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
      # Model
    if args.model == 'CNN':
        model = CNN().to(device)
    elif args.model == 'ResNet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
    else:
        raise ValueError("Model not recognized")
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Testing loop
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Loss: {val_loss / len(val_loader)}, Accuracy: {100 * correct / total}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CNN on Handwritten Chinese Dataset")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/cnn_epoch_20.pth", help="Path to the trained model checkpoint")
    parser.add_argument("--model", type=str, default="CNN", choices=["CNN", "ResNet18"], help="Model to use for testing")
    args = parser.parse_args()

    test(args)