import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import argparse
from dataset import HandwrittenChineseDataset
from model import CNN, ResNet, BasicBlock

def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Transformations

    transform_train = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1), translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Dataset
    dataset = HandwrittenChineseDataset(root_dir=args.dataset_path, transform=transform_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

     # Model
    if args.model == 'CNN':
        model = CNN().to(device)
    elif args.model == 'ResNet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
    else:
        raise ValueError("Model not recognized")

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader)}, Learning Rate: {optimizer.param_groups[0]["lr"]}')
        # Learning rate decay
        if (epoch + 1) % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        # Validation
        model.eval()
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

        print(f'Validation Loss: {val_loss / len(val_loader)}, Accuracy: {100 * correct / total}')

        # Save checkpoint
        if epoch % args.save_interval == args.save_interval - 1:
            torch.save(model.state_dict(), f'{args.checkpoint_path}/model_{args.model}_bs_{args.batch_size}_lr_{args.learning_rate}_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN on Handwritten Chinese Dataset")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--save_interval", type=int, default=5, help="Interval for saving checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Path to save model checkpoints")
    parser.add_argument("--model", type=str, default="CNN", choices=["CNN", "ResNet18"], help="Model to use for training")
    args = parser.parse_args()

    train(args)
