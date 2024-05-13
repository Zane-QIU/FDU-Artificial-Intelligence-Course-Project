import argparse
from train import train
from test import test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test CNN on Handwritten Chinese Dataset")
    parser.add_argument("mode", type=str, choices=["train", "test"], help="Mode: train or test")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs (for training mode)")
    parser.add_argument("--save_interval", type=int, default=5, help="Interval for saving checkpoints (for training mode)")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Path to save model checkpoints (for training mode)")
    parser.add_argument("--model", type=str, default="CNN", choices=["CNN", "ResNet18"], help="Model to use (for both training and testing)")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
