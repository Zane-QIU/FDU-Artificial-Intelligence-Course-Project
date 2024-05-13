import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import AdamW
from tqdm import tqdm

# 特征提取器
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# 数据预处理
def transform_image(image):
    return feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze()

# ImageNet数据加载
def load_dataset(path, train=True):
    dataset = datasets.ImageFolder(
        root=path,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transform_image
        ])
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=train, num_workers=4)
    return loader

train_loader = load_dataset('/path/to/imagenet/train', train=True)
val_loader = load_dataset('/path/to/imagenet/val', train=False)

# 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=1000)
model.to(device)

# 损失函数和优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练和验证函数
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 训练循环
for epoch in range(1):
    train_loss = train_epoch(model, train_loader)
    val_accuracy = validate(model, val_loader)
    print(f"Epoch {epoch+1}: Loss = {train_loss:.2f}, Val Accuracy = {val_accuracy:.2f}%")
