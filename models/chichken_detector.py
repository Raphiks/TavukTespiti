import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

print("Kod başladı")


# ==== YOLO formatından PASCAL VOC formatına çeviri ====
def yolo_to_voc(box, img_width, img_height):
    x_center, y_center, width, height = box
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    xmin = int(x_center - width / 2)
    xmax = int(x_center + width / 2)
    ymin = int(y_center - height / 2)
    ymax = int(y_center + height / 2)
    return [xmin, ymin, xmax, ymax]

# ==== Dataset Sınıfı ====
class ChickenDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpeg")))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size

        boxes = []
        labels = []

        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = map(float, parts)
                box = yolo_to_voc([x, y, w, h], img_width, img_height)
                boxes.append(box)
                labels.append(1)  # Chicken class = 1

        # 🔴 Eğer hiç kutu yoksa, geçerli bir örnek bulana kadar sıradaki indeks denenir
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


# ==== Transformlar ====
def get_transform(train=True):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# ==== Model Oluşturma ====
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ==== Eğitim Fonksiyonu ====
def train_model(model, train_loader, device, num_epochs=10):
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} başladı...")
        epoch_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            print(f"  Batch {i+1} işleniyor...")
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}")
    return model

# ==== Test ve Görselleştirme ====
def predict_and_display(model, image_path, device, threshold=0.5):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)

    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    img = np.array(image)
    for box, score in zip(boxes, scores):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.imshow(img)
    plt.axis('off')
    plt.show()

# ==== Ana Çalıştırma ====
if __name__ == "__main__":
    print("Ana kod bloğuna girildi")

    base_path = r"C:/Users/erkan/OneDrive/Desktop/projes/Tavuk_tespiti/Tavuk_sayimi/data"
    train_img_dir = os.path.join(base_path, "images/train")
    train_label_dir = os.path.join(base_path, "labels/train")


    dataset = ChickenDataset(train_img_dir, train_label_dir, get_transform(train=True))
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=2)

    print("Eğitim başlıyor...")
    print(f"Toplam örnek sayısı: {len(dataset)}")
    sample = dataset[0]
    print("Birinci örnek başarıyla alındı:", sample)

    model = train_model(model, data_loader, device, num_epochs=10)
    torch.save(model.state_dict(), "chicken_model.pth")

    # Test
    test_image_path = r"C:\Users\erkan\OneDrive\Desktop\projes\Tavuk_tespiti\Tavuk_sayimi\data\test\images\test_image.jpg"  # örnek bir test resmi
    model.load_state_dict(torch.load("chicken_model.pth"))
    model.to(device)
    predict_and_display(model, test_image_path, device)
