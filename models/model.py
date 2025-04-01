import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
from torchvision.transforms import functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

# Model oluşturma fonksiyonu
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Tahmin ve görselleştirme fonksiyonu
def predict_and_display(model, image_path, device, threshold=0.5, max_boxes=20):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    # Skor eşiği üstü kutuları filtrele
    filtered = [(box, score) for box, score in zip(boxes, scores) if score > threshold]

    # En iyi skorla sıralayıp ilk N kutuyu al
    filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:max_boxes]

    img = np.array(image)
    print(f"🎯 Gösterilecek kutu sayısı: {len(filtered)}")

    for box, score in filtered:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Görüntüyü kaydet
    save_path = f"prediction_{os.path.basename(image_path)}"
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"💾 Tahmin sonucu kaydedildi: {save_path}")

    # Görseli göster (isteğe bağlı)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Tahmin Sonucu")
    plt.show()


def evaluate_model(model, test_img_dir, device, threshold=0.5):
    print("🧪 evaluate_model() fonksiyonu çalıştı.")
    print(f"📁 Test klasörü: {test_img_dir}")
    
    # jpg, jpeg, png dosyalarını tara
    img_paths = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")) +
                       glob.glob(os.path.join(test_img_dir, "*.jpeg")) +
                       glob.glob(os.path.join(test_img_dir, "*.png")))

    print(f"🔍 Bulunan görsel sayısı: {len(img_paths)}")
    if len(img_paths) == 0:
        print("🚫 Test klasöründe uygun görsel bulunamadı.")
        return

    model.eval()
    total = len(img_paths)
    detected = 0

    for idx, path in enumerate(img_paths):
        print(f"[{idx+1}/{total}] Görsel işleniyor: {os.path.basename(path)}")

        image = Image.open(path).convert("RGB")
        img_tensor = F.to_tensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)

        scores = outputs[0]['scores'].cpu().numpy()
        print(f"  🔸 Tahmin edilen kutu sayısı: {len(scores)}")
        print(f"  🔸 Skorlar: {scores}")

        if any(score > threshold for score in scores):
            detected += 1
            print("  ✅ En az bir tespit var (eşik üstü)")
        else:
            print("  ❌ Tespit yok veya skorlar düşük")

    print("\n" + "=" * 50)
    print(f"📊 Toplam Görsel Sayısı: {total}")
    print(f"🎯 Tespit Yapılan Görsel Sayısı: {detected}")
    print(f"📈 Tespit Oranı: %{(detected / total * 100):.2f}")
    print("=" * 50)