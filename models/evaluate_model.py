import os
import glob
from PIL import Image
from torchvision.transforms import functional as F
import torch

def evaluate_model(model, test_img_dir, device, threshold=0.5):
    print(f"📁 Test klasörü: {test_img_dir}")
    
    # jpg, jpeg, png uzantılarını al
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
