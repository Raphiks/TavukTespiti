import cv2
import albumentations as A
import os

# Veri artırma işlemleri
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Yatay çevirme
    A.VerticalFlip(p=0.5),    # Dikey çevirme
    A.Rotate(limit=30, p=0.5),  # -30 ile +30 derece arasında döndürme
    A.RandomBrightnessContrast(p=0.2),  # Parlaklık ve kontrast ayarlama
    A.GaussianBlur(p=0.1),  # Gauss bulanıklığı ekleme
], bbox_params=A.BboxParams(format='yolo'))  # YOLO formatında bounding box'lar

# Veri artırma fonksiyonu
def augment_data(image_path, label_path, output_image_path, output_label_path):
    # Görseli yükle
    image = cv2.imread(image_path)
    if image is None:
        print(f"Hata: Görsel yüklenemedi: {image_path}")
        return False

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Etiket dosyasının var olup olmadığını kontrol et
    if not os.path.exists(label_path):
        print(f"Uyarı: Etiket dosyası bulunamadı: {label_path}")
        return False

    # Etiketleri yükle
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    # Bounding box'ları saklamak için liste
    bboxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.split())
        bboxes.append([x_center, y_center, width, height, class_id])
    
    # Veri artırma uygula
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    
    # Dönüştürülmüş görseli kaydet
    cv2.imwrite(output_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
    print(f"Artırılmış görsel kaydedildi: {output_image_path}")
    
    # Dönüştürülmüş bounding box'ları kaydet
    with open(output_label_path, 'w') as file:
        for bbox in transformed_bboxes:
            x_center, y_center, width, height, class_id = bbox
            file.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
    print(f"Artırılmış etiket dosyası kaydedildi: {output_label_path}")
    
    return True

# Doğru dosya yolları
train_images_path = r"C:\Users\erkan\OneDrive\Desktop\projes\Tavuk_tespiti\train\images"
train_labels_path = r"C:\Users\erkan\OneDrive\Desktop\projes\Tavuk_tespiti\train\labels"

# Etiket dosyası olmayan görsellerin listesi
missing_label_files = []

# Tüm görseller ve etiketler için veri artırma uygula
for image_name in os.listdir(train_images_path):
    if image_name.endswith(".jpg"):  # Görsellerin uzantısı .jpg ise
        image_path = os.path.join(train_images_path, image_name)
        label_path = os.path.join(train_labels_path, image_name.replace(".jpg", ".txt"))
        
        # Yeni dosya yolları
        output_image_path = os.path.join(train_images_path, f"aug_{image_name}")
        output_label_path = os.path.join(train_labels_path, f"aug_{image_name.replace('.jpg', '.txt')}")
        
        # Veri artırma uygula
        success = augment_data(image_path, label_path, output_image_path, output_label_path)
        if not success:
            missing_label_files.append(image_name)  # Etiket dosyası olmayan görselleri kaydet

# Etiket dosyası olmayan görselleri raporla
if missing_label_files:
    print("Etiket dosyası olmayan görseller:")
    for image_name in missing_label_files:
        print(image_name)
else:
    print("Tüm görsellerin etiket dosyaları mevcut.")