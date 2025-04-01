from ultralytics import YOLO

# YOLOv8 modelini yükle
model5 = YOLO("yolov5s.pt")  # YOLOv8 Nano modeli

# Modeli eğit
results = model5.train(
    data=r"C:\Users\erkan\OneDrive\Desktop\projes\Tavuk_tespiti\data.yaml",
    epochs=10,
    imgsz=640,
    batch=32,
    device="0",  # GPU'yu kullan
    amp=False,  # AMP'yi devre dışı bırak
    name="Tavuk_tespiti"
)