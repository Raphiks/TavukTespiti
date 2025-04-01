import torch
from model import get_model, predict_and_display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Görselin yolu (gerekirse değiştir)
test_image_path = r"C:\Users\erkan\OneDrive\Desktop\projes\Tavuk_tespiti\Tavuk_sayimi\data\test\images\test_image.jpg"

# Model yükle ve test yap
model = get_model(num_classes=2)
model.load_state_dict(torch.load("chicken_model.pth", map_location=device))
model.to(device)

predict_and_display(model, test_image_path, device, threshold=0.5)

