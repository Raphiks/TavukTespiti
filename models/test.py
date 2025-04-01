from model import get_model, evaluate_model, predict_and_display
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(num_classes=2)
model.load_state_dict(torch.load("chicken_model.pth", map_location=device))
model.to(device)

test_img_dir = "C:/Users/erkan/OneDrive/Desktop/projes/Tavuk_tespiti/Tavuk_sayimi/data/test/images"
predict_and_display(model, "C:/Users/erkan/OneDrive/Desktop/projes/Tavuk_tespiti/Tavuk_sayimi/data/test/images/test_image4.jpg", device, threshold=0.85, max_boxes=20)

evaluate_model(model, test_img_dir, device, threshold=0.5)
