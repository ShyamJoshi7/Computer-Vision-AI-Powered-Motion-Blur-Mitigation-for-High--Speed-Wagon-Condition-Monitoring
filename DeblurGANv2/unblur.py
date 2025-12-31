import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import glob
import os

# Load model
model = torch.hub.load('VITA-Group/DeblurGANv2', 'DeblurGANv2', pretrained=True)
model.eval()

transform = ToTensor()
inv_transform = ToPILImage()

input_folder = "input_images/"
output_folder = "output_images/"
os.makedirs(output_folder, exist_ok=True)

for img_path in glob.glob(input_folder + "*.*"):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    result = inv_transform(output.squeeze(0))
    filename = os.path.basename(img_path)
    result.save(output_folder + "deblurred_" + filename)

print("Finished deblurring!")