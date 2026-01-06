import torch
import torchvision
import os
import time
import glob
import numpy as np
from PIL import Image
import model

# device handling (CPU-safe)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lowlight(image_path):
    scale_factor = 12

    # load image
    data_lowlight = Image.open(image_path).convert("RGB")
    data_lowlight = np.asarray(data_lowlight) / 255.0
    data_lowlight = torch.from_numpy(data_lowlight).float()

    # resize to multiple of scale_factor
    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]

    # CHW + batch
    data_lowlight = data_lowlight.permute(2, 0, 1).unsqueeze(0).to(device)

    # load model
    DCE_net = model.enhance_net_nopool(scale_factor).to(device)
    DCE_net.load_state_dict(
        torch.load("snapshots_Zero_DCE++/Epoch99.pth", map_location=device)
    )
    DCE_net.eval()

    start = time.time()
    with torch.no_grad():
        enhanced_image, _ = DCE_net(data_lowlight)
    end_time = time.time() - start

    print(f"Time: {end_time:.4f}s")
    
    result_path = "result_single_image.png"
    torchvision.utils.save_image(enhanced_image, result_path)
    print(f"Saved enhanced image to: {result_path}")

if __name__ == "__main__":
    lowlight("100.png")


    # save result
    # result_path = image_path.replace("test_data", "result_Zero_DCE++")
    # os.makedirs(os.path.dirname(result_path), exist_ok=True)
    # torchvision.utils.save_image(enhanced_image, result_path)
    


# if __name__ == "__main__":

#     filePath = "data/test_data/"
#     sum_time = 0.0

#     for folder_name in os.listdir(filePath):
#         test_list = glob.glob(os.path.join(filePath, folder_name, "*"))
#         for image in test_list:
#             print(image)
#             sum_time += lowlight(image)

#     print("Total time:", sum_time)
    

        

