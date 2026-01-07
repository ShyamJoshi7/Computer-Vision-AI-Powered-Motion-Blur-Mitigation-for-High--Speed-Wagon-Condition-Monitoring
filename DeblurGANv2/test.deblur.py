import cv2
import numpy as np
from models.networks import get_generator

# Load model
model = get_generator('fpn_inception')
model.load_weights('/DeblurGANv2/weights/fpn_inception.h5')

# Read image
img = cv2.imread('10.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0
img = np.expand_dims(img, 0)

# Predict
out = model.predict(img)[0]
out = (out * 255).astype('uint8')

# Save result
cv2.imwrite('output.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

print("✅ Deblurring completed successfully!")
