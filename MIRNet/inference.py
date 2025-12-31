import cv2 # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mirnet.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read input image
img = cv2.imread("input.jpg")
if img is None:
    raise FileNotFoundError("input.jpg not found!")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize according to model input
h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
img = cv2.resize(img, (w, h))

img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# Post-process
output = np.squeeze(output)
output = (output * 255).clip(0, 255).astype(np.uint8)

# Save result
cv2.imwrite("output.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

print("✅ DONE: Enhanced image saved as output.jpg")
