import cv2
import numpy as np
import onnxruntime as ort
import time

# 1. Setup NPU Provider
providers = ['VitisAIExecutionProvider']
provider_options = [{
    'config_file': 'vaip_config.json',
    'xclbin': '1x4.xclbin'  # Standard firmware for Ryzen 8040 (Phoenix/Hawk Point)
}]

# 2. Load Model
print("Loading model to NPU... (First run takes ~2 mins to compile)")
try:
    session = ort.InferenceSession(
        "models\hand-trained.int8.onnx", 
        providers=providers,
        provider_options=provider_options
    )
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 3. Webcam Loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # --- Preprocessing ---
    t1 = time.time()
    input_img = cv2.resize(frame, (640, 640))
    input_img = input_img.transpose(2, 0, 1)
    input_img = np.expand_dims(input_img, axis=0).astype(np.float32)
    input_img /= 255.0
    
    # --- INFERENCE (NPU) ---
    outputs = session.run(None, {"images": input_img})
    t2 = time.time()
    
    # Calculate NPU Latency
    latency_ms = (t2 - t1) * 1000
    cv2.putText(frame, f"NPU Latency: {latency_ms:.2f}ms", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # (Add your post-processing / skeleton drawing code here)
    
    cv2.imshow("Ryzen AI NPU Test", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()