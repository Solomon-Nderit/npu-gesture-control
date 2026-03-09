import onnxruntime as ort
import numpy as np
import cv2
import os

class YoloNpuEngine:
    def __init__(self, model_path, xclbin_path=None, config_file_path=None):
        self.model_path = model_path
        self.target_size = (224, 224)
        self.padding_color = (114, 114, 114)
        
        # NMS Parameters
        self.conf_threshold = 0.5
        self.nms_threshold = 0.45
        
        # Initialize Session
        self.session = self._initialize_session(xclbin_path, config_file_path)
        self.input_name = self.session.get_inputs()[0].name

    def _initialize_session(self, xclbin_path, config_file_path):
        # 1. Look for XCLBIN
        if xclbin_path is None:
            ryzen_path = os.environ.get('RYZEN_AI_INSTALLATION_PATH', r"C:\Windows\System32\AMD")
            search_paths = [
                os.path.join(ryzen_path, "RyzenAI", "xclbin", "phoenix", "1x4.xclbin"),
                os.path.join(ryzen_path, "voe-4.0-win_amd64", "xclbins", "phoenix", "1x4.xclbin"),
            ]
            for p in search_paths:
                if os.path.exists(p):
                    xclbin_path = p
                    break
        
        if xclbin_path:
            print(f"YoloNpuEngine: Using XCLBIN: {xclbin_path}")
        else:
            print("YoloNpuEngine: Warning - XCLBIN not found. NPU might fail.")

        # 2. Look for/Create Config
        if config_file_path is None:
            config_file_path = "vaip_config.json"
        
        if not os.path.exists(config_file_path):
            with open(config_file_path, 'w') as f:
                f.write('{"vaip": {}}')

        # 3. Create Session
        # Prepare provider options
        provider_options = [{
            'target': 'X1',
            'xlnx_enable_py3_round': 0,
            'xclbin': xclbin_path,
        }]

        print("Creating Inference Session with VitisAIExecutionProvider...")
        try:
            session = ort.InferenceSession(
                self.model_path, 
                providers=['VitisAIExecutionProvider'],
                provider_options=provider_options
            )
            print("Session created successfully!")
        except Exception as e:
            print(f"Failed to create NPU session: {e}")
            # Fallback to CPU for debugging flow if NPU fails
            print("Falling back to CPUExecutionProvider...")
            session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            
        return session


    def _letterbox_image(self, image):
        pad_left  = 0
        pad_top  = 0
        SCALE = 1
        
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scaling ratio
        ratio = min(target_w / w, target_h / h)
        SCALE = ratio
        new_w, new_h = int(w * ratio), int(h * ratio)

        if target_h > new_h:
            pad_top = (target_h - new_h) / 2
        if target_w > new_w:
            pad_left = (target_w - new_w) / 2
            
        resized_image = cv2.resize(image, (new_w, new_h))
        
        # Create canvas
        canvas = np.full((target_h, target_w, 3), self.padding_color, dtype=np.uint8)
        
        # Center image
        top = int(pad_top)
        left = int(pad_left)
        
        # Handle potential rounding errors in sizing
        canvas[top:top+new_h, left:left+new_w] = resized_image
        
        return canvas, pad_left, pad_top, SCALE

    def _postprocess(self, raw_tensor, pad_left, pad_top, scale):
        # [1, 84, 8400] -> squeeze -> [84, 8400] -> transpose -> [8400, 84]
        squeezed = np.squeeze(raw_tensor)
        outputs = np.transpose(squeezed)
        
        # Filter by confidence
        # Index 4 is objectness/class score for single class
        mask = outputs[:, 4] >= self.conf_threshold
        detections = outputs[mask]
        
        boxes = []
        confidences = []
        keypoints_list = []
        
        for row in detections:
            cx, cy, w, h = row[0:4]
            x_min = cx - (w/2)
            y_min = cy - (h/2)
            
            boxes.append([x_min, y_min, w, h])
            confidences.append(float(row[4]))
            keypoints_list.append(row[5:])
            
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        final_results = []
        if len(indices) > 0:
            for i in indices.flatten():
                # Recover original box
                box = boxes[i]
                orig_x = (box[0] - pad_left) / scale
                orig_y = (box[1] - pad_top) / scale
                orig_w = box[2] / scale
                orig_h = box[3] / scale
                
                # Recover Keypoints
                kps_flat = keypoints_list[i]
                num_kps = len(kps_flat) // 3
                kps_2d = np.array(kps_flat).reshape(num_kps, 3)
                
                kps_2d[:, 0] = (kps_2d[:, 0] - pad_left) / scale
                kps_2d[:, 1] = (kps_2d[:, 1] - pad_top) / scale
                
                final_results.append({
                    "box": [int(orig_x), int(orig_y), int(orig_w), int(orig_h)],
                    "score": confidences[i],
                    "keypoints": kps_2d.astype(int)
                })
                
        return final_results

    def infer(self, image):
        # 1. Preprocess
        preprocessed, pad_left, pad_top, scale = self._letterbox_image(image)
        
        # 2. Blob preparation
        rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        float_img = rgb.astype(np.float32) / 255.0
        # HWC -> CHW -> Batch
        tensor_in = np.transpose(float_img, (2, 0, 1))[np.newaxis, :]
        
        # 3. Run Session
        outputs = self.session.run(None, {self.input_name: tensor_in})
        
        # 4. Postprocess
        results = self._postprocess(outputs[0], pad_left, pad_top, scale)
        
        return results

    def draw_results(self, image, results):
        annotated = image.copy()
        
        for res in results:
            x, y, w, h = res["box"]
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            for kp in res["keypoints"]:
                cv2.circle(annotated, (kp[0], kp[1]), 3, (0, 0, 255), -1)
                
        return annotated
