import cv2
import time
import torch
import onnxruntime
import numpy as np
import argparse
import sys
import pathlib

# Add local dir to path BEFORE importing local modules
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from utils import non_max_suppression

# Configuration
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45

def preprocess(img, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    # HWC to CHW
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img)
    img = img.float()  
    img /= 255.0  
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    return img, ratio, (dw, dh)

class DFL(torch.nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    def __init__(self, c1=16):
        super().__init__()
        self.conv = torch.nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = torch.nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

def post_process(x):
    # This logic matches what was in infer_onnx.py
    dfl = DFL(16)
    anchors = torch.tensor(np.load("./anchors.npy", allow_pickle=True))
    strides = torch.tensor(np.load("./strides.npy", allow_pickle=True))
    
    box, cls = torch.cat([xi.view(x[0].shape[0], 144, -1) for xi in x], 2).split((16 * 4, 80), 1)
    dbox = dist2bbox(dfl(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y

def run_live_inference(onnx_model_path, use_ipu=True):
    # 1. Setup Session
    if use_ipu:
        print("Initializing NPU Session...")
        providers = ["VitisAIExecutionProvider"]
        provider_options = [{
            "target": "X1", 
            "xclbin": r"C:\Program Files\RyzenAI\1.7.0\voe-4.0-win_amd64\xclbins\phoenix\4x4.xclbin",
            "xlnx_enable_py3_round": 0
        }]
        session = onnxruntime.InferenceSession(onnx_model_path, providers=providers, provider_options=provider_options)
    else:
        print("Initializing CPU Session...")
        session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

    # 2. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Inference Loop. Press 'q' to exit.")
    
    # Load class names
    classnames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
    names = {k: classnames[k] for k in range(80)}

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Preprocess Frame
        img_tensor, ratio, pad = preprocess(frame)
        
        # 4. Run Inference
        # Note: Quantized models input layout is usually NHWC for NPU efficiency, 
        # but preprocess returns NCHW. We permute it here.
        # Check model first input shape in Netron if unsure. Assuming NHWC for Vitis.
        input_name = session.get_inputs()[0].name
        
        input_data = img_tensor.permute(0, 2, 3, 1).cpu().numpy() # NCHW -> NHWC
        
        start = time.time()
        outputs = session.run(None, {input_name: input_data})
        inference_time = (time.time() - start) * 1000
        
        # 5. Post Process
        # Convert outputs back to torch for NMS
        # Vitis usually returns NHWC output, we might need to transpose back to NCHW
        # outputs shape check
        outputs_torch = [torch.tensor(item).permute(0, 3, 1, 2) for item in outputs]
        
        preds = post_process(outputs_torch)
        preds = non_max_suppression(preds, CONF_THRES, IOU_THRES, agnostic=False, max_det=10, classes=None)

        # 6. Visualize
        # Scale coords back to original image size
        for i, det in enumerate(preds):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=3)

        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {inference_time:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Ryzen AI YOLO Live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8m_quantized.onnx", help="Path to quantized ONNX model")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of NPU")
    args = parser.parse_args()

    run_live_inference(args.model, use_ipu=not args.cpu)
