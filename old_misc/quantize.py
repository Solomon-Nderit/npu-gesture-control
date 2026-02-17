import os
import sys
import numpy as np
import onnx
import pathlib
import argparse
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

# Add local dir to path to find utils.py
sys.path.append(str(pathlib.Path(__file__).parent))
from utils import LoadImages

def preprocess(img):
    # Matches the preprocess logic in infer_onnx.py
    # Check if image is uint8, if so convert to float32 and normalize
    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img

class YoloReader(CalibrationDataReader):
    def __init__(self, model_path, image_dir, limit=100):
        # Load model to find input name
        model = onnx.load(model_path)
        self.input_name = model.graph.input[0].name
        
        # Initialize loader from utils.py
        print(f"Loading images from {image_dir}...")
        self.dataset = LoadImages(image_dir, imgsz=[640, 640], stride=32, auto=False)
        self.iterator = iter(self.dataset)
        self.count = 0
        self.max_count = limit 

    def get_next_input(self):
        if self.count >= self.max_count:
            return None
        
        try:
            # batch = path, im, im0s, vid_cap, s
            _, im, _, _, _ = next(self.iterator)
        except StopIteration:
            return None

        # Preprocess
        data = preprocess(im)
        
        # Add batch dim -> (1, 3, 640, 640)
        if len(data.shape) == 3:
            data = data[np.newaxis, ...]
        
        self.count += 1
        return {self.input_name: data}

def quantize(input_model, output_model, image_dir):
    print(f"Starting quantization for {input_model}")
    print(f"Output will be saved to {output_model}")
    
    # Create Data Reader
    dr = YoloReader(input_model, image_dir)
    
    # Run Static Quantization
    # This produces a QDQ model compatible with the NPU
    quantize_static(
        input_model,
        output_model,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,     # NPU requires QDQ format
        per_channel=False,                # NPU usually prefers per-tensor for activations
        weight_type=QuantType.QInt8,      
        activation_type=QuantType.QUInt8
    )
    print("Quantization Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model", type=str, required=True, help="Input float ONNX model")
    parser.add_argument("--output_model", type=str, default="model_quantized.onnx", help="Output quantized ONNX model")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing calibration images")
    args = parser.parse_args()

    quantize(args.onnx_model, args.output_model, args.image_dir)