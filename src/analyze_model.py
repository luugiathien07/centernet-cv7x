"""
Model Information and DCNv2 Handling Guide for CV72/CV7x Deployment

This document provides technical details about the CenterNet model
and how to handle DCNv2 (Deformable Convolution) for CNNGEN/Cavalry_gen pipeline.
"""

import torch
import sys
sys.path.insert(0, 'lib')

from models.model import create_model, load_model

def analyze_model(model_path, arch='dla_34'):
    """Analyze model architecture and DCNv2 layer locations."""
    
    print("=" * 70)
    print("CenterNet Model Analysis for CV72/CV7x Deployment")
    print("=" * 70)
    
    # Create and load model
    heads = {'hm': 80, 'wh': 2, 'reg': 2}
    model = create_model(arch, heads, head_conv=-1)
    model = load_model(model, model_path)
    model.eval()
    
    print(f"\n[1] MODEL ARCHITECTURE")
    print(f"  Backbone: {arch}")
    print(f"  Input: RGB 512×512 images")
    print(f"  Output heads: {heads}")
    print(f"    - hm: {heads['hm']} (heatmap, 80 COCO classes)")
    print(f"    - wh: {heads['wh']} (width/height)")
    print(f"    - reg: {heads['reg']} (offset refinement)")
    
    print(f"\n[2] DCNv2 LAYER LOCATIONS")
    dcn_count = 0
    dcn_layers = []
    for name, module in model.named_modules():
        if 'DCN' in module.__class__.__name__ or 'dcn' in name.lower():
            dcn_count += 1
            dcn_layers.append(name)
            print(f"  Layer {dcn_count}: {name}")
            if hasattr(module, 'in_channels'):
                print(f"    - Input channels: {module.in_channels}")
                print(f"    - Output channels: {module.out_channels}")
                print(f"    - Kernel size: {module.kernel_size}")
                print(f"    - Stride: {module.stride}")
                print(f"    - Padding: {module.padding}")
    
    if dcn_count == 0:
        print("  No DCNv2 layers found!")
    else:
        print(f"\n  Total DCNv2 layers: {dcn_count}")
    
    print(f"\n[3] TOTAL PARAMETERS")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print(f"\n[4] PREPROCESSING REQUIREMENTS")
    print(f"  Image format: RGB (NOT BGR)")
    print(f"  Input size: 512×512 pixels")
    print(f"  Normalization (COCO dataset):")
    print(f"    - Mean: [0.408, 0.447, 0.470]")
    print(f"    - Std: [0.289, 0.274, 0.278]")
    print(f"  Steps:")
    print(f"    1. Resize to 512×512")
    print(f"    2. Divide by 255.0")
    print(f"    3. Subtract mean and divide by std")
    print(f"    4. Transpose to CHW format")
    
    print(f"\n[5] OUTPUT INTERPRETATION")
    print(f"  Output 1 - Heatmap (hm): [1, 80, 128, 128]")
    print(f"    - 80 COCO classes")
    print(f"    - 128×128 feature map (512 / 4x downsampling)")
    print(f"    - Values after sigmoid: [0, 1]")
    print(f"    - Threshold for detection: typically 0.3-0.5")
    print(f"")
    print(f"  Output 2 - Box size (wh): [1, 2, 128, 128]")
    print(f"    - 2 channels: width, height")
    print(f"    - Units: pixels")
    print(f"")
    print(f"  Output 3 - Offset (reg): [1, 2, 128, 128]")
    print(f"    - 2 channels: x offset, y offset")
    print(f"    - Range: [0, 1]")
    
    print(f"\n[6] DCNv2 TO STANDARD CONVOLUTION CONVERSION")
    print(f"  Challenge: DCNv2 learns adaptive spatial sampling patterns")
    print(f"  Options:")
    print(f"    A) Direct ONNX export (requires CNNGEN DCNv2 support)")
    print(f"    B) Manual implementation in C/C++ on CV7x")
    print(f"    C) Use model without DCNv2 (train new model with standard Conv)")
    print(f"    D) Approximate DCNv2 with grouped convolutions")
    
    print(f"\n[7] QUANTIZATION CONSIDERATIONS")
    print(f"  FP32 model size: ~{total_params * 4 / (1024**2):.1f} MB")
    print(f"  INT8 model size: ~{total_params / (1024**2):.1f} MB (75% reduction)")
    print(f"  Accuracy impact: typically 1-3% mAP loss on COCO")
    print(f"  Latency improvement: 2-4x on CV72/CV7x")
    
    print(f"\n[8] INFERENCE PIPELINE FOR CV7x")
    print(f"  1. Load preprocessed image")
    print(f"  2. Run forward pass:")
    print(f"     - Input: image [1, 3, 512, 512]")
    print(f"     - Output: [hm, wh, reg]")
    print(f"  3. Decode predictions:")
    print(f"     - Apply sigmoid to heatmap")
    print(f"     - Find local maxima > threshold")
    print(f"     - Compute bounding boxes from wh + reg")
    print(f"  4. NMS (Non-Maximum Suppression)")
    print(f"  5. Return sorted detections")
    
    print(f"\n[9] NEXT STEPS")
    print(f"  1. Check CNNGEN/Cavalry_gen DCNv2 support")
    print(f"  2. If not supported, implement DCNv2 in C/C++:")
    print(f"     - Reference: src/lib/models/networks/DCNv2/")
    print(f"     - CPU version in src/lib/external/")
    print(f"  3. Or retrain model without DCNv2:")
    print(f"     - Use ResNet backbone instead of DLA")
    print(f"     - Compare accuracy/speed tradeoff")
    
    print("\n" + "=" * 70)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'dcn_layers': dcn_count,
        'dcn_layer_names': dcn_layers
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--arch', default='dla_34', help='Model architecture')
    args = parser.parse_args()
    
    analyze_model(args.model, args.arch)
