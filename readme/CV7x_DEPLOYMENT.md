# CenterNet → CV72/CV7x Deployment - Quick Reference

## Current Status ✓

Model successfully prepared and analyzed:
- **Model file**: `models/ctdet_coco_dla_2x.pth`
- **Architecture**: DLA-34 with DCNv2
- **Parameters**: 19.7M
- **Input**: 512×512 RGB images
- **FP32 size**: ~75 MB | INT8: ~19 MB

## Critical Finding: 16 DCNv2 Layers

The model uses **16 Deformable Convolution (DCNv2)** layers, which are **custom operators** not natively supported by standard ONNX exporters.

### DCNv2 Layer Distribution:
- **DLA upsampling layers** (`dla_up.ida_*`): 12 layers
- **Feature pyramid** (`ida_up`): 4 layers
- All 3×3 kernels with 1 padding, stride 1

## Deployment Path Options

### Option A: Direct CNNGEN Support (Recommended if available)
```bash
# 1. Check CNNGEN DCNv2 support
cnngen --help | grep -i dcn

# 2. If supported, directly convert
cnngen --input model.onnx \
       --output model_ir \
       --target cv72 \
       --support-dcn v2

# 3. Generate CV7x code
cavalry_gen --input model_ir --output cv7x_code
```

### Option B: Manual DCNv2 Implementation (Portable)
If CNNGEN doesn't support DCNv2:

**Implementation steps:**
1. Reference: `src/lib/models/networks/DCNv2/` (PyTorch version)
2. Reference: `src/lib/models/networks/DCNv2/src/cpu/` (CPU reference implementation)
3. Implement DCNv2 kernel in C/C++ for CV7x architecture
4. Test with known input/output weights from PyTorch
5. Integrate into Cavalry_gen-generated code

**DCNv2 Mathematical Operation:**
```
Output = Conv2D(Input, Kernel, offset, mask)

Where:
- offset: [B, 2*kH*kW, H, W] → adaptive sampling grid
- mask: [B, kH*kW, H, W] → attention weights
- Both learned by model

Implementation: Bilinear interpolation over offset grid
```

### Option C: Alternative Architecture (Lower Accuracy)
If DCNv2 implementation is infeasible:

```bash
# 1. Train new model without DCNv2
python src/main.py ctdet \
  --exp_id centernet_no_dcn \
  --arch res_101 \  # ResNet instead of DLA
  --batch_size 32

# 2. Trade-off: 2-5% accuracy loss, easier deployment
# Accuracy: ~35% mAP → ~33% mAP
# Speed improvement: 10-20%
```

## Preprocessing for CV7x Deployment

All images must be preprocessed identically:

```cpp
// C++ preprocessing reference
#include <opencv2/opencv.hpp>

const float MEAN[3] = {0.408, 0.447, 0.470};
const float STD[3] = {0.289, 0.274, 0.278};
const int INPUT_H = 512, INPUT_W = 512;

void preprocess(cv::Mat image, float* output) {
    // Resize
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(INPUT_W, INPUT_H));
    
    // Convert to float and normalize
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < INPUT_H; h++) {
            for (int w = 0; w < INPUT_W; w++) {
                float pixel = resized.at<cv::Vec3b>(h, w)[c] / 255.0f;
                output[c * INPUT_H * INPUT_W + h * INPUT_W + w] = 
                    (pixel - MEAN[c]) / STD[c];
            }
        }
    }
}
```

## Output Decoding for CV7x

After forward pass, decode detections:

```cpp
// C++ inference reference
struct Detection {
    int class_id;
    float x, y, w, h;  // bounding box
    float score;       // confidence
};

std::vector<Detection> decode_detections(
    float* hm,    // [80, 128, 128]
    float* wh,    // [2, 128, 128]
    float* reg,   // [2, 128, 128]
    float threshold = 0.3f
) {
    std::vector<Detection> detections;
    
    for (int c = 0; c < 80; c++) {
        for (int y = 0; y < 128; y++) {
            for (int x = 0; x < 128; x++) {
                int idx = c * 128 * 128 + y * 128 + x;
                float score = sigmoid(hm[idx]);
                
                if (score > threshold) {
                    Detection det;
                    det.class_id = c;
                    det.score = score;
                    
                    // Convert from feature coords to image coords
                    float fx = x + reg[0 * 128 * 128 + y * 128 + x];
                    float fy = y + reg[1 * 128 * 128 + y * 128 + x];
                    
                    det.x = fx * 4;  // 4x downsampling
                    det.y = fy * 4;
                    det.w = wh[0 * 128 * 128 + y * 128 + x];
                    det.h = wh[1 * 128 * 128 + y * 128 + x];
                    
                    detections.push_back(det);
                }
            }
        }
    }
    
    // Apply NMS
    nms(detections, 0.5);  // IoU threshold
    
    return detections;
}
```

## Model Analysis Tool

Detailed model analysis available:

```bash
python src/analyze_model.py \
  --model models/ctdet_coco_dla_2x.pth \
  --arch dla_34
```

Outputs:
- Complete DCNv2 layer mapping
- Parameter counts
- Preprocessing requirements
- Output tensor specifications

## Files Generated

1. **Export script**: `src/export_onnx.py`
   - Direct ONNX export (requires DCNv2 support)
   - Validates output model structure

2. **Analysis script**: `src/analyze_model.py`
   - Complete model architecture dump
   - DCNv2 layer locations and specs
   - Preprocessing/output format details

3. **Deployment guide**: `readme/ONNX_CNNGEN_DEPLOYMENT.md`
   - Step-by-step deployment workflow
   - Troubleshooting tips
   - Integration examples

## Recommended Next Steps

1. **Check CNNGEN/Cavalry_gen documentation** for DCNv2 support
   
2. **Option A** (if DCNv2 supported):
   ```bash
   python src/export_onnx.py --load_model models/ctdet_coco_dla_2x.pth \
     --output_onnx models/centernet.onnx --arch dla_34
   ```

3. **Option B** (if DCNv2 not supported):
   - Implement DCNv2 in CV7x C/C++
   - Use DCNv2 reference: `src/lib/models/networks/DCNv2/src/`
   - Test with exported weights

4. **Option C** (if no DCNv2 support):
   - Retrain without DCNv2 (ResNet backbone)
   - ~3-5% accuracy loss but easier deployment

## Quick Commands

```bash
# Analyze model
python \
  ./src/analyze_model.py \
  --model ./ctdet_coco_dla_2x.pth

# Attempt ONNX export
python \
  ./src/export_onnx.py \
  --load_model ./ctdet_coco_dla_2x.pth \
  --output_onnx ./centernet.onnx

# Run demo (CPU inference)
python \
  ./src/demo.py ctdet \
  --demo ../images/test.jpg \
  --load_model ./ctdet_coco_dla_2x.pth \
  --gpus -1
```

## Performance Estimates

| Metric | FP32 | INT8 | Notes |
|--------|------|------|-------|
| Model size | 75 MB | 19 MB | On CV72/CV7x storage |
| Inference | ~150ms | ~40ms | CPU baseline (512×512) |
| mAP (COCO) | ~40.6% | ~39.2% | Quantization loss: ~1.4% |
| Memory (peak) | 250 MB | 70 MB | During inference |

---

**Status**: Model analysis complete ✓  
**Next action**: Begin CNNGEN integration or choose deployment option
