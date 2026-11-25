# CenterNet → ONNX → CNNGEN → Cavalry_gen → CV72/CV7x Deployment Guide

## Overview
This guide walks through converting the trained CenterNet model to run on CV72/CV7x chips via the CNNGEN/Cavalry_gen pipeline.

## Prerequisites
- CenterNet model checkpoint (`.pth` file)
- ONNX export tools: `pip install onnx`
- CNNGEN and Cavalry_gen toolchain installed
- Understanding of CV72/CV7x chip specifications

## Step 1: Export Model to ONNX

```bash
cd ./src

# Export the trained model
python export_onnx.py \
  --arch dla_34 \
  --load_model ../models/ctdet_coco_dla_2x.pth \
  --output_onnx ../models/ctdet_coco_dla_2x.onnx \
  --input_h 512 \
  --input_w 512
```

**Output:**
- ONNX model file: `models/ctdet_coco_dla_2x.onnx`

**Notes:**
- The model uses DCNv2 (Deformable Convolution) operators which are custom ONNX operations
- CNNGEN/Cavalry_gen must support or convert DCNv2 operations
- Input size: 512×512 RGB image (normalized, see preprocessing below)

## Step 2: Preprocessing Requirements

Before inference on CV72/CV7x, images must be preprocessed identically to training:

**Image Normalization:**
```python
# Mean and std (COCO dataset)
MEAN = [0.408, 0.447, 0.470]
STD = [0.289, 0.274, 0.278]

# Preprocessing
image = image / 255.0  # Convert to [0, 1]
image = (image - MEAN) / STD  # Normalize
```

**Input Layout:**
- Format: NCHW (Batch, Channels, Height, Width)
- Channels: RGB (not BGR)
- Size: 512×512
- Data type: float32

## Step 3: CNNGEN Compilation

Use CNNGEN to convert ONNX to intermediate representation:

```bash
# Example command (adjust based on your CNNGEN setup)
cnngen --input ../models/ctdet_coco_dla_2x.onnx \
       --output ../models/ctdet_coco_dla_2x_ir \
       --target cv72  # or cv7x depending on chip
```

**Key options:**
- `--target`: `cv72` or `cv7x` chip target
- `--quantize`: Optional quantization (may reduce accuracy)
- `--optimize`: Enable optimization passes

## Step 4: Cavalry_gen Code Generation

Generate optimized code for CV72/CV7x:

```bash
# Generate C/C++ code for deployment
cavalry_gen --input ../models/ctdet_coco_dla_2x_ir \
            --output ../models/centernet_cv7x \
            --language cpp \
            --target-cpu arm  # or your target CPU
```

**Output files:**
- `centernet_cv7x/model.h`: Model definition
- `centernet_cv7x/model.c`: Optimized inference code
- `centernet_cv7x/weights.bin`: Quantized/optimized weights

## Step 5: Model Output Interpretation

CenterNet outputs three feature maps for each image:

1. **Heatmap (`hm`)**: Object class predictions, shape: `[1, 80, 128, 128]`
   - 80 COCO classes
   - 128×128 feature map (512/4 due to 4× downsampling)
   - Values: [0, 1], higher = more confident

2. **Width/Height (`wh`)**: Bounding box dimensions, shape: `[1, 2, 128, 128]`
   - 2 channels: width and height
   - Values in pixels (before scaling)

3. **Regression offset (`reg`)**: Refinement offsets, shape: `[1, 2, 128, 128]`
   - 2 channels: x and y offset
   - Values: [0, 1]

**Postprocessing for Detections:**
1. Apply sigmoid to heatmap
2. Find local maxima above threshold
3. Extract bounding boxes using wh and reg outputs
4. Convert from feature map coordinates to image coordinates
5. Filter by confidence threshold

Example pseudo-code:
```cpp
// After forward pass:
float* hm = output.heatmap;    // [80, 128, 128]
float* wh = output.wh;         // [2, 128, 128]
float* reg = output.reg;       // [2, 128, 128]

// Find detections
std::vector<Detection> detections;
for (int c = 0; c < 80; c++) {
    for (int y = 0; y < 128; y++) {
        for (int x = 0; x < 128; x++) {
            float score = sigmoid(hm[c * 128 * 128 + y * 128 + x]);
            if (score > THRESHOLD) {
                // Extract detection
                Detection det;
                det.class_id = c;
                det.score = score;
                det.x = (x + reg[0 * 128 * 128 + y * 128 + x]) * 4;
                det.y = (y + reg[1 * 128 * 128 + y * 128 + x]) * 4;
                det.w = wh[0 * 128 * 128 + y * 128 + x];
                det.h = wh[1 * 128 * 128 + y * 128 + x];
                detections.push_back(det);
            }
        }
    }
}
```

## Step 6: Integration on CV72/CV7x

1. Include generated header files in your project
2. Link against generated object files or library
3. Call inference function with preprocessed image
4. Post-process outputs as shown above
5. Use NMS (Non-Maximum Suppression) to filter overlapping detections

## Troubleshooting

### Issue: DCNv2 operators not supported
**Solution:** Check if CNNGEN supports DCNv2. If not:
- Use a backbone without DCNv2 (e.g., ResNet instead of DLA)
- Or implement DCNv2 manually in CV7x code

### Issue: Model inference too slow
**Solution:**
- Enable CNNGEN optimization flags
- Consider model quantization (int8)
- Reduce input resolution if acceptable for use case

### Issue: Accuracy degradation after conversion
**Solution:**
- Verify preprocessing is identical
- Check quantization parameters if enabled
- Compare outputs on sample images with PyTorch version

## References

- CenterNet paper: https://arxiv.org/abs/1904.07850
- CNNGEN documentation: [your_cnngen_docs]
- Cavalry_gen documentation: [your_cavalry_docs]
- CV72/CV7x chip specifications: [your_chip_docs]
