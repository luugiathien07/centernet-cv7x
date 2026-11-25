# CenterNet → ONNX for CV7x edge deployment

Fork of [CenterNet (Objects as Points)](https://github.com/xingyizhou/CenterNet) prepared for deployment on an edge NPU (CV72/CV7x class SoC).

What this fork adds:

- `src/export_centernet_onnx.py`: exports `ctdet` DLA-34 to ONNX. `--mode conv` replaces the 16 DCNv2 layers with standard Conv2d so the graph only uses operators an edge compiler supports; `--mode direct` keeps DCNv2.
- `src/analyze_model.py`: layer and operator summary of a checkpoint.
- `benchmark_cpu.py`: ONNX Runtime CPU latency benchmark.
- `config/ea_cvt_centernet.yaml`, `tools/run_onnxparser.py`, `dra_img/`: configuration for the vendor graph converter (calibration images, input shape, fp16 outputs).
- `readme/DEPLOYMENT_READY.md`, `readme/CV7x_DEPLOYMENT.md`, `readme/ONNX_CNNGEN_DEPLOYMENT.md`, `DEPLOYMENT_SUMMARY.txt`: preprocessing/postprocessing contract (RGB 512×512, mean/std, sigmoid heatmap, offset/size decoding, NMS) and the export → compile → integrate workflow.

Not included: model weights and ONNX files (see `readme/MODEL_ZOO.md` for upstream checkpoints), the vendor SDK and its tools, which are licensed separately.

```bash
python src/export_centernet_onnx.py ctdet --load_model models/ctdet_coco_dla_2x.pth \
  --arch dla_34 --mode conv --output_onnx models/ctdet_coco_dla_2x_conv.onnx
```

Upstream documentation: [README_UPSTREAM.md](README_UPSTREAM.md). License: MIT (see `LICENSE`, `NOTICE`).
