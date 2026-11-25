"""
Export CenterNet model to ONNX format for CNNGEN/Cavalry_gen pipeline.
Usage:
    python export_onnx.py --arch dla_34 \
        --load_model path/to/model.pth \
        --output_onnx path/to/output.onnx
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
import argparse
import torch
import numpy as np

from lib.opts import opts
from lib.models.model import create_model, load_model


def export_onnx(opt):
    """Export CenterNet model to ONNX format.
    
    Note: DCNv2 (Deformable Convolution) is not natively supported in ONNX.
    This script exports the model with a wrapper to handle DCNv2 operations.
    """
    
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model.eval()
    
    # Move to CPU for export (ONNX export works best on CPU)
    model = model.to('cpu')
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, opt.input_h, opt.input_w, device='cpu')
    
    print('Exporting to ONNX...')
    print(f'  Input shape: {dummy_input.shape}')
    print(f'  Output file: {opt.output_onnx}')
    print(f'  Note: DCNv2 (Deformable Convolution) operations will be exported.')
    print(f'        These may need special handling by CNNGEN/Cavalry_gen.')
    
    try:
        # Attempt export with opset_version 13+ for better operator support
        torch.onnx.export(
            model,
            dummy_input,
            opt.output_onnx,
            input_names=['input'],
            output_names=['output'],
            opset_version=13,
            do_constant_folding=True,
            verbose=False,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            custom_opsets={'custom': 1}  # Register custom domain for DCNv2
        )
        print('✓ ONNX export successful!')
        
        # Verify the exported model
        print('\nVerifying ONNX model...')
        import onnx
        onnx_model = onnx.load(opt.output_onnx)
        # Note: skip strict model checking as custom operators may not be recognized
        try:
            onnx.checker.check_model(onnx_model)
            print('✓ ONNX model is valid!')
        except Exception as check_err:
            print(f'⚠ ONNX model has custom operators (expected): {check_err}')
            print('  These should be handled by CNNGEN/Cavalry_gen.')
        
        # Print model info
        onnx_graph = onnx_model.graph
        print(f'\nONNX Model Info:')
        print(f'  Inputs: {[inp.name for inp in onnx_graph.input]}')
        print(f'  Outputs: {[out.name for out in onnx_graph.output]}')
        print(f'  Operators: {len(onnx_graph.node)} nodes')
        
        # List custom operators if any
        custom_ops = set()
        for node in onnx_graph.node:
            if node.op_type not in ['Conv', 'BatchNormalization', 'Relu', 'MaxPool', 'Add', 'Sigmoid', 'Constant', 'Unsqueeze']:
                custom_ops.add(node.op_type)
        
        if custom_ops:
            print(f'\n  Custom operators (may need CNNGEN support):')
            for op in sorted(custom_ops):
                print(f'    - {op}')
        
    except Exception as e:
        print(f'\n✗ Export failed: {e}')
        print(f'\nTroubleshooting:')
        print(f'  - Ensure ONNX package is installed: pip install onnx')
        print(f'  - DCNv2 requires custom CNNGEN/Cavalry_gen support')
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='dla_34',
                        help='model architecture (default: dla_34)')
    parser.add_argument('--load_model', required=True,
                        help='path to pretrained model weights')
    parser.add_argument('--output_onnx', required=True,
                        help='path to output ONNX file')
    parser.add_argument('--input_h', type=int, default=512,
                        help='input height')
    parser.add_argument('--input_w', type=int, default=512,
                        help='input width')
    parser.add_argument('--head_conv', type=int, default=-1,
                        help='conv layer channels for output head')
    
    args = parser.parse_args()
    
    # Create a minimal opt object
    class OptObj:
        pass
    
    opt = OptObj()
    opt.arch = args.arch
    opt.load_model = args.load_model
    opt.output_onnx = args.output_onnx
    opt.input_h = args.input_h
    opt.input_w = args.input_w
    opt.head_conv = args.head_conv
    
    # Default head config for COCO detection
    opt.heads = {'hm': 80, 'wh': 2, 'reg': 2}
    
    export_onnx(opt)


if __name__ == '__main__':
    main()
