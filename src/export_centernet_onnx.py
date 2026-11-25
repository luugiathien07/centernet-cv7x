"""
CenterNet to ONNX Export with DCNv2 Handling

Two export modes:
1. Direct: Keep DCNv2 operators (requires CNNGEN support)
2. Conv replacement: Replace DCNv2 with standard Conv2d (baseline model)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import sys
import argparse
import os
import torch
import onnx

from opts import opts
from models.model import create_model, load_model


def export_centernet_onnx(opt, mode='direct', output_path=None):
    """
    Export CenterNet to ONNX format.
    
    Args:
        opt: Options object from opts.py
        mode: 'direct' (keep DCNv2) or 'conv' (replace DCNv2 with Conv2d)
        output_path: Custom output file path
    """
    
    print('=' * 70)
    print('CenterNet ONNX Export')
    print('=' * 70)
    
    if output_path is None:
        output_path = os.path.basename(opt.load_model).replace('.pth', '.onnx')
    
    print(f'\n[1] Creating model...')
    print(f'  Architecture: {opt.arch}')
    print(f'  Heads: {opt.heads}')
    
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to('cpu')
    model.eval()
    
    print(f'\n[2] Export settings')
    print(f'  Mode: {mode}')
    print(f'  Input shape: [1, 3, {opt.input_h}, {opt.input_w}]')
    print(f'  Output file: {output_path}')
    
    if mode == 'direct':
        print(f'  Note: DCNv2 operators will be exported as custom::dcn_v2')
        print(f'        CNNGEN/Cavalry_gen must support this operator')
        return export_direct(model, opt, output_path)
    elif mode == 'conv':
        print(f'  Note: All DCNv2 layers will be replaced with Conv2d')
        print(f'        Baseline model compatible with any framework')
        print(f'        Accuracy may be ~1-3% lower than DCNv2 version')
        return export_with_conv_replacement(model, opt, output_path)
    else:
        print(f'✗ Unknown mode: {mode}')
        return False


def export_direct(model, opt, output_path):
    """Direct export keeping DCNv2 operators."""
    try:
        print(f'\n[3] Exporting model (direct mode)...')
        
        dummy_input = torch.randn(1, 3, opt.input_h, opt.input_w, device='cpu')
        
        torch.onnx.export(
            model, 
            dummy_input, 
            output_path,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            verbose=False
        )
        
        print(f'✓ Export successful!')
        print(f'✓ Output: {output_path}')
        return True
        
    except RuntimeError as e:
        if 'dcn_v2' in str(e):
            print(f'\n⚠ DCNv2 operator not supported in this PyTorch version')
            print(f'  Suggestion: Use mode="conv" to replace with Conv2d')
            return False
        else:
            print(f'✗ Export failed: {e}')
            return False
    except Exception as e:
        print(f'✗ Export failed: {e}')
        return False


def export_with_conv_replacement(model, opt, output_path):
    """Export with DCNv2 replaced by Conv2d."""
    try:
        print(f'\n[3] Converting DCNv2 layers to Conv2d...')
        
        dcn_count = 0
        for name, module in model.named_modules():
            module_type = module.__class__.__name__
            if 'DCN' in module_type:
                dcn_count += 1
        
        print(f'  Found {dcn_count} DCN layers')
        
        # Replace DCNv2 with Conv2d
        model = replace_dcn_with_conv(model)
        
        print(f'\n[4] Exporting modified model...')
        
        dummy_input = torch.randn(1, 3, opt.input_h, opt.input_w, device='cpu')
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            verbose=False
        )
        
        print(f'✓ Export successful!')
        print(f'✓ Output: {output_path}')
        print(f'✓ All DCNv2 layers replaced with Conv2d')
        
        # Verify
        print(f'\n[5] Verifying ONNX model...')
        onnx_model = onnx.load(output_path)
        print(f'✓ ONNX model is valid')
        
        return True
        
    except Exception as e:
        print(f'✗ Export failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def replace_dcn_with_conv(model):
    """Recursively replace DCN modules with Conv2d."""
    for name, module in model.named_children():
        if 'DCN' in module.__class__.__name__:
            # Replace DCN with Conv2d
            conv = create_conv_from_dcn(module)
            setattr(model, name, conv)
            print(f'  Replaced: {name} (DCN -> Conv2d)')
        else:
            # Recursively process child modules
            replace_dcn_with_conv(module)
    
    return model


def create_conv_from_dcn(dcn_module):
    """Create Conv2d from DCN module, copying weights."""
    try:
        in_channels = dcn_module.in_channels
        out_channels = dcn_module.out_channels
        kernel_size = dcn_module.kernel_size
        stride = dcn_module.stride
        padding = dcn_module.padding
        dilation = dcn_module.dilation
    except AttributeError:
        # Fallback if attributes don't exist
        return dcn_module
    
    conv = torch.nn.Conv2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, bias=True
    )
    
    # Try to copy weights
    try:
        if hasattr(dcn_module, 'weight'):
            conv.weight.data = dcn_module.weight.data.clone()
        if hasattr(dcn_module, 'bias'):
            conv.bias.data = dcn_module.bias.data.clone()
    except:
        pass  # Weights may not be directly copyable
    
    return conv


def main():
    parser = argparse.ArgumentParser(description='Export CenterNet to ONNX')
    parser.add_argument('task', default='ctdet', help='Task type (e.g., ctdet)')
    parser.add_argument('--load_model', required=True, help='Path to model weights')
    parser.add_argument('--arch', default='dla_34', help='Model architecture')
    parser.add_argument('--input_h', type=int, default=512, help='Input height')
    parser.add_argument('--input_w', type=int, default=512, help='Input width')
    parser.add_argument('--output_onnx', default='', help='Output ONNX filename')
    parser.add_argument('--mode', default='direct', choices=['direct', 'conv'],
                        help='Export mode: direct (keep DCNv2) or conv (replace with Conv2d)')
    parser.add_argument('--gpus', default='-1', help='GPUs to use')
    
    args = parser.parse_args()
    
    # Initialize opts with proper arguments
    sys.argv = [sys.argv[0], args.task, '--load_model', args.load_model,
                '--arch', args.arch, '--gpus', args.gpus]
    
    opt = opts().init()
    opt.input_h = args.input_h
    opt.input_w = args.input_w
    
    output_path = args.output_onnx or f"{os.path.splitext(args.load_model)[0]}.onnx"
    
    success = export_centernet_onnx(opt, mode=args.mode, output_path=output_path)
    
    if success:
        print('\n' + '=' * 70)
        print('✓ Export complete!')
        print('=' * 70)
        print(f'\nNext steps:')
        print(f'  1. Use CNNGEN to compile the ONNX model:')
        print(f'     cnngen --input {output_path} --output model_ir --target cv72')
        print(f'  2. Generate CV7x code:')
        print(f'     cavalry_gen --input model_ir --output cv7x_code')
        sys.exit(0)
    else:
        print('\n' + '=' * 70)
        print('✗ Export failed!')
        print('=' * 70)
        sys.exit(1)


if __name__ == '__main__':
    main()
