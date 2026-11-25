"""
DCNv2 Operator Registration for ONNX Export

This module registers custom DCNv2 operator with ONNX so it can be exported.
Then external tools (CNNGEN/Cavalry_gen) can handle the custom operator.
"""

import torch
import torch.onnx
from torch.onnx import register_custom_op_symbolic

# Register DCNv2 as a custom ONNX operator
@register_custom_op_symbolic('custom::dcn_v2', 9)
def symbolic_dcn_v2(g, input, offset, mask, weight, bias, 
                    kernel_h, kernel_w, stride_h, stride_w, 
                    pad_h, pad_w, dilation_h, dilation_w, deformable_groups):
    """
    Custom symbolic function for DCNv2.
    This tells ONNX how to represent DCNv2 in the exported graph.
    
    The actual DCNv2 computation will be handled by:
    - CNNGEN/Cavalry_gen during compilation
    - Or manual implementation on CV7x
    """
    return g.op(
        'custom::dcn_v2_op',
        input, offset, mask, weight, bias,
        kernel_h_i=kernel_h, kernel_w_i=kernel_w,
        stride_h_i=stride_h, stride_w_i=stride_w,
        pad_h_i=pad_h, pad_w_i=pad_w,
        dilation_h_i=dilation_h, dilation_w_i=dilation_w,
        deformable_groups_i=deformable_groups
    )

# Alternative approach: Replace DCNv2 with standard Conv2D for export
class DCNv2ToConvWrapper(torch.nn.Module):
    """
    Wrapper that replaces DCNv2 with standard Conv2D for ONNX export.
    This provides a baseline model that can run on CV72/CV7x.
    Accuracy may be slightly lower than using DCNv2.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._replace_dcn_with_conv()
    
    def _replace_dcn_with_conv(self):
        """Replace all DCN layers with standard Conv2d."""
        for name, module in self.model.named_modules():
            if 'DCN' in module.__class__.__name__:
                # Get parent module and attribute name
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                
                parent = self._get_submodule(self.model, parent_name)
                if parent is not None:
                    # Create equivalent Conv2d
                    conv = torch.nn.Conv2d(
                        module.in_channels,
                        module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        bias=True
                    )
                    # Copy weights if available
                    if hasattr(module, 'weight'):
                        conv.weight.data = module.weight.data
                    if hasattr(module, 'bias'):
                        conv.bias.data = module.bias.data
                    
                    setattr(parent, attr_name, conv)
    
    def _get_submodule(self, model, name):
        """Get submodule by dotted name."""
        if not name:
            return model
        parts = name.split('.')
        module = model
        for part in parts:
            module = getattr(module, part)
        return module
    
    def forward(self, x):
        return self.model(x)


def export_with_dcn_handling(model, example_input, output_path, 
                              use_conv_replacement=False, opset_version=11):
    """
    Export model to ONNX with DCNv2 handling.
    
    Args:
        model: PyTorch model
        example_input: Example input tensor for tracing
        output_path: Output ONNX file path
        use_conv_replacement: If True, replace DCNv2 with Conv2d
        opset_version: ONNX opset version (11 or 13+)
    
    Returns:
        success: Boolean indicating export success
    """
    
    try:
        if use_conv_replacement:
            print("Converting DCNv2 layers to Conv2d for export...")
            model = DCNv2ToConvWrapper(model)
            model.eval()
            
            torch.onnx.export(
                model, example_input, output_path,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                verbose=False
            )
            print(f"✓ Exported model with Conv2d (DCNv2 replaced)")
            print(f"  Note: Accuracy may be slightly lower than original DCNv2 version")
        else:
            print("Attempting direct export with registered DCNv2 operator...")
            
            torch.onnx.export(
                model, example_input, output_path,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                verbose=False
            )
            print(f"✓ Successfully exported model with DCNv2 operator")
            print(f"  CNNGEN/Cavalry_gen must support custom::dcn_v2 operator")
        
        return True
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return False


if __name__ == '__main__':
    print("DCNv2 ONNX Export Utilities")
    print("\nUsage:")
    print("  from dcnv2_onnx import export_with_dcn_handling")
    print("  export_with_dcn_handling(model, example_input, 'output.onnx')")
    print("\nOptions:")
    print("  use_conv_replacement=True  # Replace DCNv2 with Conv2d")
    print("  opset_version=13           # Use newer ONNX opset")
