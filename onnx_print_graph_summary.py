
import onnx
import sys
import numpy as np

def print_graph_summary(onnx_path):
    model = onnx.load(onnx_path)
    graph = model.graph
    print(f"Model: {onnx_path}")
    print(f"Inputs:")
    for inp in graph.input:
        print(f"  {inp.name}: {inp.type}")
    print(f"Outputs:")
    for out in graph.output:
        print(f"  {out.name}: {out.type}")
    print(f"Nodes: {len(graph.node)}")
    print(f"Initializers: {len(graph.initializer)}")
    print("Node types:")
    types = {}
    for node in graph.node:
        types[node.op_type] = types.get(node.op_type, 0) + 1
    for k, v in types.items():
        print(f"  {k}: {v}")
    
    # Xuất initializers (weights) ra file .bin
    print("Saving initializers to bin files:")
    for init in graph.initializer:
        data = onnx.numpy_helper.to_array(init)
        filename = f"{init.name}.bin"
        data.tofile(filename)
        print(f"  Saved {init.name} to {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python onnx_print_graph_summary.py <onnx_model_path>")
        sys.exit(1)
    print_graph_summary(sys.argv[1])