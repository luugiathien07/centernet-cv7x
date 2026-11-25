import time
import numpy as np
import onnxruntime as ort


def build_dummy_input(session):
    input_meta = session.get_inputs()[0]
    shape = []
    for dim in input_meta.shape:
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        else:
            shape.append(1)
    return input_meta.name, np.random.rand(*shape).astype(np.float32)


def benchmark(session, input_name, input_data, runs=100):
    session.run(None, {input_name: input_data})
    times = []
    for _ in range(runs):
        start = time.time()
        session.run(None, {input_name: input_data})
        times.append(time.time() - start)
    return np.mean(times) * 1000


if __name__ == "__main__":
    session = ort.InferenceSession(
        "./models/ctdet_coco_dla_2x_conv.onnx"
    )
    input_name, input_data = build_dummy_input(session)
    avg_ms = benchmark(session, input_name, input_data)
    print(f"Average time: {avg_ms:.2f} ms")