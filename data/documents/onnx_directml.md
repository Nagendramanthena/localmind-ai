# ONNX Runtime and DirectML

## Overview

ONNX Runtime is a high-performance inference engine for machine learning models in the Open Neural Network Exchange (ONNX) format. It provides cross-platform acceleration with support for hardware-specific execution providers.

DirectML (Direct Machine Learning) is a high-performance, hardware-accelerated DirectX 12 library for machine learning on Windows. It provides GPU acceleration across all DirectX 12-capable hardware, including AMD, NVIDIA, Intel, and Qualcomm GPUs.

## ONNX Runtime Execution Providers

ONNX Runtime supports multiple execution providers (EPs) that optimize inference for specific hardware:

### CPU Execution Provider
- Default provider available on all platforms
- Optimized with MLAS (Microsoft Linear Algebra Subprograms)
- Supports x86/x64 and ARM architectures
- Good for models under 100M parameters

### DirectML Execution Provider
- Windows-specific GPU acceleration via DirectX 12
- Vendor-agnostic: works with AMD, NVIDIA, Intel, Qualcomm GPUs
- Install: `pip install onnxruntime-directml`
- Do NOT install alongside `onnxruntime` or `onnxruntime-gpu`

### CUDA Execution Provider
- NVIDIA GPU acceleration
- Requires CUDA toolkit installation
- Best for NVIDIA-only deployments

### TensorRT Execution Provider
- NVIDIA's high-performance inference optimizer
- Built on top of CUDA
- Provides additional optimizations like layer fusion and precision calibration

## Using ONNX Runtime with DirectML

### Installation

```bash
pip install onnxruntime-directml
```

### Basic Inference

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    "model.onnx",
    providers=['DmlExecutionProvider', 'CPUExecutionProvider']
)

input_data = {session.get_inputs()[0].name: np.array(data, dtype=np.float32)}
results = session.run(None, input_data)
```

### Best Practices

1. **Provider Fallback**: Always list `CPUExecutionProvider` as a fallback
2. **Graph Optimization**: Enable `ORT_ENABLE_ALL` for automatic graph optimizations
3. **Session Reuse**: Create the `InferenceSession` once and reuse it
4. **Batch Processing**: Process multiple inputs in a single call for better GPU utilization
5. **Model Optimization**: Use Microsoft Olive to optimize models for DirectML

## Exporting Models to ONNX

### From PyTorch

Using Hugging Face Optimum (recommended):

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction

model = ORTModelForFeatureExtraction.from_pretrained("model-name", export=True)
model.save_pretrained("./onnx_output")
```

### From TensorFlow

```python
import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")
tf2onnx.convert.from_keras(model, output_path="model.onnx")
```

## Performance Considerations

- **Quantization**: INT8 quantization can reduce model size by 4x with minimal accuracy loss
- **Graph Optimization**: ONNX Runtime can fuse operations, eliminate redundant computations
- **Memory Management**: DirectML manages GPU memory automatically through DirectX 12
- **Warm-up**: First inference call is slower due to graph compilation; subsequent calls are faster
