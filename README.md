# Voice Authentication with ECAPA-TDNN and ONNX Inference Optimization

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.0%2B-red.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/onnxruntime-green.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Overview

This project develops an efficient, production-ready pipeline for speaker verification, focusing on optimizing a state-of-the-art ECAPA-TDNN deep learning model for deployment using ONNX. The core objective is to achieve fast, cross-platform inference for voice authentication tasks by engineering custom components for ONNX compatibility and exporting the model to an optimized format.

### Key Features

*   **Robust Speaker Embeddings:** Utilizes the powerful ECAPA-TDNN architecture to extract discriminative speaker embeddings from raw audio.
*   **ONNX-Compatible Custom Layers:**
    *   **Custom Mel-Spectrogram (`MelConv`):** Replaces standard `torchaudio` Mel spectrogram transformations with an ONNX-serializable implementation for pre-processing.
    *   **ONNX-Safe Attentive Statistics Pooling (`ONNXAttentiveStatsPool`):** Implements an ONNX-friendly version of the ASP layer, critical for model accuracy and deployment within the ONNX ecosystem.
*   **Efficient Model Export:** Seamlessly exports the entire ECAPA-TDNN pipeline, including custom layers, to the ONNX format, supporting dynamic input shapes.
*   **Optimized Inference:** Enables highly efficient, CPU-optimized inference using ONNX Runtime, suitable for resource-constrained environments (e.g., edge devices) or scalable backend services.

### Technical Deep Dive & My Contribution

The primary engineering challenge addressed in this project was making the entire ECAPA-TDNN inference pipeline fully compatible and efficient within the ONNX ecosystem. Standard `torchaudio` transforms and certain `SpeechBrain` components (like its ASP) are often not directly traceable or optimizable for ONNX export. My key contributions involved:

1.  **Mel-Spectrogram Re-implementation:** Developed `src/models/mel_conv.py` to manually re-create the Mel-spectrogram calculation using ONNX-compatible PyTorch operations (detailed framing, windowing, rFFT, Mel filterbank via `Conv1d`, log-scaling).
2.  **Custom ONNXAttentiveStatsPooling:** Engineered `src/models/onnx_pool.py`, a custom version of the Attentive Statistics Pooling layer that matches the functionality of the original while ensuring full traceability and optimization for ONNX export.
3.  **Integrated Wrapper:** Created `src/models/ecapawrapper.py` to seamlessly integrate the pre-trained SpeechBrain ECAPA-TDNN base with these custom ONNX-compatible pre- and post-processing layers, forming a unified model for export.
4.  **ONNX Export Script:** Developed `export_onnx.py` to correctly export the wrapped model to ONNX with dynamic input axes, ensuring flexibility for various audio lengths.

This work demonstrates a deep understanding of PyTorch model internals, ONNX graph tracing, and strategic optimization for deploying complex deep learning models beyond simple training.

### How it Works (Architectural Flow)

```mermaid
graph TD
    A[Raw Audio (WAV)] --> B(MelConv: Custom Mel-Spectrogram)
    B --> C(ECAPA-TDNN: Feature Extraction)
    C --> D(ONNXAttentiveStatsPool: Custom ASP)
    D --> E(L2 Normalization)
    E --> F(Speaker Embedding (192-dim))
    F --> G[ONNX Runtime (Deployment)]
# Voice-ID-ONNX-Deployment
"Efficient Voice Authentication leveraging ECAPA-TDNN, custom ONNX-compatible layers (MelSpec, ASP), and optimized ONNX export for high-performance, cross-platform inference."
