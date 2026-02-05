# Unity Image Recognition Project

This project demonstrates a real-time image classification system built with **Unity 6 (6000.0.58f2)** and **Unity Sentis** (formerly Barracuda). It uses a deep learning model to classify input images directly within the Unity engine.

## Features

- **In-Engine Inference**: Runs AI models natively in Unity without external dependencies using Sentis.
- **GPU Acceleration**: Utilizes GPU Compute for efficient model execution.
- **Image Processing**: Automatically converts and resizes `Texture2D` inputs to the required tensor shape `(1, 3, 224, 224)`.
- **Top Prediction**: identifies the most likely class from the model's output.
- **Target Search**: Ability to check confidence for specific target objects (e.g. "goldfish").

## Technical Details

- **Model**: [MobileNetV2](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet) (`mobilenetv2-10.onnx`)
- **Input Resolution**: 224x224 RGB
- **Backend**: Unity Sentis `BackendType.GPUCompute`

## Requirements

- **Unity Version**: 6000.0.58f2 or later (should work with later unity 6 versions too)
- **Packages**:
  - `com.unity.ai.inference` (Sentis) v2.2.2

## Setup and Usage

1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   ```
2. **Open in Unity**:
   - Open the project via Unity Hub.
   - Ensure the Unity version matches or is compatible.
3. **Configuration**:
   - The main logic is in `ClassifyImage.cs`.
   - Ensure a valid `.onnx` model is assigned to the `Model Asset` field.
   - Ensure `synset.txt` (ImageNet labels) is assigned to the `Labels Field`.
   - Assign an input image to the `Input Image` field in the inspector.
4. **Run**:
   - Enter Play Mode.
   - The classification result will be logged to the Console.

## File Structure

- `Assets/Scripts/ClassifyImage.cs`: Core logic for loading model, processing input, and interpreting results.
- `Assets/AI Models/`: Contains the `mobilenetv2-10.onnx` model file.
- `Assets/synset.txt`: List of class labels (ImageNet).

## References

- [Unity Sentis Documentation](https://docs.unity3d.com/Packages/com.unity.sentis@latest)
- [ONNX Model Zoo - MobileNet](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet)
