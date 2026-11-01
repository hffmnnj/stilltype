# Local face tracking

Stilltype uses the OpenCV Zoo **YuNet 2026may dynamic-input ONNX export**,
redistributed by Pollen Robotics. This artifact was published in 2026; its learned
weights are shared with the earlier YuNet model. It is not a newly trained 2026
architecture. Silent-speech recognition is separately provided by USR 2.0 (2026).

- Source: https://huggingface.co/pollen-robotics/face_detection_yunet_2026may
- Revision: `2b8e922362946a0db67e861bae0f77826980effd`
- SHA256: `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`
- License: [MIT](LICENSE.yunet.txt).

ONNX Runtime Web runs the small detector locally with WebAssembly. Its runtime
assets are copied from the locked npm dependency by `npm run prepare:vision`.
License: [MIT](LICENSE.onnxruntime.txt). No model or runtime assets are fetched
from a CDN while the application is running.
