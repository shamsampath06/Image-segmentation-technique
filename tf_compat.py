"""
TensorFlow Compatibility Fix
----------------------------
This file applies the PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION environment variable
fix for TF 2.11 + protobuf compatibility. Import this before any TensorFlow import.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
