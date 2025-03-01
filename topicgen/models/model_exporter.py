import logging
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class ModelExporter:
    """Exports trained PyTorch models to ONNX format for production use."""

    def __init__(self, model, tokenizer_name: str, output_path: str | None = None):
        """
        Initialize the model exporter.

        Args:
            model: Trained PyTorch model
            tokenizer_name: Name of the tokenizer used with the model
            output_path: Path for exported model (default: "models/model.onnx")
        """
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.output_path = output_path or str(Path("models") / "model.onnx")

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def export_to_onnx(self,
                      sequence_length: int = 512,
                      batch_size: int = 1) -> str:
        """
        Export model to ONNX format.

        Args:
            sequence_length: Maximum sequence length
            batch_size: Batch size for ONNX model

        Returns:
            Path to the exported model
        """
        try:
            logger.info(f"Exporting model to ONNX: {self.output_path}")

            # Prepare model for export
            self.model.eval()

            # Create dummy input
            dummy_input = self.tokenizer(
                "This is a sample input for ONNX export",
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=sequence_length
            )

            # Move to the same device as model
            dummy_input = {k: v.to(self.model.device) for k, v in dummy_input.items()}

            # Define dynamic axes for variable batch size and sequence length
            dynamic_axes = {
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size'},
            }

            # Add token_type_ids if present
            if 'token_type_ids' in dummy_input:
                dynamic_axes['token_type_ids'] = {0: 'batch_size', 1: 'sequence'}

            # Export to ONNX
            torch.onnx.export(
                self.model,
                (dummy_input,),
                self.output_path,
                input_names=list(dummy_input.keys()),
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                opset_version=13,  # Higher opset for advanced operations
                do_constant_folding=True,  # Optimize constants
                export_params=True  # Export model parameters
            )

            logger.info(f"Model successfully exported to: {self.output_path}")
            return self.output_path

        except Exception as e:
            logger.error(f"Error exporting model to ONNX: {e!s}")
            raise

    def validate_onnx_model(self) -> bool:
        """
        Validate the exported ONNX model.

        Returns:
            True if validation successful
        """
        try:
            import onnx
            import onnxruntime as ort

            # Load and check ONNX model
            onnx_model = onnx.load(self.output_path)
            onnx.checker.check_model(onnx_model)

            logger.info("ONNX model structure validated successfully")

            # Create sample input for inference test
            sample_text = "Testing ONNX model inference with a sample repository description"
            inputs = self.tokenizer(
                sample_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )

            # Convert to numpy for ONNX Runtime
            onnx_inputs = {k: v.numpy() for k, v in inputs.items()}

            # Run PyTorch model for comparison
            self.model.eval()
            with torch.no_grad():
                torch_inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                torch_outputs = self.model(**torch_inputs).logits.cpu().numpy()

            # Run ONNX model
            ort_session = ort.InferenceSession(self.output_path)
            ort_inputs = dict(onnx_inputs.items())
            ort_outputs = ort_session.run(None, ort_inputs)[0]

            # Compare outputs (with tolerance for numerical differences)
            is_close = np.allclose(torch_outputs, ort_outputs, rtol=1e-3, atol=1e-5)

            if is_close:
                logger.info("ONNX model validation successful: PyTorch and ONNX outputs match")
            else:
                logger.warning("PyTorch and ONNX outputs have significant differences")

            return is_close

        except Exception as e:
            logger.error(f"Error validating ONNX model: {e!s}")
            return False