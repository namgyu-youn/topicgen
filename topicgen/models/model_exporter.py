import logging
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class ModelExporter:
    """Exports trained PyTorch models for production use."""

    def __init__(self, model, tokenizer_name: str, output_path: str | None = None, device: str | None = None):
        """
        Initialize the model exporter.

        Args:
            model: Trained PyTorch model
            tokenizer_name: Name of the tokenizer used with the model
            output_path: Path for exported model (default: "models/pytorch_model")
            device: PyTorch device (auto-detected if None)
        """
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.output_path = output_path or str(Path("models") / "pytorch_model")

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Model exporter using device: {self.device}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def export_model(self) -> str:
        """
        Export PyTorch model for production use.

        Returns:
            Path to the exported model
        """
        try:
            logger.info(f"Exporting PyTorch model to: {self.output_path}")

            # Prepare model for export
            self.model.eval()

            # Move model to CPU for export (to ensure compatibility)
            cpu_model = self.model.to("cpu")

            # Save the model
            torch.save(cpu_model.state_dict(), f"{self.output_path}/model.pt")

            # Save the config with device info
            if hasattr(self.model, 'config'):
                self.model.config.save_pretrained(self.output_path)

            # Save the tokenizer
            self.tokenizer.save_pretrained(self.output_path)

            # Move model back to original device
            self.model.to(self.device)

            # Save metadata about the model
            with open(f"{self.output_path}/model_info.txt", "w") as f:
                f.write(f"Exported with device: {self.device}\n")
                f.write(f"PyTorch version: {torch.__version__}\n")
                f.write(f"CUDA available: {torch.cuda.is_available()}\n")
                if torch.cuda.is_available():
                    f.write(f"CUDA version: {torch.version.cuda}\n")
                    f.write(f"GPU device: {torch.cuda.get_device_name(0)}\n")

            logger.info(f"Model successfully exported to: {self.output_path}")
            return self.output_path

        except Exception as e:
            logger.error(f"Error exporting model: {e!s}")
            raise

    def validate_model(self) -> bool:
        """
        Validate the exported PyTorch model.

        Returns:
            True if validation successful
        """
        try:
            # Create sample input for inference test
            sample_text = "Testing model inference with a sample repository description"
            inputs = self.tokenizer(
                sample_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )

            # Run PyTorch model
            self.model.eval()
            with torch.no_grad():
                # Move inputs to the same device as the model
                torch_inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Ensure model is on the correct device
                self.model.to(self.device)

                # Run inference
                outputs = self.model(**torch_inputs)

            # Basic validation - check if outputs have expected shape
            if hasattr(outputs, 'logits'):
                # Check if logits have the expected shape
                expected_shape = (1, self.model.config.num_labels)
                actual_shape = tuple(outputs.logits.shape)

                if actual_shape == expected_shape:
                    logger.info(f"Model validation successful: outputs have expected format {actual_shape}")

                    # Test sigmoid activation for multi-label classification
                    logits = outputs.logits
                    probs = torch.sigmoid(logits)

                    # Check if probabilities are in valid range [0,1]
                    if torch.all((probs >= 0) & (probs <= 1)):
                        logger.info("Probability outputs are valid")
                        return True
                    else:
                        logger.warning("Invalid probability values detected")
                        return False
                else:
                    logger.warning(f"Model validation failed: expected shape {expected_shape}, got {actual_shape}")
                    return False
            else:
                logger.warning("Model validation failed: outputs don't have expected format")
                return False

        except Exception as e:
            logger.error(f"Error validating model: {e!s}")
            return False
