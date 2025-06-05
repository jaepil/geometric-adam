#!/usr/bin/env python3
"""
Multi-Architecture Geometric Adam Optimizer Testing Framework

This framework tests the Geometric Adam optimizer across multiple neural
network architectures (Transformer, CNN, LSTM, GNN) with appropriate datasets
for each architecture type.

Author: Jaepil Jeong
Python: 3.12+
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import os
import json
import time
import math
import logging
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# Configure matplotlib and seaborn for better plots - with fallback
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    try:
        plt.style.use("seaborn-darkgrid")
    except OSError:
        plt.style.use("default")
        print("Warning: Seaborn style not available, using default")

sns.set_palette("husl")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def ensure_reproducibility(seed: int = 42):
    """
    Ensure complete reproducibility across all random operations.
    This should be called once at the start of the program.

    Args:
        seed: Random seed for all libraries
    """

    # Python's built-in random
    import random

    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA-specific settings
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        # These settings trade performance for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Set environment variable for better reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Reproducibility ensured with seed: {seed}")
    print(
        f"  - Deterministic mode: {torch.backends.cudnn.deterministic if torch.cuda.is_available() else 'N/A'}"
    )
    print(
        f"  - Benchmark mode: {torch.backends.cudnn.benchmark if torch.cuda.is_available() else 'N/A'}"
    )


class LoggerManager:
    """
    Unified logging manager supporting console, file, TensorBoard, and W&B.
    This centralizes all logging to make experiments easier to track and share.

    Enhanced with robust image logging that handles format conversions properly.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        use_tensorboard: bool = True,
        use_wandb: bool = True,
        wandb_project: str = "geometric-adam",
        config: Optional[Dict] = None,
    ):
        """
        Initialize the logger manager with multiple backends.

        Args:
            experiment_name: Name of the current experiment
            log_dir: Directory for logs
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            config: Configuration dictionary to log
        """

        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Setup standard Python logging
        self._setup_file_logging()

        # Setup TensorBoard
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        if use_tensorboard:
            self._setup_tensorboard()

        # Setup Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            self._setup_wandb(wandb_project, config)

        self.step = 0

    def _setup_file_logging(self):
        """Setup traditional file and console logging."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.experiment_name}_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def _setup_tensorboard(self):
        """Setup TensorBoard writer."""

        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = self.log_dir / "tensorboard" / self.experiment_name
            self.tb_writer = SummaryWriter(str(tb_dir))
            self.logger.info(
                f"TensorBoard initialized. Run: tensorboard --logdir {tb_dir.parent}"
            )
        except ImportError:
            self.logger.warning(
                "TensorBoard not available. Install with: pip install tensorboard"
            )
            self.use_tensorboard = False

    def _setup_wandb(self, project: str, config: Optional[Dict]):
        """Setup Weights & Biases logging."""

        try:
            import wandb

            wandb.init(
                project=project,
                name=self.experiment_name,
                config=config or {},
                reinit=True,
            )
            self.logger.info(f"W&B initialized for project: {project}")
        except ImportError:
            self.logger.warning("W&B not available. Install with: pip install wandb")
            self.use_wandb = False

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to all enabled backends.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (uses internal counter if not provided)
        """

        if step is None:
            step = self.step
            self.step += 1

        # Log to TensorBoard
        if self.use_tensorboard and self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)

        # Log to W&B
        if self.use_wandb:
            import wandb

            wandb.log(metrics, step=step)

    def log_histogram(
        self, name: str, values: torch.Tensor, step: Optional[int] = None
    ):
        """Log histogram data (useful for weight distributions)."""

        if step is None:
            step = self.step

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_histogram(name, values, step)

        if self.use_wandb:
            import wandb

            wandb.log({name: wandb.Histogram(values.cpu().numpy())}, step=step)

    def _validate_image_array(self, img_array):
        """
        Validate that the image array has a sensible shape and data.

        This validation prevents the mysterious shape errors we encountered
        by catching malformed image data before it reaches TensorBoard.
        """

        if not isinstance(img_array, np.ndarray):
            return False, "Not a numpy array"

        if len(img_array.shape) not in [2, 3]:
            return False, f"Invalid dimensions: {len(img_array.shape)}"

        if len(img_array.shape) == 3:
            h, w, c = img_array.shape
            if c not in [1, 3, 4]:
                return False, f"Invalid channel count: {c}"
            if h < 10 or w < 10:
                return False, f"Image too small: {h}x{w}"
            if h > 10000 or w > 10000:
                return False, f"Image too large: {h}x{w}"

        return True, "Valid"

    def log_image(
        self,
        name: str,
        image: Union[torch.Tensor, np.ndarray],
        step: Optional[int] = None,
    ):
        """
        Log image data with comprehensive validation and error recovery.

        This enhanced version fixes the original TensorBoard image logging issues by:
        1. Properly validating image array shapes and data types
        2. Converting between HWC and CHW formats as needed
        3. Handling different color channels (RGB, RGBA, grayscale)
        4. Ensuring data is in the correct range [0,1] for TensorBoard
        """

        if step is None:
            step = self.step

        # Handle different input types and convert to proper format
        if isinstance(image, np.ndarray):
            # Validate the input image first - this prevents the mysterious shape errors
            is_valid, reason = self._validate_image_array(image)
            if not is_valid:
                self.logger.warning(f"Invalid image for logging '{name}': {reason}")
                return

            # Ensure the image has the right shape and data type
            if len(image.shape) == 3:
                # Determine if we need to transpose: TensorBoard expects CHW format
                if image.shape[2] in [1, 3, 4]:  # HWC format (channels last)
                    image = np.transpose(image, (2, 0, 1))  # Convert to CHW
                elif image.shape[0] in [1, 3, 4]:  # Already CHW format
                    pass  # Keep as is
                else:
                    self.logger.warning(
                        f"Ambiguous image shape for '{name}': {image.shape}"
                    )
                    return

                # Handle alpha channel by removing it (RGBA -> RGB)
                if image.shape[0] == 4:  # RGBA -> RGB
                    image = image[:3, :, :]

                # Ensure values are in [0, 1] range for TensorBoard
                # This fixes issues where matplotlib generates images in [0, 255] range
                if image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                elif image.max() > 1.0:
                    # Clamp to valid range to prevent overflow issues
                    image = np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)

            # Convert to torch tensor for TensorBoard
            image = torch.from_numpy(image.astype(np.float32))

        # Log to TensorBoard with comprehensive error handling
        if self.use_tensorboard and self.tb_writer:
            try:
                self.tb_writer.add_image(name, image, step)
            except Exception as e:
                self.logger.warning(f"Failed to log image '{name}' to TensorBoard: {e}")

        # Log to W&B with format conversion
        if self.use_wandb:
            try:
                import wandb

                # Convert back to numpy for W&B (expects HWC format)
                if isinstance(image, torch.Tensor):
                    if len(image.shape) == 3:  # CHW format
                        img_array = image.permute(1, 2, 0).numpy()  # CHW -> HWC
                    else:
                        img_array = image.numpy()
                else:
                    img_array = image

                # Ensure valid range for W&B display
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)

                wandb.log({name: wandb.Image(img_array)}, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log image '{name}' to W&B: {e}")

    def log_text(self, name: str, text: str, step: Optional[int] = None):
        """Log text data (useful for generated samples)."""

        if step is None:
            step = self.step

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_text(name, text, step)

        if self.use_wandb:
            import wandb

            # W&B logs text as HTML table
            wandb.log({name: wandb.Html(f"<p>{text}</p>")}, step=step)

    def close(self):
        """Close all logging connections."""

        if self.tb_writer:
            self.tb_writer.close()

        if self.use_wandb:
            import wandb

            wandb.finish()


def _get_best_device() -> str:
    """
    Determine the best available device for computation.

    Priority: CUDA > MPS > CPU
    MPS (Metal Performance Shaders) is Apple's GPU acceleration framework
    available on Apple Silicon Macs (M1, M2, M3, etc.).

    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@dataclass
class ExperimentConfig:
    """Enhanced configuration for multi-architecture optimization experiments."""

    # Architecture selection
    architecture: str = "transformer"  # "transformer", "cnn", "lstm", "gnn"
    task_type: str = "classification"  # "classification", "regression", "generation"
    dataset_name: str = "wikitext2"  # dataset appropriate for architecture

    # Model size selection
    model_size: str = "2.5M"  # "2.5M", "5M", "10M", "30M", "50M", "100M"

    # Common model parameters - will be auto-adjusted based on model_size
    model_dim: int = 96  # Base dimension for transformer/lstm
    hidden_dim: int = 256  # Hidden dimension for RNN/GNN
    num_layers: int = 4  # Number of layers
    dropout: float = 0.1

    # Transformer specific
    num_heads: int = 8
    ff_dim: int = 384
    vocab_size: int = 10000
    max_seq_len: int = 128

    # CNN specific
    num_classes: int = 10
    image_channels: int = 3
    image_size: int = 32
    base_channels: int = 64  # Base channel count for CNN

    # LSTM specific
    embed_dim: int = 128
    bidirectional: bool = False

    # GNN specific
    num_node_features: int = 1433
    num_edge_features: int = 0
    graph_hidden_dim: int = 64

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Geometric optimizer specific parameters
    refraction_sensitivity: float = 0.1
    curvature_memory: float = 0.95
    geometric_momentum_beta: float = 0.9

    # Experiment settings
    device: str = field(default_factory=_get_best_device)
    seed: int = 42
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50
    eval_interval: int = 100
    patience: int = 5  # Early stopping patience

    # Logging settings
    use_tensorboard: bool = True
    use_wandb: bool = True
    wandb_project: str = "geometric-adam-multi-arch"
    experiment_name: str = field(
        default_factory=lambda: f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    def __post_init__(self):
        """Post-initialization validation and adjustments."""

        # Adjust batch size for MPS if needed
        if self.device == "mps" and self.batch_size > 16:
            self.batch_size = 16
            print(f"Adjusted batch size to {self.batch_size} for MPS optimization")

        # Adjust parameters based on architecture
        if self.architecture == "cnn" and self.dataset_name == "wikitext2":
            self.dataset_name = "cifar10"  # Default to CIFAR-10 for CNN
        elif self.architecture == "lstm" and self.dataset_name == "wikitext2":
            self.dataset_name = "penn_treebank"  # Default to PTB for LSTM

        # Adjust model parameters based on target size
        self._adjust_model_params_for_size()

    def _adjust_model_params_for_size(self):
        """Adjust model hyperparameters to achieve target parameter count."""

        # Model size configurations
        model_configs = {
            "2.5M": {
                "transformer": {
                    "model_dim": 96,
                    "num_heads": 8,
                    "num_layers": 4,
                    "ff_dim": 384,
                },
                "cnn": {
                    "base_channels": 64,
                    "num_layers": 4,
                    "layer_blocks": [2, 2, 2, 2],
                },
                "lstm": {"embed_dim": 128, "hidden_dim": 256, "num_layers": 4},
            },
            "5M": {
                "transformer": {
                    "model_dim": 128,
                    "num_heads": 8,
                    "num_layers": 6,
                    "ff_dim": 512,
                },
                "cnn": {
                    "base_channels": 80,
                    "num_layers": 4,
                    "layer_blocks": [2, 3, 3, 2],
                },
                "lstm": {"embed_dim": 192, "hidden_dim": 384, "num_layers": 4},
            },
            "10M": {
                "transformer": {
                    "model_dim": 192,
                    "num_heads": 8,
                    "num_layers": 8,
                    "ff_dim": 768,
                },
                "cnn": {
                    "base_channels": 96,
                    "num_layers": 4,
                    "layer_blocks": [3, 4, 4, 3],
                },
                "lstm": {"embed_dim": 256, "hidden_dim": 512, "num_layers": 5},
            },
            "30M": {
                "transformer": {
                    "model_dim": 320,
                    "num_heads": 10,
                    "num_layers": 12,
                    "ff_dim": 1280,
                },
                "cnn": {
                    "base_channels": 128,
                    "num_layers": 5,
                    "layer_blocks": [3, 4, 6, 4, 3],
                },
                "lstm": {"embed_dim": 384, "hidden_dim": 768, "num_layers": 6},
            },
            "50M": {
                "transformer": {
                    "model_dim": 384,
                    "num_heads": 12,
                    "num_layers": 16,
                    "ff_dim": 1536,
                },
                "cnn": {
                    "base_channels": 160,
                    "num_layers": 5,
                    "layer_blocks": [3, 4, 8, 4, 3],
                },
                "lstm": {"embed_dim": 512, "hidden_dim": 1024, "num_layers": 6},
            },
            "100M": {
                "transformer": {
                    "model_dim": 512,
                    "num_heads": 16,
                    "num_layers": 24,
                    "ff_dim": 2048,
                },
                "cnn": {
                    "base_channels": 224,
                    "num_layers": 5,
                    "layer_blocks": [3, 4, 12, 4, 3],
                },
                "lstm": {"embed_dim": 768, "hidden_dim": 1536, "num_layers": 8},
            },
        }

        if self.model_size not in model_configs:
            raise ValueError(
                f"Invalid model size: {self.model_size}. "
                f"Choose from: {list(model_configs.keys())}"
            )

        config = model_configs[self.model_size].get(self.architecture, {})

        # Apply architecture-specific configurations
        if self.architecture == "transformer":
            self.model_dim = config.get("model_dim", self.model_dim)
            self.num_heads = config.get("num_heads", self.num_heads)
            self.num_layers = config.get("num_layers", self.num_layers)
            self.ff_dim = config.get("ff_dim", self.ff_dim)

            # Adjust batch size for larger models
            if self.model_size in ["50M", "100M"]:
                self.batch_size = min(self.batch_size, 8)

        elif self.architecture == "cnn":
            self.base_channels = config.get("base_channels", self.base_channels)
            self.num_layers = config.get("num_layers", self.num_layers)
            self.layer_blocks = config.get("layer_blocks", [2, 2, 2, 2])

        elif self.architecture == "lstm":
            self.embed_dim = config.get("embed_dim", self.embed_dim)
            self.hidden_dim = config.get("hidden_dim", self.hidden_dim)
            self.num_layers = config.get("num_layers", self.num_layers)

            # For larger LSTM models, consider using bidirectional
            if self.model_size in ["50M", "100M"]:
                self.bidirectional = True
                self.batch_size = min(self.batch_size, 16)

        print(f"Configured {self.architecture} for {self.model_size} parameters")


class ExperimentTracker:
    """Track and store experiment metrics efficiently with step-wise best tracking."""

    def __init__(self, logger_manager: Optional[LoggerManager] = None):
        self.metrics = defaultdict(list)
        self.step_metrics = defaultdict(list)
        self.best_metrics = {}

        # Step-wise best tracking for discovering exceptional performance
        self.step_best_metrics = defaultdict(lambda: float("inf"))
        self.step_best_info = {}

        # Critical phase detection
        self.critical_phases = []
        self.loss_variance_history = []
        self.gradient_norm_history = []

        # Logger for advanced logging
        self.logger_manager = logger_manager

    def log_step(self, metrics: Dict[str, float], step: int):
        """Log metrics for a single training step with enhanced tracking."""

        for key, value in metrics.items():
            self.step_metrics[key].append((step, value))

        # Enhanced step-wise best tracking
        # Track best perplexity (critical for paper claims like 62.97)
        if (
            "perplexity" in metrics
            and metrics["perplexity"] < self.step_best_metrics["perplexity"]
        ):
            self.step_best_metrics["perplexity"] = metrics["perplexity"]
            self.step_best_info["perplexity"] = {
                "step": step,
                "value": metrics["perplexity"],
                "loss": metrics.get("loss", None),
                "improvement_from_avg": None,  # Will calculate later
                "all_metrics": metrics.copy(),
            }

            # Calculate improvement from recent average
            recent_perplexities = [
                m[1] for m in self.step_metrics["perplexity"][-100:] if m[0] < step
            ]
            if recent_perplexities:
                avg_recent = np.mean(recent_perplexities)
                improvement = (avg_recent - metrics["perplexity"]) / avg_recent * 100
                self.step_best_info["perplexity"]["improvement_from_avg"] = improvement

                # Log exceptional discoveries
                if self.logger_manager and improvement > 20:  # 20% improvement
                    self.logger_manager.logger.info(
                        f"🎯 EXCEPTIONAL DISCOVERY: PPL {metrics['perplexity']:.2f} "
                        f"({improvement:.1f}% better than recent average)"
                    )

        # Track other step-wise bests
        for metric in ["loss", "accuracy"]:
            if metric in metrics:
                if (
                    metric == "loss"
                    and metrics[metric] < self.step_best_metrics[metric]
                ):
                    self.step_best_metrics[metric] = metrics[metric]
                    self.step_best_info[metric] = {
                        "step": step,
                        "value": metrics[metric],
                    }
                elif metric == "accuracy" and metrics[
                    metric
                ] > self.step_best_metrics.get(metric, 0):
                    self.step_best_metrics[metric] = metrics[metric]
                    self.step_best_info[metric] = {
                        "step": step,
                        "value": metrics[metric],
                    }

        # Collect data for critical phase detection
        if "loss" in metrics:
            self.loss_variance_history.append((step, metrics["loss"]))
        if "grad_norm" in metrics:
            self.gradient_norm_history.append((step, metrics["grad_norm"]))

        # Log to TensorBoard/W&B if available
        if self.logger_manager:
            prefixed_metrics = {f"train/{k}": v for k, v in metrics.items()}
            self.logger_manager.log_metrics(prefixed_metrics, step)

    def detect_critical_phases(
        self, window_size: int = 20, variance_percentile: float = 90
    ):
        """
        Detect critical optimization phases where the model behavior changes dramatically.

        Critical phases are characterized by:
        - High variance in loss values
        - Rapid changes in gradient norms
        - Significant direction changes (if geometric metrics available)

        Args:
            window_size: Size of sliding window for variance calculation
            variance_percentile: Percentile threshold for detecting high variance

        Returns:
            List of tuples (start_step, end_step, severity) for each critical phase
        """

        if len(self.loss_variance_history) < window_size:
            return []

        # Calculate rolling variance of loss
        variances = []
        for i in range(window_size, len(self.loss_variance_history)):
            window_losses = [
                loss for _, loss in self.loss_variance_history[i - window_size : i]
            ]
            variance = np.var(window_losses)
            variances.append((self.loss_variance_history[i][0], variance))

        # Find high variance periods
        if not variances:
            return []

        variance_values = [v for _, v in variances]
        if not variance_values:  # Additional safety check
            return []

        threshold = np.percentile(variance_values, variance_percentile)

        # Identify critical phases
        critical_phases = []
        in_critical_phase = False
        phase_start = None

        for step, variance in variances:
            if variance > threshold and not in_critical_phase:
                in_critical_phase = True
                phase_start = step
            elif variance <= threshold and in_critical_phase:
                in_critical_phase = False
                if phase_start is not None:
                    # Calculate severity based on max variance in the phase
                    phase_variances = [
                        v for s, v in variances if phase_start <= s <= step
                    ]
                    severity = (
                        max(phase_variances) / threshold if threshold > 0 else 1.0
                    )
                    critical_phases.append((phase_start, step, severity))

        # Handle case where we're still in a critical phase
        if in_critical_phase and phase_start is not None:
            severity = (
                max(v for s, v in variances if s >= phase_start) / threshold
                if threshold > 0
                else 1.0
            )
            critical_phases.append((phase_start, variances[-1][0], severity))

        self.critical_phases = critical_phases

        # Log critical phases
        if self.logger_manager and critical_phases:
            self.logger_manager.logger.info(
                f"Detected {len(critical_phases)} critical phases:"
            )
            for start, end, severity in critical_phases:
                duration = end - start
                self.logger_manager.logger.info(
                    f"  Phase {start}-{end}: duration={duration}, severity={severity:.2f}x baseline"
                )

        return critical_phases

    def analyze_step_wise_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in step-wise best achievements.

        Returns insights about:
        - Frequency of new bests over time
        - Clustering of improvements
        - Correlation with critical phases
        """

        analysis = {
            "best_achievement_rate": {},
            "improvement_clusters": {},
            "critical_phase_correlation": {},
        }

        # Calculate rate of best achievements over time
        for metric, info in self.step_best_info.items():
            if metric in self.step_metrics:
                # Find all steps where new bests were achieved
                best_steps = []
                current_best = float("inf") if metric in ["loss", "perplexity"] else 0

                for step, value in self.step_metrics[metric]:
                    if metric in ["loss", "perplexity"] and value < current_best:
                        best_steps.append(step)
                        current_best = value
                    elif metric == "accuracy" and value > current_best:
                        best_steps.append(step)
                        current_best = value

                # Analyze distribution of improvements
                if len(best_steps) > 1:
                    intervals = np.diff(best_steps)
                    analysis["best_achievement_rate"][metric] = {
                        "total_improvements": len(best_steps),
                        "avg_steps_between_improvements": np.mean(intervals),
                        "improvement_acceleration": np.polyfit(
                            range(len(intervals)), intervals, 1
                        )[0],
                    }

        return analysis

    def log_epoch(self, metrics: Dict[str, float], epoch: int):
        """Log metrics for an epoch."""

        for key, value in metrics.items():
            self.metrics[key].append(value)

        # Log to advanced loggers
        if self.logger_manager:
            prefixed_metrics = {f"epoch/{k}": v for k, v in metrics.items()}
            self.logger_manager.log_metrics(prefixed_metrics, epoch)

    def update_best(self, metric_name: str, value: float, mode: str = "min"):
        """Update best metric value."""

        if metric_name not in self.best_metrics:
            self.best_metrics[metric_name] = value
        else:
            if mode == "min" and value < self.best_metrics[metric_name]:
                self.best_metrics[metric_name] = value
            elif mode == "max" and value > self.best_metrics[metric_name]:
                self.best_metrics[metric_name] = value

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics."""

        summary = {
            "epoch_metrics": dict(self.metrics),
            "best_metrics": self.best_metrics,
            "step_best_metrics": dict(self.step_best_metrics),
            "step_best_info": self.step_best_info,
            "final_metrics": {
                key: values[-1] if values else None
                for key, values in self.metrics.items()
            },
        }
        return summary


class GeometricAdam(torch.optim.Optimizer):
    """
    Geometrically-inspired Adam optimizer that incorporates ray tracing concepts.

    This optimizer treats the optimization landscape as a geometric space where:
    - Gradients are surface normals
    - Momentum changes represent ray direction changes
    - Adaptive learning rates simulate refraction coefficients
    - Curvature information guides path selection

    Key improvements in this version:
    - Proper state_dict() support for all geometric states
    - Better numerical stability
    - Complete checkpointing support
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        refraction_sensitivity=0.1,
        curvature_memory=0.95,
        weight_decay=0.0,
    ):
        """
        Initialize the Geometric Adam optimizer.

        Args:
            params: Model parameters to optimize
            lr: Base learning rate
            betas: Coefficients for momentum and variance estimates
            eps: Small constant for numerical stability
            refraction_sensitivity: Controls how much direction changes affect step size
            curvature_memory: Memory factor for curvature estimation
            weight_decay: L2 penalty coefficient
        """

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            refraction_sensitivity=refraction_sensitivity,
            curvature_memory=curvature_memory,
            weight_decay=weight_decay,
        )
        super(GeometricAdam, self).__init__(params, defaults)

        # Track optimizer statistics
        self.stats = {"refraction_coeffs": [], "angle_changes": [], "curvatures": []}

    def state_dict(self):
        """
        Returns the state of the optimizer as a dict.
        This includes all geometric states for proper checkpoint resumption.
        """

        # Get base state dict
        state_dict = super().state_dict()

        # Add our custom statistics
        state_dict["stats"] = self.stats

        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        Properly restores all geometric states.
        """

        # Extract our custom statistics if present
        if "stats" in state_dict:
            self.stats = state_dict.pop("stats")

        # Load the rest
        super().load_state_dict(state_dict)

    def step(self, closure=None):
        """Perform a single optimization step with geometric adaptations."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Collect statistics for this step
        step_refraction_coeffs = []
        step_angle_changes = []
        step_curvatures = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                device = grad.device

                # Handle mixed precision for MPS compatibility
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                # Add weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)
                    # Previous gradient direction for geometric calculations
                    state["prev_direction"] = torch.zeros_like(
                        p.data, dtype=torch.float32
                    )
                    # Curvature estimate - critical for landscape understanding
                    state["curvature_est"] = torch.zeros_like(
                        p.data, dtype=torch.float32
                    )
                    # Initialize refraction coefficient
                    state["refraction_coeff"] = torch.ones_like(
                        p.data, dtype=torch.float32
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                prev_direction = state["prev_direction"]
                curvature_est = state["curvature_est"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Update biased first moment estimate (momentum)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate (variance)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute current gradient direction (normalized)
                grad_norm = torch.norm(grad.reshape(-1), p=2)
                if grad_norm > group["eps"]:
                    current_direction = grad / grad_norm
                else:
                    current_direction = grad

                # Calculate geometric properties
                if state["step"] > 1:
                    # Compute angle between current and previous gradient directions
                    direction_dot = (current_direction * prev_direction).sum()

                    # Clamp for numerical stability
                    direction_dot = torch.clamp(direction_dot, -1.0 + 1e-7, 1.0 - 1e-7)

                    # Compute angle change
                    abs_dot = torch.abs(direction_dot)

                    # Safe acos computation with improved MPS handling
                    try:
                        angle_change = torch.acos(abs_dot)
                    except RuntimeError as e:
                        if "mps" in str(e).lower() or device.type == "mps":
                            # Fallback for MPS
                            angle_change = torch.sqrt(2 * (1 - abs_dot))
                        else:
                            raise

                    # Update curvature estimate
                    momentum_norm = torch.norm(exp_avg.reshape(-1), p=2)
                    if momentum_norm > group["eps"]:
                        new_curvature = angle_change / momentum_norm
                    else:
                        new_curvature = angle_change

                    curvature_est.mul_(group["curvature_memory"]).add_(
                        new_curvature, alpha=1 - group["curvature_memory"]
                    )

                    # Compute refraction coefficient
                    refraction_coeff = torch.exp(
                        -angle_change * group["refraction_sensitivity"]
                    )
                    state["refraction_coeff"] = refraction_coeff

                    # Store statistics
                    step_refraction_coeffs.append(refraction_coeff.mean().item())
                    step_angle_changes.append(angle_change.item())
                    step_curvatures.append(curvature_est.mean().item())

                    # Apply geometric adaptation to momentum
                    geometric_factor = 1.0 + curvature_est * refraction_coeff
                    geometric_factor = torch.clamp(geometric_factor, min=group["eps"])
                    exp_avg = exp_avg / geometric_factor

                # Update previous direction for next iteration
                prev_direction.copy_(current_direction)

                # Compute bias-corrected estimates
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                # Apply geometric refraction to learning rate
                if state["step"] > 1 and "refraction_coeff" in state:
                    geometric_lr = group["lr"] * state["refraction_coeff"].mean().item()
                else:
                    geometric_lr = group["lr"]

                # Compute step size
                denom = corrected_exp_avg_sq.sqrt().add_(group["eps"])
                step_size = geometric_lr * corrected_exp_avg / denom

                # Update parameters
                p.data.add_(-step_size)

        # Update optimizer statistics
        if step_refraction_coeffs:
            self.stats["refraction_coeffs"].append(np.mean(step_refraction_coeffs))
            self.stats["angle_changes"].append(np.mean(step_angle_changes))
            self.stats["curvatures"].append(np.mean(step_curvatures))

        return loss


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Factory function for creating optimizers.
    This modular approach makes ablation studies much cleaner.

    Args:
        model: The model to optimize
        optimizer_name: Name of the optimizer ('geometric_adam', 'adam', 'adamw', etc.)
        lr: Learning rate
        weight_decay: Weight decay coefficient
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Initialized optimizer

    Example:
        >>> optimizer = create_optimizer(model, 'geometric_adam', lr=0.001,
        ...                             refraction_sensitivity=0.1)
    """

    # Get all parameters
    params = model.parameters()

    # Create optimizer based on name
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "geometric_adam":
        # Extract geometric-specific parameters
        refraction_sensitivity = kwargs.pop("refraction_sensitivity", 0.1)
        curvature_memory = kwargs.pop("curvature_memory", 0.95)
        betas = kwargs.pop("betas", (0.9, 0.999))
        eps = kwargs.pop("eps", 1e-8)

        return GeometricAdam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            refraction_sensitivity=refraction_sensitivity,
            curvature_memory=curvature_memory,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "adam":
        betas = kwargs.pop("betas", (0.9, 0.999))
        eps = kwargs.pop("eps", 1e-8)
        return torch.optim.Adam(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

    elif optimizer_name == "adamw":
        betas = kwargs.pop("betas", (0.9, 0.999))
        eps = kwargs.pop("eps", 1e-8)
        # AdamW has weight decay built-in
        return torch.optim.AdamW(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

    elif optimizer_name == "sgd":
        momentum = kwargs.pop("momentum", 0.9)
        nesterov = kwargs.pop("nesterov", False)
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    elif optimizer_name == "rmsprop":
        alpha = kwargs.pop("alpha", 0.99)
        eps = kwargs.pop("eps", 1e-8)
        momentum = kwargs.pop("momentum", 0)
        return torch.optim.RMSprop(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Available: geometric_adam, adam, adamw, sgd, rmsprop"
        )


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for the Transformer."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert model_dim % num_heads == 0

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        self.dropout = nn.Dropout(dropout)

        # Scale factor for attention scores
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project input to Q, K, V
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores.masked_fill_(mask == 0, float("-inf"))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.model_dim)
        )
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    """Single Transformer block with multi-head attention and feed-forward layers."""

    def __init__(
        self, model_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)

        return x


class TestTransformer(nn.Module):
    """
    Transformer model for testing optimizers.

    This compact model is designed to have approximately 2.5M~100M parameters,
    representing the practical lower bound for meaningful transformer architecture
    while maintaining essential structural integrity for optimizer comparison studies.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()

        self.model_dim = config.model_dim
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len

        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.model_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.model_dim, config.num_heads, config.ff_dim, config.dropout
                )
                for _ in range(config.num_layers)
            ]
        )

        # Output layers
        self.norm = nn.LayerNorm(config.model_dim)
        self.output_proj = nn.Linear(config.model_dim, config.vocab_size)

        # Initialize parameters
        self.apply(self._init_weights)

        # Print model size
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"Transformer initialized with {total_params:,} parameters "
            f"({trainable_params:,} trainable, target: {config.model_size})"
        )

    def _init_weights(self, module):
        """Initialize model weights using Xavier initialization."""

        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Create position indices
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Embed tokens and positions
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = self.embedding_dropout(token_embeds + position_embeds)

        # Create causal mask for autoregressive modeling
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(
            0
        )  # Add batch and head dimensions

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)

        # Final normalization and projection
        x = self.norm(x)
        logits = self.output_proj(x)

        return logits


class TestCNN(nn.Module):
    """
    CNN model for image classification tasks.

    Designed to scale from ~2.5M to ~100M parameters based on configuration.
    Uses a ResNet-inspired design with residual connections.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()

        self.num_classes = config.num_classes

        # Get layer configuration
        base_channels = config.base_channels
        layer_blocks = getattr(config, "layer_blocks", [2, 2, 2, 2])

        # Channel progression: base -> 2*base -> 4*base -> 8*base
        channels = [base_channels * (2**i) for i in range(len(layer_blocks))]

        # Initial convolution
        self.conv1 = nn.Conv2d(
            config.image_channels,
            base_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layers = nn.ModuleList()
        in_channels = base_channels

        for i, (num_blocks, out_channels) in enumerate(zip(layer_blocks, channels)):
            stride = 1 if i == 0 else 2
            layer = self._make_layer(in_channels, out_channels, num_blocks, stride)
            self.layers.append(layer)
            in_channels = out_channels

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], config.num_classes)

        # Initialize weights
        self.apply(self._init_weights)

        # Print model size
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"CNN initialized with {total_params:,} parameters "
            f"(target: {config.model_size})"
        )

    def _make_layer(
        self, in_channels: int, out_channels: int, blocks: int, stride: int = 1
    ):
        """Create a residual layer with specified number of blocks."""

        layers = []

        # First block handles stride and channel change
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self, module):
        """Initialize weights using He initialization for ReLU networks."""

        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        for layer in self.layers:
            x = layer(x)

        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    """Basic residual block for CNN."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class TestLSTM(nn.Module):
    """
    LSTM model for sequence modeling tasks.

    Designed to have 2.5M~100M parameters for fair comparison.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.bidirectional = config.bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        # LSTM layers
        self.lstm = nn.LSTM(
            config.embed_dim,
            config.hidden_dim,
            config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

        # Output projection
        output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.output_proj = nn.Linear(output_dim, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

        # Print model size
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"LSTM initialized with {total_params:,} parameters "
            f"(target: {config.model_size})"
        )

    def _init_weights(self, module):
        """Initialize LSTM weights."""

        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Embedding
        x = self.embedding(input_ids)
        x = self.embed_dropout(x)

        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)

        # Output projection
        logits = self.output_proj(lstm_out)

        return logits


class MultiArchitectureComparison:
    """Extended comparison framework supporting multiple architectures and datasets."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize logger manager
        self.logger_manager = LoggerManager(
            experiment_name=config.experiment_name,
            use_tensorboard=config.use_tensorboard,
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
            config=asdict(config),
        )
        self.logger = self.logger_manager.logger

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(exist_ok=True)

        # Results storage
        self.results = {}

        # Initialize critical phases tracker
        self._current_critical_phases = []

        self.logger.info(f"Initialized multi-architecture experiment on {self.device}")
        self.logger.info(f"Configuration: {asdict(config)}")

    def create_model(self) -> nn.Module:
        """Factory method for creating different architectures."""

        if self.config.architecture == "transformer":
            return TestTransformer(self.config)
        elif self.config.architecture == "cnn":
            return TestCNN(self.config)
        elif self.config.architecture == "lstm":
            return TestLSTM(self.config)
        else:
            raise ValueError(f"Unknown architecture: {self.config.architecture}")

    def create_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Factory method for creating appropriate datasets for each architecture."""

        if self.config.dataset_name == "wikitext2":
            return self.create_wikitext2_dataset()
        elif self.config.dataset_name == "cifar10":
            return self.create_cifar10_dataset()
        elif self.config.dataset_name == "penn_treebank":
            return self.create_penn_treebank_dataset()
        elif self.config.dataset_name == "mnist":
            return self.create_mnist_dataset()
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset_name}")

    def create_cifar10_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Create CIFAR-10 dataset for CNN experiments."""

        self.logger.info("Loading CIFAR-10 dataset...")

        # Data augmentation for training
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for MPS compatibility
            drop_last=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )

        self.logger.info(
            f"CIFAR-10 loaded: {len(train_dataset)} train, {len(test_dataset)} test samples"
        )
        return train_loader, test_loader

    def create_mnist_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Create MNIST dataset (simpler alternative for CNN)."""

        self.logger.info("Loading MNIST dataset...")

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )

        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        # Update config for MNIST
        self.config.image_channels = 1
        self.config.image_size = 28
        self.config.num_classes = 10

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )

        self.logger.info(
            f"MNIST loaded: {len(train_dataset)} train, {len(test_dataset)} test samples"
        )
        return train_loader, test_loader

    def create_penn_treebank_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Create Penn Treebank dataset for LSTM experiments."""

        self.logger.info("Loading Penn Treebank dataset...")

        # For simplicity, we'll use the same WikiText-2 loader
        # In practice, you'd implement a proper PTB loader
        return self.create_wikitext2_dataset()

    def create_wikitext2_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Create WikiText-2 dataset (reusing existing implementation)."""
        return self.create_real_dataset()

    def download_and_prepare_wikitext2(self) -> Tuple[List[str], List[str]]:
        """
        Download and prepare WikiText-2 dataset.

        WikiText-2 is a collection of over 100 Wikipedia articles and is commonly
        used as a benchmark for language modeling. It provides realistic text with
        complex linguistic patterns that create challenging optimization landscapes.

        Returns:
            Tuple of (train_texts, valid_texts) where each is a list of text strings
        """

        try:
            from datasets import load_dataset
        except ImportError:
            self.logger.info("Installing datasets library...")
            import subprocess
            import sys

            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
            from datasets import load_dataset

        self.logger.info("Downloading WikiText-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        # Extract text from train and validation splits
        train_texts = [
            item["text"] for item in dataset["train"] if item["text"].strip()
        ]
        valid_texts = [
            item["text"] for item in dataset["validation"] if item["text"].strip()
        ]

        self.logger.info(
            f"Loaded {len(train_texts)} training texts and {len(valid_texts)} validation texts"
        )
        return train_texts, valid_texts

    def build_vocabulary(
        self, texts: List[str], vocab_size: int = 10000
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary from text data using simple word-level tokenization."""

        self.logger.info("Building vocabulary...")

        # Count word frequencies
        word_counts = {}
        for text in texts:
            words = text.lower().replace("\n", " ").split()
            for word in words:
                word = "".join(c for c in word if c.isalnum() or c in ".,!?'-")
                if word:
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency and take top vocab_size-4 words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocab_words = [word for word, count in sorted_words[: vocab_size - 4]]

        # Create vocabulary dictionaries with special tokens
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        all_words = special_tokens + vocab_words

        word_to_id = {word: idx for idx, word in enumerate(all_words)}
        id_to_word = {idx: word for word, idx in word_to_id.items()}

        self.logger.info(f"Built vocabulary with {len(word_to_id)} words")
        self.logger.info(f"Most common words: {vocab_words[:10]}")

        return word_to_id, id_to_word

    def tokenize_texts(
        self, texts: List[str], word_to_id: Dict[str, int], max_length: int
    ) -> List[torch.Tensor]:
        """Convert text strings to sequences of token IDs."""

        self.logger.info("Tokenizing texts...")

        sequences = []
        unk_id = word_to_id["<unk>"]
        bos_id = word_to_id["<bos>"]
        eos_id = word_to_id["<eos>"]
        pad_id = word_to_id["<pad>"]

        for text in texts:
            # Tokenize the text
            words = text.lower().replace("\n", " ").split()
            cleaned_words = []
            for word in words:
                word = "".join(c for c in word if c.isalnum() or c in ".,!?'-")
                if word:
                    cleaned_words.append(word)

            if not cleaned_words:
                continue

            # Convert words to IDs
            token_ids = [bos_id]
            for word in cleaned_words:
                token_id = word_to_id.get(word, unk_id)
                token_ids.append(token_id)
            token_ids.append(eos_id)

            # Handle sequence length
            if len(token_ids) > max_length:
                for i in range(0, len(token_ids) - max_length + 1, max_length // 2):
                    subseq = token_ids[i : i + max_length]
                    if len(subseq) == max_length:
                        sequences.append(torch.tensor(subseq, dtype=torch.long))
            elif len(token_ids) >= 10:
                padded = token_ids + [pad_id] * (max_length - len(token_ids))
                sequences.append(torch.tensor(padded, dtype=torch.long))

        self.logger.info(f"Created {len(sequences)} tokenized sequences")
        return sequences

    def create_real_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Create DataLoaders for WikiText-2 dataset."""

        # Download and prepare the dataset
        train_texts, valid_texts = self.download_and_prepare_wikitext2()

        # Build vocabulary from training texts
        word_to_id, id_to_word = self.build_vocabulary(
            train_texts, self.config.vocab_size
        )

        # Store vocabulary for later use
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

        # Tokenize training and validation texts
        train_sequences = self.tokenize_texts(
            train_texts, word_to_id, self.config.max_seq_len
        )
        valid_sequences = self.tokenize_texts(
            valid_texts, word_to_id, self.config.max_seq_len
        )

        # Create input-target pairs for next token prediction
        def create_input_target_pairs(sequences):
            inputs, targets = [], []
            for seq in sequences:
                if len(seq) > 1:
                    inputs.append(seq[:-1])
                    targets.append(seq[1:])
            return inputs, targets

        train_inputs, train_targets = create_input_target_pairs(train_sequences)
        valid_inputs, valid_targets = create_input_target_pairs(valid_sequences)

        # Create datasets and dataloaders
        train_dataset = TensorDataset(
            torch.stack(train_inputs), torch.stack(train_targets)
        )
        valid_dataset = TensorDataset(
            torch.stack(valid_inputs), torch.stack(valid_targets)
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,  # Set to 0 for MPS compatibility
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
        )

        self.logger.info(
            f"Created train dataloader with {len(train_dataloader)} batches"
        )
        self.logger.info(
            f"Created validation dataloader with {len(valid_dataloader)} batches"
        )

        return train_dataloader, valid_dataloader

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, num_training_steps: int
    ):
        """Create learning rate scheduler with warmup."""

        def lr_lambda(current_step: int):
            if current_step < self.config.warmup_steps:
                # Ensure minimum learning rate during warmup
                return max(
                    0.01, float(current_step) / float(max(1, self.config.warmup_steps))
                )
            return max(
                0.01,  # Minimum learning rate multiplier
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - self.config.warmup_steps)),
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def get_loss_function(self):
        """Get appropriate loss function for the task."""

        if self.config.task_type == "classification":
            if self.config.architecture in ["transformer", "lstm"]:
                # For language modeling, ignore padding tokens
                return nn.CrossEntropyLoss(ignore_index=0)
            else:
                # For image classification
                return nn.CrossEntropyLoss()
        elif self.config.task_type == "regression":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown task type: {self.config.task_type}")

    def train_model(self, optimizer_type: str) -> Dict:
        """Train a model with the specified optimizer and return training metrics."""

        self.logger.info(
            f"\nTraining {self.config.architecture} with {optimizer_type} optimizer on {self.device}..."
        )

        # Create fresh model
        model = self.create_model().to(self.device)

        # Create optimizer using factory function
        optimizer = create_optimizer(
            model,
            optimizer_type,
            lr=self.config.learning_rate,
            refraction_sensitivity=self.config.refraction_sensitivity,
            curvature_memory=self.config.curvature_memory,
        )

        # Get loss function
        criterion = self.get_loss_function()

        # Create dataset
        train_dataloader, valid_dataloader = self.create_dataset()

        # Create learning rate scheduler
        num_training_steps = len(train_dataloader) * self.config.num_epochs
        scheduler = self.create_scheduler(optimizer, num_training_steps)

        # Initialize tracker with logger
        tracker = ExperimentTracker(self.logger_manager)

        # Training loop
        model.train()
        total_steps = 0
        best_valid_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(train_dataloader):
                # Handle different batch formats for different architectures
                if self.config.architecture in ["transformer", "lstm"]:
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                else:  # CNN
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                step_start = time.time()

                # Forward pass
                optimizer.zero_grad()

                if self.config.architecture in ["transformer", "lstm"]:
                    logits = model(inputs)
                    # Reshape for loss calculation
                    batch_size, seq_len, vocab_size = logits.shape
                    logits = logits.reshape(-1, vocab_size)
                    targets = targets.reshape(-1)
                else:  # CNN
                    logits = model(inputs)

                loss = criterion(logits, targets)

                # Backward pass
                loss.backward()

                # Calculate gradient norm
                total_grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** (1.0 / 2)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=self.config.gradient_clip
                )

                # Optimizer step
                optimizer.step()
                scheduler.step()

                step_time = time.time() - step_start

                # Calculate accuracy
                if self.config.architecture in ["transformer", "lstm"]:
                    mask = targets != 0
                    if mask.sum() > 0:
                        predictions = torch.argmax(logits, dim=-1)
                        accuracy = (predictions == targets)[mask].float().mean().item()
                    else:
                        accuracy = 0.0
                    # Calculate perplexity
                    perplexity = math.exp(min(loss.item(), 10))
                else:  # CNN
                    predictions = torch.argmax(logits, dim=-1)
                    accuracy = (predictions == targets).float().mean().item()
                    perplexity = None

                # Log step metrics
                step_metrics = {
                    "loss": loss.item(),
                    "accuracy": accuracy,
                    "grad_norm": total_grad_norm,
                    "lr": scheduler.get_last_lr()[0],
                    "step_time": step_time,
                }

                if perplexity is not None:
                    step_metrics["perplexity"] = perplexity

                # Add optimizer-specific metrics if available
                if hasattr(optimizer, "stats") and optimizer.stats["refraction_coeffs"]:
                    if len(optimizer.stats["refraction_coeffs"]) > 0:
                        step_metrics["refraction_coeff"] = optimizer.stats[
                            "refraction_coeffs"
                        ][-1]
                        step_metrics["angle_change"] = optimizer.stats["angle_changes"][
                            -1
                        ]
                        step_metrics["curvature"] = optimizer.stats["curvatures"][-1]

                tracker.log_step(step_metrics, total_steps)
                total_steps += 1

                # Print progress with enhanced metrics
                if batch_idx % self.config.log_interval == 0:
                    log_msg = (
                        f"Epoch {epoch+1}/{self.config.num_epochs}, "
                        f"Step {batch_idx}/{len(train_dataloader)}, "
                        f"Loss: {loss.item():.4f}, "
                        f"Acc: {accuracy:.4f}, "
                        f"LR: {scheduler.get_last_lr()[0]:.6f}"
                    )

                    if perplexity is not None:
                        log_msg += f", PPL: {perplexity:.2f}"

                    self.logger.info(log_msg)

                    # Log step-wise best if available
                    if (
                        perplexity is not None
                        and "perplexity" in tracker.step_best_info
                    ):
                        best_info = tracker.step_best_info["perplexity"]
                        self.logger.info(
                            f"  [Step-wise Best PPL: {best_info['value']:.2f} at step {best_info['step']}]"
                        )

                    # Log geometric states if available
                    if (
                        hasattr(optimizer, "stats")
                        and optimizer.stats["refraction_coeffs"]
                    ):
                        recent_refraction = np.mean(
                            optimizer.stats["refraction_coeffs"][-10:]
                        )
                        recent_angle = np.mean(optimizer.stats["angle_changes"][-10:])
                        self.logger.info(
                            f"  [Geometric: refraction={recent_refraction:.3f}, "
                            f"angle_change={recent_angle:.3f}]"
                        )

            # Validation phase
            model.eval()
            valid_losses = []
            valid_accuracies = []

            with torch.no_grad():
                for batch in valid_dataloader:
                    if self.config.architecture in ["transformer", "lstm"]:
                        val_inputs, val_targets = batch
                        val_inputs, val_targets = val_inputs.to(
                            self.device
                        ), val_targets.to(self.device)

                        val_logits = model(val_inputs)
                        val_batch_size, val_seq_len, val_vocab_size = val_logits.shape
                        val_logits = val_logits.reshape(-1, val_vocab_size)
                        val_targets = val_targets.reshape(-1)
                    else:  # CNN
                        val_inputs, val_targets = batch
                        val_inputs, val_targets = val_inputs.to(
                            self.device
                        ), val_targets.to(self.device)
                        val_logits = model(val_inputs)

                    val_loss = criterion(val_logits, val_targets)

                    # Calculate validation accuracy
                    if self.config.architecture in ["transformer", "lstm"]:
                        mask = val_targets != 0
                        if mask.sum() > 0:
                            val_predictions = torch.argmax(val_logits, dim=-1)
                            val_acc = (
                                (val_predictions == val_targets)[mask]
                                .float()
                                .mean()
                                .item()
                            )
                            valid_accuracies.append(val_acc)
                    else:  # CNN
                        val_predictions = torch.argmax(val_logits, dim=-1)
                        val_acc = (val_predictions == val_targets).float().mean().item()
                        valid_accuracies.append(val_acc)

                    valid_losses.append(val_loss.item())

            # Calculate epoch metrics
            epoch_train_loss = np.mean(
                [m[1] for m in tracker.step_metrics["loss"][-len(train_dataloader) :]]
            )
            epoch_train_acc = np.mean(
                [
                    m[1]
                    for m in tracker.step_metrics["accuracy"][-len(train_dataloader) :]
                ]
            )
            epoch_valid_loss = np.mean(valid_losses)
            epoch_valid_acc = np.mean(valid_accuracies) if valid_accuracies else 0.0
            epoch_time = time.time() - epoch_start_time

            # Log epoch metrics
            epoch_metrics = {
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_acc,
                "valid_loss": epoch_valid_loss,
                "valid_accuracy": epoch_valid_acc,
                "epoch_time": epoch_time,
            }

            if self.config.architecture in ["transformer", "lstm"]:
                epoch_metrics["train_perplexity"] = math.exp(min(epoch_train_loss, 10))
                epoch_metrics["valid_perplexity"] = math.exp(min(epoch_valid_loss, 10))

            tracker.log_epoch(epoch_metrics, epoch)

            # Update best metrics
            tracker.update_best("valid_loss", epoch_valid_loss, mode="min")
            tracker.update_best("valid_accuracy", epoch_valid_acc, mode="max")

            # Print epoch summary
            self.logger.info(f"\nEpoch {epoch+1} Summary:")
            self.logger.info(
                f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}"
            )
            self.logger.info(
                f"  Valid Loss: {epoch_valid_loss:.4f}, Valid Acc: {epoch_valid_acc:.4f}"
            )
            if "train_perplexity" in epoch_metrics:
                self.logger.info(
                    f"  Train PPL: {epoch_metrics['train_perplexity']:.2f}, Valid PPL: {epoch_metrics['valid_perplexity']:.2f}"
                )
            self.logger.info(f"  Epoch Time: {epoch_time:.2f}s")

            # Early stopping check
            if epoch_valid_loss < best_valid_loss:
                best_valid_loss = epoch_valid_loss
                patience_counter = 0

                # Save checkpoint with complete optimizer state
                if self.config.save_checkpoints:
                    checkpoint_path = (
                        Path(self.config.checkpoint_dir)
                        / f"{self.config.architecture}_{optimizer_type}_best.pt"
                    )
                    checkpoint = {
                        "epoch": epoch,
                        "architecture": self.config.architecture,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),  # This now includes geometric states
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_valid_loss": best_valid_loss,
                        "config": asdict(self.config),
                        "tracker_summary": tracker.get_summary(),  # Include all tracking info
                    }

                    torch.save(checkpoint, checkpoint_path)
                    self.logger.info(f"  Saved best checkpoint: {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    self.logger.info(
                        f"  Early stopping triggered after {epoch+1} epochs"
                    )
                    break

            # Check for divergence
            if (
                "valid_perplexity" in epoch_metrics
                and epoch_metrics["valid_perplexity"] > 1000
            ):
                self.logger.warning(f"  Model diverged with perplexity > 1000")
                break

        # Generate sample if applicable
        sample_text = None
        if self.config.architecture in ["transformer", "lstm"]:
            sample_text = self.generate_sample_text(model, max_length=50)
            self.logger.info(f"\nSample generated text: {sample_text}")

            # Log the sample text to TensorBoard/W&B
            if self.logger_manager:
                self.logger_manager.log_text(
                    f"{optimizer_type}/sample_text", sample_text, total_steps
                )

        # Perform comprehensive optimization dynamics analysis
        self.logger.info(f"\nPerforming comprehensive analysis for {optimizer_type}...")
        critical_phases, pattern_analysis = self._analyze_optimization_dynamics(
            tracker, optimizer_type
        )

        # Get final summary
        summary = tracker.get_summary()

        # Add analysis results to summary
        summary["critical_phases"] = critical_phases
        summary["pattern_analysis"] = pattern_analysis
        summary["architecture"] = self.config.architecture
        summary["dataset"] = self.config.dataset_name

        # Add optimizer-specific stats if available
        if hasattr(optimizer, "stats") and optimizer.stats["refraction_coeffs"]:
            summary["optimizer_stats"] = {
                "avg_refraction_coeff": np.mean(optimizer.stats["refraction_coeffs"]),
                "avg_angle_change": np.mean(optimizer.stats["angle_changes"]),
                "avg_curvature": np.mean(optimizer.stats["curvatures"]),
                "min_refraction_coeff": np.min(optimizer.stats["refraction_coeffs"]),
                "max_angle_change": np.max(optimizer.stats["angle_changes"]),
                "final_curvature": (
                    optimizer.stats["curvatures"][-1]
                    if optimizer.stats["curvatures"]
                    else None
                ),
            }

            # Create enhanced geometric state plots if this is Geometric Adam
            if optimizer_type == "geometric_adam":
                self._plot_geometric_states(optimizer.stats, optimizer_type)

        # Add additional info
        summary["sample_text"] = sample_text
        summary["total_steps"] = total_steps
        summary["device_type"] = self.device.type

        return summary

    def _plot_geometric_states(self, stats: Dict, optimizer_name: str):
        """Enhanced visualization of geometric adaptation over time with fixed dimension issues."""

        if not stats["refraction_coeffs"]:
            return

        # Create comprehensive figure with multiple insights
        fig = plt.figure(figsize=(15, 20))
        gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)

        steps = range(len(stats["refraction_coeffs"]))

        # 1. Refraction coefficients over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(steps, stats["refraction_coeffs"], "b-", alpha=0.7, linewidth=1)
        ax1.set_ylabel("Average Refraction Coefficient")
        ax1.set_title("Learning Rate Adaptation Through Refraction")
        ax1.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="No refraction")
        ax1.set_ylim(0, 1.1)

        # Mark critical phases if available
        if hasattr(self, "_current_critical_phases"):
            for start, end, severity in self._current_critical_phases:
                ax1.axvspan(
                    start,
                    end,
                    alpha=0.2 * severity,
                    color="red",
                    label=f"Critical phase (severity: {severity:.1f}x)",
                )

        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Angle changes with moving average - FIXED dimension mismatch
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(
            steps, stats["angle_changes"], "g-", alpha=0.3, linewidth=1, label="Raw"
        )

        # Add moving average with corrected dimensions
        window = min(100, len(stats["angle_changes"]) // 10)
        if len(stats["angle_changes"]) > window:
            moving_avg = np.convolve(
                stats["angle_changes"], np.ones(window) / window, mode="valid"
            )
            # Correct the x-axis range for moving average
            ma_steps = range(window // 2, len(stats["angle_changes"]) - window // 2 + 1)

            # Ensure lengths match exactly
            ma_steps = list(ma_steps)[: len(moving_avg)]

            ax2.plot(ma_steps, moving_avg, "g-", linewidth=2, label=f"MA({window})")

        ax2.set_ylabel("Angle Change (radians)")
        ax2.set_title("Gradient Direction Changes")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Curvature estimates with percentiles
        ax3 = fig.add_subplot(gs[1, 1])
        curvatures = stats["curvatures"]
        ax3.plot(steps, curvatures, "r-", alpha=0.7, linewidth=1)

        # Add percentile bands
        if len(curvatures) > 100:
            percentiles = np.percentile(curvatures, [25, 50, 75])
            ax3.axhline(
                y=percentiles[0], color="r", linestyle=":", alpha=0.5, label="25th %ile"
            )
            ax3.axhline(
                y=percentiles[1], color="r", linestyle="--", alpha=0.5, label="Median"
            )
            ax3.axhline(
                y=percentiles[2], color="r", linestyle=":", alpha=0.5, label="75th %ile"
            )

        ax3.set_ylabel("Estimated Curvature")
        ax3.set_title("Loss Landscape Curvature Evolution")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Phase diagram: Refraction vs Angle Change
        ax4 = fig.add_subplot(gs[2, 0])
        scatter = ax4.scatter(
            stats["angle_changes"],
            stats["refraction_coeffs"],
            c=range(len(stats["angle_changes"])),
            cmap="viridis",
            alpha=0.6,
            s=10,
        )
        ax4.set_xlabel("Angle Change (radians)")
        ax4.set_ylabel("Refraction Coefficient")
        ax4.set_title("Phase Diagram: Geometric Adaptation")

        # Add theoretical curve
        theoretical_angles = np.linspace(0, np.pi, 100)
        theoretical_refraction = np.exp(-0.1 * theoretical_angles)  # λ=0.1
        ax4.plot(
            theoretical_angles,
            theoretical_refraction,
            "k--",
            label="Theoretical (λ=0.1)",
            alpha=0.5,
        )
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label("Training Step")

        # 5. Curvature vs Refraction correlation
        ax5 = fig.add_subplot(gs[2, 1])
        if len(stats["curvatures"]) == len(stats["refraction_coeffs"]):
            ax5.scatter(
                stats["curvatures"],
                stats["refraction_coeffs"],
                alpha=0.5,
                s=10,
                c=range(len(stats["curvatures"])),
                cmap="viridis",
            )
            ax5.set_xlabel("Curvature Estimate")
            ax5.set_ylabel("Refraction Coefficient")
            ax5.set_title("Curvature-Refraction Relationship")
            ax5.grid(True, alpha=0.3)

        # 6. Histogram of angle changes
        ax6 = fig.add_subplot(gs[3, 0])
        ax6.hist(
            stats["angle_changes"], bins=50, alpha=0.7, color="green", edgecolor="black"
        )
        ax6.set_xlabel("Angle Change (radians)")
        ax6.set_ylabel("Frequency")
        ax6.set_title("Distribution of Gradient Direction Changes")
        ax6.axvline(
            x=np.mean(stats["angle_changes"]),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(stats['angle_changes']):.3f}",
        )
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis="y")

        # 7. Effective learning rate over time
        ax7 = fig.add_subplot(gs[3, 1])
        if hasattr(self.config, "learning_rate"):
            effective_lr = [
                self.config.learning_rate * r for r in stats["refraction_coeffs"]
            ]
            ax7.plot(steps, effective_lr, "purple", linewidth=1)
            ax7.set_ylabel("Effective Learning Rate")
            ax7.set_xlabel("Training Steps")
            ax7.set_title("Adaptive Learning Rate Through Geometric Refraction")
            ax7.axhline(
                y=self.config.learning_rate,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Base LR",
            )
            ax7.legend()
            ax7.grid(True, alpha=0.3)

        # 8. Rolling statistics
        ax8 = fig.add_subplot(gs[4, :])

        # Calculate rolling statistics
        window = min(1000, len(stats["refraction_coeffs"]) // 5)
        if len(stats["refraction_coeffs"]) > window:
            rolling_mean = pd.Series(stats["refraction_coeffs"]).rolling(window).mean()
            rolling_std = pd.Series(stats["refraction_coeffs"]).rolling(window).std()

            ax8.plot(steps, rolling_mean, "b-", label=f"Rolling Mean (window={window})")
            ax8.fill_between(
                steps,
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.3,
                color="blue",
                label="±1 STD",
            )
            ax8.set_ylabel("Refraction Coefficient")
            ax8.set_xlabel("Training Steps")
            ax8.set_title("Refraction Coefficient Stability Analysis")
            ax8.legend()
            ax8.grid(True, alpha=0.3)

        plt.suptitle(
            f"{optimizer_name} - Comprehensive Geometric State Analysis ({self.config.architecture})",
            fontsize=16,
        )
        plt.tight_layout()

        # Save the plot
        plot_path = (
            Path(self.config.checkpoint_dir)
            / f"{self.config.architecture}_{optimizer_name}_geometric_analysis.png"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Log to TensorBoard/W&B if available
        if self.logger_manager:
            # Read the saved image for logging
            try:
                import PIL.Image

                img = PIL.Image.open(plot_path)
                img_array = np.array(img)
                self.logger_manager.log_image(
                    f"{optimizer_name}/geometric_analysis", img_array
                )
            except Exception as e:
                self.logger.warning(f"Failed to log geometric analysis image: {e}")

        self.logger.info(f"Saved comprehensive geometric analysis to {plot_path}")

    def _analyze_optimization_dynamics(
        self, tracker: ExperimentTracker, optimizer_name: str
    ):
        """
        Perform comprehensive analysis of optimization dynamics.

        This includes:
        - Critical phase detection and visualization
        - Step-wise best pattern analysis
        - Convergence behavior characterization
        """

        # Detect critical phases
        critical_phases = tracker.detect_critical_phases(window_size=50)
        self._current_critical_phases = critical_phases  # Store for visualization

        # Analyze step-wise patterns
        pattern_analysis = tracker.analyze_step_wise_patterns()

        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # 1. Loss trajectory with critical phases
        ax1 = axes[0]
        if "loss" in tracker.step_metrics:
            losses = [(step, loss) for step, loss in tracker.step_metrics["loss"]]
            steps, loss_values = zip(*losses)
            ax1.plot(steps, loss_values, "b-", alpha=0.7, linewidth=1)

            # Highlight critical phases
            for start, end, severity in critical_phases:
                ax1.axvspan(
                    start,
                    end,
                    alpha=0.2 * min(severity, 2),
                    color="red",
                    label=f"Critical phase",
                )

            # Mark step-wise bests
            if "loss" in tracker.step_best_info:
                best_step = tracker.step_best_info["loss"]["step"]
                best_value = tracker.step_best_info["loss"]["value"]
                ax1.scatter(
                    [best_step],
                    [best_value],
                    color="gold",
                    s=100,
                    marker="*",
                    label="Best loss",
                    zorder=5,
                )

            ax1.set_ylabel("Loss")
            ax1.set_title(
                f"{self.config.architecture} - {optimizer_name} - Loss Trajectory with Critical Phases"
            )
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Gradient norm evolution
        ax2 = axes[1]
        if "grad_norm" in tracker.step_metrics:
            grad_norms = [
                (step, norm) for step, norm in tracker.step_metrics["grad_norm"]
            ]
            steps, norm_values = zip(*grad_norms)
            ax2.semilogy(steps, norm_values, "g-", alpha=0.7, linewidth=1)

            # Add variance bands
            window = 100
            if len(norm_values) > window:
                rolling_mean = pd.Series(norm_values).rolling(window).mean()
                rolling_std = pd.Series(norm_values).rolling(window).std()
                ax2.fill_between(
                    steps,
                    np.maximum(rolling_mean - rolling_std, 1e-8),
                    rolling_mean + rolling_std,
                    alpha=0.3,
                    color="green",
                )

            ax2.set_ylabel("Gradient Norm (log scale)")
            ax2.set_title("Gradient Norm Evolution")
            ax2.grid(True, alpha=0.3)

        # 3. Step-wise improvement frequency
        ax3 = axes[2]
        if pattern_analysis["best_achievement_rate"]:
            metrics = list(pattern_analysis["best_achievement_rate"].keys())
            improvements = [
                pattern_analysis["best_achievement_rate"][m]["total_improvements"]
                for m in metrics
            ]

            bars = ax3.bar(metrics, improvements)
            ax3.set_ylabel("Number of Improvements")
            ax3.set_title("Step-wise Best Achievements by Metric")

            # Add value labels
            for bar, value in zip(bars, improvements):
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value}",
                    ha="center",
                    va="bottom",
                )

            ax3.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        # Save analysis plot
        analysis_path = (
            Path(self.config.checkpoint_dir)
            / f"{self.config.architecture}_{optimizer_name}_dynamics_analysis.png"
        )
        plt.savefig(analysis_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Log analysis results
        self.logger.info(f"\n{optimizer_name} Optimization Dynamics Analysis:")
        self.logger.info(f"  Critical phases detected: {len(critical_phases)}")
        for i, (start, end, severity) in enumerate(critical_phases):
            self.logger.info(
                f"    Phase {i+1}: steps {start}-{end}, severity {severity:.2f}x"
            )

        if pattern_analysis["best_achievement_rate"]:
            self.logger.info("  Step-wise improvement patterns:")
            for metric, stats in pattern_analysis["best_achievement_rate"].items():
                self.logger.info(
                    f"    {metric}: {stats['total_improvements']} improvements, "
                    f"avg interval {stats['avg_steps_between_improvements']:.1f} steps"
                )

        return critical_phases, pattern_analysis

    def generate_sample_text(self, model: nn.Module, max_length: int = 50) -> str:
        """Generate sample text from the trained model (for Transformer/LSTM)."""

        if not hasattr(self, "word_to_id") or not hasattr(self, "id_to_word"):
            return "No vocabulary available for generation"

        model.eval()
        with torch.no_grad():
            # Start with beginning-of-sequence token
            bos_id = self.word_to_id.get("<bos>", 0)
            current_sequence = [bos_id]

            for _ in range(max_length):
                # Convert current sequence to tensor
                input_tensor = torch.tensor([current_sequence], dtype=torch.long).to(
                    self.device
                )

                # Get model predictions
                logits = model(input_tensor)

                # Get probabilities for the last token
                probs = torch.softmax(logits[0, -1, :], dim=-1)

                # Sample from the distribution
                temperature = 0.8
                probs = probs ** (1 / temperature)
                probs = probs / probs.sum()

                next_token = torch.multinomial(probs, 1).item()

                # Stop if we hit end-of-sequence
                if next_token == self.word_to_id.get("<eos>", 1):
                    break

                current_sequence.append(next_token)

            # Convert back to text
            words = []
            for token_id in current_sequence[1:]:  # Skip the <bos> token
                word = self.id_to_word.get(token_id, "<unk>")
                if word not in ["<bos>", "<eos>", "<pad>"]:
                    words.append(word)

            return " ".join(words)

    def run_comparison(self, optimizers: List[str]) -> Dict:
        """Run comparison across multiple optimizers."""

        self.logger.info("Starting optimizer comparison experiment...")
        self.logger.info(f"Architecture: {self.config.architecture}")
        self.logger.info(f"Dataset: {self.config.dataset_name}")
        self.logger.info(f"Optimizers to test: {optimizers}")

        for optimizer_type in optimizers:
            try:
                results = self.train_model(optimizer_type)
                self.results[optimizer_type] = results

                # Log summary for this optimizer
                final_metrics = results["final_metrics"]
                best_metrics = results["best_metrics"]

                self.logger.info(f"\n{optimizer_type} completed:")
                self.logger.info(
                    f"  Final train loss: {final_metrics['train_loss']:.4f}"
                )
                self.logger.info(
                    f"  Final valid loss: {final_metrics['valid_loss']:.4f}"
                )
                self.logger.info(f"  Best valid loss: {best_metrics['valid_loss']:.4f}")
                self.logger.info(
                    f"  Best valid accuracy: {best_metrics['valid_accuracy']:.4f}"
                )

                # Log step-wise best if available
                if (
                    "step_best_info" in results
                    and "perplexity" in results["step_best_info"]
                ):
                    best_ppl_info = results["step_best_info"]["perplexity"]
                    self.logger.info(
                        f"  Step-wise best PPL: {best_ppl_info['value']:.2f} at step {best_ppl_info['step']}"
                    )

            except Exception as e:
                self.logger.error(f"Error training with {optimizer_type}: {e}")
                import traceback

                self.logger.error(traceback.format_exc())
                continue

        return self.results

    def plot_results(self, save_path: Optional[str] = None):
        """
        Create comprehensive plots comparing optimizer performance.

        Enhanced with robust image logging that prevents TensorBoard format errors
        by using multiple fallback strategies for reliable visualization logging.
        """

        if not self.results:
            self.logger.warning("No results to plot. Run comparison first.")
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Training Loss
        ax1 = fig.add_subplot(gs[0, 0])
        for optimizer_name, results in self.results.items():
            losses = results["epoch_metrics"]["train_loss"]
            epochs = range(1, len(losses) + 1)
            ax1.plot(epochs, losses, label=optimizer_name, linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss")
        ax1.set_title(f"Training Loss Over Time ({self.config.architecture})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Validation Loss
        ax2 = fig.add_subplot(gs[0, 1])
        for optimizer_name, results in self.results.items():
            losses = results["epoch_metrics"]["valid_loss"]
            epochs = range(1, len(losses) + 1)
            ax2.plot(epochs, losses, label=optimizer_name, linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation Loss")
        ax2.set_title(f"Validation Loss Over Time ({self.config.architecture})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Validation Perplexity (if available)
        ax3 = fig.add_subplot(gs[0, 2])
        has_perplexity = False
        for optimizer_name, results in self.results.items():
            if "valid_perplexity" in results["epoch_metrics"]:
                has_perplexity = True
                perplexities = results["epoch_metrics"]["valid_perplexity"]
                epochs = range(1, len(perplexities) + 1)
                ax3.plot(epochs, perplexities, label=optimizer_name, linewidth=2)

        if has_perplexity:
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Validation Perplexity")
            ax3.set_title("Validation Perplexity Over Time")
            ax3.set_yscale("log")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(
                0.5,
                0.5,
                "N/A for this architecture",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
            ax3.set_title("Validation Perplexity")

        # 4. Training Accuracy
        ax4 = fig.add_subplot(gs[1, 0])
        for optimizer_name, results in self.results.items():
            accuracies = results["epoch_metrics"]["train_accuracy"]
            epochs = range(1, len(accuracies) + 1)
            ax4.plot(epochs, accuracies, label=optimizer_name, linewidth=2)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Training Accuracy")
        ax4.set_title("Training Accuracy Over Time")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Validation Accuracy
        ax5 = fig.add_subplot(gs[1, 1])
        for optimizer_name, results in self.results.items():
            accuracies = results["epoch_metrics"]["valid_accuracy"]
            epochs = range(1, len(accuracies) + 1)
            ax5.plot(epochs, accuracies, label=optimizer_name, linewidth=2)
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Validation Accuracy")
        ax5.set_title("Validation Accuracy Over Time")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Epoch Time
        ax6 = fig.add_subplot(gs[1, 2])
        for optimizer_name, results in self.results.items():
            times = results["epoch_metrics"]["epoch_time"]
            epochs = range(1, len(times) + 1)
            ax6.plot(epochs, times, label=optimizer_name, linewidth=2)
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("Time (seconds)")
        ax6.set_title("Training Time per Epoch")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Final Performance Comparison
        ax7 = fig.add_subplot(gs[2, :2])
        optimizers = list(self.results.keys())

        # Prepare data for bar plot
        metrics_to_compare = ["valid_loss", "valid_accuracy"]
        if has_perplexity:
            metrics_to_compare.append("valid_perplexity")

        metric_values = {metric: [] for metric in metrics_to_compare}

        for optimizer in optimizers:
            final_metrics = self.results[optimizer]["final_metrics"]
            for metric in metrics_to_compare:
                metric_values[metric].append(final_metrics.get(metric, 0))

        x = np.arange(len(optimizers))
        width = 0.8 / len(metrics_to_compare)

        # Create bars
        for i, (metric, values) in enumerate(metric_values.items()):
            offset = (i - len(metrics_to_compare) / 2 + 0.5) * width
            bars = ax7.bar(
                x + offset, values, width, label=metric.replace("_", " ").title()
            )

            # Add value labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax7.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax7.set_xlabel("Optimizer")
        ax7.set_ylabel("Metric Value")
        ax7.set_title("Final Performance Comparison")
        ax7.set_xticks(x)
        ax7.set_xticklabels(optimizers)
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis="y")

        # 8. Best Performance Summary
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis("off")

        # Create summary text
        summary_text = f"Best Performance Summary\n"
        summary_text += f"Architecture: {self.config.architecture}\n"
        summary_text += f"Dataset: {self.config.dataset_name}\n"
        summary_text += "=" * 30 + "\n\n"

        # Find best performers
        best_loss_opt = min(
            self.results.items(), key=lambda x: x[1]["best_metrics"]["valid_loss"]
        )
        best_acc_opt = max(
            self.results.items(), key=lambda x: x[1]["best_metrics"]["valid_accuracy"]
        )

        summary_text += f"Best Valid Loss:\n{best_loss_opt[0]}: {best_loss_opt[1]['best_metrics']['valid_loss']:.4f}\n\n"
        summary_text += f"Best Valid Accuracy:\n{best_acc_opt[0]}: {best_acc_opt[1]['best_metrics']['valid_accuracy']:.4f}\n\n"

        # Add optimizer-specific stats for geometric adam if available
        if "geometric_adam" in self.results:
            geo_results = self.results["geometric_adam"]
            if "optimizer_stats" in geo_results:
                stats = geo_results["optimizer_stats"]
                summary_text += f"\nGeometric Adam Stats:\n"
                summary_text += f"Avg Refraction: {stats['avg_refraction_coeff']:.4f}\n"
                summary_text += f"Avg Angle Change: {stats['avg_angle_change']:.4f}\n"

        ax8.text(
            0.1,
            0.9,
            summary_text,
            transform=ax8.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.suptitle(
            f"Optimizer Comparison Results - {self.config.architecture} on {self.config.dataset_name}",
            fontsize=16,
        )
        plt.tight_layout()

        # Save the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Plot saved to {save_path}")

        plt.show()

        # Enhanced multi-strategy image logging to prevent errors
        if self.logger_manager:
            logged_successfully = False

            # Strategy 1: Direct file reading with PIL (most reliable for saved plots)
            if not logged_successfully and save_path and os.path.exists(save_path):
                try:
                    import PIL.Image

                    # Read using PIL which handles matplotlib outputs well
                    pil_image = PIL.Image.open(save_path)
                    img_array = np.array(pil_image)

                    # Handle different image formats properly
                    if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
                        img_array = img_array[:, :, :3]  # Remove alpha channel

                    self.logger_manager.log_image("comparison/final_results", img_array)
                    logged_successfully = True
                    self.logger.info("Successfully logged image using PIL from file")

                except ImportError:
                    self.logger.warning("PIL not available for image logging")
                except Exception as e:
                    self.logger.warning(f"Strategy 1 (PIL from file) failed: {e}")

            # Strategy 2: Use buffer_rgba() instead of tostring_rgb() for macOS compatibility
            if not logged_successfully:
                try:
                    fig.canvas.draw()

                    # Use buffer_rgba() which is more widely supported
                    buf = fig.canvas.buffer_rgba()
                    img_array = np.asarray(buf)

                    # RGBA format, need to remove alpha and flip
                    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                        img_array = img_array[:, :, :3]  # Remove alpha

                    self.logger_manager.log_image("comparison/final_results", img_array)
                    logged_successfully = True
                    self.logger.info(
                        "Successfully logged image using canvas buffer_rgba"
                    )

                except Exception as e:
                    self.logger.warning(f"Strategy 2 (buffer_rgba) failed: {e}")

            # Report final status
            if not logged_successfully:
                self.logger.warning(
                    "Image logging failed but plot was saved to file successfully."
                )

            # Clean up matplotlib figure to prevent memory leaks
            plt.close("all")

    def save_results(self, filepath: str):
        """Save results to JSON file for later analysis."""

        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for optimizer, results in self.results.items():
            json_results[optimizer] = self._convert_to_json_serializable(results)

        # Add experiment configuration
        json_results["config"] = asdict(self.config)
        json_results["timestamp"] = datetime.now().isoformat()

        with open(filepath, "w") as f:
            json.dump(json_results, f, indent=2)

        self.logger.info(f"Results saved to {filepath}")

    def _convert_to_json_serializable(self, obj):
        """Recursively convert numpy arrays and other non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: self._convert_to_json_serializable(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj


def run_architecture_comparison(model_size: str = "2.5M"):
    """
    Run comparison across multiple architectures.

    Args:
        model_size: Model size to use for all architectures (default: "2.5M")
    """

    # CRITICAL: Set reproducibility once at the program start
    ensure_reproducibility(seed=42)

    # Define architectures and their appropriate datasets
    architecture_configs = [
        {
            "architecture": "transformer",
            "dataset_name": "wikitext2",
            "batch_size": 16,
            "model_size": model_size,
        },
        {
            "architecture": "cnn",
            "dataset_name": "cifar10",
            "batch_size": 32,
            "model_size": model_size,
        },
        {
            "architecture": "lstm",
            "dataset_name": "penn_treebank",
            "batch_size": 20,
            "model_size": model_size,
        },
    ]

    all_results = {}
    saved_files = []

    for arch_config in architecture_configs:
        print("\n" + "=" * 80)
        print(
            f"Testing {arch_config['architecture'].upper()} architecture ({arch_config['model_size']})"
        )
        print("=" * 80)

        # Create configuration for this architecture
        config = ExperimentConfig(
            architecture=arch_config["architecture"],
            dataset_name=arch_config["dataset_name"],
            batch_size=arch_config["batch_size"],
            model_size=arch_config["model_size"],
            num_epochs=50,
            learning_rate=1e-3,
            warmup_steps=1000,
            gradient_clip=1.0,
            refraction_sensitivity=0.1,
            curvature_memory=0.95,
            patience=5,
            log_interval=50,
            save_checkpoints=True,
            use_tensorboard=True,
            use_wandb=False,
            experiment_name=f"geometric_adam_{arch_config['architecture']}_{arch_config['model_size']}",
        )

        # Initialize comparison framework
        comparison = MultiArchitectureComparison(config)

        # Run comparison
        optimizers_to_test = ["adam", "geometric_adam"]
        results = comparison.run_comparison(optimizers_to_test)

        # Store results
        all_results[arch_config["architecture"]] = results

        # Create architecture-specific plots
        plot_file = (
            f"{arch_config['architecture']}_comparison_{arch_config['model_size']}.png"
        )
        results_file = (
            f"{arch_config['architecture']}_results_{arch_config['model_size']}.json"
        )

        comparison.plot_results(plot_file)
        comparison.save_results(results_file)

        saved_files.append((arch_config["architecture"], plot_file, results_file))

        # Clean up
        comparison.logger_manager.close()

    # Create cross-architecture comparison plot
    create_cross_architecture_plot(all_results)

    # Print saved files
    print("\n" + "=" * 60)
    print("SAVED FILES")
    print("=" * 60)
    for arch, plot, results in saved_files:
        print(f"- {plot} / {results}")
    print("- cross_architecture_comparison.png")

    return all_results


def run_model_size_scaling_experiment(
    architecture: str = "transformer",
    dataset: str | None = None,
    model_sizes: list[str] | None = None,
):
    """
    Run scaling experiment for a single architecture across different model sizes.

    Args:
        architecture: Architecture to test ("transformer", "cnn", "lstm")
        dataset: Dataset to use (if None, uses default for architecture)
        model_sizes: List of model sizes to test (default: all sizes)
    """

    ensure_reproducibility(seed=42)

    # Default model sizes if not specified
    if model_sizes is None:
        model_sizes = ["2.5M", "5M", "10M", "30M", "50M", "100M"]

    # Default datasets
    default_datasets = {
        "transformer": "wikitext2",
        "cnn": "cifar10",
        "lstm": "penn_treebank",
    }

    if dataset is None:
        dataset = default_datasets.get(architecture, "wikitext2")

    all_results = {}

    for model_size in model_sizes:
        print("\n" + "=" * 80)
        print(f"Testing {architecture.upper()} with {model_size} parameters")
        print("=" * 80)

        # Adjust batch size for larger models
        base_batch_size = 32 if architecture == "cnn" else 16
        if model_size in ["50M", "100M"]:
            batch_size = base_batch_size // 4
        elif model_size in ["30M"]:
            batch_size = base_batch_size // 2
        else:
            batch_size = base_batch_size

        # Create configuration
        config = ExperimentConfig(
            architecture=architecture,
            dataset_name=dataset,
            model_size=model_size,
            batch_size=batch_size,
            num_epochs=30,  # Fewer epochs for scaling experiment
            learning_rate=1e-3,
            warmup_steps=1000,
            gradient_clip=1.0,
            refraction_sensitivity=0.1,
            curvature_memory=0.95,
            patience=5,
            log_interval=50,
            save_checkpoints=True,
            use_tensorboard=True,
            use_wandb=False,
            experiment_name=f"scaling_{architecture}_{model_size}",
        )

        try:
            # Initialize comparison framework
            comparison = MultiArchitectureComparison(config)

            # Run comparison
            optimizers_to_test = ["adam", "geometric_adam"]
            results = comparison.run_comparison(optimizers_to_test)

            # Store results
            all_results[model_size] = results

            # Save results
            comparison.save_results(f"scaling_{architecture}_{model_size}_results.json")

            # Clean up
            comparison.logger_manager.close()

        except Exception as e:
            print(f"Error with {model_size}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Create scaling analysis plot
    create_scaling_analysis_plot(architecture, all_results)

    return all_results


def create_scaling_analysis_plot(architecture: str, results: Dict):
    """Create plots analyzing how performance scales with model size."""

    model_sizes = list(results.keys())
    model_params = {"2.5M": 2.5, "5M": 5, "10M": 10, "30M": 30, "50M": 50, "100M": 100}

    # Extract model sizes in millions
    sizes_m = [model_params.get(size, 0) for size in model_sizes]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Final loss vs model size
    ax = axes[0, 0]
    for optimizer in ["adam", "geometric_adam"]:
        losses = []
        for size in model_sizes:
            if size in results and optimizer in results[size]:
                losses.append(results[size][optimizer]["final_metrics"]["valid_loss"])
            else:
                losses.append(None)

        # Filter out None values
        valid_points = [(s, l) for s, l in zip(sizes_m, losses) if l is not None]
        if valid_points:
            sizes, losses = zip(*valid_points)
            ax.semilogx(sizes, losses, "o-", label=optimizer, markersize=8)

    ax.set_xlabel("Model Size (M parameters)")
    ax.set_ylabel("Final Validation Loss")
    ax.set_title(f"{architecture.upper()}: Loss Scaling with Model Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Relative improvement of Geometric Adam
    ax = axes[0, 1]
    improvements = []
    valid_sizes = []

    for size, size_m in zip(model_sizes, sizes_m):
        if (
            size in results
            and "adam" in results[size]
            and "geometric_adam" in results[size]
        ):
            adam_loss = results[size]["adam"]["final_metrics"]["valid_loss"]
            geo_loss = results[size]["geometric_adam"]["final_metrics"]["valid_loss"]
            improvement = (adam_loss - geo_loss) / adam_loss * 100
            improvements.append(improvement)
            valid_sizes.append(size_m)

    if improvements:
        ax.semilogx(valid_sizes, improvements, "go-", markersize=8, linewidth=2)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("Model Size (M parameters)")
        ax.set_ylabel("Improvement over Adam (%)")
        ax.set_title("Geometric Adam Relative Improvement")
        ax.grid(True, alpha=0.3)

    # Plot 3: Training time per epoch
    ax = axes[1, 0]
    for optimizer in ["adam", "geometric_adam"]:
        times = []
        for size in model_sizes:
            if size in results and optimizer in results[size]:
                epoch_times = results[size][optimizer]["epoch_metrics"]["epoch_time"]
                avg_time = np.mean(epoch_times) if epoch_times else None
                times.append(avg_time)
            else:
                times.append(None)

        valid_points = [(s, t) for s, t in zip(sizes_m, times) if t is not None]
        if valid_points:
            sizes, times = zip(*valid_points)
            ax.loglog(sizes, times, "o-", label=optimizer, markersize=8)

    ax.set_xlabel("Model Size (M parameters)")
    ax.set_ylabel("Avg Time per Epoch (seconds)")
    ax.set_title("Training Time Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Memory efficiency (if available)
    ax = axes[1, 1]
    ax.text(
        0.5,
        0.5,
        "Model Scaling Summary\n\n"
        + f"Architecture: {architecture.upper()}\n"
        + f'Sizes tested: {", ".join(model_sizes)}\n'
        + f"Best improvement: {max(improvements):.1f}% at {valid_sizes[improvements.index(max(improvements))]}M\n"
        + f"Avg improvement: {np.mean(improvements):.1f}%",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"),
    )
    ax.axis("off")

    plt.suptitle(f"{architecture.upper()} Model Size Scaling Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{architecture}_scaling_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"SCALING ANALYSIS SUMMARY - {architecture.upper()}")
    print(f"{'='*60}")

    for size in model_sizes:
        if size in results:
            print(f"\nModel Size: {size}")
            if "adam" in results[size]:
                adam_loss = results[size]["adam"]["final_metrics"]["valid_loss"]
                print(f"  Adam Loss: {adam_loss:.4f}")
            if "geometric_adam" in results[size]:
                geo_loss = results[size]["geometric_adam"]["final_metrics"][
                    "valid_loss"
                ]
                print(f"  Geometric Adam Loss: {geo_loss:.4f}")
                if "adam" in results[size]:
                    improvement = (adam_loss - geo_loss) / adam_loss * 100
                    print(f"  Improvement: {improvement:.1f}%")


def create_cross_architecture_plot(all_results: Dict):
    """Create a plot comparing results across architectures."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    architectures = list(all_results.keys())
    optimizers = ["adam", "geometric_adam"]

    # Metrics to compare
    metrics = ["valid_loss", "valid_accuracy"]

    for i, metric in enumerate(metrics):
        ax = axes[i, 0]

        # Bar plot comparing architectures
        x = np.arange(len(architectures))
        width = 0.35

        for j, optimizer in enumerate(optimizers):
            values = []
            for arch in architectures:
                if arch in all_results and optimizer in all_results[arch]:
                    values.append(all_results[arch][optimizer]["best_metrics"][metric])
                else:
                    values.append(0)

            bars = ax.bar(x + j * width, values, width, label=optimizer)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

        ax.set_xlabel("Architecture")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f'Best {metric.replace("_", " ").title()} by Architecture')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(architectures)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    # Geometric Adam advantage plot
    ax = axes[0, 1]
    advantages = []
    labels = []

    for arch in architectures:
        if (
            arch in all_results
            and "adam" in all_results[arch]
            and "geometric_adam" in all_results[arch]
        ):
            adam_loss = all_results[arch]["adam"]["best_metrics"]["valid_loss"]
            geo_loss = all_results[arch]["geometric_adam"]["best_metrics"]["valid_loss"]
            advantage = (adam_loss - geo_loss) / adam_loss * 100
            advantages.append(advantage)
            labels.append(arch)

    bars = ax.bar(labels, advantages)
    ax.set_ylabel("Improvement (%)")
    ax.set_title("Geometric Adam Loss Improvement over Adam")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, advantages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.1f}%",
            ha="center",
            va="bottom" if value > 0 else "top",
        )

    ax.grid(True, alpha=0.3, axis="y")

    # Training time comparison
    ax = axes[1, 1]

    for arch in architectures:
        if arch in all_results:
            for optimizer in optimizers:
                if optimizer in all_results[arch]:
                    epoch_times = all_results[arch][optimizer]["epoch_metrics"][
                        "epoch_time"
                    ]
                    epochs = range(1, len(epoch_times) + 1)
                    ax.plot(
                        epochs,
                        epoch_times,
                        label=f"{arch}-{optimizer}",
                        linestyle="-" if optimizer == "adam" else "--",
                    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Training Time Comparison Across Architectures")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Cross-Architecture Optimizer Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig("cross_architecture_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_experiment_summary(results: Dict, mode: str, architecture: str = None):
    """Print a summary of experiment results."""

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    if mode == "single":
        # Single architecture summary
        print(f"\nArchitecture: {architecture}")
        for optimizer, opt_results in results.items():
            print(f"\n{optimizer.upper()}:")
            final = opt_results["final_metrics"]
            best = opt_results["best_metrics"]
            print(f"  Final Valid Loss: {final['valid_loss']:.4f}")
            print(f"  Best Valid Loss: {best['valid_loss']:.4f}")
            print(f"  Best Valid Accuracy: {best['valid_accuracy']:.4f}")

            if "valid_perplexity" in final:
                print(f"  Final Valid Perplexity: {final['valid_perplexity']:.2f}")

            if "optimizer_stats" in opt_results:
                stats = opt_results["optimizer_stats"]
                print("  [Geometric Stats]")
                print(f"    Avg Refraction: {stats['avg_refraction_coeff']:.4f}")
                print(f"    Avg Angle Change: {stats['avg_angle_change']:.4f}")

        # Compare optimizers
        if "adam" in results and "geometric_adam" in results:
            adam_loss = results["adam"]["best_metrics"]["valid_loss"]
            geo_loss = results["geometric_adam"]["best_metrics"]["valid_loss"]
            improvement = (adam_loss - geo_loss) / adam_loss * 100
            print(f"\n🎯 Geometric Adam Improvement: {improvement:.2f}% over Adam")

    elif mode == "multi_arch":
        # Multi-architecture summary
        for arch, arch_results in results.items():
            print(f"\n{arch.upper()}:")

            if "adam" in arch_results and "geometric_adam" in arch_results:
                adam_best = arch_results["adam"]["best_metrics"]["valid_loss"]
                geo_best = arch_results["geometric_adam"]["best_metrics"]["valid_loss"]
                improvement = (adam_best - geo_best) / adam_best * 100

                print(f"  Adam Best Loss: {adam_best:.4f}")
                print(f"  Geometric Adam Best Loss: {geo_best:.4f}")
                print(f"  Improvement: {improvement:.2f}%")

    elif mode == "scaling":
        # Scaling experiment summary
        print(f"\nArchitecture: {architecture}")
        print("\nModel Size | Adam Loss | Geo Adam Loss | Improvement")
        print("-" * 55)

        for size, size_results in sorted(
            results.items(), key=lambda x: float(x[0].rstrip("M"))
        ):
            if "adam" in size_results and "geometric_adam" in size_results:
                adam_loss = size_results["adam"]["best_metrics"]["valid_loss"]
                geo_loss = size_results["geometric_adam"]["best_metrics"]["valid_loss"]
                improvement = (adam_loss - geo_loss) / adam_loss * 100

                print(
                    f"{size:>10} | {adam_loss:>9.4f} | {geo_loss:>13.4f} | {improvement:>10.2f}%"
                )


def main():
    """Main function to run the multi-architecture optimizer comparison experiment."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Architecture Geometric Adam Comparison"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "multi_arch", "scaling"],
        help="Experiment mode: single architecture, multi-architecture comparison, or scaling analysis",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="transformer",
        choices=["transformer", "cnn", "lstm"],
        help="Architecture to test",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="2.5M",
        choices=["2.5M", "5M", "10M", "30M", "50M", "100M"],
        help="Model size for single architecture or multi-architecture test",
    )
    parser.add_argument(
        "--model_sizes",
        type=str,
        nargs="+",
        default=["2.5M", "5M", "10M", "30M", "50M", "100M"],
        help="Model sizes for scaling experiment",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (if None, uses default for architecture)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )

    args = parser.parse_args()

    # Ensure reproducibility
    ensure_reproducibility(seed=42)

    print("=" * 80)
    print("MULTI-ARCHITECTURE GEOMETRIC ADAM OPTIMIZER COMPARISON")
    print("=" * 80)

    experiment_results = None

    if args.mode == "single":
        # Single architecture test with specified model size
        print(
            f"\nRunning single architecture test: {args.architecture} ({args.model_size})"
        )

        # Determine dataset
        dataset = args.dataset
        if dataset is None:
            dataset_map = {
                "transformer": "wikitext2",
                "cnn": "cifar10",
                "lstm": "penn_treebank",
            }
            dataset = dataset_map[args.architecture]

        config = ExperimentConfig(
            architecture=args.architecture,
            dataset_name=dataset,
            model_size=args.model_size,
            num_epochs=args.epochs,
            batch_size=16 if args.architecture != "cnn" else 32,
            learning_rate=1e-3,
            experiment_name=f"geometric_adam_{args.architecture}_{args.model_size}",
        )

        comparison = MultiArchitectureComparison(config)
        optimizers_to_test = ["geometric_adam", "adam", "adamw"]
        results = comparison.run_comparison(optimizers_to_test)
        comparison.plot_results(f"{args.architecture}_{args.model_size}_comparison.png")
        comparison.save_results(f"{args.architecture}_{args.model_size}_results.json")
        comparison.logger_manager.close()

        # Print summary
        print_experiment_summary(results, args.mode, args.architecture)
        experiment_results = results

    elif args.mode == "multi_arch":
        # Multi-architecture comparison at specified or default size
        print(f"\nRunning multi-architecture comparison at {args.model_size}")
        all_results = run_architecture_comparison(model_size=args.model_size)

        # Print summary
        print_experiment_summary(all_results, args.mode)
        experiment_results = all_results

    elif args.mode == "scaling":
        # Scaling experiment for specified architecture
        print(f"\nRunning scaling experiment for {args.architecture}")
        print(f"Model sizes to test: {args.model_sizes}")

        results = run_model_size_scaling_experiment(
            architecture=args.architecture,
            dataset=args.dataset,
            model_sizes=args.model_sizes,
        )

        # Print summary
        print_experiment_summary(results, args.mode, args.architecture)
        experiment_results = results

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)

    if args.mode == "single":
        print("\nResults saved:")
        print(f"- {args.architecture}_{args.model_size}_comparison.png")
        print(f"- {args.architecture}_{args.model_size}_results.json")
    elif args.mode == "multi_arch":
        print("\nResults saved for each architecture:")
        print(
            f"- transformer_comparison_{args.model_size}.png / transformer_results_{args.model_size}.json"
        )
        print(
            f"- cnn_comparison_{args.model_size}.png / cnn_results_{args.model_size}.json"
        )
        print(
            f"- lstm_comparison_{args.model_size}.png / lstm_results_{args.model_size}.json"
        )
        print("- cross_architecture_comparison.png")
    elif args.mode == "scaling":
        print("\nScaling analysis results saved:")
        print(f"- {args.architecture}_scaling_analysis.png")
        print(f"- scaling_{args.architecture}_*_results.json (for each model size)")

    print("\nCheck the logs directory for detailed experiment logs.")
    print("Check the checkpoints directory for saved models.")

    # Return results for potential programmatic use
    return experiment_results


if __name__ == "__main__":
    main()
