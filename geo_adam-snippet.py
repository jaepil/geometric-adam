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
