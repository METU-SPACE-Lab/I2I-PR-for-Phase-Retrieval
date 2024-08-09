from typing import Callable, Tuple

import numpy as np
import torch
from einops import rearrange, repeat
from torch.nn.functional import mse_loss
from tqdm import tqdm

from . import utils
from .initializer import get_initializer

__ALGORITHMS__ = {}


def register_algorithm(name: str):
    def wrapper(func):
        if __ALGORITHMS__.get(name, None):
            raise NameError(f"Name {name} is already registered.")
        __ALGORITHMS__[name] = func
        return func

    return wrapper


def get_algorithm(name: str) -> Callable:
    if __ALGORITHMS__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __ALGORITHMS__[name]


@register_algorithm(name="ER")
def error_reduction_algorithm(
    amplitude: torch.Tensor, support: torch.Tensor, iteration: int
):
    # initial guess
    init_fn = get_initializer("gaussian")
    random_phase = init_fn(amplitude.shape).to(amplitude.device)
    G = amplitude * torch.exp(1j * random_phase * 2 * np.pi)

    pbar = tqdm(range(iteration), miniters=100)
    for i in pbar:
        G_prime = apply_fourier_constraint(G, amplitude)
        g_prime = torch.real(utils.ifft2d(G_prime))
        g = apply_image_constraint(g_prime, support)
        G = utils.fft2d(g)

        loss = mse_loss(G.abs(), amplitude)
        pbar.set_description(f"Iteration {i+1}", refresh=False)
        pbar.set_postfix({"MSE": loss.item()}, refresh=False)

    g = torch.real(utils.ifft2d(G))
    final_loss = mse_loss(G.abs(), amplitude)
    return g, final_loss


@register_algorithm(name="HIO")
def hybrid_input_output_algorithm(
    amplitude: torch.Tensor,
    support: torch.Tensor,
    iteration: int = 1000,
    initial_value: torch.Tensor = None,
    reduce: bool = True,
    use_tqdm: bool = True
):
    # initial guess
    if initial_value is None:
        init_fn = get_initializer("gaussian")
        initial_phase = init_fn(amplitude.shape).to(amplitude.device) * 2 * np.pi
    else:
        initial_phase = utils.fft2d(initial_value).angle()

    G = torch.polar(amplitude, initial_phase)
    g = torch.real(utils.ifft2d(G))

    pbar = range(iteration)
    pbar = tqdm(pbar, miniters=100) if use_tqdm else pbar
    
    for i in pbar:
        G_prime = apply_fourier_constraint(G, amplitude)
        g_prime = torch.real(utils.ifft2d(G_prime))

        g = apply_image_constraint_hio(g_prime, g, support)
        G = utils.fft2d(g)

        loss = mse_loss(G.abs(), amplitude)
        if use_tqdm:
            pbar.set_description(f"Iteration {i+1}", refresh=False)
            pbar.set_postfix({"MSE": loss.item()}, refresh=False)

    g = torch.real(utils.ifft2d(G))
    final_loss = (
        mse_loss(G.abs(), amplitude)
        if reduce
        else mse_loss(G.abs(), amplitude, reduction="none").mean((-1, -2))
    )
    return g, final_loss



@register_algorithm(name="HIO_ER_Combined")
def hio_er_combined_algorithm(
    amplitude: torch.Tensor,
    support: torch.Tensor,
    iteration: int = 1000,
    initial_value: torch.Tensor = None,
    reduce: bool = True,
    use_tqdm: bool = True,
    hio_period: int = 200,
    er_period: int = 100,
):
    # initial guess
    if initial_value is None:
        init_fn = get_initializer("gaussian")
        initial_phase = init_fn(amplitude.shape).to(amplitude.device) * 2 * np.pi
    else:
        initial_phase = utils.fft2d(initial_value).angle()

    G = torch.polar(amplitude, initial_phase)
    g = torch.real(utils.ifft2d(G))

    pbar = range(iteration)
    pbar = tqdm(pbar, miniters=100) if use_tqdm else pbar
    
    for i in pbar:
        G_prime = apply_fourier_constraint(G, amplitude)
        g_prime = torch.real(utils.ifft2d(G_prime))

        g = apply_image_constraint_hio(g_prime, g, support) if i % (hio_period + er_period) <= hio_period else apply_image_constraint(g_prime, support)
        G = utils.fft2d(g)

        loss = mse_loss(G.abs(), amplitude)
        if use_tqdm:
            pbar.set_description(f"Iteration {i+1}", refresh=False)
            pbar.set_postfix({"MSE": loss.item()}, refresh=False)

    g = torch.real(utils.ifft2d(G))
    final_loss = (
        mse_loss(G.abs(), amplitude)
        if reduce
        else mse_loss(G.abs(), amplitude, reduction="none").mean((-1, -2))
    )
    return g, final_loss

# acceleration
# current = np.random.rand(*image.shape)
# last = None

# for index in range(iterations):
#     F_proj = apply_fourier_constraint(current, original_magnitude)
#     S_proj = apply_space_constraint(F_proj, support)
    
#     if last is None:
#         last = (F_proj + S_proj) / 2
        
#     if index % 20 == 19:
#         discrepancy = np.linalg.norm(F_proj - S_proj)
#         cur_combined = (F_proj + S_proj) / 2
#         current = cur_combined + gamma * discrepancy * (cur_combined - last)
#         last = cur_combined
#     else:
#         current = S_proj
        
# return current

@register_algorithm(name="HIO_Accelerated_ER_Combined")
def hio_accelerated_er_combined_algorithm(
    amplitude: torch.Tensor,
    support: torch.Tensor,
    iteration: int = 1777,
    initial_value: torch.Tensor = None,
    reduce: bool = True,
    use_tqdm: bool = True,
    hio_period: int = 700,
    er_period: int = 300,
    acceleration_period: int = 4,
    gamma: float = 0.6
):
    # Initial guess
    if initial_value is None:
        init_fn = get_initializer("gaussian")
        initial_phase = init_fn(amplitude.shape).to(amplitude.device) * 2 * np.pi
    else:
        initial_phase = utils.fft2d(initial_value).angle()

    G = torch.polar(amplitude, initial_phase)
    g = torch.real(utils.ifft2d(G))
    
    last = None
    pbar = range(iteration)
    pbar = tqdm(pbar, miniters=100) if use_tqdm else pbar
    
    for i in pbar:
        G_prime = apply_fourier_constraint(G, amplitude)
        g_prime = torch.real(utils.ifft2d(G_prime))

        if i % (hio_period + er_period) <= hio_period:
            g = apply_image_constraint_hio(g_prime, g, support)
            last = None
        else:
            g = apply_image_constraint(g_prime, support)

            # Acceleration only during ER steps
            if last is None:
                last = (g_prime + g) / 2
                
            if i % acceleration_period == acceleration_period-1:
                discrepancy = torch.norm(g_prime - g) # use ema in acceleration?
                g_combined = (g_prime + g) / 2
                g_combined_minus_last = g_combined - last
                g = g_combined + gamma * 0.5 * discrepancy * g_combined_minus_last / torch.norm(g_combined_minus_last)
                last = g_combined

        G = utils.fft2d(g)
        
        # Loss computation for monitoring
        loss = mse_loss(G.abs(), amplitude)
        if use_tqdm:
            pbar.set_description(f"Iteration {i+1}", refresh=False)
            pbar.set_postfix({"MSE": loss.item()}, refresh=False)

    g = torch.real(utils.ifft2d(G))
    final_loss = (
        mse_loss(G.abs(), amplitude)
        if reduce
        else mse_loss(G.abs(), amplitude, reduction="none").mean((-1, -2))
    )
    return g, final_loss



def random_best(
    amplitude: torch.Tensor,
    support: torch.Tensor,
    algorithm: str = "HIO",
    iterations: int = 50,
    different_random_count: int = 50,
    use_tqdm: bool = True,
):
    amplitude_repeated = repeat(
        amplitude, "b c h w -> (repeat b) c h w", repeat=different_random_count
    )
    support_repeated = repeat(
        support, "b c h w -> (repeat b) c h w", repeat=different_random_count
    )

    g_repeated, loss_hio_repeat = get_algorithm(algorithm)(
        amplitude_repeated, support_repeated, iterations, reduce=False, use_tqdm=use_tqdm
    )

    arranged_g_repeated = rearrange(
        g_repeated, "(repeat b) c h w -> b c repeat h w", repeat=different_random_count
    )
    arranged_loss_hio_repeat = rearrange(
        loss_hio_repeat, "(repeat b) c -> b c repeat", repeat=different_random_count
    )

    best_indices_loss_hio_repeat = torch.argmin(
        arranged_loss_hio_repeat, dim=2, keepdim=True
    )

    best_indices_loss_hio_repeat_for_gather = repeat(
        best_indices_loss_hio_repeat,
        "b c repeat -> b c repeat gather_repeat_0 gather_repeat_1",
        gather_repeat_0=arranged_g_repeated.shape[-2],
        gather_repeat_1=arranged_g_repeated.shape[-1],
    )

    return torch.gather(
        arranged_g_repeated, 2, best_indices_loss_hio_repeat_for_gather
    ).squeeze(2)


@register_algorithm(name="RobustHIO")
def robust_hio(
    amplitude: torch.Tensor,
    support: torch.Tensor,
    iterations: int = 1000,
    random_iterations: int = 50,
    different_random_count: int = 50,
    use_tqdm: bool = False,
):
    best_g_random = random_best(
        amplitude, support, "HIO", random_iterations, different_random_count, use_tqdm
    )

    return get_algorithm("HIO")(
        amplitude, support, iterations, initial_value=best_g_random, use_tqdm=use_tqdm
    )


def random_best_top_k(
    amplitude: torch.Tensor,
    support: torch.Tensor,
    algorithm: str = "HIO",
    iterations: int = 50,
    different_random_count: int = 50,
    top_k: int = 10,
    use_tqdm: bool = True,
):
    amplitude_repeated = repeat(
        amplitude, "b c h w -> (repeat b) c h w", repeat=different_random_count
    )
    support_repeated = repeat(
        support, "b c h w -> (repeat b) c h w", repeat=different_random_count
    )

    g_repeated, loss_hio_repeat = get_algorithm(algorithm)(
        amplitude_repeated, support_repeated, iterations, reduce=False, use_tqdm=use_tqdm
    )

    arranged_g_repeated = rearrange(
        g_repeated, "(repeat b) c h w -> b c repeat h w", repeat=different_random_count
    )
    arranged_loss_hio_repeat = rearrange(
        loss_hio_repeat, "(repeat b) c -> b c repeat", repeat=different_random_count
    )

    best_indices_loss_hio_repeat = torch.topk(
        arranged_loss_hio_repeat, top_k, dim=2, largest=False
    ).indices

    best_indices_loss_hio_repeat_for_gather = repeat(
        best_indices_loss_hio_repeat,
        "b c repeat -> b c repeat gather_repeat_0 gather_repeat_1",
        gather_repeat_0=arranged_g_repeated.shape[-2],
        gather_repeat_1=arranged_g_repeated.shape[-1],
    )

    result_final = torch.gather(
        arranged_g_repeated, 2, best_indices_loss_hio_repeat_for_gather
    )

    result_final = result_final.squeeze(dim=1) # now only 1 channel support
    
    return result_final
    

@register_algorithm(name="MultiOutputRobustHIO")
def multi_output_robust_hio(
    amplitude: torch.Tensor,
    support: torch.Tensor,
    iterations: int = 1000,
    random_iterations: int = 50,
    different_random_count: int = 50,
    top_k: int = 10,
    use_tqdm: bool = False,
    algorithm_str: str = "HIO",
):
    best_g_random = random_best_top_k(
        amplitude, support, "HIO", random_iterations, different_random_count, top_k, use_tqdm
    )
    
    amplitude_repeated = repeat(
        amplitude, "b c h w -> b (repeat c) h w", repeat=top_k
    )
    support_repeated = repeat(
        support, "b c h w -> b (repeat c) h w", repeat=top_k
    )

    return get_algorithm(algorithm_str)(
        amplitude_repeated, support_repeated, iterations, initial_value=best_g_random, use_tqdm=use_tqdm
    )


@register_algorithm(name="OSS")
def oversampling_smoothness_algorithm(
    amplitude: torch.Tensor, support: torch.Tensor, iteration: int
):
    """https://arxiv.org/ftp/arxiv/papers/1211/1211.4519.pdf"""

    # prepare alpha for gaussian filter (following the paper)
    n_filters = int(iteration / 200)
    x_alpha = torch.linspace(
        2 * amplitude.size(-2), 0.2 * amplitude.size(-2), n_filters
    )
    y_alpha = torch.linspace(
        2 * amplitude.size(-1), 0.2 * amplitude.size(-1), n_filters
    )

    # initial guess
    init_fn = get_initializer("gaussian")
    random_phase = init_fn(amplitude.shape).to(amplitude.device)
    G = amplitude * torch.exp(1j * random_phase * 2 * np.pi)
    g = torch.real(utils.ifft2d(G))

    # initial loss for choosing best recon.
    loss = 1e9

    for i in range(n_filters):
        best = {"recon": G, "loss": loss}
        pbar = tqdm(range(int(iteration / n_filters)), miniters=100)
        for _ in pbar:
            G_prime = apply_fourier_constraint(G, amplitude)
            g_prime = torch.real(utils.ifft2d(G_prime))
            gaussian_filter = generate_gaussian_filter(
                amplitude, x_alpha[i], y_alpha[i]
            )
            g = apply_image_constraint_oss(g_prime, g, support, gaussian_filter)
            G = utils.fft2d(g)

            loss = mse_loss(G.abs(), amplitude)
            if torch.isnan(loss):
                loss = torch.tensor(np.inf)
            if best.get("loss") > loss:
                best.update({"recon": G, "loss": loss})

            pbar.set_description(f"Iteration {i+1}", refresh=False)
            pbar.set_postfix({"MSE": loss.item()})

        G = best.get("recon")

    g = torch.real(utils.ifft2d(G))
    final_loss = mse_loss(G.abs(), amplitude)
    return g, final_loss


@register_algorithm(name="WF")
def wirtinger_flow_algorithm(
    amplitude: torch.Tensor, support: torch.Tensor, iteration: int
):
    # initialize step size schedule
    tau0 = 330
    mu_max = 0.4
    mu = [min(1 - np.exp(-t / tau0), mu_max) for t in range(1, iteration + 1)]

    # initial guess
    init_fn = get_initializer("spectral")
    z = init_fn(amplitude, power_iteration=500)

    pbar = tqdm(range(iteration), miniters=100)
    for i in pbar:
        estimate = utils.fft2d(z)
        temp = (estimate.abs() ** 2 - amplitude**2) * estimate
        grad = utils.ifft2d(temp) / torch.numel(temp)

        z = z - mu[i] / (amplitude**2).mean() * grad

        loss = mse_loss(utils.fft2d(z).abs(), amplitude)
        pbar.set_description(f"Iteration {i+1}", refresh=False)
        pbar.set_postfix({"MSE": loss.item()}, refresh=False)

    final_loss = mse_loss(utils.fft2d(z).abs(), amplitude)
    return z.abs(), final_loss


# =================
# Helper functions
# =================


def generate_gaussian_filter(amplitude: torch.Tensor, x_alpha: float, y_alpha: float):
    x = torch.arange(
        -round((amplitude.size(-2) - 1) / 2),
        round((amplitude.size(-2) - 1) / 2),
        amplitude.size(-2),
    )
    y = torch.arange(
        -round((amplitude.size(-1) - 1) / 2),
        round((amplitude.size(-1) - 1) / 2),
        amplitude.size(-1),
    )
    X, Y = torch.meshgrid(x, y)

    gaussian_filter = torch.exp(-0.5 * ((X / x_alpha) ** 2)) * torch.exp(
        -0.5 * ((Y / y_alpha) ** 2)
    )

    # normalize and repeat filter
    gaussian_filter /= gaussian_filter.max()
    gaussian_filter = gaussian_filter.expand(amplitude.shape)
    gaussian_filter = gaussian_filter.to(amplitude.device)

    return gaussian_filter


def generate_random_phase(
    amplitude: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    random_uniform = torch.rand(amplitude.shape).to(support.device)
    random_phase = random_uniform * support
    return random_phase


def apply_image_constraint(obj, support):
    support = support * generate_non_negative_support(obj)
    obj = obj * support
    return obj


def apply_image_constraint_hio(obj, prev_obj, support, beta=0.9):
    support = support * generate_non_negative_support(obj)
    in_support = obj * support
    out_support = (prev_obj - beta * obj) * (1 - support)
    return in_support + out_support


def apply_image_constraint_oss(obj, prev_obj, support, gaussian_filter):
    new_obj = apply_image_constraint_hio(obj, prev_obj, support)

    # for oss conditioning, don't use non-negative constraint.
    in_support = new_obj * support
    out_support = torch.real(utils.ifft2d(utils.fft2d(new_obj) * gaussian_filter)) * (
        1 - support
    )
    return in_support + out_support


def generate_non_negative_support(obj: torch.Tensor) -> torch.Tensor:
    nn_support = torch.ones_like(obj)
    nn_support[obj < 0] = 0
    return nn_support.to(obj.device)


def apply_fourier_constraint(fft_obj, measured_amplitude):
    substituted_obj = substitute_amplitude(fft_obj, measured_amplitude)
    return substituted_obj


def substitute_amplitude(
    complex_obj: torch.Tensor, measured_amplitude: torch.Tensor
) -> torch.Tensor:
    """Substitute amplitude of complex object with measured ampiltude.

    Args:
        complex_obj (torch.Tensor): Complex object that has amplitude and phase.
        measured_amplitude (torch.Tensor): Measured amplitude.

    Returns:
        torch.Tensor: Substituted complex object that has the same phase with input data.
    """
    # estimated_amplitude = complex_obj.abs()
    # substituted_obj = complex_obj / (estimated_amplitude + 1e-5) * measured_amplitude
    substituted_obj = torch.polar(measured_amplitude, complex_obj.angle())
    return substituted_obj
