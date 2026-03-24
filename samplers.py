import torch
import numpy as np


def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    return 2 * t_cur


def _velocity_step(model, x, t_val, y_cond, y_null, _dtype, device,
                   cfg_scale, guidance_low, guidance_high,
                   inference_budget, unconditional_inference_budget,
                   autoguidance=False):
    """Single velocity prediction with CFG / CCFG / autoguidance support.

    Autoguidance keeps the real class label on the "unconditional" path but
    runs it at the lower ``unconditional_inference_budget``, so guidance
    comes from the capacity gap rather than from dropping the condition.
    """
    in_guidance = cfg_scale > 1.0 and t_val <= guidance_high and t_val >= guidance_low

    if in_guidance and unconditional_inference_budget is not None:
        t_in = torch.ones(x.size(0), device=device, dtype=torch.float64) * t_val
        d_cond = model(x.to(_dtype), t_in.to(_dtype),
                       y=y_cond, inference_budget=inference_budget)[0].to(torch.float64)
        y_low = y_cond if autoguidance else y_null
        d_uncond = model(x.to(_dtype), t_in.to(_dtype),
                         y=y_low, inference_budget=unconditional_inference_budget)[0].to(torch.float64)
        return d_uncond + cfg_scale * (d_cond - d_uncond)

    if in_guidance:
        model_input = torch.cat([x, x], dim=0)
        y_cur = torch.cat([y_cond, y_null], dim=0)
    else:
        model_input = x
        y_cur = y_cond

    t_in = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_val
    d = model(model_input.to(_dtype), t_in.to(_dtype),
              y=y_cur, inference_budget=inference_budget)[0].to(torch.float64)

    if in_guidance:
        d_cond, d_uncond = d.chunk(2)
        d = d_uncond + cfg_scale * (d_cond - d_uncond)
    return d


def _drift_step_sde(model, x, t_val, y_cond, y_null, _dtype, device,
                    cfg_scale, guidance_low, guidance_high,
                    inference_budget, unconditional_inference_budget,
                    path_type, diffusion, autoguidance=False):
    """Single drift prediction for SDE with CFG / CCFG / autoguidance support."""
    in_guidance = cfg_scale > 1.0 and t_val <= guidance_high and t_val >= guidance_low

    if in_guidance and unconditional_inference_budget is not None:
        t_in = torch.ones(x.size(0), device=device, dtype=torch.float64) * t_val
        x_f64 = x.to(torch.float64)
        v_cond = model(x.to(_dtype), t_in.to(_dtype),
                       y=y_cond, inference_budget=inference_budget)[0].to(torch.float64)
        s_cond = get_score_from_velocity(v_cond, x_f64, t_in, path_type=path_type)
        d_cond = v_cond - 0.5 * diffusion * s_cond
        y_low = y_cond if autoguidance else y_null
        v_uncond = model(x.to(_dtype), t_in.to(_dtype),
                         y=y_low, inference_budget=unconditional_inference_budget)[0].to(torch.float64)
        s_uncond = get_score_from_velocity(v_uncond, x_f64, t_in, path_type=path_type)
        d_uncond = v_uncond - 0.5 * diffusion * s_uncond
        return d_uncond + cfg_scale * (d_cond - d_uncond)

    if in_guidance:
        model_input = torch.cat([x, x], dim=0)
        y_cur = torch.cat([y_cond, y_null], dim=0)
    else:
        model_input = x
        y_cur = y_cond

    t_in = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_val
    v = model(model_input.to(_dtype), t_in.to(_dtype),
              y=y_cur, inference_budget=inference_budget)[0].to(torch.float64)
    s = get_score_from_velocity(v, model_input.to(torch.float64), t_in, path_type=path_type)
    d = v - 0.5 * diffusion * s

    if in_guidance:
        d_cond, d_uncond = d.chunk(2)
        d = d_uncond + cfg_scale * (d_cond - d_uncond)
    return d


def euler_sampler(
        model, latents, y,
        num_steps=20, heun=False,
        cfg_scale=1.0, guidance_low=0.0, guidance_high=1.0,
        path_type="linear",
        inference_budget=None, unconditional_inference_budget=None,
        autoguidance=False,
        ):
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    else:
        y_null = None
    _dtype = latents.dtype
    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device

    fwd = dict(model=model, y_cond=y, y_null=y_null, _dtype=_dtype, device=device,
               cfg_scale=cfg_scale, guidance_low=guidance_low, guidance_high=guidance_high,
               inference_budget=inference_budget,
               unconditional_inference_budget=unconditional_inference_budget,
               autoguidance=autoguidance)

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            d_cur = _velocity_step(x=x_cur, t_val=t_cur, **fwd)
            x_next = x_cur + (t_next - t_cur) * d_cur

            if heun and (i < num_steps - 1):
                d_prime = _velocity_step(x=x_next, t_val=t_next, **fwd)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def euler_maruyama_sampler(
        model, latents, y,
        num_steps=20, heun=False,
        cfg_scale=1.0, guidance_low=0.0, guidance_high=1.0,
        path_type="linear",
        inference_budget=None, unconditional_inference_budget=None,
        autoguidance=False,
        ):
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    else:
        y_null = None

    _dtype = latents.dtype
    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    drift = dict(model=model, y_cond=y, y_null=y_null, _dtype=_dtype, device=device,
                 cfg_scale=cfg_scale, guidance_low=guidance_low, guidance_high=guidance_high,
                 inference_budget=inference_budget,
                 unconditional_inference_budget=unconditional_inference_budget,
                 path_type=path_type, autoguidance=autoguidance)

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next
            diffusion = compute_diffusion(t_cur)
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            d_cur = _drift_step_sde(x=x_cur, t_val=t_cur, diffusion=diffusion, **drift)
            x_next = x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

    # last step (no noise)
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    diffusion = compute_diffusion(t_cur)

    d_cur = _drift_step_sde(x=x_cur, t_val=t_cur, diffusion=diffusion, **drift)
    mean_x = x_cur + dt * d_cur

    return mean_x
