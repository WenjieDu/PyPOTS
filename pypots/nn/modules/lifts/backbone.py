"""Default LIFTS architecture and joint objective from the official implementation.

Copyright (c) 2026 LIFTS contributors. MIT license; see LICENSE.
"""

import torch
from torch import nn

from .layers import (
    InitialNoiseProcessorLearnable,
    SSM_Decoder_V2,
    SSM_Filter,
    SSM_Out_Fo_MLP,
    SSM_Out_Fx_MLP,
    SSM_Smoother,
    SSM_Transition_G,
)


def masked_distance(x, y, mask, other_mask=None, comparison_mask=None, mask_weight=1.0):
    """Reference masked L2 distance, reducing the penultimate axis before averaging.

    With two masks, compare measurements on ``comparison_mask`` (or ``mask``)
    and compare the masks themselves. Keep the reference tensor layouts at the
    call sites: feature-major for forecasts and time-major for self-masking.
    """
    if other_mask is None:
        difference = (x - y) * mask
        return torch.sqrt(difference.square().sum(dim=-2) + 1e-8).mean(dim=-1).mean()
    selected = mask if comparison_mask is None else comparison_mask
    measurement_distance = torch.sqrt((x * selected - y * selected).square().sum(dim=-2) + 1e-8)
    mask_distance = torch.sqrt((mask - other_mask).square().sum(dim=-2) + 1e-8)
    return (measurement_distance + mask_weight * mask_distance).mean(dim=-1).mean()


class BackboneLIFTS(nn.Module):
    """Reference default: causal RoPE filter, residual smoother, MLP emissions,
    stochastic transition, and feature-attention decoder.

    All public inputs use [batch, steps, features]. The official module and
    parameter names are retained to facilitate comparison with the source.
    """

    def __init__(self, n_steps, n_features, d_latent, d_model, n_heads, d_ffn, n_layers, n_inner_layers):
        super().__init__()
        # Preserve the standalone constructor's RNG consumption and initialization.
        self.noise_processor = InitialNoiseProcessorLearnable(d_latent)
        self.encoder_h = SSM_Filter(
            n_steps,
            n_features,
            d_latent,
            d_model,
            d_ffn,
            n_heads,
            d_model // n_heads,
            d_model // n_heads,
            0.0,
            n_layers,
            n_inner_layers,
        )
        self.out_fx = SSM_Out_Fx_MLP(n_features, d_latent, d_model)
        self.out_fo = SSM_Out_Fo_MLP(n_features, d_latent, d_model)
        self.transition_g = SSM_Transition_G(d_latent, d_model)
        self.decoder_u = SSM_Decoder_V2(n_features, d_latent, d_model, n_heads, dropout=0.0)
        self.backward_transition_q = SSM_Smoother(
            feature_dim=n_features,
            keep_dim=d_latent,
            time_dim=n_steps,
            d_model=d_model,
        )

    def smooth(self, filtered):
        """Sample backward recursively, conditioned on the filtered terminal state."""
        states = [filtered[:, -1]]
        for t in range(filtered.shape[1] - 2, -1, -1):
            states.append(self.backward_transition_q(filtered[:, t], states[-1]))
        return torch.stack(states[::-1], dim=1)

    def joint_loss(
        self,
        X,
        mask,
        self_mask_rate,
        self_mask_weight,
        mask_weight,
        n_sampling_times,
        self_mask_keep=None,
        has_self_mask_targets=None,
    ):
        """Joint one-step forecast and self-masked imputation energy objective.

        This implements the default joint-training path (lambda_enc=0 and
        lambda_d_o=0), without the optional experimental calibration stages.
        """
        batch, steps, features = X.shape
        # The standalone loader batches contiguous [B,P,T] arrays. Preserve
        # their strides, including torch.rand_like's self-mask RNG ordering.
        feature_X = X.transpose(1, 2).contiguous()
        feature_mask = mask.transpose(1, 2).contiguous()
        original_X, original_mask = feature_X.transpose(1, 2), feature_mask.transpose(1, 2)
        X = feature_X[:, None].expand(-1, n_sampling_times, -1, -1).reshape(-1, features, steps)
        mask = feature_mask[:, None].expand(-1, n_sampling_times, -1, -1).reshape(-1, features, steps)
        X, mask = X.transpose(1, 2), mask.transpose(1, 2)
        filtered = self.encoder_h(X, mask)
        transitioned = self.transition_g(filtered[:, :-1])
        z1 = self.noise_processor(X.shape[0], X.device)
        z2 = self.noise_processor(X.shape[0], X.device)
        z1 = torch.cat([z1, transitioned], dim=1)
        z2 = torch.cat([z2, transitioned], dim=1)
        forecast1, forecast2 = self.out_fx(z1), self.out_fx(z2)
        observation1, observation2 = self.out_fo(z1), self.out_fo(z2)

        def feature_major(value):
            return value.transpose(1, 2).reshape(batch, n_sampling_times, features, steps)

        true, observed = feature_major(X), feature_major(mask)
        f1, f2 = feature_major(forecast1), feature_major(forecast2)
        o1, o2 = feature_major(observation1), feature_major(observation2)
        forecast_loss = (
            masked_distance(true, f1, observed, o1, mask_weight=mask_weight)
            + masked_distance(true, f2, observed, o2, mask_weight=mask_weight)
            - masked_distance(f1, f2, o1, o2, comparison_mask=observed, mask_weight=mask_weight)
        )
        # The reference self-masking branch uses the original batch, not the
        # Monte Carlo expansion used above for the forecast energy objective.
        X, mask = original_X, original_mask
        keep = (torch.rand_like(mask) > self_mask_rate).to(mask.dtype) if self_mask_keep is None else self_mask_keep
        input_mask = mask * keep
        input_X = X * input_mask
        self_mask_loss = X.new_zeros(())
        if self_mask_weight > 0 and (
            (mask * (1 - keep)).any() if has_self_mask_targets is None else has_self_mask_targets
        ):
            filtered = self.encoder_h(input_X, input_mask)
            smooth1, smooth2 = self.smooth(filtered), self.smooth(filtered)
            sample1 = self.decoder_u(smooth1, input_X, input_mask)
            sample2 = self.decoder_u(smooth2, input_X, input_mask)
            self_mask_loss = (
                masked_distance(X, sample1, mask)
                + masked_distance(X, sample2, mask)
                - masked_distance(sample1, sample2, mask)
            )
        return forecast_loss + self_mask_weight * self_mask_loss, forecast_loss, self_mask_loss

    def impute(self, X, mask, n_sampling_times):
        """Return [batch, samples, steps, features], preserving observations."""
        batch, steps, features = X.shape
        repeated_X = X[:, None].expand(-1, n_sampling_times, -1, -1).reshape(-1, steps, features)
        repeated_mask = mask[:, None].expand(-1, n_sampling_times, -1, -1).reshape(-1, steps, features)
        filtered = self.encoder_h(repeated_X, repeated_mask)
        samples = self.decoder_u(self.smooth(filtered), repeated_X, repeated_mask)
        samples = torch.where(repeated_mask.bool(), repeated_X, samples)
        return samples.reshape(batch, n_sampling_times, steps, features)

    def forecast_in_sample(self, X, mask, n_sampling_times):
        """Original one-step predictions: at t, condition only on observations before t.

        Retain both original forecast branches and mask emissions to preserve
        the standalone forecasting routine's random-number consumption.
        """
        batch, steps, features = X.shape
        feature_X = X.transpose(1, 2).contiguous()
        feature_mask = mask.transpose(1, 2).contiguous()
        if n_sampling_times > 1:
            feature_X = feature_X[:, None].expand(-1, n_sampling_times, -1, -1).reshape(-1, features, steps)
            feature_mask = feature_mask[:, None].expand(-1, n_sampling_times, -1, -1).reshape(-1, features, steps)
        filtered = self.encoder_h(feature_X.transpose(1, 2), feature_mask.transpose(1, 2))
        transitioned = self.transition_g(filtered[:, :-1])
        z1 = self.noise_processor(filtered.shape[0], X.device)
        z2 = self.noise_processor(filtered.shape[0], X.device)
        z1 = torch.cat([z1, transitioned], dim=1)
        z2 = torch.cat([z2, transitioned], dim=1)
        forecast = self.out_fx(z1)
        self.out_fx(z2)
        self.out_fo(z1)
        self.out_fo(z2)
        return forecast.reshape(batch, n_sampling_times, steps, features)

    def forecast(self, X, mask, n_pred_steps, n_sampling_times):
        """Roll out the latent transition without accessing future observations."""
        batch, steps, features = X.shape
        repeated_X = X[:, None].expand(-1, n_sampling_times, -1, -1).reshape(-1, steps, features)
        repeated_mask = mask[:, None].expand(-1, n_sampling_times, -1, -1).reshape(-1, steps, features)
        latent = self.encoder_h(repeated_X, repeated_mask)[:, -1:]
        samples = []
        for _ in range(n_pred_steps):
            latent = self.transition_g(latent)
            samples.append(self.out_fx(latent))
        return torch.cat(samples, dim=1).reshape(batch, n_sampling_times, n_pred_steps, features)
