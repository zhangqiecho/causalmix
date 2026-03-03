from sklearn.mixture import BayesianGaussianMixture
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping 
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import os
import math
from torch.distributions import Normal

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# conVAE trains a generator which can generate Y | X

class conVAE(pl.LightningModule):
    def __init__(
        self,
        df,
        Ynames,  # random variable to be generated / inputted
        Xnames=[],  # random variable to condition on
        binary_cols=[],  # binary columns
        categorical_cols=[],      # list of categorical columns in Ynames
        categorical_dims={},      # dict {col_name: num_classes}
        var_bounds={},  # variable lower and upper bounds
        max_bound = 5, # variance bound [-5, 5] for each predicted log-variance for continuous variables, used to stablize NLL and vary uncertainty 
        # integer_cols = [],
        latent_dim=2,  # latent space dimensions
        hidden_dim=[16],  # perceptrons in each layer
        batch_size=10,  # batch size
        potential_outcome=False,  # indicator if the generated sample has both potential outcomes
        treatment_cols=[],  # treatment indicator column
        treatment_effect_fn=lambda x, index: 0,  # treatment effect function defined by the user
        selection_bias_fn=lambda x, t, index: 0,  # selection bias/unmeasured confounding function defined by the user
        effect_rigidity=1e20,  # strength of treatment effect constraint
        effect_mse_weight=0.5, # effect blend weight on MSE
        effect_var_weight=1e-2, # variance penalty for treatment effect residuals
        bias_rigidity=1e20,  # strength of selection bias constraint
        bias_mse_weight=0.5, # bias blend weight on MSE
        bias_var_weight=5e-3, # variance penalty for bias residuals
        kld_rigidity=0.1,  # strength of KL divergence loss
        overlap_weight=0.0,
        overlap_target=lambda x, index: 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["treatment_effect_fn", "selection_bias_fn"]) # saves all arguments passed to the __init__ method as hyperparameters for reproducibility and debugging in training pipelines

        num_cores = os.cpu_count()
        # num_cores = 16
        # print(f"Number of CPU cores: {num_cores}")
        # initializing internal variables/objects
        self.in_dim = len(Ynames)  # input dimension
        self.con_dim = len(Xnames)  # dimension of variable to condition on
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        #self.binary_cols = binary_cols 
        # categorical, binary and continuous index detection
        self.binary_output_idx = [i for i in range(self.in_dim) if Ynames[i] in binary_cols]
        self.categorical_output_idx = [Ynames.index(c) for c in categorical_cols if c in Ynames] # index detection for categorical columns 
        self.continuous_output_idx = [i for i in range(self.in_dim) if i not in set(self.binary_output_idx + self.categorical_output_idx)]
        self.categorical_dims = {Ynames.index(k): v for k, v in categorical_dims.items() if k in Ynames} # number of classes for the categorical columns 

        self.encoder_dims = [self.in_dim] + self.hidden_dim
        self.decoder_dims = [self.latent_dim + self.con_dim] + self.hidden_dim[::-1]
        self.batch_size = batch_size
        self.potential_outcome = potential_outcome
        self.kld_rigidity = kld_rigidity
        self.overlap_weight = overlap_weight
        self.overlap_target = overlap_target

        self.bgmm = None            # will hold {'bgm': <BGMM>, 'log_var_mean': np.array[D]}
        self.bgmm_n_components = self.latent_dim  # default; you can expose as __init__ arg if you want

        # store the variable bound per index
        self.var_bounds = {
            Ynames.index(k): v for k, v in var_bounds.items() if k in Ynames
        }
        self.max_bound = max_bound

        if self.overlap_weight > 0:
            self.T_col = [i for i in range(self.con_dim) if Xnames[i] in treatment_cols]
            # update January 13, 2026: to add covariate index for function specification
            self.index = VarIndex(Ynames)
            

        # if generating potential outcomes then add constraints for user defined treatment effects and selection bias function
        if self.potential_outcome:
            self.T_col = [i for i in range(self.con_dim) if Xnames[i] in treatment_cols]
            self.X_col = [
                i for i in range(self.con_dim) if Xnames[i] not in treatment_cols
            ]
            # update January 13, 2026: to add covariate index for function specification
            self.Xnames = [
            x for x in Xnames if x not in treatment_cols
            ]
            self.index = VarIndex(self.Xnames)

            self.alpha = effect_rigidity
            self.f = treatment_effect_fn
            self.mix_eff = effect_mse_weight
            self.eta_var = effect_var_weight

            self.beta = bias_rigidity
            self.g = selection_bias_fn
            self.mix_bias = bias_mse_weight
            self.zeta_var = bias_var_weight

        # loading the input data to tensor dataset from dataframe
        self.data = TensorDataset(
            torch.from_numpy(df[Xnames].values.astype(float)).float(),
            torch.from_numpy(df[Ynames].values.astype(float)).float(),
        )

        # splitting the data into training and validation
        training_split = 0.8
        self.train_data, self.val_data = random_split(
            self.data,
            [ int(df.shape[0] * training_split), (df.shape[0]) - int((df.shape[0] * training_split)) ],
        )

        # training and validation dataset loader
        self.train_loader = DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle= True, 
            # num_workers=max(2, num_cores// 2),
            num_workers = 0,
            persistent_workers=False 
        )
        self.val_loader = DataLoader(
            self.val_data, 
            batch_size=self.batch_size, 
            shuffle=False,
            # num_workers=max(2, num_cores// 2), # customize num_workers for parallel data loading
            num_workers = 0,
            persistent_workers=False 
        )

        # Encoder layers
        self.encoder_module = []
        for layer in range(len(hidden_dim)):
            self.encoder_module.append(
                nn.Sequential(
                    nn.Linear(self.encoder_dims[layer], self.encoder_dims[layer + 1]),
                    nn.LeakyReLU(),
                )
            )
        self.encoder = nn.Sequential(*self.encoder_module)

        # embedding layers
        self.en_mu = nn.Linear(self.hidden_dim[-1], self.latent_dim)
        self.en_logvar = nn.Linear(self.hidden_dim[-1], self.latent_dim)

        # Decoder layers
        self.decoder_module = []
        for layer in range(len(hidden_dim)):
            self.decoder_module.append(
                nn.Sequential(
                    nn.Linear(self.decoder_dims[layer], self.decoder_dims[layer + 1]),
                    nn.LeakyReLU(),
                )
            )
        
        self.decoder = nn.Sequential(*self.decoder_module)

        self.output_heads = nn.ModuleDict()
        for i in range(self.in_dim):
            base_dim = 1
            if i in self.categorical_output_idx:
                base_dim = self.categorical_dims[i]
            output_dim = base_dim * (2 if self.potential_outcome else 1)  # double outputs for Y(0) and Y(1)
            self.output_heads[str(i)] = nn.Linear(hidden_dim[0], output_dim)

        # to add variance heads for continuous var
        self.var_heads = nn.ModuleDict()
        for i in range(self.in_dim):
            # one logvar per scalar dim; double if potential outcomes
            out_dim = 1 * (2 if self.potential_outcome else 1)
            self.var_heads[str(i)] = nn.Linear(hidden_dim[0], out_dim)

    def forward(self, y):
        # embedding into latent space
        l = self.encoder(y)

        # mean and logvariance of the projected point in latent space
        mu = self.en_mu(l)
        logvar = self.en_logvar(l)

        return (mu, logvar)
    
    # update January 17, 2026
    def loss_fn(self, yhat, y, mu, logvar, **kwargs):
        """
        PO loss with:
        - Effect constraint: blended MSE + SmoothL1 on (y1 - y0) vs f(X)
        - Bias constraint: ORIGINAL T-gated definition, blended MSE + SmoothL1 vs g(X,T)
        - Optional Fix 1: residual variance penalties on (effect_residual, bias_residual)
          1)eta_var (η): variance penalty for treatment effect residuals
          2)zeta_var (ζ): variance penalty for unmeasured confounding bias residuals
        Non-PO branch unchanged.
        """
        import math
        LOG_2PI = math.log(2.0 * math.pi)

        cont_mu     = kwargs.get("cont_mu", {})
        cont_logvar = kwargs.get("cont_logvar", {})

        def blend_loss(pred, target, w_mse: float):
            # pred/target: same shape
            mse = (pred - target).pow(2).mean()
            hub = F.smooth_l1_loss(pred, target, reduction="mean")
            return w_mse * mse + (1.0 - w_mse) * hub

        def expand(val, B, D, ref):
            # val can be python float/int or torch tensor; output [B, D]
            if not torch.is_tensor(val):
                val = torch.tensor(val, device=ref.device, dtype=ref.dtype)
            if val.dim() == 0:
                return val.view(1, 1).expand(B, D)
            if val.dim() == 1:
                return val.view(B, 1).expand(B, D)
            if val.dim() == 2 and val.shape == (B, 1):
                return val.expand(B, D)
            if val.dim() == 2 and val.shape == (B, D):
                return val
            raise ValueError(f"Cannot expand val of shape {tuple(val.shape)} to ({B},{D})")

        # ------------------------------- POTENTIAL OUTCOMES -------------------------------
        if self.potential_outcome:
            # # update January 17, 2026: added knobs for a better fit to the causal constraint functions
            # mix_eff: weight on MSE (effect) in [0,1]
            # mix_bias: weight on MSE (bias)   in [0,1]
            # eta_var: effect residual variance penalty
            # zeta_var: bias residual variance penalty
            if len(self.categorical_output_idx) > 0:
                raise RuntimeError("no multi-categorical outcomes are allowed for potential outcomes")
            if "T" not in kwargs or "X" not in kwargs or "y_prime" not in kwargs:
                raise KeyError("loss_fn expects 'T', 'X', and 'y_prime' when potential_outcome=True")

            B, V = yhat.shape[0], self.in_dim
            if yhat.shape[1] != 2 * V:
                raise ValueError(f"Expected yhat width 2*in_dim={2*V}, got {yhat.shape[1]}")

            yhat_3d = yhat.view(B, V, 2)               # [B, V, 2]
            Tcol    = kwargs["T"].view(B, 1)           # [B, 1]
            y_obs   = yhat_3d[:, :, 0] * (1 - Tcol) + yhat_3d[:, :, 1] * Tcol   # [B, V]

            loss = yhat.new_tensor(0.0)

            # ---------- Likelihood ----------
            if len(self.binary_output_idx):
                logits_bin = y_obs[:, self.binary_output_idx]   # [B, Vb]
                target_bin = y[:,   self.binary_output_idx]     # [B, Vb]
                loss += F.binary_cross_entropy_with_logits(logits_bin, target_bin, reduction="mean")

            if len(self.continuous_output_idx):
                mu_stack = torch.stack([cont_mu[i]     for i in self.continuous_output_idx], dim=1)  # [B, Vc, 2]
                lv_stack = torch.stack([cont_logvar[i] for i in self.continuous_output_idx], dim=1)  # [B, Vc, 2]
                mu_obs = mu_stack[:, :, 0] * (1 - Tcol.squeeze(-1)) + mu_stack[:, :, 1] * Tcol.squeeze(-1)  # [B, Vc]
                lv_obs = lv_stack[:, :, 0] * (1 - Tcol.squeeze(-1)) + lv_stack[:, :, 1] * Tcol.squeeze(-1)  # [B, Vc]
                y_true = y[:, self.continuous_output_idx]  # [B, Vc]
                nll = 0.5 * (torch.exp(-lv_obs) * (y_true - mu_obs) ** 2 + lv_obs + LOG_2PI)
                loss += nll.mean()

            # ---------- PO constraints ----------
            yprime_3d = kwargs["y_prime"].view(B, V, 2)
            y0, y1 = yhat_3d[:, :, 0], yhat_3d[:, :, 1]
            y0p, y1p = yprime_3d[:, :, 0], yprime_3d[:, :, 1]

            X = kwargs["X"]
            f_val = self.f(X, self.index)                   # float or tensor
            g_val = self.g(X, Tcol[:, 0], self.index)       # float or tensor

            # ---------------- Binary constraints (on probabilities) ----------------
            if len(self.binary_output_idx):
                y0b  = torch.sigmoid(y0[:,  self.binary_output_idx])   # [B, Vb]
                y1b  = torch.sigmoid(y1[:,  self.binary_output_idx])
                y0pb = torch.sigmoid(y0p[:, self.binary_output_idx])
                y1pb = torch.sigmoid(y1p[:, self.binary_output_idx])

                D = y0b.shape[1]
                T_b = Tcol.repeat(1, D)
                f_b = expand(f_val, B, D, yhat)
                g_b = expand(g_val, B, D, yhat)

                eff_pred_b  = (y1b - y0b)
                bias_pred_b = T_b * (y0b - y0pb) + (1 - T_b) * (y1pb - y1b)

                # blended constraint losses
                loss += self.alpha * blend_loss(eff_pred_b,  f_b, self.mix_eff)
                loss += self.beta  * blend_loss(bias_pred_b, g_b, self.mix_bias)

                # optional residual variance penalties
                if self.eta_var > 0.0:
                    loss += self.eta_var * (eff_pred_b - f_b).var(dim=0, unbiased=False).mean()
                if self.zeta_var > 0.0:
                    loss += self.zeta_var * (bias_pred_b - g_b).var(dim=0, unbiased=False).mean()

            # ---------------- Continuous constraints (on means) ----------------
            if len(self.continuous_output_idx):
                y0c  = y0[:,  self.continuous_output_idx]  # [B, Vc]
                y1c  = y1[:,  self.continuous_output_idx]
                y0pc = y0p[:, self.continuous_output_idx]
                y1pc = y1p[:, self.continuous_output_idx]

                D = y0c.shape[1]
                T_c = Tcol.repeat(1, D)
                f_c = expand(f_val, B, D, yhat)
                g_c = expand(g_val, B, D, yhat)

                eff_pred_c  = (y1c - y0c)
                bias_pred_c = T_c * (y0c - y0pc) + (1 - T_c) * (y1pc - y1c)

                loss += self.alpha * blend_loss(eff_pred_c,  f_c, self.mix_eff)
                loss += self.beta  * blend_loss(bias_pred_c, g_c, self.mix_bias)

                if self.eta_var > 0.0:
                    loss += self.eta_var * (eff_pred_c - f_c).var(dim=0, unbiased=False).mean()
                if self.zeta_var > 0.0:
                    loss += self.zeta_var * (bias_pred_c - g_c).var(dim=0, unbiased=False).mean()

        # ------------------------------- NON-PO (categoricals allowed) -------------------------------
        else:
            loss = yhat.new_tensor(0.0)
            offset = 0
            for i in range(self.in_dim):
                if i in self.categorical_output_idx:
                    K = self.categorical_dims[i]
                    logits_i = yhat[:, offset:offset + K]      # [B, K]
                    target_i = y[:, i].long()                  # [B]
                    loss += F.cross_entropy(logits_i, target_i, reduction='mean')
                    offset += K
                elif i in self.binary_output_idx:
                    logits_i = yhat[:, offset:offset + 1]      # [B, 1]
                    target_i = y[:, i:i+1]                     # [B, 1]
                    loss += F.binary_cross_entropy_with_logits(logits_i, target_i, reduction='mean')
                    offset += 1
                else:
                    if (i not in cont_mu) or (i not in cont_logvar):
                        raise KeyError(f"Missing cont_mu/cont_logvar for variable index {i}")
                    mu_i = cont_mu[i]                           # [B,1]
                    lv_i = cont_logvar[i]                       # [B,1]
                    y_true = y[:, i:i+1]
                    nll = 0.5 * (torch.exp(-lv_i) * (y_true - mu_i) ** 2 + lv_i + LOG_2PI)
                    loss += nll.mean()
                    offset += 1

        # KL (mean over batch)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y.shape[0]
        loss += self.kld_rigidity * kld
        return loss


    def _activate_continuous_mean(self, i, raw_mu):
        b = self.var_bounds.get(i, None)
        if b is None or (('lower' not in b) and ('upper' not in b)):
            return raw_mu
        if ('lower' in b) and ('upper' in b):
            lo, hi = float(b['lower']), float(b['upper'])
            return lo + (hi - lo) * torch.sigmoid(raw_mu)
        elif 'lower' in b:
            lo = float(b['lower'])
            return lo + F.softplus(raw_mu)
        else:  # upper only
            hi = float(b['upper'])
            return hi - F.softplus(raw_mu)
    
    def _clamp_continuous(self, i, y):
        b = self.var_bounds.get(i, None)
        if b is None:
            return y
        if 'lower' in b:
            y = torch.maximum(y, y.new_tensor(float(b['lower'])))
        if 'upper' in b:
            y = torch.minimum(y, y.new_tensor(float(b['upper'])))
        return y

    def configure_optimizers(self):
        # initializing optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # functions added for overlap knob
    def _mk_x_with_t(self, x: torch.Tensor, t_scalar: int) -> torch.Tensor:
            """Return a copy of x with treatment column set to t_scalar (0 or 1).""" # x is just t here
            x_mod = x.clone()
            x_mod[:, self.T_col] = float(t_scalar)
            return x_mod
    
    def _gauss_logprob_1d(self, y_i: torch.Tensor, mu_i: torch.Tensor, logvar_i: torch.Tensor) -> torch.Tensor:
        """Diagonal Gaussian log-prob for a single scalar feature (shapes [B,1] -> [B])."""
        return (-0.5 * (math.log(2*math.pi) + logvar_i + (y_i - mu_i)**2 / torch.exp(logvar_i))).squeeze(1)
    
    def _log_p_y_given_T(self, y: torch.Tensor, z: torch.Tensor, x: torch.Tensor, t_scalar: int) -> torch.Tensor:
        """
        Compute per-sample log p_theta(y | T=t_scalar) by decoding with x where T=t_scalar
        and summing per-variable log-probs using your heads. Assumes potential_outcome=False.
        Returns [B].
        """
        x_t = self._mk_x_with_t(x, t_scalar)
        w_t = self.decoder(torch.cat((z, x_t), dim=1))
    
        B = y.size(0)
        total_logp = y.new_zeros(B)
    
        for i in range(self.in_dim):
            out = self.output_heads[str(i)](w_t)
    
            if i in self.binary_output_idx:
                # Bernoulli logits expected; y[:, i] in {0,1}
                logits = out if out.dim() == 2 else out.unsqueeze(-1)  # [B,1]
                logits = logits.squeeze(1)                              # [B]
                yi = y[:, i]
                logp_i = -F.binary_cross_entropy_with_logits(logits, yi, reduction="none")
    
            elif i in self.categorical_output_idx:
                # Softmax logits expected; y[:, i] stores class indices (0..K-1)
                logits = out                                           # [B,K]
                yi = y[:, i].long()                                    # [B]
                logp_i = -F.cross_entropy(logits, yi, reduction="none")
    
            else:
                # Continuous scalar: mean head + var head
                mu_i_raw = out                                        # [B,1]
                mu_i = self._activate_continuous_mean(i, mu_i_raw)    # [B,1]
                logvar_i = self.var_heads[str(i)](w_t)                # [B,1]
                logvar_i = torch.tanh(logvar_i) * self.max_bound
                yi = y[:, i].unsqueeze(1)                             # [B,1]
                logp_i = self._gauss_logprob_1d(yi, mu_i, logvar_i)
    
            total_logp = total_logp + logp_i # conditional independence assumption
    
        return total_logp  # [B]

    def _overlap_target_from_y(self, y, ref, detach=True):
        # update January 13, 2026: to add covariate index for function specification
        t = self.overlap_target(y, self.index)                      # scalar, [B], or [B,1]
        t = torch.as_tensor(t, dtype=ref.dtype, device=ref.device).flatten()
        if t.numel() == 1:
            t = t.expand_as(ref)
        elif t.numel() != ref.numel():
            raise ValueError(f"overlap_target(y) must return scalar or length-{ref.numel()} (got {tuple(t.shape)})")
        return t.detach() if detach else t
    
    def _shared_step(self, batch): # min_val=1e-3
        x, y = batch
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
    
        # Encode
        l = self.encoder(y)
        mu = self.en_mu(l)
        logvar = self.en_logvar(l)
    
        # Reparameterize
        e = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        z = mu + std * e
    
        # Decode (condition on X)
        z_ = torch.cat((z, x), dim=1)
        w = self.decoder(z_)
    
        # Build packed outputs and collect continuous params
        outputs = []
        cont_mu = {}      # var_idx -> [B, 2] if potential_outcome else [B, 1]
        cont_logvar = {}  # var_idx -> same shape as cont_mu
    
        for i in range(self.in_dim):
            out = self.output_heads[str(i)](w)
    
            if self.potential_outcome:
                if len(self.categorical_output_idx) > 0:
                    raise RuntimeError("no multi-categorical outcomes are allowed for potential outcomes")
                # Determine base dim (1 for binary/continuous, K for categorical)
                base_dim = 1
    
                # Shape to [B, 2, base_dim] for Y(0), Y(1)
                out = out.view(-1, 2, base_dim)
    
                if i in self.binary_output_idx:
                    # instead of probabilities, use logits for cross entropy calculation
                    outputs.append(out.view(-1, 2 * base_dim))
                else:
                    # continuous scalar: keep means in the packed y_hat,
                    # but also predict log-variance via var_heads
                    # means: [B, 2]
                    mu_i_raw = out.squeeze(-1)
                    mu_i = self._activate_continuous_mean(i, mu_i_raw)
                    
                    # log-variance: [B, 2]
                    logvar_i = self.var_heads[str(i)](w).view(-1, 2)
                    logvar_i = torch.tanh(logvar_i) * self.max_bound # logvar in (-5,5), so var in (-0.0067, 148)
    
                    cont_mu[i] = mu_i
                    cont_logvar[i] = logvar_i
    
                    # For packing, we keep means (2 columns)
                    outputs.append(mu_i)
    
            else:
                if i in self.binary_output_idx:
                    # instead of probabilities, use logits for cross entropy calculation
                    outputs.append(out if out.dim() == 2 else out.unsqueeze(-1))
    
                elif i in self.categorical_output_idx:
                    # keep logits
                    # out = torch.tanh(out) * 10.0  # Limit the max values
                    outputs.append(out if out.dim() == 2 else out.unsqueeze(-1))
    
                else:
                    # continuous scalar: mean + log-variance
                    mu_i_raw = out  # [B, 1]
                    mu_i = self._activate_continuous_mean(i, mu_i_raw)
                    logvar_i = self.var_heads[str(i)](w)  # [B, 1]
                    logvar_i = torch.tanh(logvar_i) * self.max_bound # logvar in (-5,5), so var in (-0.0067, 148)
    
                    cont_mu[i] = mu_i
                    cont_logvar[i] = logvar_i
    
                    outputs.append(mu_i if mu_i.dim() == 2 else mu_i.unsqueeze(-1))
    
        y_hat = torch.cat(outputs, dim=1)
    
        # If potential outcomes, also compute y_hat' with flipped treatment for constraints (if used)
        if self.potential_outcome:
            x_prime = x.clone()
            x_prime[:, self.T_col] = 1 - x_prime[:, self.T_col]
            z_prime_ = torch.cat((z, x_prime), dim=1)
            w_prime = self.decoder(z_prime_)
    
            outputs_prime = []
            for i in range(self.in_dim):
                out_p = self.output_heads[str(i)](w_prime) # out_p are raw logits
                base_dim = 1
    
                out_p = out_p.view(-1, 2, base_dim)
    
                # For constraints we just need the same packing (no need to compute var heads here)
                outputs_prime.append(out_p.view(-1, 2 * base_dim))
    
            y_hat_prime = torch.cat(outputs_prime, dim=1)
    
            loss = self.loss_fn(
                y_hat, y, mu, logvar,
                X=x[:, self.X_col], T=x[:, self.T_col],
                y_prime=y_hat_prime,
                cont_mu=cont_mu, cont_logvar=cont_logvar
            )
        else:
            loss = self.loss_fn(
                y_hat, y, mu, logvar,
                cont_mu=cont_mu, cont_logvar=cont_logvar
            )

         # ---------- NEW: overlap penalty λ * MSE( log p(y|T=0) - log p(y|T=1), α ) ----------   
        prefix = "train" if self.training else "val"
        if getattr(self, "overlap_weight", 0.0) > 0.0:
            base_loss = loss
            logp0 = self._log_p_y_given_T(y, z, x, t_scalar=0)   # [B]
            logp1 = self._log_p_y_given_T(y, z, x, t_scalar=1)   # [B]
            log_ratio = logp0 - logp1                            # [B]
    
            # target = log_ratio.new_full(log_ratio.shape, self.overlap_target)
            target = self._overlap_target_from_y(y, ref=log_ratio)
            overlap_penalty = F.mse_loss(log_ratio, target, reduction="mean")
            loss = base_loss + self.overlap_weight * overlap_penalty
            
            self.log_dict({
                f"{prefix}_loss": loss,
                f"{prefix}_base_loss": base_loss,
                f"{prefix}_overlap_penalty": overlap_penalty,
                f"{prefix}_log_ratio_mean": log_ratio.mean(),
                f"{prefix}_log_ratio_std":  log_ratio.std(),
            }, prog_bar=False, on_step=False, on_epoch=True)
        else:
            self.log(f"{prefix}_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
    
        return loss

    
    def training_step(self, train_batch, batch_idx):
        loss = self._shared_step(train_batch)
        # returning training loss and log it
        # self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._shared_step(val_batch)
        # self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False) # QZ: log validation loss to the rich progress bar
        # self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False) # stop showing progress bar
        return loss

    def fit_model(self, accelerator='auto', precision='bf16-mixed', max_epochs=1000): # QZ: updates in PyTorch Lightning's API
        logger = TensorBoardLogger(save_dir="lightning_logs", # Directory to save logs
                                   name="explore", # Experiment name (creates subfolder)
                                   )
        # 1) Slow down redraws; 
        # 2) don’t leave big finished bars; 
        # 3) force notebook-friendly console;
        # progress = RichProgressBar(
        #     # Lightning ≥1.9/2.x uses `refresh_rate` (updates every N batches).
        #     refresh_rate=10,                 # update every 10 batches (reduce output volume)
        #     leave=False,                     # don't leave a big block of output at epoch end
        #     console_kwargs={"force_jupyter": True}
        #     )
        
        early_stop = EarlyStopping(monitor="val_loss", patience=5)
        ckpt = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
        trainer = pl.Trainer(
            logger=logger,
            accelerator=accelerator,
            precision=precision,
            max_epochs=max_epochs,
            log_every_n_steps=50,  
            #callbacks=[progress, EarlyStopping(monitor="val_loss", patience=5)],
            callbacks=[early_stop,ckpt],
            enable_progress_bar=False,
            #gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        trainer.fit(self, self.train_loader, self.val_loader)
        # make them accessible later
        self.trainer = trainer
        return trainer

    @torch.no_grad()
    def _collect_latents_from_loader(self, loader):
        """
        Encodes all Y in a loader to collect posterior (mu, logvar).
        Returns concatenated tensors on CPU: mu_all [N,D], logvar_all [N,D].
        """
        mu_list, logvar_list = [], []
        was_training = self.training
        self.eval()
        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            l = self.encoder(yb)
            mu = self.en_mu(l)
            logvar = self.en_logvar(l)
            mu_list.append(mu.detach().cpu())
            logvar_list.append(logvar.detach().cpu())
        if was_training:
            self.train()
        mu_all = torch.cat(mu_list, dim=0)
        logvar_all = torch.cat(logvar_list, dim=0)
        return mu_all, logvar_all
    
    
    def fit_bgmm_from_training_latents(self, max_tries=50, random_seed=42, reg_covar=1e-5,
                                       n_init=10, max_iter=1000):
        """
        Fit a BayesianGaussianMixture on encoder means mu(Y) from the *training* split.
        Stores:
          self.bgmm = {'bgm': BGMM instance, 'log_var_mean': np.array[D]}
        """
        if getattr(self, "train_loader", None) is None:
            raise RuntimeError("train_loader not found. Make sure __init__ sets self.train_loader.")
    
        mu_all, logvar_all = self._collect_latents_from_loader(self.train_loader)
        mu_np = mu_all.numpy()
        logvar_np = logvar_all.numpy()
    
        converged = False
        last_err = None
        for t in range(max_tries):
            try:
                bgm = BayesianGaussianMixture(
                    n_components=self.bgmm_n_components,
                    random_state=random_seed + t,
                    weight_concentration_prior_type="dirichlet_process",  # DP prior
                    reg_covar=reg_covar,
                    n_init=n_init,
                    max_iter=max_iter
                ).fit(mu_np)
                converged = getattr(bgm, "converged_", True)
                if converged:
                    self.bgmm = {
                        "bgm": bgm,
                        "log_var_mean": logvar_np.mean(axis=0).astype(np.float32)  # [D]
                    }
                    return self.bgmm
            except Exception as e:
                last_err = e
                continue
    
        if not converged:
            raise RuntimeError(f"BGMM did not converge after {max_tries} attempts. "
                               f"Last error: {repr(last_err)}")
    
    
    def _sample_bgmm_latents(self, batch_size):
        """
        Draw batch_size latent vectors z from the fitted BGMM by:
          1) sampling mu ~ BGMM
          2) using global mean logvar from training encodings
          3) z = mu + exp(0.5*logvar)*epsilon
        Returns:
          z [B,D] (torch on self.device)
          mu_sample [B,D] (np)
          logvar_sample [B,D] (np)
        """
        if self.bgmm is None:
            raise RuntimeError("BGMM not fitted. Call fit_bgmm_from_training_latents(...) first.")
    
        mu_sample = self.bgmm["bgm"].sample(batch_size)[0].astype(np.float32)  # [B,D]
        logvar_mean = self.bgmm["log_var_mean"]                                 # [D]
        logvar_sample = np.tile(logvar_mean[None, :], (batch_size, 1)).astype(np.float32)
    
        mu_t = torch.from_numpy(mu_sample).to(self.device)
        logvar_t = torch.from_numpy(logvar_sample).to(self.device)
        e = torch.randn_like(mu_t)
        std_t = torch.exp(0.5 * logvar_t)
        z = mu_t + std_t * e
        return z, mu_sample, logvar_sample
    

    @torch.no_grad()
    def compute_unmeasured_bias(self, z, x, y0_means, y1_means):
        """
        Compute unmeasured confounding bias using the SAME decoder/heads by flipping T in x.
    
        Args:
            z         : [B, latent_dim]         (the same latent used to produce y0_means/y1_means)
            x         : [B, con_dim]            (conditioning; MUST include treatment at self.T_col)
            y0_means  : [B, V]                  (E[Y(0)|X] per variable: probs for binary, means for continuous)
            y1_means  : [B, V]                  (E[Y(1)|X] per variable)
    
        Returns:
            y_means_prime: [B, 2*V], concatenating
                - 'y0_prime_means'     : [B, V]   (E[Y(0)|X, flipped T])
                - 'y1_prime_means'     : [B, V]   (E[Y(1)|X, flipped T])
            bias_info : dict with
                - 'bias'               : [B, V]   = T*(y0 - y0') + (1-T)*(y1' - y1)
                - 'avg_bias'           : [V]
                - 'avg_bias_overall'   : scalar
                - 'y0_prime_means'     : [B, V]   (E[Y(0)|X, flipped T]) [removed, since it is redundant with y_means_prime)
                - 'y1_prime_means'     : [B, V]   (E[Y(1)|X, flipped T]) [removed, since it is redundant with y_means_prime)
        """
        if not self.potential_outcome:
            raise RuntimeError("Bias computation requires potential_outcome=True.")
        if len(self.categorical_output_idx) > 0:
            raise RuntimeError("PO mode forbids categorical outcomes; cannot compute bias with categoricals.")
        if self.T_col is None:
            raise RuntimeError("self.T_col must be set to locate the treatment column in x.")
        if x is None:
            raise RuntimeError("x must be provided to flip treatment for bias computation.")
    
        #device = z.device
        B = x.size(0)
        V = y0_means.size(1)
    
        # 1) Flip treatment in x
        x_flip = x.clone()
        x_flip[:, self.T_col] = 1 - x_flip[:, self.T_col]
    
        # 2) Decode with flipped T
        z_flip = torch.cat([z, x_flip], dim=1)
        w_flip = self.decoder(z_flip)
    
        # 3) Run the SAME heads to get y0', y1' expectations (probs for binary, means for continuous)
        y0p_list, y1p_list = [], []
        for i in range(self.in_dim):
            out_p = self.output_heads[str(i)](w_flip)  # expect two arms
    
            # Coerce to [B,2]
            if out_p.dim() == 3 and out_p.shape[-1] == 1 and out_p.shape[1] == 2:
                out_p = out_p.squeeze(-1)              # [B,2]
            elif out_p.dim() == 2 and out_p.shape[1] == 2:
                pass
            else:
                out_p = out_p.view(B, 2)
    
            if i in self.binary_output_idx:
                probs_p = torch.sigmoid(out_p)         # [B,2]
                y0p_list.append(probs_p[:, :1])        # keep [B,1] to concat later
                y1p_list.append(probs_p[:, 1:2])
            else:
                mu_p = out_p                           # [B,2] are the means for both arms
                mu_p = self._activate_continuous_mean(i, mu_p)
                y0p_list.append(mu_p[:, :1])
                y1p_list.append(mu_p[:, 1:2])
    
        y0_prime = torch.cat(y0p_list, dim=1)          # [B, V]
        y1_prime = torch.cat(y1p_list, dim=1)          # [B, V]
        # interleave y0_prime and y1_prime
        y_means_prime = torch.stack([y0_prime, y1_prime], dim=2)   # [B, V, 2]
        y_means_prime = y_means_prime.reshape(y0_prime.size(0), -1) # [B, 2*V]
    
        # 4) Build T mask from x (ensure {0,1})
        Tmask = x[:, self.T_col].to(dtype = y0_means.dtype) #[B,1], broadcasts over V               
    
        # 5) Bias = T*(y0 - y0') + (1-T)*(y1' - y1)
        bias = Tmask * (y0_means - y0_prime) + (1 - Tmask) * (y1_prime - y1_means)  # [B, V]
        avg_bias = bias.mean(dim=0)            # [V]
        #avg_bias_overall = avg_bias_per_var.mean()     # scalar
        bias_info = {
            "bias": bias,
            "avg_bias": avg_bias,
            #"avg_bias_overall": avg_bias_overall,
            #"y0_prime_means": y0_prime,
            #"y1_prime_means": y1_prime,
        }
    
        return y_means_prime, bias_info


    @torch.no_grad()
    def generate_bgmm(self, x=None, n_samples=None, deterministic=False, return_probs=False, return_stats = True):
        """
        Conditional generation with BGMM posterior prior.
    
        Args:
            x: [B, con_dim] tensor or None (unconditional). If None and con_dim>0, raise error.
            n_samples: int, required only when x is None.
            deterministic: if True, return hard decisions (no sampling).
            return_probs: for binary vars, return probabilities instead of 0/1.
            return_stats: if True (PO mode only), also returns a dict with:
                - 'ite' : [B, V] = E[Y(1)|X] - E[Y(0)|X]
                - 'ate' : [V]
                - 'bias' : [B, V] = T*(y0 - y0') + (1-T)*(y1' - y1)
                - 'avg_bias' : [V]
            max_bound: fallback variance bound if self.max_bound not set.
    
        Returns:
            If return_stats and self.potential_outcome:
                (y_hat [a realization/sample of the potential outcomes],
                 y_means [the expected value of the potential outcomes, i.e., probabilities or means], 
                 y_means_prime [the expected potential outcomes given flipped treatment, i.e., E[Y(0)|X, flipped T] and E[Y(1)|X, flipped T]],
                 stats)
            else:
                y_hat (a realization of the outcomes)
            where:
              - PO=False: y_hat has shape [B, V] (bin/cont scalars or class index)
              - PO=True : y_hat has shape [B, 2*V] (per-var [y0, y1] packed) 
                          y_means (y_means_prime): probabilities for binary outcomes or mean for continuous outcomes given the observed treatment (flipped treatment)
        """
        if self.bgmm is None:
            raise RuntimeError("BGMM not fitted. Call fit_bgmm_from_training_latents(...) first.")
    
        self.eval()
    
        # ---- Build latent + conditioning ----
        if x is not None and x.numel() > 0:
            x = x.to(self.device)
            B = x.shape[0]
            z, _, _ = self._sample_bgmm_latents(B)  # [B, D]
            z_ = torch.cat((z, x), dim=1) if self.con_dim > 0 else z
        else:
            if n_samples is None:
                raise ValueError("Must specify n_samples if no conditioning x is provided.")
            B = n_samples
            z, _, _ = self._sample_bgmm_latents(B)
            if self.con_dim > 0:
                raise ValueError("This model expects a conditioning tensor x, but none was provided.")
            else:
                z_ = z
    
        # ---- Decode ----
        w = self.decoder(z_)  # [B, hidden_dim[0]]
    
        outputs = []

        for i in range(self.in_dim):
            out_mu = self.output_heads[str(i)](w)  # mean/logits head
    
            if self.potential_outcome:
                # For stats in PO mode: collect per-var means (probabilities for binary, means for continuous)
                y0_means, y1_means = [], []
                if len(self.categorical_output_idx) > 0:
                    raise RuntimeError("no multi-categorical outcomes are allowed for potential outcomes")
                base_dim = 1
                out_mu = out_mu.view(-1, 2, base_dim)  # [B, 2, base_dim]
    
                if i in self.binary_output_idx:
                    probs = torch.sigmoid(out_mu).squeeze(-1)  # [B, 2]
                    # record means for stats
                    y0_means.append(probs[:, :1])
                    y1_means.append(probs[:, 1:2])
                    # output sampling/threshold
                    if deterministic:
                        out = probs if return_probs else (probs > 0.5).float()
                    else:
                        out = probs if return_probs else torch.bernoulli(probs)
                        # out = probs # to generate probabilities
                    outputs.append(out)  # [B,2]
    
                else:
                    # continuous scalar for both potential outcomes
                    mu_raw = out_mu.squeeze(-1)  # [B,2]
                    mu = self._activate_continuous_mean(i, mu_raw) # enforce bounds on mean
                    # record means for stats
                    y0_means.append(mu[:, :1])
                    y1_means.append(mu[:, 1:2])
                    ## variance
                    if str(i) in self.var_heads:
                        lv = self.var_heads[str(i)](w).view(-1, 2)  # [B,2]
                        lv = torch.tanh(lv) * self.max_bound
                        # lv = self._logvar_from_raw(lv)
                        if deterministic:
                            out = mu
                        else:
                            std = torch.exp(0.5 * lv)
                            out = mu + std * torch.randn_like(mu)
                    else:
                        out = mu  # fallback to mean
                    out = self._clamp_continuous(i, out)
                    outputs.append(out)  # [B,2]

            else:
            # ----------------------------- Non-PO mode -----------------------------
                if i in self.binary_output_idx:
                    probs = torch.sigmoid(out_mu)  # [B,1]
                    if deterministic:
                        out = probs if return_probs else (probs > 0.5).float()
                    else:
                        out = probs if return_probs else torch.bernoulli(probs)  # [B,2]
                    outputs.append(out if out.dim() == 2 else out.unsqueeze(-1))
    
                elif i in self.categorical_output_idx:
                    probs = F.softmax(out_mu, dim=-1)  # [B,K]
                    if deterministic:
                        idx = torch.argmax(probs, dim=-1, keepdim=True).float()  # [B,1]
                    else:
                        idx = torch.multinomial(probs, 1).float()  # [B,1]
                    outputs.append(idx)
    
                else:
                    # continuous scalar
                    mu_raw = out_mu  # [B,1]
                    mu = self._activate_continuous_mean(i, mu_raw) # enforce bounds on mean
                    if str(i) in self.var_heads:
                        lv = self.var_heads[str(i)](w)  # [B,1]
                        lv = torch.tanh(lv) * self.max_bound
                        # lv = self._logvar_from_raw(lv)
                        if deterministic:
                            out = mu
                        else:
                            std = torch.exp(0.5 * lv)
                            out = mu + std * torch.randn_like(mu)
                    else:
                        out = mu
                    out = self._clamp_continuous(i, out)
                    outputs.append(out if out.dim() == 2 else out.unsqueeze(-1))
    
        # Pack like sample(): concat along feature axis
        y_hat = torch.cat(outputs, dim=1)

        # ------------------------ Stats (PO mode only) ------------------------
        if self.potential_outcome and return_stats:
            # Means for the current (unflipped) pass
            y0m = torch.cat(y0_means, dim=1)  # [B, V]
            y1m = torch.cat(y1_means, dim=1)  # [B, V]
    
            # ATE components: ITE = E[Y(1)|X] - E[Y(0)|X]
            ite = (y1m - y0m)                          # [B, V]
            ate_per_var = ite.mean(dim=0)              # [V]
            # ate_overall = ate_per_var.mean()           # scalar

            # interleave y0m and y1m
            y_means = torch.stack([y0m, y1m], dim=2)   # [B, V, 2]
            y_means = y_means.reshape(y0m.size(0), -1) # [B, 2*V]
    
            # Build y_hat_prime by flipping T in x and reusing SAME decoder/heads
            y_means_prime, bias_info = self.compute_unmeasured_bias(z = z, x = x, y0_means = y0m, y1_means = y1m)
            stats = {
                "ite": (y1m - y0m),
                "ate": ate_per_var,
                **bias_info,
            }
            return y_hat, y_means, y_means_prime, stats
    
        return y_hat

    @torch.no_grad()
    def generate_gauss(self, x, n_samples=None, deterministic=False, return_probs=False, return_stats=True):
        """
        Conditional sampling with a standard Gaussian prior:
          z ~ N(0, I)
    
        Args:
            x              : [B, Xdim] conditioning features (same preprocessing as training)
            n_samples      : int, required only when x is None.
            deterministic  : if True, uses means/thresholds instead of sampling
            return_probs   : for binary vars, if True return probabilities instead of 0/1 samples
            return_stats   : if True (PO mode only), also returns a dict with:
                - 'ite' : [B, V] = E[Y(1)|X] - E[Y(0)|X]
                - 'ate' : [V]
                - 'ate_overall' : scalar [removed]
                - 'bias' : [B, V] = T*(y0 - y0') + (1-T)*(y1' - y1)
                - 'avg_bias' : [V]
                - 'avg_bias_overall' : scalar [removed]
            max_bound: fallback variance bound if self.max_bound not set.
    
        Returns:
            If return_stats and self.potential_outcome:
                (y_hat, y_means, y_means_prime, stats)
            else:
                y_hat
            where:
              - PO=False: y_hat has shape [B, V] (bin/cont scalars or class index)
              - PO=True : y_hat has shape [B, 2*V] (per-var [y0, y1] packed)
                          y_means (y_means_prime): probabilities for binary outcomes or mean for continuous outcomes given the observed treatment (flipped treatment)
        """
        self.eval()

        # ----- Determine batch size & build latent + conditioning -----
        if x is not None and x.numel() > 0:
            x = x.to(self.device)
            B = x.shape[0]
        elif n_samples is not None:
            B = n_samples
            if self.con_dim > 0:
                raise ValueError("This model expects a conditioning tensor x, but none was provided.")
            x = None
        else:
            raise ValueError("Must specify x or n_samples if no conditioning x is provided.")

        # ----- Sample latent z -----
        mu_zeros = torch.zeros(B, self.latent_dim, device=self.device)
        logvar_zeros = torch.zeros(B, self.latent_dim, device=self.device)
        e = torch.randn_like(mu_zeros)
        z = mu_zeros + torch.exp(0.5 * logvar_zeros) * e
        z_ = torch.cat((z, x), dim=1) if (x is not None and self.con_dim > 0) else z
        
        w = self.decoder(z_)   # shared trunk for heads
    
        outputs = []
    
        if self.potential_outcome:
            # For stats in PO mode: collect per-var means (probabilities for binary, means for continuous)
            y0_means, y1_means = [], []
            # Per your spec, PO mode disallows categoricals
            if len(self.categorical_output_idx) > 0:
                raise RuntimeError("No categorical outcomes allowed in potential_outcome mode.")

            # PO branch
            for i in range(self.in_dim):
                out_mu = self.output_heads[str(i)](w).view(-1, 2)  # [B,2]
    
                if i in self.binary_output_idx:
                    probs = torch.sigmoid(out_mu)  # [B,2]
                    # record means for stats
                    y0_means.append(probs[:, :1])
                    y1_means.append(probs[:, 1:2])
                    if deterministic:
                        out = probs if return_probs else (probs > 0.5).float()
                    else:
                        out = probs if return_probs else torch.bernoulli(probs)  # [B,2]
                else:  # continuous scalar
                    mu = self._activate_continuous_mean(i, out_mu)  # [B,2]
                    # record means for stats
                    y0_means.append(mu[:, :1])
                    y1_means.append(mu[:, 1:2])

                    if str(i) in self.var_heads:
                        lv = self.var_heads[str(i)](w).view(-1, 2)
                        lv = torch.tanh(lv) * self.max_bound 
                        std = torch.exp(0.5 * lv)
                        if deterministic:
                            out = mu
                        else:
                            out = mu + std * torch.randn_like(mu)
                    else:
                        out = mu
                    out = self._clamp_continuous(i, out)
                outputs.append(out)
    
        # ----------------------------- Non-PO mode -----------------------------
        else:
            # non-PO branch
            for i in range(self.in_dim):
                out_mu = self.output_heads[str(i)](w)
                if i in self.binary_output_idx:
                    probs = torch.sigmoid(out_mu)
                    if deterministic:
                        out = probs if return_probs else (probs > 0.5).float()
                    else:
                        out = probs if return_probs else torch.bernoulli(probs)  
                    outputs.append(out if out.dim() == 2 else out.unsqueeze(-1))
                elif i in self.categorical_output_idx:
                    probs = F.softmax(out_mu, dim=-1)
                    if deterministic:
                        idx = torch.argmax(probs, dim=-1, keepdim=True).float()
                    else:
                        # sample class index
                        idx = torch.multinomial(probs, 1).float()
                    outputs.append(idx)
                else:  # continuous scalar
                    mu = self._activate_continuous_mean(i, out_mu)
                    if str(i) in self.var_heads:
                        lv = self.var_heads[str(i)](w)
                        lv = torch.tanh(lv) * self.max_bound
                        std = torch.exp(0.5 * lv)
                        if deterministic:
                            out = mu                                   
                        else:
                            eps = torch.randn_like(mu)
                            out = mu + std * eps       
                    else:
                        out = mu
                    out = self._clamp_continuous(i, out)
                    outputs.append(out if out.dim() == 2 else out.unsqueeze(-1))
                    
        y_hat = torch.cat(outputs, dim=1)                       # [B, V]


        # ------------------------ Stats (PO mode only) ------------------------
        if self.potential_outcome and return_stats:
            # Means for the current (unflipped) pass
            y0m = torch.cat(y0_means, dim=1)  # [B, V]
            y1m = torch.cat(y1_means, dim=1)  # [B, V]
    
            # ATE components: ITE = E[Y(1)|X] - E[Y(0)|X]
            ite = (y1m - y0m)                          # [B, V]
            ate_per_var = ite.mean(dim=0)              # [V]
            # ate_overall = ate_per_var.mean()           # scalar

            # interleave y0m and y1m
            y_means = torch.stack([y0m, y1m], dim=2)   # [B, V, 2]
            y_means = y_means.reshape(y0m.size(0), -1) # [B, 2*V]
    
            # Build y_hat_prime by flipping T in x and reusing SAME decoder/heads
            y_means_prime, bias_info = self.compute_unmeasured_bias(z = z, x = x, y0_means = y0m, y1_means = y1m)
            stats = {
                "ite": (y1m - y0m),
                "ate": ate_per_var,
                **bias_info,
            }
            return y_hat, y_means, y_means_prime, stats

        
        return y_hat
    
        
    @torch.no_grad()
    def overlap_diagnostics(self, X_gen: torch.Tensor, T_gen: torch.Tensor,
                            tol: float = 0.25, bins: int = 50, eps: float = 0.05,
                            fit_propensity: bool = True):
        """
        Evaluate achieved overlap on generated samples.
    
        Parameters
        ----------
        X_gen : torch.Tensor  [B, in_dim]  -- generated X (these are your Ynames)
        T_gen : torch.Tensor  [B] or [B,1] -- generated T in {0,1}
        tol   : float                      -- tolerance band for |Δ - α| (decoder-based)
        bins  : int                        -- #bins for propensity histogram overlap
        eps   : float                      -- 'positivity' band for e(x) in [eps, 1-eps]
                                               eps defines how far from 0 and 1 we consider acceptable for positivity
        fit_propensity : bool              -- fit a quick logistic to get e(x)
    
        Returns
        -------
        metrics : dict of scalars
        """
    
        device = next(self.parameters()).device
        self.eval()
    
        # --- Prepare tensors ---
        y = X_gen.to(device).view(X_gen.size(0), -1)             # what decoder models
        t = T_gen.view(-1).to(device).float()                    # [B]
        x_cond = t.unsqueeze(1)                                   # since Xnames = T
    
        # --- Decoder-based Δ(x) diagnostics (closest to your loss) ---
        # use posterior mean for stability (no sampling)
        l = self.encoder(y)
        mu = self.en_mu(l)
        z = mu
    
        logp0 = self._log_p_y_given_T(y, z, x_cond, t_scalar=0)   # [B]
        logp1 = self._log_p_y_given_T(y, z, x_cond, t_scalar=1)   # [B]
        log_ratio_t = (logp0 - logp1)      # Δ(x)
        #alpha = getattr(self, "overlap_target", 0.0)
        alpha_t = self._overlap_target_from_y(y, ref=log_ratio_t, detach=True)

        log_ratio = log_ratio_t.detach().cpu().numpy()  
        alpha = alpha_t.detach().cpu().numpy()
    
        dec_mse_to_target  = float(np.mean((log_ratio - alpha) ** 2))
        # dec_mae_to_target  = float(np.mean(np.abs(log_ratio - alpha)))
        dec_band_coverage  = float(np.mean(np.abs(log_ratio - alpha) <= tol))
        dec_mean           = float(np.mean(log_ratio))
        dec_std            = float(np.std(log_ratio))
        target_mean        = float(np.mean(alpha))
        target_std         = float(np.std(alpha))
    
        metrics = {
            "dec/log_ratio_mean": round(dec_mean, 3),
            "dec/log_ratio_std":  round(dec_std, 3),
            "dec/target_mean": round(target_mean, 3),
            "dec/target_std": round(target_std, 3),
            "dec/mse_to_target":  round(dec_mse_to_target, 3),
            # "dec/mae_to_target":  dec_mae_to_target,
            "dec/fraction_within_tol": round(dec_band_coverage, 3),
        }
    
        # --- Propensity-based diagnostics (distributional overlap of e(x)) ---
        if fit_propensity:
            X_np = X_gen.detach().cpu().numpy()
            T_np = T_gen.view(-1).detach().cpu().numpy()

            clf = LogisticRegression(max_iter=2000, solver="lbfgs")
            clf.fit(X_np, T_np)
            e_hat = clf.predict_proba(X_np)[:, 1]
            auc  = float(roc_auc_score(T_np, e_hat)) 
            # auc measures how well treatment is separable by covariates
            ## AUC: 0.5 - strong overlap (treatment assignment is almost random)
    
            e0 = e_hat[T_np == 0]
            e1 = e_hat[T_np == 1]
    
            # Histogram overlap coefficient  ∫ min(f0, f1) de  \in [0,1]
            h0, edges = np.histogram(e0, bins=bins, range=(0, 1), density=True)
            h1, _     = np.histogram(e1, bins=bins, range=(0, 1), density=True)
            bin_w = edges[1] - edges[0]
            overlap_coeff = float(np.sum(np.minimum(h0, h1)) * bin_w)
    
            # Positivity / common support mass<}: "Common support" = where both treated and control units have non-extreme propensity scores (not near 0 or 1).
            in_band = (e_hat > eps) & (e_hat < 1 - eps)
            frac_common_support = float(np.mean(in_band)) # Overall fraction of samples in the common support
            frac_common_support_t0 = float(np.mean(in_band[T_np == 0])) if np.any(T_np==0) else 0.0 # Fraction of untreated units with overlap
            frac_common_support_t1 = float(np.mean(in_band[T_np == 1])) if np.any(T_np==1) else 0.0 # Fraction of treated units with overlap
            '''
            # Range-overlap ratio on e(x): the fraction of the unioned range that overlaps.
            ## this is not used, since it is sensitive to sample size and outliers, ingoring the mass and optimistic
            lo0, hi0 = float(np.min(e0)), float(np.max(e0))
            lo1, hi1 = float(np.min(e1)), float(np.max(e1))
            inter = max(0.0, min(hi0, hi1) - max(lo0, lo1))
            union = max(hi0, hi1) - min(lo0, lo1) + 1e-12
            range_overlap = float(inter / union)
    
            # Overlap index from AUC: 1 at perfect overlap (AUC=0.5), 0 at perfect separation (AUC=1)
            ## this is not used, since it is a simple rescaling of AUC
            overlap_index = float(2.0 * (1.0 - auc))  # maps [0.5,1] -> [1,0]
            ''' 
            metrics.update({
                "ps/auc": round(auc, 3),
                #"ps/overlap_index": overlap_index,
                "ps/hist_overlap_coeff": round(overlap_coeff, 3),
                "ps/frac_common_support": round(frac_common_support, 3),
                "ps/frac_common_support_t0": round(frac_common_support_t0, 3),
                "ps/frac_common_support_t1": round(frac_common_support_t1, 3),
                #"ps/range_overlap": range_overlap,
            })
    
        return metrics
