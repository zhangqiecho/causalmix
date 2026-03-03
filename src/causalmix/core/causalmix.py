# updated CausalMix class Nov/25/2025
# import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import tqdm
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
# import conVAE
# import fit_metadata, preprocess_with_met,

from ..models.convae import conVAE
from ..data.preprocess import fit_metadata, preprocess_with_meta, postprocess_generated
from ..eval.causal_eval import CausalEval
from ..eval.synth_eval import SynthEval
from ..data.schema import DataSchema

class CausalMix:
    def __init__(
        self,
        data,  # dataframe
        post_treatment_var,  # list of post treatment variables
        treatment_var,  # list of treatment variable(s)
        binary_var=[], # list of variables which are binary
        categorical_var=[],  # list of variables which are categorical
        numerical_var=[],  # list of variables which are numerical
        integer_var=[], # list of continuous variables stored as integers
        categorical_dims={},      # dict {col_name: num_classes}
        var_bounds={},  # dictionary of bounds if certain variable is bounded   
    ):
        self.data_raw = data
        self.Ynames = post_treatment_var
        self.Tnames = treatment_var

        self.binary_var = binary_var
        self.categorical_var = categorical_var
        self.categorical_dims = categorical_dims
        self.numerical_var = numerical_var
        self.integer_var = integer_var
        
        # preprocess data
        self.data_meta = fit_metadata(
            df = self.data_raw, 
            numerical_var = self.numerical_var, 
            binary_var = self.binary_var,
            categorical_var = self.categorical_var, 
            integer_numerical_var = self.integer_var
        )
        self.data_processed = preprocess_with_meta(df = self.data_raw, meta = self.data_meta)
        # set the var_bounds for numerical_var
        if var_bounds == {}:
            self.var_bounds = {
                var: {
                    "lower": self.data_processed[var].min(),
                    "upper": self.data_processed[var].max()
                }
                for var in self.numerical_var
            }
        else:
            # standardize var_bounds based on self.data_meta
            self.var_bounds = {}
            for var, bounds in var_bounds.items():
                if var in self.numerical_var:
                    mu = self.data_meta["num_mean"][var]
                    sigma = self.data_meta["num_std"][var]
                
                    self.var_bounds[var] = {
                        "lower": (bounds["lower"] - mu) / sigma,
                        "upper": (bounds["upper"] - mu) / sigma,
                }
        
        self.Xnames = [
            x for x in self.data_processed.columns if x not in self.Ynames + self.Tnames
        ]
        # update January 13, 2026 to include the index of the covariates for function specification
        self.index = VarIndex(self.Xnames)

        self.X_latent = len(self.Xnames)
        self.Y_latent = len(self.Ynames) * 2
        self.T_latent = len(self.Tnames)

    # train generator
    def fit(
        self,
        # latent_dim=4,
        hidden_dim=[64],
        batch_size=10,
        treatment_effect_fn=lambda x, index: 0,
        selection_bias_fn=lambda x, t, index: 0,
        overlap_target= lambda x, index: 0,
        effect_rigidity=1e3,
        effect_mse_weight=0.5, # effect blend weight on MSE
        effect_var_weight=1e-2, # variance penalty for treatment effect residuals
        bias_rigidity=1e3,
        bias_mse_weight=0.5, # bias blend weight on MSE
        bias_var_weight=5e-3, # variance penalty for bias residuals
        overlap_weight = 1e1, # 1e2 is good for a constant value of the overlap_target; 1e1 is good for a function of the overlap_target (even if it is constant result); 
        kld_rigidity=0.1,
        max_epochs=1000,
        accelerator="auto",
        precision='32-true', # use 'bf16-mixed' for better performance on laptop, otherwise use "32-true" for full precision
    ):

        # generator for T
        self.m_treat = self.data_processed[self.Tnames].mean()
        '''
        # alternatively, we can use conVAE to fit T, which may be time comsuming
        self.m_treat = conVAE(
            df=self.data_processed,
            Xnames=[],
            Ynames=self.Tnames,
            binary_cols=self.binary_var,
            categorical_cols=self.categorical_var,      
            categorical_dims=self.categorical_dims,      # dict {col_name: num_classes}
            var_bounds=self.var_bounds,
            latent_dim=self.T_latent,
            hidden_dim=hidden_dim,
        ) 
        self.m_treat.fit_model(precision = precision, max_epochs = max_epochs)
        # fit the bgmm for treatment
        self.m_treat.fit_bgmm_from_training_latents()
        '''
        # generator for X | T
        # update 11/5/2025
        self.m_pre = conVAE(
            df=self.data_processed,
            Xnames=self.Tnames,
            Ynames=self.Xnames,
            treatment_cols = self.Tnames, # fixed January 17,2026: added the omitted treatment column
            binary_cols=self.binary_var, # cat_cols=self.categorical_var,
            categorical_cols=self.categorical_var,      # QZ: list of categorical columns in Ynames
            categorical_dims=self.categorical_dims,      # QZ: dict {col_name: num_classes}
            var_bounds=self.var_bounds,
            latent_dim=self.X_latent,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            potential_outcome=False,
            overlap_weight=overlap_weight,
            overlap_target= overlap_target
        ) 

        self.m_pre.fit_model(precision = precision, max_epochs = max_epochs)
        # fit the bgmm for X
        self.m_pre.fit_bgmm_from_training_latents()

        # generator for Y(1),Y(0) | X, T
        self.m_post = conVAE(
            df=self.data_processed,
            Xnames=self.Xnames + self.Tnames,
            Ynames=self.Ynames,
            binary_cols=self.binary_var,
            categorical_cols=self.categorical_var,      # list of categorical columns in Ynames
            categorical_dims=self.categorical_dims,      # dict {col_name: num_classes}
            var_bounds=self.var_bounds,
            latent_dim= self.Y_latent,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            potential_outcome=True,
            treatment_cols=self.Tnames,
            treatment_effect_fn=treatment_effect_fn,
            selection_bias_fn=selection_bias_fn,
            effect_rigidity=effect_rigidity,
            effect_mse_weight=effect_mse_weight, # effect blend weight on MSE
            effect_var_weight=effect_var_weight, # variance penalty for treatment effect residuals
            bias_rigidity=bias_rigidity,
            bias_mse_weight=bias_mse_weight, # bias blend weight on MSE
            bias_var_weight=bias_var_weight, # variance penalty for bias residuals
            kld_rigidity=kld_rigidity,
            overlap_weight=0.0,
        )  

        self.m_post.fit_model(precision = precision, max_epochs = max_epochs)
        # fit the bgmm for Y
        self.m_post.fit_bgmm_from_training_latents()
        
        # returning trained generators
        return [self.m_treat, self.m_pre, self.m_post]
    
    def post_sample(self, Tgen, Xgen, Ygen, Ymean, Ymean_prime):
        # wrapping in a dataframe
        df = pd.DataFrame(Xgen.detach().numpy(), columns=self.Xnames)
        df_T = pd.DataFrame(Tgen.detach().numpy(), columns=self.Tnames)
        # Create column names using Ynames
        Y_columns = [f"{y}{suffix}" for y in self.Ynames for suffix in [0, 1]]
        df_Y = pd.DataFrame(Ygen.detach().numpy(), columns=Y_columns)
        df_Ymean = pd.DataFrame(Ymean.detach().numpy(), columns=Y_columns)
        df_Ymean_prime = pd.DataFrame(Ymean_prime.detach().numpy(), columns=Y_columns)

        df = df.join(df_T).join(df_Y)
        # Create observed outcomes based on treatment T
        for y in self.Ynames:
            df[f"{y}"] = np.where(df[self.Tnames[0]] == 1, df[f"{y}1"], df[f"{y}0"])
        
        df_gen = postprocess_generated(y_gen = df, meta = self.data_meta)
        return df_gen, df_Y, df_Ymean, df_Ymean_prime
        
    # sample from generator
    def sample_bgmm(self, 
                    n_samples=1000, 
                    deterministic=False, 
                    return_probs = False, 
                    overlap = True, 
                    return_latent=False, #whether to return Xgen, Tgen, stats as-is
                   ): 
        # generate the treatment vars
        Tgen = torch.bernoulli(torch.full((n_samples, 1), float(self.m_treat)))
        '''
        # alternatively, generating T using conVAE, which may be time consuming
        Tgen = self.m_treat.generate_bgmm(x = None, n_samples = n_samples, deterministic=deterministic, return_probs = return_probs)
        '''
        Xgen = self.m_pre.generate_bgmm(x = Tgen, deterministic=deterministic, return_probs = return_probs)
        Ygen, Ymean, Ymean_prime, stats = self.m_post.generate_bgmm(x = torch.cat((Xgen, Tgen), 1), deterministic=deterministic, return_probs = return_probs)
        if overlap:
            overlap_metrics = self.m_pre.overlap_diagnostics(Xgen, Tgen)
            print("Overlap metrics:", {k: round(v, 4) for k, v in overlap_metrics.items()})

        df_gen, df_Y, df_Ymean, df_Ymean_prime = self.post_sample(Tgen = Tgen, Xgen = Xgen, Ygen = Ygen, Ymean = Ymean, Ymean_prime = Ymean_prime)
        if return_latent:
            # return tensors as well for downstream evaluation
            if overlap:
                return df_gen, df_Y, df_Ymean, df_Ymean_prime, stats, Xgen, Tgen, overlap_metrics
            else:
                return df_gen, df_Y, df_Ymean, df_Ymean_prime, stats, Xgen, Tgen
        else:
            if overlap:
                return df_gen, df_Y, df_Ymean, df_Ymean_prime, stats, overlap_metrics
            else:
                return df_gen, df_Y, df_Ymean, df_Ymean_prime, stats

    def causal_bgmm(
        self,
        n_samples=1000,
        deterministic=False,
        return_probs=False,
        plot=True,
        savepath=None,
    ):
        """
        Generate synthetic data via BGMM prior and evaluate
        treatment-effect, confounding, and overlap diagnostics.
        Returns a dict of metrics.
        """
        # 1) Generate + keep latent tensors
        df_gen, df_Y, df_Ymean, df_Ymean_prime, stats, Xgen, Tgen, overlap_metrics = self.sample_bgmm(
            n_samples=n_samples,
            deterministic=deterministic,
            return_probs=return_probs,
            overlap=True,   # still uses model.overlap_diagnostics + plot_propensity_overlap
            return_latent=True,
        )

        # 2) Move to numpy
        X_np = Xgen.detach().cpu().numpy() # shape (N, V)
        T_np = Tgen.detach().cpu().numpy().reshape(-1) # shape (N, )

        # 3) Predicted τ and κ from m_post stats
        tau_pred = stats["ite"].detach().cpu().numpy()      # shape [N, 1] 
        kappa_pred = stats["bias"].detach().cpu().numpy()   # shape [N, 1]

        # update Jan 11 2026:
        tau_pred = np.squeeze(tau_pred) # shape (N, )
        kappa_pred = np.squeeze(kappa_pred) # shape (N, )
        # 4) Target τ(X), κ(X, T) from user’s functions f, g
        #    (these are exactly your effect / bias functions passed to conVAE)
        # update January 13, 2026 to include the index of the covariates for function specification and use tensors as the input
        # tau_target = np.asarray(self.m_post.f(X_np, self.index))         # should be broadcastable to tau_pred
        # kappa_target = np.asarray(self.m_post.g(X_np, T_np, self.index)) # should be broadcastable to kappa_pred
        def _to_numpy(x):
            """Convert torch.Tensor or scalar to numpy array safely."""
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            else:
                return np.asarray(x)
            
        tau_target = _to_numpy(self.m_post.f(Xgen, self.index))
        kappa_target = _to_numpy(self.m_post.g(Xgen, Tgen[:, 0], self.index)) # shape of Tgen[:,0] is [N]

        if tau_target.shape == ():
            tau_target = np.full(n_samples, float(tau_target))
        else:
            tau_target = np.squeeze(tau_target)

        if kappa_target.shape == ():
            kappa_target = np.full(n_samples, float(kappa_target))
        else:
            kappa_target = np.squeeze(kappa_target)

        # 5) Instantiate CausalEval (the class we sketched before)
        print("checking")
        print("X_np:", X_np.shape)
        print("T_np:", T_np.shape)
        print("tau_pred:", tau_pred.shape)
        print("kappa_pred:", kappa_pred.shape)
        print("tau_target:", np.asarray(tau_target).shape)
        print("kappa_target:", np.asarray(kappa_target).shape)

        ce = CausalEval(
            tau_pred=tau_pred,
            tau_target=tau_target,
            kappa_pred=kappa_pred,
            kappa_target=kappa_target,
            T=T_np,
            X=X_np
        )

        # 6) Collect everything
        metrics = ce.all_metrics(plot = plot, savepath = savepath)

        return {
            "df_gen": df_gen,
            "df_Y": df_Y,
            "df_Ymean": df_Ymean,
            "df_Ymean_prime": df_Ymean_prime,
            "stats": stats,
            "causal_metrics": metrics,
            "overlap": overlap_metrics
        }


    def sample_gauss(self, 
                     n_samples=1000, 
                     deterministic=False, 
                     return_probs = False,
                     overlap = True,
                     return_latent = False):
        # generate the treatment vars
        Tgen = torch.bernoulli(torch.full((n_samples, 1), float(self.m_treat)))
        '''
        Tgen = self.m_treat.generate_gauss(x = None, n_samples = n_samples, deterministic=deterministic, return_probs = return_probs)
        '''
        Xgen = self.m_pre.generate_gauss(x = Tgen, deterministic=deterministic, return_probs = return_probs)
        Ygen, Ymean, Ymean_prime, stats = self.m_post.generate_gauss(x = torch.cat((Xgen, Tgen), 1), deterministic=deterministic, return_probs = return_probs)
        #df_gen, df_Y, df_Ymean, df_Ymean_prime = self.post_sample(Tgen = Tgen, Xgen = Xgen, Ygen = Ygen, Ymean = Ymean, Ymean_prime = Ymean_prime)
        # added 12/23/2025
        if overlap:
            overlap_metrics = self.m_pre.overlap_diagnostics(Xgen, Tgen)
            print("Overlap metrics:", {k: round(v, 4) for k, v in overlap_metrics.items()})
        df_gen, df_Y, df_Ymean, df_Ymean_prime = self.post_sample(Tgen = Tgen, Xgen = Xgen, Ygen = Ygen, Ymean = Ymean, Ymean_prime = Ymean_prime)
        if return_latent:
            # return tensors as well for downstream evaluation
            if overlap:
                return df_gen, df_Y, df_Ymean, df_Ymean_prime, stats, Xgen, Tgen, overlap_metrics
            else:
                return df_gen, df_Y, df_Ymean, df_Ymean_prime, stats, Xgen, Tgen
        else:
            if overlap:
                return df_gen, df_Y, df_Ymean, df_Ymean_prime, stats, overlap_metrics
            else:
                return df_gen, df_Y, df_Ymean, df_Ymean_prime, stats

    def causal_gauss(
        self,
        n_samples=1000,
        deterministic=False,
        return_probs=False,
        plot=True,
        savepath=None,
    ):
        """
        Generate synthetic data via standard gaussian distributions and evaluate
        treatment-effect, confounding, and overlap diagnostics.
        Returns a dict of metrics.
        """
        # 1) Generate + keep latent tensors
        df_gen, df_Y, df_Ymean, df_Ymean_prime, stats, Xgen, Tgen, overlap_metrics = self.sample_gauss(
            n_samples=n_samples,
            deterministic=deterministic,
            return_probs=return_probs,
            overlap=True,   # still uses model.overlap_diagnostics + plot_propensity_overlap
            return_latent=True,
        )

        # 2) Move to numpy
        X_np = Xgen.detach().cpu().numpy()
        T_np = Tgen.detach().cpu().numpy().reshape(-1)

        # 3) Predicted τ and κ from m_post stats
        tau_pred = stats["ite"].detach().cpu().numpy()      # shape [N, V] or [N]
        kappa_pred = stats["bias"].detach().cpu().numpy()   # shape [N, V] or [N]

        # update Jan 11 2026:
        tau_pred = np.squeeze(tau_pred)
        kappa_pred = np.squeeze(kappa_pred)
        # 4) Target τ(X), κ(X, T) from user’s functions f, g
        #    (these are exactly your effect / bias functions passed to conVAE)
        # update January 13, 2026 to include the index of the covariates for function specification
        # tau_target = np.asarray(self.m_post.f(X_np, self.index))         # should be broadcastable to tau_pred
        # kappa_target = np.asarray(self.m_post.g(X_np, T_np, self.index)) # should be broadcastable to kappa_pre
        def _to_numpy(x):
            """Convert torch.Tensor or scalar to numpy array safely."""
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            else:
                return np.asarray(x)
            
        tau_target = _to_numpy(self.m_post.f(Xgen, self.index))
        kappa_target = _to_numpy(self.m_post.g(Xgen, Tgen[:, 0], self.index)) # shape of Tgen[:,0] is [N]
        if tau_target.shape == ():
            tau_target = np.full(n_samples, float(tau_target))
        else:
            tau_target = np.squeeze(tau_target)

        if kappa_target.shape == ():
            kappa_target = np.full(n_samples, float(kappa_target))
        else:
            kappa_target = np.squeeze(kappa_target)

        # 5) Instantiate CausalEval (the class we sketched before)
        ce = CausalEval(
            tau_pred=tau_pred,
            tau_target=tau_target,
            kappa_pred=kappa_pred,
            kappa_target=kappa_target,
            T=T_np,
            X=X_np
        )

        # 6) Collect everything
        metrics = ce.all_metrics(plot = plot, savepath = savepath)

        return {
            "df_gen": df_gen,
            "df_Y": df_Y,
            "df_Ymean": df_Ymean,
            "df_Ymean_prime": df_Ymean_prime,
            "stats": stats,
            "causal_metrics": metrics,
            "overlap": overlap_metrics
        }
