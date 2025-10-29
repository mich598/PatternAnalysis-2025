"""
“modules.py" containing the source code of the components of your model. Each component must be
implementated as a class or a function

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Github Link: https://github.com/jsyoon0823/TimeGAN/blob/master/data_loading.py

Based on timegan.py code
"""

"""
modules.py — Optimized TimeGAN Components and Training
Reference: Jinsung Yoon et al., "Time-series Generative Adversarial Networks" (NeurIPS 2019)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import extract_time, random_generator, batch_generator

import torch.backends.cudnn as cudnn
cudnn.enabled = False  # avoid CuDNN double-backward errors

# -------------------------
# Helper utilities
# -------------------------
def create_rnn(module_name, input_size, hidden_size, num_layers, batch_first=True):
    module_name = module_name.lower()
    if module_name in ("gru", "rnn-gru"):
        return nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first)
    elif module_name in ("lstm", "rnn-lstm"):
        return nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
    else:
        return nn.RNN(input_size, hidden_size, num_layers, nonlinearity="tanh", batch_first=batch_first)

def sequence_mask(lengths, max_len=None, device=None):
    if max_len is None:
        max_len = lengths.max().item()
    batch_size = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=device).unsqueeze(0).expand(batch_size, -1)
    return seq_range < lengths.unsqueeze(1)


# -------------------------
# Model blocks
# -------------------------
class Embedder(nn.Module):
    def __init__(self, module_name, input_dim, hidden_dim, num_layers, dropout_rate=0.3):
        super().__init__()
        self.rnn = create_rnn(module_name, input_dim, hidden_dim, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self.rnn(x)
        out = self.norm(out)
        out = self.fc(out)
        return out


class Recovery(nn.Module):
    def __init__(self, module_name, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.rnn = create_rnn(module_name, hidden_dim, hidden_dim, num_layers)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, h):
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self.rnn(h)
        out = self.fc(out)
        return out


class Generator(nn.Module):
    def __init__(self, module_name, z_dim, hidden_dim, num_layers, dropout_rate=0.1):
        super().__init__()
        self.rnn = create_rnn(module_name, z_dim, hidden_dim, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self.rnn(z)
        out = self.fc(out)
        return out


class Supervisor(nn.Module):
    def __init__(self, module_name, hidden_dim, num_layers_minus1, dropout_rate=0.1):
        super().__init__()
        self.rnn = create_rnn(
            module_name, hidden_dim, hidden_dim, max(num_layers_minus1, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self.rnn(z)
        out = self.fc(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, module_name, hidden_dim, num_layers):
        super().__init__()
        self.rnn = create_rnn(module_name, hidden_dim, hidden_dim, num_layers)
        # speed: remove spectral norm
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        out, _ = self.rnn(h)  # CuDNN ok here
        out = self.fc(out)
        return out

# -------------------------
# TimeGAN training
# -------------------------
def timegan(ori_data, parameters, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ori_data_np = np.asarray(ori_data)
    no, seq_len, dim = ori_data_np.shape
    ori_time, max_seq_len = extract_time(ori_data)
    ori_time = np.asarray(ori_time, dtype=np.int64)

    # normalization
    def MinMaxScaler(data):
        min_val = np.min(np.min(data, 0), 0)
        data_adj = data - min_val
        max_val = np.max(np.max(data_adj, 0), 0)
        return data_adj / (max_val + 1e-7), min_val, max_val

    ori_data_norm, min_val, max_val = MinMaxScaler(ori_data_np)

    hidden_dim = parameters["hidden_dim"]
    num_layers = parameters["num_layer"]
    iterations = parameters["iterations"]
    batch_size = parameters["batch_size"]
    module_name = parameters["module"]
    z_dim = 32
    gamma = 1

    # hyperparams
    lr = 5e-5
    lr_supervised = parameters.get("lr_supervised", 2e-4)
    iterations_supervise = parameters.get("iterations_supervise", iterations * 2)
    beta1, beta2 = 0.4, 0.9
    lambda_stats = 300.0
    inst_noise_std = 0.03
    real_label_smooth = 0.85

    # networks
    embedder = Embedder(module_name, dim, hidden_dim, num_layers).to(device)
    recovery = Recovery(module_name, hidden_dim, dim, num_layers).to(device)
    generator = Generator(module_name, z_dim, hidden_dim, num_layers).to(device)
    supervisor = Supervisor(module_name, hidden_dim, num_layers - 1, 0.1).to(device)
    discriminator = Discriminator(module_name, hidden_dim, num_layers).to(device)

    # optimizers
    E0_optimizer = optim.Adam(
        list(embedder.parameters()) + list(recovery.parameters()), lr=lr, betas=(beta1, beta2)
    )
    E_optimizer = optim.Adam(
        list(embedder.parameters()) + list(recovery.parameters()), lr=lr, betas=(beta1, beta2)
    )
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr * 0.1, betas=(0.4, 0.9))
    G_optimizer = optim.Adam(list(generator.parameters()) + list(supervisor.parameters()), lr=lr * 2.0, betas=(0.4, 0.9))

    bce_logits = nn.BCEWithLogitsLoss(reduction="none")
    mse_loss = nn.MSELoss(reduction="none")

    def to_torch(x_np, t_np):
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
        t_t = torch.tensor(t_np, dtype=torch.int64, device=device)
        mask = sequence_mask(t_t, max_len=x_t.size(1), device=device).unsqueeze(-1).float()
        return x_t, t_t, mask

    # 1. Embedding phase
    print("Start Embedding Network Training")
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data_norm, ori_time, batch_size)
        X_mb_t, T_mb_t, mask = to_torch(X_mb, T_mb)
        embedder.train(); recovery.train()
        E0_optimizer.zero_grad()
        H = embedder(X_mb_t)
        X_tilde = recovery(H)
        loss = (mse_loss(X_tilde, X_mb_t) * mask).sum() / mask.sum().clamp_min(1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(embedder.parameters()) + list(recovery.parameters()), 1.0)
        E0_optimizer.step()
        if itt % 500 == 0:
            print(f"step:{itt}/{iterations}, e_loss:{np.sqrt(loss.item()):.4f}")
    print("Finish Embedding Network Training")

    print("Start Supervised Loss Training")
    # Use a dedicated optimizer for Supervisor only (not G) to focus learning.
    S_optimizer = optim.Adam(list(supervisor.parameters()), lr=parameters.get("lr_supervised", 2e-3), betas=(0.4, 0.9))

    def masked_mse(a, b, m):
        # a,b: (B,T,H), m: (B,T,1) float {0,1}
        return ((a - b) ** 2 * m).sum() / m.sum().clamp_min(1.0)

    iterations_supervise = parameters.get("iterations_supervise", iterations * 4)  # give it more steps

    for itt in range(iterations_supervise):
        X_mb, T_mb = batch_generator(ori_data_norm, ori_time, batch_size)
        X_mb_t, T_mb_t, mask = to_torch(X_mb, T_mb)

        # freeze E during this phase
        embedder.eval()
        with torch.no_grad():
            H_real = embedder(X_mb_t)  # (B,T,H)

        # predict next latent from current latent
        supervisor.train()
        S_optimizer.zero_grad()

        H_pred = supervisor(H_real)  # (B,T,H)

        # next-step target on valid timesteps (T-1 of them)
        valid = sequence_mask(T_mb_t - 1, max_len=max_seq_len - 1, device=device).unsqueeze(-1).float()
        loss_s = masked_mse(H_real[:, 1:, :], H_pred[:, :-1, :], valid)
        loss_s.backward()
        torch.nn.utils.clip_grad_norm_(supervisor.parameters(), 1.0)
        S_optimizer.step()

        if itt % 500 == 0:
            print(f"supervised step:{itt}/{iterations_supervise}, s_loss:{loss_s.item()**0.5:.4f}")  # sqrt to match your display

    print("Finish Supervised Training")

    print("Extra fine-tuning for Supervisor alignment")
    for itt in range(1000):
        X_mb, T_mb = batch_generator(ori_data_norm, ori_time, batch_size)
        X_mb_t, T_mb_t, mask = to_torch(X_mb, T_mb)
        embedder.eval()
        with torch.no_grad():
            H_real = embedder(X_mb_t)
        supervisor.train()
        S_optimizer.zero_grad()
        H_pred = supervisor(H_real)
        valid = sequence_mask(T_mb_t - 1, max_len=max_seq_len - 1, device=device).unsqueeze(-1).float()
        loss_s = masked_mse(H_real[:, 1:, :], H_pred[:, :-1, :], valid)
        loss_s.backward()
        torch.nn.utils.clip_grad_norm_(supervisor.parameters(), 1.0)
        S_optimizer.step()
    print("Finished fine-tuning supervisor.")

    # -------------------------
    # 3. Joint Training (simplified, TF-style)
    # -------------------------
    print("Start Joint Training")

    # --- EMA tracking for smoothed logging ---
    ema_g_loss_u = None
    decay = 0.9  # 0.9–0.95 works best

    for itt in range(iterations):

        # -------------------------
        # Generator/Embedder training (twice per outer iter)
        # -------------------------
        for kk in range(2):
            # Mini-batch
            X_mb, T_mb = batch_generator(ori_data_norm, ori_time, batch_size)
            X_mb_t, T_mb_t, mask = to_torch(X_mb, T_mb)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            Z_mb_t = torch.tensor(Z_mb, dtype=torch.float32, device=device)

            # --- G (generator + supervisor) step ---
            generator.train(); supervisor.train(); embedder.train(); recovery.train()
            G_optimizer.zero_grad()

            # forward: G->S->R
            E_hat = generator(Z_mb_t)          # (B, T, H)
            H_hat = supervisor(E_hat)          # (B, T, H)
            X_hat = recovery(H_hat)            # (B, T, D)

            # discriminator logits for fake paths
            D_fake   = discriminator(H_hat)    # (B, T, 1)
            D_fake_e = discriminator(E_hat)    # (B, T, 1)

            # G_loss_U (unsupervised/adversarial on D)
            ones = torch.ones_like(D_fake, device=device)
            g_loss_u   = (bce_logits(D_fake,   ones) * mask).sum() / mask.sum().clamp_min(1.0)
            g_loss_u_e = (bce_logits(D_fake_e, ones) * mask).sum() / mask.sum().clamp_min(1.0)

            # G_loss_S (supervised: next-step in latent space)
            with torch.no_grad():
                H_real = embedder(X_mb_t)  # target latent
            valid_mask = sequence_mask(T_mb_t - 1, max_len=max_seq_len - 1, device=device).unsqueeze(-1).float()
            H_target = H_real[:, 1:, :]
            H_pred   = supervisor(H_real)[:, :-1, :]
            g_loss_s = (mse_loss(H_target * valid_mask, H_pred * valid_mask)).sum() / valid_mask.sum().clamp_min(1.0)

            # G_loss_V (two-moment matching on X)
            X_hat_m = X_hat * mask
            X_real_m = X_mb_t * mask
            mean_diff = torch.mean(torch.abs(X_hat_m.mean(dim=0) - X_real_m.mean(dim=0)))
            std_diff  = torch.mean(torch.abs(torch.sqrt(X_hat_m.var(dim=0, unbiased=False) + 1e-6) -
                                            torch.sqrt(X_real_m.var(dim=0, unbiased=False) + 1e-6)))
            g_loss_v = mean_diff + std_diff

            # total generator loss (match original weighting)
            total_g_loss = 1.0 * (g_loss_u + gamma * g_loss_u_e) + 100.0 * torch.sqrt(g_loss_s + 1e-8) + 100.0 * g_loss_v

            total_g_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(generator.parameters()) + list(supervisor.parameters()), 1.0)
            G_optimizer.step()

            # keep last values for logging
            step_g_loss_u  = g_loss_u.detach().item()

            # --- EMA smoothing for generator unsupervised loss ---
            if ema_g_loss_u is None:
                ema_g_loss_u = step_g_loss_u
            else:
                ema_g_loss_u = decay * ema_g_loss_u + (1 - decay) * step_g_loss_u

            step_g_loss_s  = g_loss_s.detach().item()
            step_g_loss_v  = g_loss_v.detach().item()

            # --- E (embedder + recovery) step ---
            E_optimizer.zero_grad()
            H = embedder(X_mb_t)
            X_tilde = recovery(H)

            # E_loss_T0 (reconstruction on X)
            E_loss_T0 = (mse_loss(X_tilde, X_mb_t) * mask).sum() / mask.sum().clamp_min(1.0)
            E_loss0 = 10.0 * torch.sqrt(E_loss_T0 + 1e-8)

            # small supervised term to align E with S
            H_hat_sup = supervisor(H)
            H_left  = H[:, 1:, :]
            H_right = H_hat_sup[:, :-1, :]
            G_loss_S_val = (mse_loss(H_left * valid_mask, H_right * valid_mask)).sum() / valid_mask.sum().clamp_min(1.0)

            E_loss = E_loss0 + 0.1 * G_loss_S_val
            E_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(embedder.parameters()) + list(recovery.parameters()), 1.0)
            E_optimizer.step()

            step_e_loss_t0 = E_loss_T0.detach().item()

        # -------------------------
        # Discriminator training (only if needed)
        # -------------------------
        X_mb, T_mb = batch_generator(ori_data_norm, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Z_mb_t = torch.tensor(Z_mb, dtype=torch.float32, device=device)

        discriminator.train()
        noise_std = max(0.01, 0.05 * (1 - itt / iterations))
        with torch.no_grad():
            H_real = embedder(X_mb_t) + noise_std * torch.randn_like(H_real)
            E_hat  = generator(Z_mb_t) + noise_std * torch.randn_like(E_hat)
            H_hat  = supervisor(E_hat) + noise_std * torch.randn_like(H_hat)

        D_real = torch.clamp(discriminator(H_real), -10, 10)
        D_fake = torch.clamp(discriminator(H_hat), -10, 10)
        D_fake_e = torch.clamp(discriminator(E_hat), -10, 10)

        labels_real = torch.ones_like(D_real, device=device) * real_label_smooth
        labels_fake = torch.zeros_like(D_fake, device=device)

        # --- Label smoothing and flipping ---
        flip_prob = 0.05  # 5% chance to swap real/fake labels
        if torch.rand(1).item() < flip_prob:
            labels_real, labels_fake = labels_fake, labels_real

        def gradient_penalty(D, real, fake):
            alpha = torch.rand(real.size(0), 1, 1, device=real.device)
            interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
            out = D(interp)
            grad = torch.autograd.grad(outputs=out, inputs=interp,
                                    grad_outputs=torch.ones_like(out),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            return ((grad.norm(2, dim=[1,2]) - 1) ** 2).mean()

        d_loss_real   = (bce_logits(D_real,   labels_real) * mask).sum() / mask.sum().clamp_min(1.0)
        d_loss_fake   = (bce_logits(D_fake,   labels_fake) * mask).sum() / mask.sum().clamp_min(1.0)
        d_loss_fake_e = (bce_logits(D_fake_e, labels_fake) * mask).sum() / mask.sum().clamp_min(1.0)

        # GP term is expensive, half GP computation
        if itt % 2 == 0:
            gp_loss = 2.0 * gradient_penalty(discriminator, H_real, H_hat)
        else:
            gp_loss = 0.0

        d_loss = d_loss_real + d_loss_fake + gamma * d_loss_fake_e + gp_loss

        step_d_loss = d_loss.detach().item()

        if step_d_loss < 0.5:  # discriminator already strong enough
            continue  # skip update

        if itt % 2 == 0 and step_d_loss > 0.15:
            D_optimizer.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            D_optimizer.step()

        # -------------------------
        # Logging every 1000 iters
        # -------------------------
        if itt % 1000 == 0:
            print(
                f"step: {itt}/{iterations}, "
                f"d_loss: {np.round(step_d_loss,4)}, "
                f"g_loss_u: {np.round(ema_g_loss_u,4)}, "
                f"g_loss_s: {np.round(np.sqrt(step_g_loss_s),4)}, "
                f"g_loss_v: {np.round(step_g_loss_v,4)}, "
                f"e_loss_t0: {np.round(np.sqrt(step_e_loss_t0),4)}"
            )

    print("Finish Joint Training")

    # synthesize
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    Z_mb_t = torch.tensor(Z_mb, dtype=torch.float32, device=device)
    with torch.no_grad():
        X_hat = recovery(supervisor(generator(Z_mb_t))).float().cpu().numpy()
    generated = [(X_hat[i, : int(ori_time[i]), :] * max_val + min_val) for i in range(no)]
    return generated