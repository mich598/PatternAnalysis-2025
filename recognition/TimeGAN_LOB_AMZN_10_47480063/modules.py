"""
“modules.py" containing the source code of the components of your model. Each component must be
implementated as a class or a function

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Github Link: https://github.com/jsyoon0823/TimeGAN/blob/master/data_loading.py

Based on timegan.py code
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import extract_time, random_generator, batch_generator

# -------------------------
# Helper utilities
# -------------------------
def create_rnn(module_name, input_size, hidden_size, num_layers, batch_first=True):
    """Return a PyTorch RNN module matching the requested type."""
    module_name = module_name.lower()
    if module_name in ("gru", "rnn-gru"):
        return nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first)
    elif module_name in ("lstm", "rnn-lstm"):
        return nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
    else:
        # default: vanilla RNN with tanh
        return nn.RNN(input_size, hidden_size, num_layers, nonlinearity='tanh', batch_first=batch_first)

def sequence_mask(lengths, max_len=None, device=None):
    """Create boolean mask (batch, max_len) where True indicates valid timestep."""
    if max_len is None:
        max_len = lengths.max().item()
    batch_size = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=device).unsqueeze(0).expand(batch_size, -1)
    return seq_range < lengths.unsqueeze(1)

# -------------------------
# Modules
# -------------------------
class Embedder(nn.Module):
    def __init__(self, module_name, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = create_rnn(module_name, input_dim, hidden_dim, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, dim)
        out, _ = self.rnn(x)  # out: (batch, seq_len, hidden_dim)
        out = self.fc(out)
        return out

class Recovery(nn.Module):
    def __init__(self, module_name, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.rnn = create_rnn(module_name, hidden_dim, hidden_dim, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, h):
        out, _ = self.rnn(h)
        out = self.fc(out)
        return out

class Generator(nn.Module):
    def __init__(self, module_name, z_dim, hidden_dim, num_layers, dropout_rate=0.3):
        super().__init__()
        self.rnn = create_rnn(module_name, z_dim, hidden_dim, num_layers)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        out, _ = self.rnn(z)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class Supervisor(nn.Module):
    def __init__(self, module_name, hidden_dim, num_layers_minus1, dropout_rate=0.3):
        super().__init__()
        # note: original used num_layers-1 in TF; match that here
        self.rnn = create_rnn(module_name, hidden_dim, hidden_dim, num_layers_minus1 if num_layers_minus1>0 else 1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        out, _ = self.rnn(z)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, module_name, hidden_dim, num_layers):
        super().__init__()
        self.rnn = create_rnn(module_name, hidden_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, 1)  # logits

    def forward(self, h):
        out, _ = self.rnn(h)
        out = self.fc(out)  # (batch, seq_len, 1)
        return out

# -------------------------
# TimeGAN training function
# -------------------------
def timegan(ori_data, parameters, device=None):
    """
    PyTorch reimplementation of TimeGAN

    Args:
      - ori_data: list or numpy array shaped (no, variable_seq_len, dim) or padded array (no, max_seq_len, dim)
      - parameters: dict with keys:
          'hidden_dim', 'num_layer', 'iterations', 'batch_size', 'module'
      - device: 'cpu' or 'cuda'

    Returns:
      - generated_data: list of numpy arrays, each sequence renormalized to original scale
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Preprocessing (same MinMax scaling logic)
    # -------------------------
    ori_data_np = np.asarray(ori_data)
    no, seq_len, dim = ori_data_np.shape

    ori_time, max_seq_len = extract_time(ori_data)  # ori_time is list/array of lengths
    ori_time = np.asarray(ori_time, dtype=np.int64)

    def MinMaxScaler(data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        data_adj = data - min_val
        max_val = np.max(np.max(data_adj, axis=0), axis=0)
        norm_data = data_adj / (max_val + 1e-7)
        return norm_data, min_val, max_val

    ori_data_norm, min_val, max_val = MinMaxScaler(ori_data_np)

    # Cast min/max to numpy arrays for later renormalization
    min_val = np.array(min_val)
    max_val = np.array(max_val)

    # -------------------------
    # Hyperparameters
    # -------------------------
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = 32
    gamma = 0.85

    lr = 5e-5
    beta1 = 0.4
    beta2 = 0.9

    # statistics loss weight (targets spread & midprice return matching)
    lambda_stats = 200.0
    # instance-noise (stddev) added to discriminator inputs to reduce memorisation
    inst_noise_std = 0.03
    # label smoothing for real labels (helps stability)
    real_label_smooth = 0.9

    # -------------------------
    # Instantiate networks
    # -------------------------
    embedder = Embedder(module_name, dim, hidden_dim, num_layers).to(device)
    recovery = Recovery(module_name, hidden_dim, dim, num_layers).to(device)
    generator = Generator(module_name, z_dim, hidden_dim, num_layers).to(device)
    supervisor = Supervisor(module_name, hidden_dim, max(num_layers-1, 1)).to(device)
    discriminator = Discriminator(module_name, hidden_dim, num_layers).to(device)

    # -------------------------
    # Optimizers
    # -------------------------
    E0_optimizer = optim.Adam(list(embedder.parameters()) + list(recovery.parameters()), lr=lr, betas=(beta1, beta2))
    E_optimizer = optim.Adam(list(embedder.parameters()) + list(recovery.parameters()), lr=lr, betas=(beta1, beta2))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    G_optimizer = optim.Adam(list(generator.parameters()) + list(supervisor.parameters()), lr=lr, betas=(beta1, beta2))
    GS_optimizer = optim.Adam(list(generator.parameters()) + list(supervisor.parameters()), lr=lr, betas=(beta1, beta2))

    # Loss functions
    bce_logits = nn.BCEWithLogitsLoss(reduction='none')  # we'll mask and average manually
    mse_loss = nn.MSELoss(reduction='none')

    # Move normalized data to numpy for batch_generator which likely expects numpy
    ori_data_norm_np = ori_data_norm.copy()

    # -------------------------
    # Helper to convert numpy batch to torch tensors and create masks
    # -------------------------
    def to_torch(x_np, time_lengths):
        # x_np: (batch, max_seq_len, dim)
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
        t_t = torch.tensor(time_lengths, dtype=torch.int64, device=device)
        mask = sequence_mask(t_t, max_len=x_t.size(1), device=device).unsqueeze(-1).float()  # (batch, seq_len, 1)
        return x_t, t_t, mask

    # --- helper to compute spreads and midprice returns from a batch tensor ---
    # expects x_t: (batch, seq_len, dim), mask: (batch, seq_len, 1)
    def compute_spread_midret_from_tensor(x_t, mask):
        # Assumes feature 0 = best bid, feature 1 = best ask
        # x_t is torch.Tensor on device
        bid = x_t[..., 0]  # (batch, seq_len)
        ask = x_t[..., 1]  # (batch, seq_len)
        mid = (bid + ask) / 2.0
        spread = (ask - bid) * mask.squeeze(-1)  # masked spreads
        # midprice returns: r_t = (mid_t - mid_{t-1}) / mid_{t-1}
        mid_shift = mid[:, :-1]
        mid_next = mid[:, 1:]
        # avoid division by zero
        denom = (mid_shift.abs() + 1e-8)
        midret = (mid_next - mid_shift) / denom
        # also apply mask for valid positions (exclude timesteps where mask==0)
        mask_mid = (mask[:, 1:, 0])  # (batch, seq_len-1)
        spread_valid = spread[:, :-1] * mask_mid
        midret_valid = midret * mask_mid
        # flatten across batch/time but keep torch tensors (used for moment computation)
        return spread_valid, midret_valid


    # -------------------------
    # Training steps
    # -------------------------
    print("Start Embedding Network Training")
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data_norm_np, ori_time, batch_size)
        X_mb_t, T_mb_t, mask = to_torch(X_mb, T_mb)

        # Train embedder (E0 step): minimize E_loss_T0 = MSE(X, X_tilde)
        embedder.train(); recovery.train()
        E0_optimizer.zero_grad()
        H = embedder(X_mb_t)                   # (batch, seq_len, hidden_dim)
        X_tilde = recovery(H)                  # (batch, seq_len, dim)
        loss_T0_all = mse_loss(X_tilde, X_mb_t) * mask  # mask invalid timesteps
        loss_T0 = loss_T0_all.sum() / mask.sum().clamp_min(1.0)
        loss_T0.backward()
        E0_optimizer.step()
        
        if itt % 1000 == 0:
            print(f"step: {itt}/{iterations}, e_loss: {np.round(np.sqrt(loss_T0.item()),4)}")

    print("Finish Embedding Network Training")

    # 2. Supervised training only
    print("Start Training with Supervised Loss Only")
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data_norm_np, ori_time, batch_size)
        X_mb_t, T_mb_t, mask = to_torch(X_mb, T_mb)

        # random Z
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)  # numpy
        Z_mb_t = torch.tensor(Z_mb, dtype=torch.float32, device=device)

        generator.train(); supervisor.train()
        GS_optimizer.zero_grad()

        E_hat = generator(Z_mb_t)               # (batch, seq_len, hidden_dim)
        H_hat_supervise = supervisor(E_hat)     # (batch, seq_len, hidden_dim)

        # supervised loss: MSE(H[:,1:,:], H_hat_supervise[:,:-1,:]) with masking where applicable
        embedder.eval()
        with torch.no_grad():
            H = embedder(X_mb_t)

        # create shifted versions and mask for valid positions (exclude last timestep of each sequence)
        valid_mask = sequence_mask(T_mb_t - 1, max_len=max_seq_len-1, device=device).unsqueeze(-1).float()
        left = H[:, 1:, :]                        # target next-step
        right = H_hat_supervise[:, :-1, :]        # predicted
        loss_s_all = mse_loss(left * valid_mask, right * valid_mask)
        loss_s = loss_s_all.sum() / valid_mask.sum().clamp_min(1.0)

        loss_s.backward()
        GS_optimizer.step()

        if itt % 1000 == 0:
            print(f"step: {itt}/{iterations}, s_loss: {np.round(np.sqrt(loss_s.item()),4)}")

    print("Finish Training with Supervised Loss Only")

    # 3. Joint Training
    print("Start Joint Training")
    for itt in range(iterations):
        # Generator training (twice)
        for kk in range(2):
            X_mb, T_mb = batch_generator(ori_data_norm_np, ori_time, batch_size)
            X_mb_t, T_mb_t, mask = to_torch(X_mb, T_mb)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            Z_mb_t = torch.tensor(Z_mb, dtype=torch.float32, device=device)

            # Train generator + supervisor
            generator.train(); supervisor.train(); embedder.train(); recovery.train()
            G_optimizer.zero_grad()

            E_hat = generator(Z_mb_t)
            H_hat = supervisor(E_hat)
            X_hat = recovery(H_hat)

            # Discriminator outputs
            D_fake = discriminator(H_hat)          # (batch, seq_len, 1) logits
            D_fake_e = discriminator(E_hat)

            # Adversarial losses (generator wants discriminator to predict 1 for generated)
            labels_ones = torch.ones_like(D_fake, device=device)
            g_loss_u_all = bce_logits(D_fake, labels_ones)
            g_loss_u = (g_loss_u_all * mask).sum() / mask.sum().clamp_min(1.0)

            g_loss_u_e_all = bce_logits(D_fake_e, torch.ones_like(D_fake_e, device=device))
            g_loss_u_e = (g_loss_u_e_all * mask).sum() / mask.sum().clamp_min(1.0)

            # Supervised loss: between H[:,1:,:] and supervisor(H) shifted
            with torch.no_grad():
                H = embedder(X_mb_t)
            valid_mask = sequence_mask(T_mb_t - 1, max_len=max_seq_len-1, device=device).unsqueeze(-1).float()
            H_shift_target = H[:, 1:, :]
            H_hat_supervise = supervisor(H)
            H_hat_shift = H_hat_supervise[:, :-1, :]
            g_loss_s_all = mse_loss(H_shift_target * valid_mask, H_hat_shift * valid_mask)
            g_loss_s = g_loss_s_all.sum() / valid_mask.sum().clamp_min(1.0)

            # Two moments loss on X and X_hat across batch dimension (axis=0 in TF)
            # compute mean and std over batch dimension (dim=0)
            X_hat_masked = X_hat * mask
            X_masked = X_mb_t * mask
            # compute feature-wise mean over batch dimension and time dimension combined for stability
            # we follow TF axes loosely: compute across batch only (dim=0)
            mean_x_hat = X_hat_masked.mean(dim=0)   # (seq_len, dim)
            mean_x = X_masked.mean(dim=0)
            var_x_hat = X_hat_masked.var(dim=0, unbiased=False)
            var_x = X_masked.var(dim=0, unbiased=False)

            g_loss_v1 = torch.mean(torch.abs(torch.sqrt(var_x_hat + 1e-6) - torch.sqrt(var_x + 1e-6)))
            g_loss_v2 = torch.mean(torch.abs(mean_x_hat - mean_x))
            g_loss_v = g_loss_v1 + g_loss_v2

            # ---------- statistics loss: match spread & midprice return mean/std ----------
            # compute spreads & midreturns for real X_mb_t and generated X_hat
            with torch.no_grad():
                # compute real spreads/midret on this minibatch (use mask)
                real_spread, real_midret = compute_spread_midret_from_tensor(X_mb_t, mask)

            # generated is X_hat: shape (batch, seq_len, dim)
            gen_spread, gen_midret = compute_spread_midret_from_tensor(X_hat, mask)

            # compute batch-wise mean and std for each (ignore zeros from masking)
            # for stability, compute means dividing by nonzero counts
            def masked_mean_std(x):
                # x: (batch, time) with zeros in invalid positions
                valid = (x != 0).float()
                counts = valid.sum()
                if counts.item() < 1.0:
                    return torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device)
                m = x.sum() / counts
                centered = (x - m) * valid
                var = (centered.pow(2).sum() / counts).clamp_min(1e-8)
                return m, torch.sqrt(var)

            real_sp_m, real_sp_s = masked_mean_std(real_spread)
            gen_sp_m, gen_sp_s = masked_mean_std(gen_spread)
            real_mr_m, real_mr_s = masked_mean_std(real_midret)
            gen_mr_m, gen_mr_s = masked_mean_std(gen_midret)

            # absolute differences for mean/std
            sp_diff_mean = torch.abs(real_sp_m - gen_sp_m)
            sp_diff_std = torch.abs(real_sp_s - gen_sp_s)
            mr_diff_mean = torch.abs(real_mr_m - gen_mr_m)
            mr_diff_std = torch.abs(real_mr_s - gen_mr_s)

            g_loss_stats = sp_diff_mean + sp_diff_std + mr_diff_mean + mr_diff_std

            # ---------- total generator loss (add stats loss weighted by lambda_stats) ----------
            total_g_loss = (
                g_loss_u
                + gamma * g_loss_u_e
                + 100.0 * torch.sqrt(g_loss_s + 1e-8)
                + 100.0 * g_loss_v
                + lambda_stats * g_loss_stats
            )

            # ---------- backward & step ----------
            total_g_loss.backward()
            G_optimizer.step()

            # Train embedder (E_solver)
            E_optimizer.zero_grad()
            # E_loss_T0 and E_loss
            H = embedder(X_mb_t)
            X_tilde = recovery(H)
            E_loss_T0_all = mse_loss(X_tilde, X_mb_t) * mask
            E_loss_T0 = E_loss_T0_all.sum() / mask.sum().clamp_min(1.0)
            E_loss0 = 10.0 * torch.sqrt(E_loss_T0 + 1e-8)
            # G_loss_S contribution
            # compute G_loss_S again for gradient purposes (supervised loss using H and H_hat_supervise)
            H_hat_supervise = supervisor(H)
            valid_mask = sequence_mask(T_mb_t - 1, max_len=max_seq_len-1, device=device).unsqueeze(-1).float()
            left = H[:, 1:, :]
            right = H_hat_supervise[:, :-1, :]
            G_loss_S_all = mse_loss(left * valid_mask, right * valid_mask)
            G_loss_S_val = G_loss_S_all.sum() / valid_mask.sum().clamp_min(1.0)
            E_loss = E_loss0 + 0.1 * G_loss_S_val

            E_loss.backward()
            E_optimizer.step()

        # Discriminator training
        X_mb, T_mb = batch_generator(ori_data_norm_np, ori_time, batch_size)
        X_mb_t, T_mb_t, mask = to_torch(X_mb, T_mb)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Z_mb_t = torch.tensor(Z_mb, dtype=torch.float32, device=device)

        discriminator.train()
        D_optimizer.zero_grad()

        with torch.no_grad():
            H = embedder(X_mb_t)
            E_hat = generator(Z_mb_t)
            H_hat = supervisor(E_hat)

        # add small instance noise to discriminator inputs (reduces memorisation)
        if inst_noise_std > 0:
            noise_real = torch.randn_like(H) * inst_noise_std
            noise_fake = torch.randn_like(H_hat) * inst_noise_std
            noise_fake_e = torch.randn_like(E_hat) * inst_noise_std
            H_noisy = H + noise_real
            H_hat_noisy = H_hat + noise_fake
            E_hat_noisy = E_hat + noise_fake_e
        else:
            H_noisy, H_hat_noisy, E_hat_noisy = H, H_hat, E_hat

        D_real = discriminator(H_noisy)
        D_fake = discriminator(H_hat_noisy)
        D_fake_e = discriminator(E_hat_noisy)

        labels_real = torch.ones_like(D_real, device=device) * real_label_smooth
        labels_fake = torch.zeros_like(D_fake, device=device)

        d_loss_real_all = bce_logits(D_real, labels_real)
        d_loss_fake_all = bce_logits(D_fake, labels_fake)
        d_loss_fake_e_all = bce_logits(D_fake_e, labels_fake)

        d_loss_real = (d_loss_real_all * mask).sum() / mask.sum().clamp_min(1.0)
        d_loss_fake = (d_loss_fake_all * mask).sum() / mask.sum().clamp_min(1.0)
        d_loss_fake_e = (d_loss_fake_e_all * mask).sum() / mask.sum().clamp_min(1.0)

        d_loss = d_loss_real + d_loss_fake + gamma * d_loss_fake_e

        # conditional update (if discriminator not too good)
        if d_loss.item() > 0.15:
            d_loss.backward()
            D_optimizer.step()
            step_d_loss = d_loss.item()
        else:
            step_d_loss = d_loss.item()

        # print logs
        if itt % 1000 == 0:
            # for printing g_loss_u/g_loss_s/g_loss_v we reuse values computed earlier (they are in scope if last generator loop executed)
            print(f"step: {itt}/{iterations}, d_loss: {np.round(step_d_loss,4)}, "
                  f"g_loss_u: {np.round(g_loss_u.item(),4)}, g_loss_s: {np.round(np.sqrt(g_loss_s.item()),4)}, "
                  f"g_loss_v: {np.round(g_loss_v.item(),4)}, e_loss_t0: {np.round(np.sqrt(E_loss_T0.item()),4)}")

    print("Finish Joint Training")

    # -------------------------
    # Synthetic data generation (use all samples)
    # -------------------------
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)  # shape (no, max_seq_len, z_dim)
    Z_mb_t = torch.tensor(Z_mb, dtype=torch.float32, device=device)

    with torch.no_grad():
        E_hat = generator(Z_mb_t)
        H_hat = supervisor(E_hat)
        X_hat = recovery(H_hat)  # (no, max_seq_len, dim)

    X_hat_np = X_hat.cpu().numpy()

    generated_data = []
    for i in range(no):
        length = int(ori_time[i])
        temp = X_hat_np[i, :length, :]
        # renormalize: generated_data = generated_data * max_val + min_val
        # max_val and min_val are arrays of length dim
        temp = temp * max_val + min_val
        generated_data.append(temp)

    return generated_data