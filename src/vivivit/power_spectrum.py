import warnings
import numpy as np
import torch
from typing import Dict, Tuple, Optional

class PkEstimator:

    def __init__(self, BoxSize: float, device: str | torch.device='cpu'):
        if 'cuda' in str(device) and (not torch.cuda.is_available()):
            warnings.warn('CUDA not available – falling back to CPU.')
            device = 'cpu'
        self.device = torch.device(device)
        self.BoxSize = float(BoxSize)
        self._kgrid_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self._stats_cache: Dict[Tuple[int, torch.device], Tuple] = {}
        self._mas_cache: Dict[Tuple[int, str, torch.device], torch.Tensor] = {}

    def compute(self, delta_r_1: torch.Tensor | np.ndarray, delta_r_2: Optional[torch.Tensor | np.ndarray]=None, MAS: Tuple[Optional[str], Optional[str]]=(None, None)) -> torch.Tensor:
        if delta_r_1.ndim == 5 and delta_r_1.shape[1] == 1:
            delta_r_1 = delta_r_1.squeeze(1)
        d1 = self._as_tensor(delta_r_1)
        if delta_r_2 is not None:
            if delta_r_2.ndim == 5 and delta_r_2.shape[1] == 1:
                delta_r_2 = delta_r_2.squeeze(1)
            d2 = self._as_tensor(delta_r_2)
        else:
            d2 = d1
        if d1.shape != d2.shape:
            raise ValueError('delta_r_1 and delta_r_2 must have identical shapes.')
        (B, N, _, _) = d1.shape
        k_max_idx = N // 2
        num_bins = max(k_max_idx - 1, 0)
        (k_mag, kF, valid_mask, k_idx_valid0, k_center, n_modes) = self._get_geometry(N, k_max_idx)
        d1k = torch.fft.rfftn(d1, dim=(-3, -2, -1))
        d2k = d1k if d1 is d2 else torch.fft.rfftn(d2, dim=(-3, -2, -1))
        if MAS[0]:
            d1k *= self._get_mas(N, MAS[0]).unsqueeze(0)
        if MAS[1]:
            d2k *= self._get_mas(N, MAS[1]).unsqueeze(0)
        V = self.BoxSize ** 3
        norm = V / N ** 6
        P1 = (d1k.conj() * d1k).real * norm
        P2 = (d2k.conj() * d2k).real * norm
        PX = (d1k.conj() * d2k).real * norm

        def _bin(P):
            P_flat = P.permute(1, 2, 3, 0)[valid_mask].permute(1, 0)
            sums = torch.zeros(B, num_bins, device=self.device)
            idx = k_idx_valid0.unsqueeze(0).expand(B, -1)
            sums.scatter_add_(1, idx, P_flat)
            Nm_safe = n_modes.clone()
            Nm_safe[Nm_safe == 0] = 1.0
            return sums / Nm_safe.unsqueeze(0)
        Pk1 = _bin(P1)
        Pk2 = _bin(P2)
        PkX = _bin(PX)
        kcen_b = k_center.unsqueeze(0).expand(B, -1)
        Nm_b = n_modes.unsqueeze(0).expand(B, -1)
        return torch.stack([kcen_b, Pk1, Pk2, PkX, Nm_b], dim=2)

    def _as_tensor(self, arr):
        if arr is None:
            raise ValueError('Input cannot be None here.')
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr.astype(np.float32, copy=False))
        arr = arr.to(self.device, dtype=torch.float32)
        if arr.ndim == 3:
            arr = arr.unsqueeze(0)
        if arr.ndim != 4:
            raise ValueError(f'Input must be 3- or 4-D. Got {arr.ndim}-D.')
        return arr

    def _get_geometry(self, N, k_max_idx):
        key = (N, self.device)
        if key in self._stats_cache:
            return self._stats_cache[key]
        (k_mag, kF) = self._get_kgrid(N)
        k_idx = (k_mag / kF).round().long()
        valid_m = (k_idx > 0) & (k_idx < k_max_idx)
        valid_m_rfft = (k_idx > 0) & (k_idx < k_max_idx)
        k_idx_valid0 = k_idx[valid_m_rfft] - 1
        k_flat = k_mag[valid_m_rfft]
        num_bins = k_max_idx - 1
        k_sum = torch.zeros(num_bins, device=self.device)
        Nmodes = torch.zeros(num_bins, device=self.device)
        k_sum.scatter_add_(0, k_idx_valid0, k_flat)
        Nmodes.scatter_add_(0, k_idx_valid0, torch.ones_like(k_flat))
        Nm_safe = torch.where(Nmodes == 0, torch.ones_like(Nmodes), Nmodes)
        k_center = k_sum / Nm_safe
        self._stats_cache[key] = (k_mag, kF, valid_m_rfft, k_idx_valid0, k_center.detach(), Nmodes.detach())
        return self._stats_cache[key]

    def _get_kgrid(self, N):
        key = (N, self.device)
        if key in self._kgrid_cache:
            return self._kgrid_cache[key]
        cell = self.BoxSize / N
        kx = 2 * np.pi * torch.fft.fftfreq(N, d=cell, device=self.device)
        ky = kx
        kz = 2 * np.pi * torch.fft.rfftfreq(N, d=cell, device=self.device)
        (kxg, kyg, kzg) = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = torch.sqrt(kxg ** 2 + kyg ** 2 + kzg ** 2).detach()
        kF = torch.tensor(2 * np.pi / self.BoxSize, device=self.device)
        self._kgrid_cache[key] = (k_mag, kF)
        return (k_mag, kF)

    def _get_mas(self, N, scheme: str):
        key = (N, scheme, self.device)
        if key in self._mas_cache:
            return self._mas_cache[key]
        p = 2 if scheme == 'CIC' else None
        if p is None:
            raise ValueError(f"Unknown MAS scheme '{scheme}'")
        cell = self.BoxSize / N
        kx = 2 * np.pi * torch.fft.fftfreq(N, d=cell, device=self.device)
        ky = kx
        kz = 2 * np.pi * torch.fft.rfftfreq(N, d=cell, device=self.device)
        (kxg, kyg, kzg) = torch.meshgrid(kx, ky, kz, indexing='ij')
        fac = lambda k: k * cell / 2

        def inv_sinc_pow(x):
            s = torch.ones_like(x)
            nz = torch.abs(x) > 1e-09
            s[nz] = torch.sin(x[nz]) / x[nz]
            return s.pow(-p)
        mas = inv_sinc_pow(fac(kxg)) * inv_sinc_pow(fac(kyg)) * inv_sinc_pow(fac(kzg))
        self._mas_cache[key] = mas.detach()
        return mas


class ShellPkEstimator:

    def __init__(self, BoxSize: float, device: str | torch.device='cpu'):
        if 'cuda' in str(device) and (not torch.cuda.is_available()):
            warnings.warn('CUDA not available – falling back to CPU.')
            device = 'cpu'
        self.device = torch.device(device)
        self.BoxSize = float(BoxSize)
        self._cache: Dict[Tuple[int, torch.device], Tuple] = {}

    def compute(self, delta_r_1, delta_r_2=None) -> torch.Tensor:
        d1 = self._as_tensor(delta_r_1)
        d2 = d1 if delta_r_2 is None else self._as_tensor(delta_r_2)
        if d1.shape != d2.shape:
            raise ValueError('delta_r_1 and delta_r_2 must have identical shapes.')
        (B, N, _, _) = d1.shape
        d1k = torch.fft.rfftn(d1, dim=(-3, -2, -1))
        d2k = d1k if d2 is d1 else torch.fft.rfftn(d2, dim=(-3, -2, -1))
        norm = self.BoxSize ** 3 / N ** 6
        P1 = (d1k.conj() * d1k).real * norm
        P2 = (d2k.conj() * d2k).real * norm
        PX = (d1k.conj() * d2k).real * norm
        (valid, index, k_center, n_modes, weight, num_bins) = self._geometry(N)

        def _bin(P):
            flat = P.permute(1, 2, 3, 0)[valid].permute(1, 0) * weight.unsqueeze(0)
            sums = torch.zeros(B, num_bins, device=self.device, dtype=P.dtype)
            sums.scatter_add_(1, index.unsqueeze(0).expand(B, -1), flat)
            return sums / torch.where(n_modes == 0, torch.ones_like(n_modes), n_modes).unsqueeze(0)
        return torch.stack([k_center.unsqueeze(0).expand(B, -1), _bin(P1), _bin(P2), _bin(PX), n_modes.unsqueeze(0).expand(B, -1)], dim=2)

    def _as_tensor(self, arr):
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr.astype(np.float32, copy=False))
        arr = arr.to(self.device, dtype=torch.float32)
        if arr.ndim == 3:
            arr = arr.unsqueeze(0)
        if arr.ndim != 4:
            raise ValueError(f'Input must be 3- or 4-D. Got {arr.ndim}-D.')
        return arr

    def _geometry(self, N):
        key = (N, self.device)
        if key in self._cache:
            return self._cache[key]
        middle = N // 2
        kF = float(torch.tensor(2 * np.pi / self.BoxSize, dtype=torch.float32))
        axis = torch.arange(N, device=self.device, dtype=torch.long)
        n = torch.where(axis > middle, axis - N, axis)
        nz = torch.arange(N // 2 + 1, device=self.device, dtype=torch.long)
        n2 = n.view(N, 1, 1) ** 2 + n.view(1, N, 1) ** 2 + nz.view(1, 1, -1) ** 2
        k_over_kF = torch.sqrt(n2.to(torch.float64))
        k_idx = torch.floor(k_over_kF).long()
        num_bins = int(np.floor(np.sqrt(3.0) * middle + 1e-12))
        valid = (k_idx >= 1) & (k_idx <= num_bins)
        index = k_idx[valid] - 1
        weight = torch.ones((N, N, N // 2 + 1), device=self.device, dtype=torch.float32)
        corners = [0, middle] if N % 2 == 0 else [0]
        for plane in corners:
            weight[:, :, plane] = 0.5
            for i in corners:
                for j in corners:
                    weight[i, j, plane] = 1.0
        weight = weight[valid]
        n_modes = torch.zeros(num_bins, device=self.device, dtype=torch.float64)
        k_sum = torch.zeros(num_bins, device=self.device, dtype=torch.float64)
        n_modes.scatter_add_(0, index, weight.to(torch.float64))
        k_sum.scatter_add_(0, index, k_over_kF[valid] * kF * weight.to(torch.float64))
        k_center = (k_sum / torch.where(n_modes == 0, torch.ones_like(n_modes), n_modes)).to(torch.float32)
        self._cache[key] = (valid, index, k_center, n_modes.to(torch.float32), weight, num_bins)
        return self._cache[key]
