import numpy as np
import torch

class MuResolvedBispectrumGPU:

    def __init__(self, BoxSize, grid, n_mu=5, device='cuda'):
        if 'cuda' in str(device) and (not torch.cuda.is_available()):
            print('[MuResolvedBispectrumGPU] CUDA not available -- falling back to CPU.')
            device = 'cpu'
        self.device = torch.device(device)
        self.BoxSize = float(BoxSize)
        self.grid = int(grid)
        self.n_mu = int(n_mu)
        self.kF = 2.0 * np.pi / self.BoxSize
        self.mu_edges = np.linspace(0.0, 1.0, self.n_mu + 1)
        self.dtype = torch.float64
        self._bk_cache = {}
        self._build_kgrid()

    def _build_kgrid(self):
        g = self.grid
        cell_size = self.BoxSize / g
        kz_los = 2 * np.pi * np.fft.fftfreq(g, cell_size)
        ky = 2 * np.pi * np.fft.fftfreq(g, cell_size)
        kx = 2 * np.pi * np.fft.rfftfreq(g, cell_size)
        (KZ, KY, KX) = np.meshgrid(kz_los, ky, kx, indexing='ij')
        kmag_np = np.sqrt(KZ ** 2 + KY ** 2 + KX ** 2)
        with np.errstate(invalid='ignore', divide='ignore'):
            mu_np = np.abs(KZ) / kmag_np
        mu_np[~np.isfinite(mu_np)] = 0.0
        self.kmag = torch.tensor(kmag_np, dtype=self.dtype, device=self.device)
        self.mu = torch.tensor(mu_np, dtype=self.dtype, device=self.device)

    def read_real(self, delta_r, MAS='CIC'):
        g = self.grid
        if isinstance(delta_r, torch.Tensor):
            arr = delta_r.to(dtype=self.dtype, device=self.device)
        else:
            arr = torch.tensor(np.ascontiguousarray(delta_r.astype(np.float64)), dtype=self.dtype, device=self.device)
        assert arr.shape == (g, g, g), f'Expected ({g},{g},{g}), got {arr.shape}'
        self.delta_c = torch.fft.rfftn(arr, dim=(0, 1, 2))
        if MAS:
            self._compensate_MAS(MAS)

    def _compensate_MAS(self, MAS):
        p = {'CIC': 2}[MAS]
        g = self.grid
        wz = torch.sinc(torch.tensor(np.fft.fftfreq(g), dtype=self.dtype, device=self.device)) ** p
        wy = torch.sinc(torch.tensor(np.fft.fftfreq(g), dtype=self.dtype, device=self.device)) ** p
        wx = torch.sinc(torch.tensor(np.fft.rfftfreq(g), dtype=self.dtype, device=self.device)) ** p
        window = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
        self.delta_c = self.delta_c / window.to(self.delta_c.dtype)

    def _mu_shell_tensor(self, k_low, k_high, a):
        dc = self.delta_c.clone()
        sel = (self.kmag >= k_low) & (self.kmag < k_high) & (self.mu >= self.mu_edges[a]) & (self.mu < self.mu_edges[a + 1] + (1e-09 if a == self.n_mu - 1 else 0.0))
        dc[~sel] = 0.0
        return torch.fft.irfftn(dc, s=(self.grid,) * 3, dim=(0, 1, 2))

    def _count_shell_tensor(self, k_low, k_high, a):
        ones = torch.ones_like(self.delta_c)
        sel = (self.kmag >= k_low) & (self.kmag < k_high) & (self.mu >= self.mu_edges[a]) & (self.mu < self.mu_edges[a + 1] + (1e-09 if a == self.n_mu - 1 else 0.0))
        ones[~sel] = 0.0
        return torch.fft.irfftn(ones, s=(self.grid,) * 3, dim=(0, 1, 2))

    def Bk_mu(self, fc=1, dk=1, Nbins=42, triangle_type='All'):
        g = self.grid
        nm = self.n_mu
        dev = self.device
        dtype = self.dtype
        k_centers = self.kF * (fc + dk * np.arange(Nbins))
        cache_key = (fc, dk, Nbins, triangle_type)
        cached = self._bk_cache.get(cache_key)
        S_list = []
        if cached is None:
            C_list = []
        for i in range(Nbins):
            klo = self.kF * (fc + dk * i - dk / 2)
            khi = self.kF * (fc + dk * i + dk / 2)
            si_row = []
            if cached is None:
                ci_row = []
            for a in range(nm):
                s = self._mu_shell_tensor(klo, khi, a).reshape(-1)
                si_row.append(s)
                if cached is None:
                    c = self._count_shell_tensor(klo, khi, a).reshape(-1)
                    ci_row.append(c)
            S_list.append(torch.stack(si_row, dim=0))
            if cached is None:
                C_list.append(torch.stack(ci_row, dim=0))
        S = torch.stack(S_list, dim=0)
        if cached is None:
            C = torch.stack(C_list, dim=0)
            counts_P_t = (C ** 2).sum(dim=2) * g ** 3
            counts_P = counts_P_t.cpu().numpy()
            dk_phys = dk * self.kF
            tris = []
            for i in range(Nbins):
                for j in range(i, Nbins):
                    for l in range(j, Nbins):
                        if triangle_type == 'All' and (not k_centers[l] <= k_centers[i] + k_centers[j] + dk_phys):
                            continue
                        tris.append((i, j, l))
            tri_index = np.array(tris, dtype=int)
            Ntri = len(tri_index)
            counts_out_t = torch.empty((Ntri, nm, nm, nm), dtype=dtype, device=dev)
            for (t, (i, j, l)) in enumerate(tris):
                counts_out_t[t] = torch.einsum('aN,bN,cN->abc', C[i], C[j], C[l]) * g ** 6
            counts_out = counts_out_t.to(torch.float64).cpu().numpy()
            self._bk_cache[cache_key] = dict(C=C, counts_P_t=counts_P_t, counts_P=counts_P, tri_index=tri_index, counts_out_t=counts_out_t, counts_out=counts_out)
        else:
            C = cached['C']
            counts_P_t = cached['counts_P_t']
            counts_P = cached['counts_P']
            tri_index = cached['tri_index']
            counts_out_t = cached['counts_out_t']
            counts_out = cached['counts_out']
        Ntri = len(tri_index)
        tris = [tuple(tri_index[t]) for t in range(Ntri)]
        V = self.BoxSize ** 3
        norm_P = V / g ** 3
        Pwr = (S ** 2).sum(dim=2)
        cnt_safe = torch.where(counts_P_t > 0, counts_P_t, torch.ones_like(counts_P_t))
        Pkmu_t = torch.where(counts_P_t > 0, Pwr * norm_P / cnt_safe, torch.zeros_like(Pwr))
        Pkmu = Pkmu_t.cpu().numpy()
        norm_B = self.BoxSize ** 6 / g ** 3
        B_out_t = torch.empty((Ntri, nm, nm, nm), dtype=torch.float64, device=dev)
        for (t, (i, j, l)) in enumerate(tris):
            B_raw = torch.einsum('aN,bN,cN->abc', S[i], S[j], S[l])
            cnt = counts_out_t[t]
            cnt_safe2 = torch.where(cnt > 0, cnt, torch.ones_like(cnt))
            B_phys = torch.where(cnt > 0, B_raw * norm_B / cnt_safe2, torch.zeros_like(B_raw))
            B_out_t[t] = B_phys.to(torch.float64)
        B_out = B_out_t.cpu().numpy()
        return dict(B=B_out, counts=counts_out, P=Pkmu, counts_P=counts_P, tri_index=tri_index, k_centers=k_centers, mu_edges=self.mu_edges.copy())
