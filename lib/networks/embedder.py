import torch
from lib.config import cfg


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # input_enc = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        # start, end = opt.barf_c2f
        # L = self.kwargs['num_freqs']
        # alpha = (self.progress.data - start) / (end - start) * L
        # k = torch.arange(L, dtype=torch.float32, device=device)
        # weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
        # # apply weights
        # shape = input_enc.shape
        # input_enc = (input_enc.view(-1, L) * weight).view(*shape)



def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

#pos_embedder, xyz_dim = get_embedder(cfg.xyz_res)#used for point position
xyz_embedder, xyz_dim = get_embedder(cfg.xyz_res)
view_embedder, view_dim = get_embedder(cfg.view_res)
node_embedder, xyz_dim = get_embedder(cfg.xyz_res)#used for graph node
time_embedder, time_dim = get_embedder(cfg.xyz_res, input_dims=1)#used for time
