
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentAggregation(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, feats):
        out, _ = self.lstm(feats)  # (N, C, d_hid)
        return torch.mean(out, dim=1)


class NeighborAggregation(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.lstm = nn.LSTM(emb_dim, emb_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, embeds):
        out, _ = self.lstm(embeds)  # (N, Nt, d)
        return torch.mean(out, dim=1)


class TypesCombination(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.w = nn.Parameter(torch.range(emb_dim, emb_dim))
        self.wk = nn.Parameter(torch.range(emb_dim, emb_dim))
        self.wv = nn.Parameter(torch.range(emb_dim, emb_dim))
        self.attn = nn.Parameter(torch.ones(1, 2 * emb_dim))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, content_embed, neighbor_embeds):
        neighbor_embeds = torch.cat(
            [content_embed.unsqueeze(0), neighbor_embeds], dim=0
        )  # (A+1, N, d)
        cat_embeds = torch.cat(
            [content_embed.repeat(neighbor_embeds.shape[0], 1, 1), neighbor_embeds], dim=-1
        )  # (A+1, N, 2d)
        attn_scores = self.leaky_relu((self.attn * cat_embeds).sum(dim=-1, keepdim=True))
        attn_scores = F.softmax(attn_scores, dim=0)  # (A+1, N, 1)
        out = (attn_scores * neighbor_embeds).sum(dim=0)  # (N, d)
        return out


class HHfan(nn.Module):

    def __init__(self, in_dim, hidden_dim, ntypes):
        super().__init__()
        self.content_aggs = nn.ModuleDict({
            ntype: ContentAggregation(in_dim, hidden_dim) for ntype in ntypes})
        self.neighbor_aggs = nn.ModuleDict({
            ntype: NeighborAggregation(hidden_dim) for ntype in ntypes})
        self.combs = nn.ModuleDict({
            ntype: TypesCombination(hidden_dim) for ntype in ntypes})

    def forward(self, g, feats):
        with g.local_scope():
            for ntype in g.ntypes:
                g.nodes[ntype].data['c'] = self.content_aggs[ntype](feats[ntype])  # (N_i, d_hid)
            neighbor_embeds = {}
            for dt in g.ntypes:
                tmp = []
                for st in g.ntypes:
                    g.multi_update_all({f'{st}-{dt}': (fn.copy_u('c', 'm'), stack_reducer)}, 'sum')
                    tmp.append(self.neighbor_aggs[st](g.nodes[dt].data.pop('nc')))
                neighbor_embeds[dt] = torch.stack(tmp)  # (A, N_dt, d_hid)

            # 3.类型组合
            out = {
                ntype: self.combs[ntype](g.nodes[ntype].data['c'], neighbor_embeds[ntype])
                for ntype in g.ntypes
            }
            return out

    def calc_score(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            for etype in g.etypes:
                g.apply_edges(fn.u_dot_v('h', 'h', 's'), etype=etype)
            return torch.cat(list(g.edata['s'].values())).squeeze(dim=-1)  # (A*E,)


def stack_reducer(nodes):
    return {'nc': nodes.mailbox['m']}