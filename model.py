import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


class HeteroDotProductPredictor(nn.Module):
    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = (h_u * h_v).sum(dim=-1)
        return {'score': score}

    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['score']


class CosineSimilarityPredictor(nn.Module):
    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = F.cosine_similarity(h_u, h_v, dim=-1)
        return {'score': score}

    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['score']


class RGCN(nn.Module):
    def __init__(self, hid_feats, out_feats, rel_in_feats):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel[1]: dglnn.GraphConv(in_feat, hid_feats)
            for rel, in_feat in rel_in_feats.items()}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel[1]: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in list(rel_in_feats.keys())}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel[1]: dglnn.GraphConv(hid_feats, out_feats)
            for rel in list(rel_in_feats.keys())}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        return h


class Model(nn.Module):
    def __init__(self, hidden_features, out_features, rel_in_feats):
        super().__init__()
        self.sage = RGCN(hidden_features, out_features, rel_in_feats)
        self.pred = CosineSimilarityPredictor()
        # self.pred = HeteroDotProductPredictor()

    def forward(self, g, x, pred_etype):
        h = self.sage(g, x)
        return self.pred(g, h, pred_etype)