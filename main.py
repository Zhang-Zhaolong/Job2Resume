from data_utils import PersonRecruitGraphDataset
from model import Model
import torch
from metrics import *


def get_dataset(method, path_args):
    g_dataset = PersonRecruitGraphDataset()
    if method == 'create':
        g_dataset.extract_data_from_file(path_args['dataset_path'])
        g_dataset.build_graph('cpu')
        # g_dataset.graph_train_test_split_by_nodes(0.2)
        g_dataset.save_dataset(path_args['save_path'])
    elif method == 'load':
        g_dataset.load_data(path_args['save_path'])
    else:
        raise ValueError('method should be either create or load')

    return g_dataset


def evaluate(scores, edge_labels):
    with torch.no_grad():
        auc = compute_auc(scores, edge_labels)
        acc, best_threshold = find_best_acc_and_threshold(scores, edge_labels, True)
        pred_label = torch.zeros(scores.cpu().shape[0], device='cuda:0')
        pred_label[scores.ge(best_threshold)] = 1
        f1 = f1_score(edge_labels.cpu().numpy(), pred_label.cpu().detach().numpy())
        return 'acc:{:4f}, auc:{:6f}, f1:{:6f},threshold:{:6f}'.format(acc, auc, f1, best_threshold)


def train(train_graph, epochs, hidden_dim, learning_rate, margin):
    rel_in_feats = {
        train_graph.to_canonical_etype(k):
            train_graph.nodes[train_graph.to_canonical_etype(k)[0]].data['feature'].shape[1]
        for k in train_graph.etypes
    }
    model = Model(hidden_dim, hidden_dim, rel_in_feats).to('cuda:0')
    person_feats = train_graph.nodes['person'].data['feature']
    job_feats = train_graph.nodes['job'].data['feature']
    node_features = {'person': person_feats, 'job': job_feats}
    edge_label = train_graph.edges['p_apply_j'].data['label']
    pred_etype = ('person', 'p_apply_j', 'job')
    train_mask = train_graph.edges['p_apply_j'].data['train_mask']
    train_mask = train_mask.bool()

    result = []
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    for epoch in range(epochs):
        scores = model(train_graph, node_features, pred_etype)
        loss = compute_loss(scores[train_mask], edge_label[train_mask], margin)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('training: loss:{:6f} auc:{:6f}'.format(loss.item(), compute_auc(scores, edge_label)))
        if epoch % 10 == 0:
            result_str = 'epoch:{} '.format(epoch) + evaluate(scores[~train_mask], edge_label[~train_mask])
            print('testing: ', result_str)
            result.append(result_str)
    return model


if __name__ == '__main__':
    path = {'save_path': './dataset/saved_dataset/',
            'dataset_path': './dataset/'}
    # dataset = get_dataset('create', path)
    dataset = get_dataset('create', path)
    graph = dataset.graph
