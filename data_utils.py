import pandas as pd
import dgl
import torch
from itertools import permutations


class PersonRecruitGraphDataset:
    def __init__(self):
        self.full_data = None
        self.person_data = None
        self.recruit_data = None
        self.folder_data = None
        self.graph = None

    def extract_data_from_file(self, source_file_path):
        person = pd.read_csv(source_file_path + 'person.csv')
        recruit = pd.read_csv(source_file_path + 'recruit.csv')
        recruit_folder = pd.read_csv(source_file_path + 'recruit_folder.csv')

        person.rename(columns={'MAJOR': 'MAJOR_PERSON'}, inplace=True)
        recruit.rename(columns={'MAJOR': 'MAJOR_RECRUIT'}, inplace=True)

        self.full_data = pd.merge(recruit_folder, person, on='PERSON_ID', how='left')
        self.full_data = pd.merge(self.full_data, recruit, on='RECRUIT_ID', how='left')
        self.full_data = self.full_data[['PERSON_ID', 'SPECILTY', 'MAJOR_PERSON',
                                         'RECRUIT_ID', 'DETAIL', 'MAJOR_RECRUIT', 'LABEL']].copy().dropna()
        self.preprocess_data()

    def preprocess_data(self):
        person_id = self.full_data['PERSON_ID'].drop_duplicates().tolist()
        person_id_map = dict(zip(person_id, range(len(person_id))))

        recruit_id = self.full_data['RECRUIT_ID'].drop_duplicates().tolist()
        recruit_id_map = dict(zip(recruit_id, range(len(recruit_id))))

        self.full_data['PERSON_ID'] = self.full_data['PERSON_ID'].map(person_id_map)
        self.full_data['RECRUIT_ID'] = self.full_data['RECRUIT_ID'].map(recruit_id_map)

        new_folder = self.full_data[['PERSON_ID', 'RECRUIT_ID', 'LABEL']].copy()
        df1 = self.full_data[['PERSON_ID', 'MAJOR_PERSON', 'SPECILTY']].drop_duplicates().copy()
        df2 = self.full_data[['RECRUIT_ID', 'MAJOR_RECRUIT', 'DETAIL']].drop_duplicates().copy()

        df1 = df1.sort_values(by=['PERSON_ID'], ascending=True).reset_index(drop=True)
        df2 = df2.sort_values(by=['RECRUIT_ID'], ascending=True).reset_index(drop=True)

        self.person_data = df1
        self.recruit_data = df2
        self.folder_data = new_folder

    def build_graph(self, graph_device):
        # using recruit_id and person_id to be dgl.graph's node_id
        apply_edges_src, apply_edges_dst = self.folder_data[['PERSON_ID', 'RECRUIT_ID']].to_numpy().T.copy()

        person_major_edges_src = []
        person_major_edges_dst = []
        for major in self.person_data['MAJOR_PERSON'].drop_duplicates().tolist():
            person_id = self.person_data.loc[self.person_data['MAJOR_PERSON'] == major, 'PERSON_ID'].tolist()
            edges = list(permutations(person_id, 2))
            for (src, dst) in edges:
                person_major_edges_src.append(src)
                person_major_edges_dst.append(dst)

        recruit_major_edges_src = []
        recruit_major_edges_dst = []
        for major in self.recruit_data['MAJOR_RECRUIT'].drop_duplicates().tolist():
            recruit_id = self.recruit_data.loc[self.recruit_data['MAJOR_RECRUIT'] == major, 'RECRUIT_ID'].tolist()
            edges = list(permutations(recruit_id, 2))
            for (src, dst) in edges:
                recruit_major_edges_src.append(src)
                recruit_major_edges_dst.append(dst)

        graph = dgl.heterograph(
            data_dict={
                ('person', 'p_apply_j', 'job'): (apply_edges_src, apply_edges_dst),
                ('person', 'p_sameMajor_p', 'person'): (person_major_edges_src, person_major_edges_dst),
                ('job', 'j_sameMajor_j', 'job'): (recruit_major_edges_src, recruit_major_edges_dst),
                ('job', 'j_appliedBy_p', 'person'): (apply_edges_dst, apply_edges_src),
            },
            num_nodes_dict={'person': self.person_data.shape[0], 'job': self.recruit_data.shape[0]},
            device=graph_device
        )

        edge_labels = self.extract_edgeLabel(graph)
        graph.edges['p_apply_j'].data['label'] = edge_labels

        self.graph = graph

    def extract_edgeLabel(self, graph):
        edge_label = torch.zeros(graph.num_edges(etype='p_apply_j'))
        src, dst = graph.edges(etype='p_apply_j')
        for i in range(edge_label.shape[0]):
            pid = src[i].item()
            jid = dst[i].item()
            if self.folder_data.loc[(self.folder_data['PERSON_ID'] == pid) &
                                    (self.folder_data['RECRUIT_ID'] == jid), 'LABEL'].values[0] == 1:
                edge_label[i] = 1
            else:
                edge_label[i] = 0
        return edge_label

    def graph_train_test_split_by_edges(self, test_size):
        train_mask = torch.zeros(self.graph.num_edges(etype='p_apply_j'), dtype=torch.bool).bernoulli(1 - test_size)
        self.graph.edges['p_apply_j'].data['train_mask'] = train_mask

        edges_src, edges_dst = self.graph.edges(etype='p_apply_j')

        train_src, train_dst = edges_src[train_mask], edges_dst[train_mask]
        train_folder = pd.DataFrame({'PERSON_ID': train_src.numpy(),
                                     'RECRUIT_ID': train_dst.numpy()})
        train_data = pd.merge(train_folder, self.full_data, how='left', on=['PERSON_ID', 'RECRUIT_ID'])

        test_src, test_dst = edges_src[~train_mask], edges_dst[~train_mask]
        test_folder = pd.DataFrame({'PERSON_ID': test_src.numpy(),
                                    'RECRUIT_ID': test_dst.numpy()})
        test_data = pd.merge(test_folder, self.full_data, how='left', on=['PERSON_ID', 'RECRUIT_ID'])

        self.train_data = train_data
        self.test_data = test_data

    def graph_train_test_split_by_nodes(self, test_size):
        person_train_mask = torch.zeros(self.graph.num_nodes(ntype='person'), dtype=torch.bool).bernoulli(1 - test_size)
        self.graph.nodes['person'].data['train_mask'] = person_train_mask

        recruit_train_mask = torch.zeros(self.graph.num_nodes(ntype='job'), dtype=torch.bool).bernoulli(1 - test_size)
        self.graph.nodes['job'].data['train_mask'] = recruit_train_mask


    def save_dataset(self, save_path):
        self.full_data.to_csv(save_path+'full_data.csv', index=0)
        self.person_data.to_csv(save_path+'person_data.csv', index=0)
        self.recruit_data.to_csv(save_path+'recruit_data.csv', index=0)
        self.folder_data.to_csv(save_path+'folder_data.csv', index=0)
        dgl.save_graphs(save_path+'graph.bin', self.graph)

    def load_data(self, save_path):
        self.full_data = pd.read_csv(save_path+'full_data.csv')
        self.person_data = pd.read_csv(save_path+'person_data.csv')
        self.recruit_data = pd.read_csv(save_path+'recruit_data.csv')
        self.folder_data = pd.read_csv(save_path+'folder_data.csv')
        self.graph = dgl.load_graphs(save_path+'graph.bin')[0][0]
        self.train_data = pd.read_csv(save_path+'train_data.csv')
        self.test_data = pd.read_csv(save_path+'test_data.csv')

    def add_node_features(self, sent_encoder):  # sent_encoder is an Sentence Transformer object
        person_feature = sent_encoder.encode(self.person_data['SPECILTY'].values,
                                             convert_to_tensor=True, show_progress_bar=True)
        recruit_feature = sent_encoder.encode(self.recruit_data['DETAIL'].values,
                                              convert_to_tensor=True, show_progress_bar=True)

        self.graph.nodes['person'].data['feature'] = person_feature
        self.graph.nodes['job'].data['feature'] = recruit_feature
