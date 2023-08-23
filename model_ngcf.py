import torch
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# import logging
# path = "log.log"
# logging.basicConfig(level=logging.INFO,filename=path, format="%(asctime)s - %(filename)s - %(message)s")
# log=logging.info


class GCN(torch.nn.Module):
    def __init__(self, args, n_users, n_items, norm_adj):
        super(GCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.norm_adj = norm_adj
        self.lr = args.lr
        self.n_fold = 1
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.weight_size_list = [self.emb_dim] + self.weight_size
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self._init_weights()
        self.A_fold_hat = self._split_A_hat(self.norm_adj)
        self.result = nn.init.xavier_normal_(torch.rand((self.n_users + self.n_items, self.emb_dim))).cuda()

    def _init_weights(self):
        self.user_embedding = xavier_normal_(
            torch.nn.Parameter(torch.rand(self.n_users, self.emb_dim), requires_grad=True))
        self.item_embedding = xavier_normal_(
            torch.nn.Parameter(torch.rand(self.n_items, self.emb_dim), requires_grad=True))
        self.W_gc_1 = xavier_normal_(torch.nn.Parameter(torch.rand(self.weight_size_list[0], self.weight_size_list[1])))
        self.b_gc_1 = xavier_normal_(torch.nn.Parameter(torch.rand(1, self.weight_size_list[1])))
        self.W_bi_1 = xavier_normal_(torch.nn.Parameter(torch.rand(self.weight_size_list[0], self.weight_size_list[1])))
        self.b_bi_1 = xavier_normal_(torch.nn.Parameter(torch.rand(1, self.weight_size_list[1])))
        # self.W_gc_2 = xavier_normal_(torch.nn.Parameter(torch.rand(self.weight_size_list[0], self.weight_size_list[1])))
        # self.b_gc_2 = xavier_normal_(torch.nn.Parameter(torch.rand(1, self.weight_size_list[1])))
        # self.W_bi_2 = xavier_normal_(torch.nn.Parameter(torch.rand(self.weight_size_list[0], self.weight_size_list[1])))
        # self.b_bi_2 = xavier_normal_(torch.nn.Parameter(torch.rand(1, self.weight_size_list[1])))

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = torch.LongTensor(np.mat([coo.row, coo.col]))
        st = torch.sparse.FloatTensor(indices, torch.FloatTensor(coo.data), coo.shape).cuda()

        return st

    def forward(self):
        # A_fold_hat = self._split_A_hat(self.norm_adj)
        ego_embeddings = torch.cat((self.user_embedding, self.item_embedding), dim=0).cuda()
        all_embeddings = [ego_embeddings]

        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(torch.sparse.mm(self.A_fold_hat[f], ego_embeddings))

        side_embeddings = torch.cat(temp_embed, 0)
        sum_embeddings = F.leaky_relu(torch.mm(side_embeddings, self.W_gc_1) + self.b_gc_1)
        bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
        bi_embeddings = F.leaky_relu(torch.mm(bi_embeddings, self.W_bi_1) + self.b_bi_1)
        ego_embeddings = sum_embeddings + bi_embeddings
        norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
        all_embeddings += [norm_embeddings]
        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        self.result = all_embeddings

        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        ua_embeddings, ia_embeddings = self.forward()
        users = np.array(users)
        pos_items = np.array(pos_items)
        neg_items = np.array(neg_items)
        users = ua_embeddings[users]
        pos_items = ia_embeddings[pos_items]
        neg_items = ia_embeddings[neg_items]
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        # regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        # regularizer = regularizer / self.batch_size
        # emb_loss = self.decay * regularizer
        return loss

    def full_accuracy(self, data, step=2000, topk=10):
        user_tensor = self.result[:self.n_users]
        item_tensor = self.result[self.n_users:]

        start_index = 0
        end_index = start_index + step if step < self.n_users else self.n_users

        all_index_of_rank_list = torch.LongTensor([])
        while end_index <= self.n_users and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
            for i in range(start_index, end_index):
                row = i - start_index
                col = torch.LongTensor(list(data.train_items[i]))
                score_matrix[row][col] = 1e-5

            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()),
                                               dim=0)
            start_index = end_index

            if end_index + step < self.n_users:
                end_index += step
            else:
                end_index = self.n_users
        length = 0
        precision = recall = ndcg = 0.0
        for i in data.test_set.keys():
            pos_items = set(data.test_set[i])
            num_pos = len(pos_items)
            if num_pos == 0:
                continue
            length += 1
            items_list = all_index_of_rank_list[i].tolist()
            items = set(items_list)

            num_hit = len(pos_items.intersection(items))

            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)

            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(num_pos, topk)):
                max_ndcg_score += 1 / math.log2(i + 2)
            if max_ndcg_score == 0:
                continue

            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / math.log2(i + 2)

            ndcg += ndcg_score / max_ndcg_score

        return precision / length, recall / length, ndcg / length



