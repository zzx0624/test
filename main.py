from load_data import Data
import preargs
import time
import numpy as np
import json
import torch
from ngcf_model import GCN
from torch.autograd import no_grad
from evaluate import evaluate_model

if __name__ == '__main__':
    args = preargs.parse_args()
    evaluation_threads = 16  # mp.cpu_count()
    data = Data(args.data_path + args.dataset, args.batch_size)
    train_edge = []
    for i in data.train_items.keys():
        items = data.train_items[i]
        for j in range(len(items)):
            train_edge.append([i, items[j]])
    train_edge = np.array(train_edge)
    n_users, n_items = data.n_users, data.n_items
    plain_adj, norm_adj, mean_adj = data.get_adj_mat()
    n_train, n_test = data.n_train, data.n_test
    t0 = time.time()
    model = GCN(args, n_users, n_items, norm_adj)
    if args.cuda:
        model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    max_precision, max_recall, max_NDCG = {}, {}, {}
    for i in range(1, 11):
        max_precision[i], max_recall[i], max_NDCG[i] = 0.0, 0.0, 0.0
    num_decreases = 0
    precision, recall, ndcg = {}, {}, {} 
    print('Now, training start ...')
    for epoch in range(args.epoch):
        print('第{}次迭代'.format(epoch + 1))
        n_batch = n_train // args.batch_size
        sum_loss = 0.
        model.train()
        t1 = time.time()
        for idx in range(n_batch):
            opt.zero_grad()
            users, pos_items, neg_items = data.sample()
            loss = model.create_bpr_loss(users, pos_items, neg_items)
            loss.backward()
            opt.step()
            sum_loss += loss.item()
        print('---------------- epoch:{} loss value:{} time:{}-------------'
              .format(epoch + 1, sum_loss / n_batch, time.time() - t1))
        if (epoch + 1) % 1 == 0 and (epoch + 1) > 20:
            model.eval()
            t2 = time.time()
            with no_grad():
                (precisions, recalls, ndcgs) = evaluate_model(model, n_users, data.test_set, data.test_negtive, 10, evaluation_threads)
                for i in range(1, 11):
                  precision[i], recall[i], ndcg[i] = np.array(precisions[i]).mean(), np.array(recalls[i]).mean(), np.array(ndcgs[i]).mean()
                  print('Iteration %d [%.1f s]: precision[%d] = %.4f, recall[%d] = %.4f, ndcg[%d] = %.4f [%.1f s]'
                      % (epoch + 1, t2 - t1, i, precision[i], i, recall[i], i, ndcg[i], time.time() - t2))
                for i in range(1, 11):
                    if recall[i] > max_recall[i]:
                        max_recall[i] = recall[i]
                        max_NDCG[i] = ndcg[i]
                        max_precision[i] = precision[i]
                        print('max_precision[%d]: %f, max_recall[%d]: %f, max_ndcg[%d]: %f' % (i, max_precision[i], i, max_recall[i], i, max_NDCG[i]))
