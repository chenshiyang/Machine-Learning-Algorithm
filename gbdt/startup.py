# -*- coding:utf-8 -*-
"""
如下为credit.data.csv文件的训练信息
iter1 : train loss=0.400147
iter2 : train loss=0.267936
iter3 : train loss=0.186018
iter4 : train loss=0.133488
iter5 : train loss=0.090084
iter6 : train loss=0.068898
iter7 : train loss=0.057030
iter8 : train loss=0.047719
iter9 : train loss=0.037887
iter10 : train loss=0.030659
iter11 : train loss=0.024336
iter12 : train loss=0.020004
iter13 : train loss=0.016107
iter14 : train loss=0.014062
iter15 : train loss=0.012457
iter16 : train loss=0.010846
iter17 : train loss=0.009104
iter18 : train loss=0.007611
iter19 : train loss=0.005936
iter20 : train loss=0.005033
"""
from gbdt.data import DataSet
from gbdt.model import GBDT

# import sys
# sys.path.append("D:\GitHub-project\Machine-Learning-Algorithm\gbdt")

if __name__ == '__main__':
    data_file = '../data/credit.data.csv'
    dateset = DataSet(data_file)
    gbdt = GBDT(max_iter=20, sample_rate=0.8, learn_rate=0.5, max_depth=7, loss_type='binary-classification')
    gbdt.fit(dateset, dateset.get_instances_idset())