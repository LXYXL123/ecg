import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import wfdb
import ast
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import os

class ECGDatasetLoader(Dataset):
    def __init__(self, path, sampling_rate=100, type='train', use_cache=True):
        super().__init__()
        self.type = type
        self.use_cache = use_cache

        # 定义缓存路径
        cache_prefix = path + f'cache_{sampling_rate}_{type}'
        cache_x_path = os.path.join(cache_prefix, 'X.npy')
        cache_y_path = os.path.join(cache_prefix, 'y.npy')

        if use_cache and os.path.exists(cache_x_path) and os.path.exists(cache_y_path):
            print(f'loading cache data from {cache_x_path} and {cache_y_path}')
            self.X = np.load(cache_x_path)
            self.y = np.load(cache_y_path, allow_pickle=True)
            self.y = pd.Series(list(self.y))
            self._load_class_map()
            return


        # 1. 读取标签并解析诊断代码
        self.Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
        self.Y.scp_codes = self.Y.scp_codes.apply(ast.literal_eval)

        # 2. 读取信号
        self.X = self.load_raw_data(self.Y, sampling_rate, path)

        # 3. 读取诊断映射
        self.agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]

        # 4. 添加诊断大类
        self.Y['diagnostic_superclass'] = self.Y.scp_codes.apply(
            lambda x: self.aggregate_diagnostic(x, self.agg_df)
        )

        # 5. 多标签编码
        self.classes = sorted(set(cls for sublist in self.Y.diagnostic_superclass for cls in sublist))
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}
        self.Y['labels'] = self.Y.diagnostic_superclass.apply(self.multilabel_encode)

        # 6. 划分训练和测试
        test_fold = 10
        val_fold = 9
        if self.type == 'train':
            self.X = self.X[np.where(self.Y.strat_fold <= 8)]
            self.y = self.Y[self.Y.strat_fold <= 8].labels.reset_index(drop=True)
        elif self.type == 'val':
            self.X = self.X[np.where(self.Y.strat_fold == val_fold)]
            self.y = self.Y[self.Y.strat_fold == val_fold].labels.reset_index(drop=True)
        else:
            self.X = self.X[np.where(self.Y.strat_fold == test_fold)]
            self.y = self.Y[self.Y.strat_fold == test_fold].labels.reset_index(drop=True)

        # === 7. 保存缓存 ===
        if use_cache:
            print(f"Saving cached {type}_data...")
            os.makedirs(os.path.dirname(cache_x_path), exist_ok=True)
            np.save(cache_x_path, self.X)
            os.makedirs(os.path.dirname(cache_y_path), exist_ok=True)
            np.save(cache_y_path, self.y.to_numpy(object))
            print(f'{type}_data saved')
        

    def _load_class_map(self):
        """从标签恢复类名映射"""
        all_labels = set()
        for y in self.y:
            for i, val in enumerate(y):
                if val == 1.0:
                    all_labels.add(i)
        self.classes = sorted(list(all_labels))
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}

    def multilabel_encode(self, label_list):
        """将标签列表转换为 one-hot 向量"""
        label_vec = np.zeros(len(self.class_to_index), dtype=np.float32)
        for label in label_list:
            if label in self.class_to_index:
                label_vec[self.class_to_index[label]] = 1.0
        return label_vec

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)  # [L, 12]
        y = torch.tensor(self.y.iloc[idx], dtype=torch.float32)  # [n_classes]
        return x, y

    @staticmethod
    def load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path + f)[0] for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path + f)[0] for f in df.filename_hr]
        data = np.array(data)
        return data

    @staticmethod
    def aggregate_diagnostic(y_dic, agg_df):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
