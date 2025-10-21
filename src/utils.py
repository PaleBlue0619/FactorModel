import os,re,glob,json
import datetime
from typing import List
import bisect
import numpy as np
import pandas as pd
import pickle
import joblib

def init_path(path_dir):
    "创建当前.py目录下的文件夹"
    if not os.path.exists(path=path_dir):
        os.mkdir(path=path_dir)

def get_glob_list(path_dir) -> List:
    "返回符合条件的文件名列表"
    return [os.path.basename(i) for i in glob.iglob(pathname=path_dir,recursive=False)]

def save_model(model_obj, save_path:str, file_name:str, target_format:str):
    """
    保存模型为指定格式到本地指定路径
    target_format: ['pickle','joblib','npy','npz','bin']
    """
    init_path(path_dir=save_path)

    # 合并save_path：save_path\file_name.target_format
    save_path = rf"{save_path}\{file_name}.{target_format}"
    total_format_list = ["pickle","npy","npz","bin"]
    if target_format not in total_format_list:
        raise ValueError(f"target format must in {total_format_list}")

    if target_format == 'pickle':
        with open(save_path, 'wb') as f:
            pickle.dump(model_obj, f)

    elif target_format == 'joblib':  # 通常比pickle更适合scikit-learn模型
        joblib.dump(model_obj, save_path)

    elif target_format == 'npy':
        np.save(save_path, model_obj, allow_pickle=True)

    elif target_format == 'npz':
        np.savez(save_path, model=model_obj)

    elif target_format == 'bin':  # 自定义二进制格式
        with open(save_path, 'wb') as f:
            f.write(pickle.dumps(model_obj))


def load_model(load_path: str, file_name: str, target_format: str):
    """
    从指定路径加载模型

    Args:
        load_path: 加载路径（目录）
        file_name: 文件名（不含后缀）
        target_format: 文件格式，可选 ['pickle','joblib','npy','npz','bin']

    Returns:
        加载的模型对象

    Raises:
        ValueError: 当target_format不被支持时
        FileNotFoundError: 当文件不存在时
    """
    # 构建完整路径
    full_path = f"{load_path}/{file_name}.{target_format}"

    total_format_list = ["pickle", "joblib", "npy", "npz", "bin"]
    if target_format not in total_format_list:
        raise ValueError(f"target format must in {total_format_list}")

    if target_format == 'pickle':
        with open(full_path, 'rb') as f:
            return pickle.load(f)

    elif target_format == 'joblib':
        return joblib.load(full_path)

    elif target_format == 'npy':
        return np.load(full_path, allow_pickle=True)

    elif target_format == 'npz':
        return np.load(full_path)['model']

    elif target_format == 'bin':
        with open(full_path, 'rb') as f:
            return pickle.loads(f.read())


import pandas as pd


def generate_freq_folds(start_date, end_date, freq='D',
                        train_period: int = 20,
                        test_period: int = 5,
                        pred_period: int = 20):
    """
    生成多频率时序划分
    freq: D-日, W-周, M-月, Q-季
    """
    # 生成完整日期序列
    dates = pd.date_range(start_date, end_date, freq=freq)
    date_strs = [d.strftime('%Y%m%d') for d in dates]

    result = {}
    period = 0
    counter = 0
    while True:
        counter += 1
        # 计算各段索引
        train_test_split_idx = period + train_period - 1
        test_pred_split_idx = period + train_period + test_period - 1
        pred_end_idx = period + train_period + test_period + pred_period
        train_start = date_strs[period]
        train_end = (pd.Timestamp(date_strs[train_test_split_idx]) - pd.Timedelta(1, "d")).strftime("%Y%m%d")
        test_start = date_strs[train_test_split_idx]
        test_end = (pd.Timestamp(date_strs[test_pred_split_idx]) - pd.Timedelta(1, "d")).strftime("%Y%m%d")
        pred_start = date_strs[test_pred_split_idx]
        pred_end = (pd.Timestamp(date_strs[min(pred_end_idx, len(date_strs) - 1)]) - pd.Timedelta(1, "d")).strftime("%Y%m%d")

        # 检查是否超出范围
        if pred_end_idx >= len(date_strs):
            break

        result[str(counter)] = {
            "train": [train_start, 1500, train_end, 1500],
            "test": [test_start, 1500, test_end, 1500],
            "pred": [pred_start, 1500, pred_end, 1500]
        }
        period += pred_period + 1
    return result

if __name__ == "__main__":
    # 使用示例
    start_date = '20180101'
    end_date = '20251231'

    result = generate_freq_folds(start_date, end_date, freq="W",
                                 train_period=2, test_period=1, pred_period=10)
    with open(r"D:\DolphinDB\Project\FactorModel\src\config\period_cfg.json5", "w") as f:
        f.write(json.dumps(result))
