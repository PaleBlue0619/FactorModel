import os,re,glob
import datetime
from typing import List
import bisect

def init_path(path_dir):
    "创建当前.py目录下的文件夹"
    if os.path.exists(path=path_dir)==bool(False):
        os.mkdir(path=path_dir)

def get_glob_list(path_dir) -> List:
    "返回符合条件的文件名列表"
    return [os.path.basename(i) for i in glob.iglob(pathname=path_dir,recursive=False)]