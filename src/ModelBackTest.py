import os,sys
import json,json5
import time
import pandas as pd
import numpy as np
import dolphindb as ddb
import tqdm
from typing import List, Dict
from utils import init_path, get_glob_list
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelBackTest:
    def __init__(self, session: ddb.session, pool:ddb.DBConnectionPool,
                 factorDB: str, factorTB: str, factorList: str, freq: str, # 模型频率(D/M)
                 model_cfg: Dict, modelList: list,
                 selectDB: str, selectTB: str,  # 每期模型的特征: symbol tradeDate tradeTime featureName
                 labelDB: str, labelTB: str,    # 每期模型的标签
                 resultDB: str, resultTB: str,  # 每期模型的结果
                 savePath: str,  # 本地模型保存路径: savePath/{modelName}
                 labelFunc,
                 selectFunc,
                 seed: int = 42,
                 resultDesc: str = None):
        """
        session: DolphinDB Session
        pool: DolphinDB ConnectionPool
        resultDB: 数据库
        resultTB: 数据表
        factor_cfg: 因子池配置

        """
        # DolphinDB 配置
        self.session = session  # DolphinDB Session
        self.pool = pool        # DolphinDB Session Pool

        # 因子部分
        self.factorDB = factorDB
        self.factorTB = factorTB
        self.factor_list = factorList
        self.freq = freq

        # 模型部分
        self.model_cfg = model_cfg
        self.model_list = modelList    # 解析model_cfg
        init_path(path_dir=savePath)
        self.savePath = savePath

        # 数据库部分
        self.selectDB = selectDB
        self.selectTB = selectTB
        self.labelDB = labelDB
        self.labelTB = labelTB
        self.resultDB = resultDB
        self.resultTB = resultTB

        # 结果部分
        if not resultDesc:
            self.resultDesc = "ModelDefault"
        self.selectFunc = selectFunc
        self.labelFunc = labelFunc
        self.currentPeriod = 0
        self.periodDict = {}
        self.seed = seed

    def get_factorList(self, inplace: bool = True) -> Dict:
        """
        自动解析用户输入的dbName+tbName对应的因子, 输出一个字典
        数据格式:
        """
        factor_df = self.session.run(f"""
            select count(*) from loadTable("{self.factorDB}","{self.factorTB}") group by factor
        """)
        factor_list = factor_df["factor"].tolist()
        if inplace:
            self.factor_list = factor_list
        return factor_list

    def init_labelDB(self, dropDB: bool = False, dropTB: bool = False):
        """
        初始化标签数据库
        """
        if dropDB and self.session.existsDatabase(self.labelDB):
            self.session.dropDatabase(self.labelDB)
        if dropTB and self.session.existsTable(dbUrl=self.labelDB, tableName=self.labelTB):
            self.session.dropTable(dbPath=self.labelDB, tableName=self.labelTB)
        if not self.session.existsTable(self.labelDB, self.labelTB):
            """新建数据库表"""
            self.session.run(f"""
                db = database({self.labelDB},VALUE,2010.01M..2030.01M, engine="TSDB")
                schemaTb = table(1:0, ["symbol","TradeDate","TradeTime","method","label","MaxDate","MaxTime"],
                                    [SYMBOL,DATE,TIME,STRING,DOUBLE,DATE,TIME])
                db.createPartitionedTable(schemaTb, "{self.labelTB}", partitionColumns=`TradeDate, 
                    sortColumns=`method`symbol`TradeTime`TradeDate, keepDuplicates=LAST)
            """)

    def add_labelData(self):
        """添加标签数据至标签数据库"""
        self.labelFunc(self)
        print("标签数据添加完毕")

    def init_selectDB(self, dropDB: bool = False, dropTB: bool = False):
        """
        初始化特征选择数据库
        :param dropDB:
        :param dropTB:
        :return:
        """
        if dropDB and self.session.existsDatabase(self.selectDB):
            self.session.dropDatabase(self.selectDB)
        if dropTB and self.session.existsTable(dbUrl=self.selectDB, tableName=self.selectTB):
            self.session.dropTable(dbPath=self.selectDB, tableName=self.selectTB)
        if not self.session.existsTable(self.selectDB, self.selectTB):
            """新建数据库表"""
            self.session.run(f"""
                db = database({self.selectDB},VALUE,2010.01M..2030.01M, engine="TSDB")
                schemaTb = table(1:0, ["symbol","TradeDate","TradeTime","method","featureName"],
                                    [SYMBOL,DATE,TIME,STRING,SYMBOL])
                db.createPartitionedTable(schemaTb, "{self.selectTB}", partitionColumns=`TradeDate, 
                    sortColumns=`method`symbol`TradeTime`TradeDate, keepDuplicates=LAST)
                """)

    def add_selectData(self):
        """添加特征选择数据至特征数据库"""
        self.selectFunc(self)
        print("特征数据添加完毕")

    def init_resultDB(self, dropDB: bool = False, dropTB: bool = False):
        """
        初始化结果保存数据库 ->
        resultDesc symbol TradeDate TradeTime label labelPred method
        """
        if dropDB and self.session.existsDatabase(self.resultDB):
            self.session.dropDatabase(self.resultDB)
        if dropTB and self.session.existsTable(dbUrl=self.resultDB, tableName=self.resultTB):
            self.session.dropTable(dbPath=self.resultDB, tableName=self.resultTB)
        if not self.session.existsTable(self.resultDB, self.resultTB):
            """新建数据库表"""
            self.session.run(f"""
            db1 = database(,VALUE,2010.01M..2030.01M)
            db2 = database(,VALUE,[`Maxim,`DolphinDB])
            db = database("{self.resultDB}",partitionType=COMPO, partitionScheme=[db1, db2], engine="TSDB")
            schemaTb = table(1:0, ["resultDesc","symbol","TradeDate","TradeTime","method","label","label_pred"],
                                    [SYMBOL,SYMBOL,DATE,TIME,STRING,DOUBLE,DOUBLE])
            db.createPartitionedTable(schemaTb, "{self.resultTB}", partitionColumns=`TradeDate`resultClass, 
                sortColumns=`resultClass`method`symbol`TradeTime`TradeDate, keepDuplicates=LAST)
            """)

    def getData(self,
                startPeriod: int,
                endPeriod: int,
                factor_list: list = None,
                ) -> pd.DataFrame:
        """
        传入开始时间和结束时间
        获取数据的接口-> table(label,factor).lj(`symbol`TradeDate`TradeTime)
        factor_list： 这里如果不传, 就默认按照对应的factor_list去拿;
                      如果传了一个相等的startPeriod&endPeriod+这里传了factor_list, 就只取这些特征
        返回一张宽表
        """
        if not factor_list:
            factor_list = self.factor_list
        if startPeriod == endPeriod:
            # 说明只需要取一个Period的factor
            data = self.session.run(rf"""
                // 获取特征列名称
                featureName = select 
                
                // 从中解析对应的startTimeStamp & endTimeStamp
                
                // 获取标签数据
                
            """)
            return data

        if startPeriod < endPeriod:
            # 说明需要取一个区间的数据
            data = self.session.run(rf"""
            
            """)
            return
    def train(self, factor_list: list = None, model_list: list = None):
        self.factor_list = factor_list
        if not factor_list:
            for dbName, valueDict in factor_cfg.items():
                for tbName in valueDict.keys():
                    factor_list += valueDict[tbName]["factorList"]
                self.factor_list = factor_list
        self.trainFunc(self)

    def pred(self):
        """从本地指定路径加载模型进行增量预测"""

    def run(self, startDate:pd.Timestamp, startTime:int, endDate:pd.Timestamp, endTime:int):
        """
        策略运行的主体函数
        """
        # for loop: 时间列
            # 解析当前对应的period，若没有则跳过

            # ---train----

            # 提取对应时间的特征列数据[离他最近比他小]

            # 训练集/测试集划分

            # 进行并行训练skorch搜索参数

            # 保存模型

            # ---pred---

            # 调用模型进行预测

            # 保存至数据库


if __name__ == "__main__":
    session = ddb.session()
    session.connect("172.16.0.184",8001,"maxim","dyJmoc-tiznem-1figgu")
    pool=ddb.DBConnectionPool("172.16.0.184",8001,10,"maxim","dyJmoc-tiznem-1figgu")
    from model.Label import get_DayLabel
    from model.Select import get_DayFeature
    from model.Train import to_dolphindb

    with open(r".\config\factor_cfg.json5","r",encoding='utf-8') as f:
        factor_cfg = json5.load(f)
    with open(r".\config\model_cfg.json5","r",encoding="utf-8") as f:
        model_cfg = json5.load(f)
    M = ModelBackTest(session, pool,
                      factorDB="dfs://Dayfactor",
                      factorTB="pt", freq="D",
                      factorList= ['shio',
                       'shioStrong',
                       'shioStrong_avg20',
                       'shioStrong_std20',
                       'shio_avg20',
                       'shio_std20'
                      ],

                      model_cfg=model_cfg,
                      modelList=["lightgbm",
                                 "xgboost"],
                      savePath=r"D:\DolphinDB\Project\FactorModel\model",

                      selectDB="dfs://Select",selectTB="Select20250920",
                      resultDB="dfs://Model",resultTB="Model20250920",
                      labelDB="dfs://Label",labelTB="DayLabel",
                      selectFunc=get_DayFeature,
                      labelFunc=get_DayLabel,
                      resultDesc="result20250920")
    M.get_factorList(inplace=True)
    print(M.factor_list)
    # M.init_labelDB(dropDB=False,dropTB=True)        # 创建预测标签数据库
    # M.init_selectDB(dropDB=False,dropTB=True)       # 创建因子选择数据库
    # M.init_resultDB(dropDB=False,dropTB=True)       # 创建模型训练结果保存数据库
    # M.add_labelData()   # 添加标签数据库
    # M.add_selectData()  # 添加选择数据库
    # print(M.factor_cfg)