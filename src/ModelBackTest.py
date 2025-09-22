import os,sys
import json,json5
import time
import pandas as pd
import numpy as np
import dolphindb as ddb
import torch
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator
from typing import List, Dict
from utils import init_path, get_glob_list
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelBackTest:
    def __init__(self, session: ddb.session, pool:ddb.DBConnectionPool,
                 startDate: str, endDate: str,
                 factorDB: str, factorTB: str, factorList: str, freq: str, # 模型频率(D/M)
                 model_cfg: Dict, savePath: str,  # 本地模型保存路径: savePath/{modelName}
                 selectDB: str, selectTB: str, selectMethod: str, selectFunc, # 每期模型的特征: symbol tradeDate tradeTime featureName
                 labelDB: str, labelTB: str, labelName: str, labelFunc,   # 每期模型的标签
                 resultDB: str, resultTB: str,  # 每期模型的结果
                 seed: int = 42, nJobs: int = -1,
                 callBackPeriod: int = 1,   # 利用过去K期数据进行训练
                 splitTrain: float = 0.8,   # 训练集划分比例
                 selfModel: Dict = None,
                 modelList: list = None,
                 resultDesc: str = None):
        """
        session: DolphinDB Session
        pool: DolphinDB ConnectionPool
        resultDB: 数据库
        resultTB: 数据表
        factor_cfg: 因子池配置

        """
        # 全局配置
        self.startDate = pd.Timestamp(startDate).strftime("%Y.%m.%d")
        self.endDate = pd.Timestamp(endDate).strftime("%Y.%m.%d")   # 2020.01.04
        self.periodDF = "periodDF"

        # DolphinDB 配置
        self.session = session  # DolphinDB Session
        self.pool = pool        # DolphinDB Session Pool

        # 因子部分
        self.factorDB = factorDB
        self.factorTB = factorTB
        self.factor_list = factorList
        self.freq = freq

        # 模型部分
        self.seed = seed
        self.nJobs = nJobs
        self.splitTrain = splitTrain
        self.callBackPeriod = callBackPeriod
        self.model_cfg = model_cfg
        self.modelFunc = {"adaboost": AdaBoostRegressor,      # AdaBoost
                          "gbdt": GradientBoostingRegressor,  # GBDT回归器
                          "lightgbm": LGBMRegressor,          # LightGBM
                          "mlp": MLPRegressor,                # MLP(sklearn)
                          "randomforest": RandomForestRegressor, # RandomForest
                          "xgboost": XGBRegressor             # XGBoost
                         }
        self.basicModel = ["adaboost","gbdt","lightgbm","mlp","randomforest","xgboost"]
        self.earlyStopModel = ["lightgbm","xgboost"]
        if selfModel:
            self.modelFunc.update(selfModel)
        self.model_list = self.modelFunc.keys() if not modelList else modelList
        init_path(path_dir=savePath)
        self.savePath = savePath

        # 数据库部分
        self.selectDB = selectDB
        self.selectTB = selectTB
        self.selectMethod = selectMethod
        self.labelDB = labelDB
        self.labelTB = labelTB
        self.labelName = labelName
        self.resultDB = resultDB
        self.resultTB = resultTB

        # 结果部分
        self.resultDesc = "ModelDefault" if not resultDesc else resultDesc
        self.selectFunc = selectFunc
        self.labelFunc = labelFunc

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

    def get_periodList(self):
        """
        返回一个共享变量->TradeDate TradeTime period
        """
        self.session.run(f"""
            freq = "{self.freq.lower()}"
            if (freq == "d"){{
                pt = select 15:00:00.000 as TradeTime, 1.0 as period 
                     from loadTable("{self.factorDB}","{self.factorTB}") 
                    group by date as TradeDate
            }}else{{
                pt = select 1.0 as period from loadTable("{self.factorDB}","{self.factorTB}")
                    group by date as TradeDate, time as TradeTime
            }}
            update pt set period = cumsum(period);
            share(pt, "{self.periodDF}");  // 共享变量
        """)

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
                db = database("{self.labelDB}",VALUE,2010.01M..2030.01M, engine="TSDB")
                schemaTb = table(1:0, ["symbol","TradeDate","TradeTime","labelName","label","MaxDate","MaxTime"],
                                    [SYMBOL,DATE,TIME,STRING,DOUBLE,DATE,TIME])
                db.createPartitionedTable(schemaTb, "{self.labelTB}", partitionColumns=`TradeDate, 
                    sortColumns=`labelName`symbol`TradeTime`TradeDate, keepDuplicates=LAST)
            """)

    def add_labelData(self):
        """添加标签数据至标签数据库"""
        self.labelFunc(self)
        print("标签数据添加完毕")

    def init_selectDB(self, dropDB: bool = False, dropTB: bool = False):
        """
        初始化特征选择数据库
        TradeDate,TradeTime,selectMethod,featureName
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
                db = database("{self.selectDB}",VALUE,2010.01M..2030.01M, engine="TSDB")
                schemaTb = table(1:0, ["TradeDate","TradeTime","selectMethod","featureName"],
                                    [DATE,TIME,STRING,SYMBOL])
                db.createPartitionedTable(schemaTb, "{self.selectTB}", partitionColumns=`TradeDate, 
                    sortColumns=`selectMethod`TradeTime`TradeDate, keepDuplicates=ALL)
                """)

    def add_selectData(self):
        """添加特征选择数据至特征数据库"""
        self.selectFunc(self)
        print("特征数据添加完毕")

    def init_resultDB(self, dropDB: bool = False, dropTB: bool = False):
        """
        初始化结果保存数据库 ->
        resultDesc symbol TradeDate TradeTime method label labelPred method
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
            schemaTb = table(1:0, ["Desc","symbol","TradeDate","TradeTime","label","labelPred"],
                                    [SYMBOL,SYMBOL,DATE,TIME,DOUBLE,DOUBLE])
            db.createPartitionedTable(schemaTb, "{self.resultTB}", partitionColumns=`TradeDate`Desc, 
                sortColumns=`Desc`symbol`TradeTime`TradeDate, keepDuplicates=LAST)
            """)

    def getData(self, selectMethod: str, labelName: str, startPeriod: int, endPeriod: int) -> pd.DataFrame:
        """
        传入开始时间和结束时间
        获取数据的接口-> table(label,factor).lj(`symbol`TradeDate`TradeTime)
        返回一张宽表: symbol,TradeDate,TradeTime,label,factorList
        """
        if startPeriod == endPeriod:
            # 说明只需要取一个Period的factor
            data = self.session.run(rf"""
                // 获取时间频率
                freq = "{self.freq.lower()}"
                
                // 获取这个period对应的TradeDate & TradeTime
                periodDF = objByName("{self.periodDF}")
                idx = find(periodDF[`period],{startPeriod})
                targetDate = periodDF[`TradeDate][idx]
                targetTime = periodDF[`TradeTime][idx]
            
                // 获取特征列名称
                featureList = exec featureName from  
                        from loadTable("{self.selectDB}","{self.selectTB}") 
                        where selectMethod == "{selectMethod}" and TradeDate == targetDate and TradeTime == targetTime
                
                if (size(featureList) == 0){{
                    print("period:{startPeriod}没有找到对应的特征,请重新设置")
                }}
                
                // 从因子数据库&标签数据库中获取数据
                if (freq=="d"){{    // 日频
                    // 因子数据
                    factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}") 
                                where factor in featureList and date == targetDate 
                                pivot by symbol, date as TradeDate
                    // 标签数据
                    label_df = select symbol,TradeDate,label from loadTable("{self.labelDB}","{self.labelTB}") 
                                where labelName == "{labelName}" and TradeDate == targetDate and TradeTime == targetTime
                    matchingCols = ["symbol","TradeDate"]            
                }}else{{    // 分钟频
                    // 因子数据
                    factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}")
                                where factor in featureList and date == targetDate and time == targetTime
                                pivot by symbol, date as TradeDate, time as TradeTime
                    // 标签数据
                    label_df = select symbol,TradeDate,TradeTime,label from loadTable("{self.labelDB}","{self.labelTB}") 
                                where labelName == "{labelName}" and TradeDate == targetDate and TradeTime == targetTime
                    matchingCols = ["symbol","TradeDate","TradeTime"]
                }}
                select * from lj(factor_df,label_df, matchingCols);
            """)
            return data

        if startPeriod < endPeriod:
            # 说明需要取一个区间的数据
            data = self.session.run(rf"""
                // 获取时间频率
                freq = "{self.freq.lower()}"

                // 获取这个period对应的TradeDate & TradeTime
                periodDF = objByName("{self.periodDF}")
                startIdx = find(periodDF[`period],{startPeriod})
                endIdx = find(periodDF[`period],{endPeriod})
                startDate = periodDF[`TradeDate][startIdx]
                endDate = periodDF[`TradeDate][endIdx]
                startTime = periodDF[`TradeTime][startIdx]
                endTime = periodDF[`TradeTime][endIdx]
                startTimeStamp = concatDateTime(startDate,startTime)
                endTimeStamp = concatDateTime(endDate,endTime)
                
                // 获取特征列名称
                featureList = exec featureName from  
                        from loadTable("{self.selectDB}","{self.selectTB}") 
                        where selectMethod == "{selectMethod}" 
                        and (concatDateTime(TradeDate,TradeTime) between startTimeStamp and endTimeStamp)

                if (size(featureList) == 0){{
                    print("period:{startPeriod}~{endPeriod}没有找到对应的特征,请重新设置")
                }}

                // 从因子数据库&标签数据库中获取数据
                if (freq=="d"){{    // 日频
                    // 因子数据
                    factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}") 
                                where factor in featureList and (date between startDate and endDate) 
                                pivot by symbol, date as TradeDate
                    // 标签数据
                    label_df = select symbol,TradeDate,label from loadTable("{self.labelDB}","{self.labelTB}") 
                                where labelName == "{labelName}" and (TradeDate between startDate and endDate)
                    matchingCols = ["symbol","TradeDate"]            
                }}else{{    // 分钟频
                    // 因子数据
                    factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}")
                                where factor in featureList and (concatDateTime(date,time) between startTimeStamp and endTimeStamp)
                                pivot by symbol, date as TradeDate, time as TradeTime
                    // 标签数据
                    label_df = select symbol,TradeDate,TradeTime,label from loadTable("{self.labelDB}","{self.labelTB}") 
                                where labelName == "{labelName}" and (concatDateTime(TradeDate,TradeTime) between startTimeStamp and endTimeStamp)
                    matchingCols = ["symbol","TradeDate","TradeTime"]
                }}
                select * from lj(factor_df,label_df, matchingCols);
            """)
            return data

    def train(self, x: any, y: any, modelName: str, cv:int = 5, evalSet: List[Tuple] = None) -> BaseEstimator:
        """
        Sklearn 网格搜索最优模型参数
        :param x: 训练集
        :param y: 测试集
        :param model_cfg: 模型待选参数
        :param modelName: 模型名称
        :param modelFunc: 模型函数方法
        :param cv: k折交叉验证
        :param evalSet: 验证集
        :return: Sklearn.BaseEstimator
        """

        # 参数准备 （解析model_cfg）
        default_params, grid_params = model_cfg["default_params"], model_cfg["grid_params"]
        input_dim = x.shape[1]
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if modelName not in self.basicModel:    # 说明不是基础模型, 是用户自定义模型, 需要调用getter方法初始化模型
            modelMethod = self.modelFunc[modelName][1](input_dim, **default_params)
        else:
            modelMethod = self.modelFunc[modelName](**default_params)
        grid = GridSearchCV(estimator=modelMethod,
                            param_grid=grid_params,
                            cv=cv,
                            n_jobs=self.nJobs)
        # 是否早停
        if modelName in self.earlyStopModel and evalSet:
            grid.fit(x, y, eval_set = evalSet)
        else:
            grid.fit(x, y)

        # 返回最佳模型
        best_params, best_scores = grid.best_params_, grid.best_score_
        if modelName not in self.basicModel:    # 说明不是基础模型, 是用户自定义模型, 需要调用getter方法初始化模型
            bestModel = self.modelFunc[modelName][1](input_dim, **best_params)
        else:
            bestModel = self.modelFunc[modelName](**best_params)

        return bestModel.fit(x, y)

    def pick(self, startPeriod: int, endPeriod: int, model_list: list, zscore: bool = True):
        """skorch训练模型并将最优模型保存至本地"""

    def pred(self, startPeriod: int, endPeriod: int, model_list: list):
        """从本地指定路径加载模型进行增量预测"""

    def run(self, startDate: pd.Timestamp, startTime: int, endDate: pd.Timestamp, endTime:int):
        """
        策略运行的主体函数
        startPeriod & endPeriod 的调度系统
        """
        # 准备
        startDate = pd.Timestamp(startDate).strftime("%Y.%m.%d")
        endDate = pd.Timestamp(endDate).strftime("%Y.%m.%d")
        startTime = str(startTime).zfill(4)
        startTime = f"{startTime[:2]}{startTime[2:]}:00.000"
        endTime = str(endTime).zfill(4)
        endTime = f"{endTime[:2]}{endTime[2:]}:00.000"
        for model in self.model_list:
            init_path(path_dir=rf"{self.savePath}/{model}")
        appender = ddb.PartitionedTableAppender(dbPath=self.resultDB,
                                                tableName=self.resultTB,
                                                partitionColName="date",
                                                dbConnectionPool=self.pool)  # 写入数据的appender
        totalPeriodList = self.session.run(f"""
        exec period from objByName("{self.periodDF}") 
                where concatDateTime(TradeDate, TradeTime) 
                between concatDateTime({startDate},"{startTime}") 
                and concatDateTime({endDate},"{endTime}")
        """)    # 所有需要回测的period list
        # 训练集&测试集划分函数
        self.session.run(f"""
        def trainTestSplit(data){{
            trainSplit = {self.splitTrain};
            setRandomSeed({self.seed});
            n = rows(data)
            allScores = randNormal(0,1,n)   // 所有行的得分
            allRows = 0..(n-1)
            trainRows = allRows[allScores>=percentile(allScores,trainSplit*100,"linear")]
            testRows = allRows[!(allRows in trainRows)]
            trainData = x[trainRows, ]
            testData = x[testRows, ]
            return trainData, testData
        }}
        """)


        for period in tqdm.tqdm(totalPeriodList, desc="Model BackTesting..."):
            if period<=self.callBackPeriod:
                continue
            # 解析当前对应的period，若没有则跳过
            totalData = self.getData(selectMethod=self.selectMethod,
                                     labelName=self.labelName,
                                     startPeriod=period-1-self.callBackPeriod,
                                     endPeriod=period-1)

            # Method
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
    from model.Dnn import CustomDNN,get_DNN
    from model.Resnet import CustomResNet,get_RESNET

    with open(r".\config\factor_cfg.json5","r",encoding='utf-8') as f:
        factor_cfg = json5.load(f)
    with open(r".\config\model_cfg.json5","r",encoding="utf-8") as f:
        model_cfg = json5.load(f)
    M = ModelBackTest(session, pool,
                      startDate="20200101", endDate="20250430",
                      factorDB="dfs://Dayfactor",
                      factorTB="pt", freq="D",
                      factorList=['shio',
                       'shioStrong',
                       'shioStrong_avg20',
                       'shioStrong_std20',
                       'shio_avg20',
                       'shio_std20'
                      ],
                      model_cfg=model_cfg, nJobs=-1, callBackPeriod=1,
                      modelList=["lightgbm","xgboost"], selfModel={"dnn": [CustomDNN, get_DNN], "resnet": [CustomResNet, get_RESNET]},
                      savePath=r"D:\DolphinDB\Project\FactorModel\model",
                      selectDB="dfs://Select",selectTB="Select20250920", selectMethod="ret5D",
                      labelDB="dfs://Label", labelTB="Label20250920", labelName="ret5D",
                      resultDB="dfs://Model",resultTB="Model20250920",
                      selectFunc=get_DayFeature,
                      labelFunc=get_DayLabel,
                      resultDesc="result20250920")
    M.get_factorList(inplace=True)
    M.get_periodList()
    print(M.factor_list)
    # M.init_labelDB(dropDB=False,dropTB=True)        # 创建预测标签数据库
    # M.init_selectDB(dropDB=False,dropTB=True)       # 创建因子选择数据库
    # M.init_resultDB(dropDB=False,dropTB=True)       # 创建模型训练结果保存数据库
    # M.add_labelData()   # 添加标签数据库
    # M.add_selectData()  # 添加选择数据库
    # print(M.factor_cfg)