import os,sys
import json,json5
import time
import tqdm
import pandas as pd
import numpy as np
import dolphindb as ddb
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostRegressor  # AdaBoost
from sklearn.ensemble import RandomForestRegressor  # RandomForest
from sklearn.ensemble import GradientBoostingRegressor  # GBDT回归器
from sklearn.neural_network import MLPRegressor # MLP(sklearn版)
# from catboost import CatBoostRegressor  # CatBoost
from lightgbm import LGBMRegressor      # LGBM
from xgboost import XGBRegressor        # XGB
from typing import List, Dict, Tuple
from utils import init_path, get_glob_list, save_model, load_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelBackTest:
    def __init__(self, session: ddb.session, pool:ddb.DBConnectionPool,
                 startDate: str, endDate: str,
                 factorDB: str, factorTB: str, factorList: str, freq: str, toFactorDB: bool, # 模型频率(D/M)
                 model_cfg: Dict, savePath: str,  # 本地模型保存路径: savePath/{modelName}
                 selectDB: str, selectTB: str, selectMethod: str, selectFunc, # 每期模型的特征: symbol tradeDate tradeTime featureName
                 labelDB: str, labelTB: str, labelName: str, labelFunc,   # 每期模型的标签
                 resultDB: str, resultTB: str,  # 每期模型的结果
                 seed: int = 42, nJobs: int = -1, cv: int = 5, # K折交叉验证
                 earlyStopping: bool = False,   # 是否对能够使用早停的模型进行早停
                 callBackPeriod: int = 1,   # 利用过去K期数据进行训练
                 splitTrain: float = 0.8,   # 训练集划分比例
                 selfModel: Dict = None,
                 modelList: list = None,
                 factorPrefix: str = ""     # toFactorDB == True时生效, 保存至因子数据库的因子为factorPrefix+ModelName
                 ):
        """
        :param session: DolphinDB session
        :param pool: DolphinDB Connection Pool
        :param startDate: 回测开始日期
        :param endDate: 回测结束日期
        :param factorDB: 因子数据库名称
        :param factorTB: 因子数据表名称
        :param factorList: 所有候选的因子池
        :param freq: 因子频率->目前只支持同频训练+预测
        :param toFactorDB: 是否将合成后的因子保存回FactorDB
        :param model_cfg: 模型参数字典
        :param savePath: 模型保存路径, 实际路径为savePath/modelName/xxx.bin
        :param selectDB: 因子(X)选择数据库
        :param selectTB: 因子(X)选择数据表
        :param selectMethod: 特征选择方式
        :param selectFunc: 特征选择函数
        :param labelDB: 标签(Y)选择数据库
        :param labelTB: 标签(Y)选择数据表
        :param labelName: 标签构造名称
        :param labelFunc: 标签构造函数
        :param resultDB: 结果数据库: Y~Y_Pred
        :param resultTB: 结果数据表
        :param seed: 随机种子
        :param nJobs: 训练多个模型的并行度
        :param cv: K折交叉验证
        :param earlyStopping: 是否对于能够早停的模型进行早停
        :param callBackPeriod: 回看期:利用[t-callBackPeriod,t-1]的样本进行训练
        :param splitTrain: 训练集-测试集划分比例
        :param selfModel: 自定义模型配置项
        :param modelList: 所有使用的模型list
        :param factorPrefix: 保存至因子数据库的合成因子前缀, 完整因子名为factorPrefix+ModelName
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
        self.toFactorDB = toFactorDB
        self.factorPrefix = factorPrefix

        # 模型部分
        self.seed = seed
        self.cv = cv
        self.nJobs = nJobs
        self.splitTrain = splitTrain
        self.callBackPeriod = callBackPeriod
        self.earlyStopping = earlyStopping
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
        self.model_list = [i.lower() for i in self.modelFunc.keys()] if not modelList else modelList
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
        最终格式: symbol TradeDate TradeTime period label factor
        共享变量: TradeDate TradeTime MaxDate MaxTime period
        """
        self.session.run(f"""
            freq = "{self.freq.lower()}"
            if (freq == "d"){{ // 日频
                factorPt = select 15:00:00.000 as TradeTime, 1.0 as period 
                     from loadTable("{self.factorDB}","{self.factorTB}") 
                    group by date as TradeDate
                labelPt = select TradeDate, TradeTime, MaxDate, MaxTime 
                    from loadTable("{self.labelDB}","{self.labelTB}")
                    where labelName == "{self.labelName}" context by TradeDate,TradeTime limit 1
                pt = lj(factorPt, labelPt, `TradeDate`TradeTime)
            }}else{{ // 分钟频
                factorPt = select 1.0 as period from loadTable("{self.factorDB}","{self.factorTB}")
                    group by date as TradeDate, time as TradeTime
                labelPt = select TradeDate, TradeTime, MaxDate, MaxTime 
                    from loadTable("{self.labelDB}","{self.labelTB}")
                    where labelName == "{self.labelName}" context by TradeDate,TradeTime limit 1
                pt = lj(factorPt, labelPt, `TradeDate`TradeTime)
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
            db = database("{self.resultDB}",VALUE,2010.01M..2030.01M, engine="TSDB")
            schemaTb = table(1:0, ["ModelName","symbol","TradeDate","TradeTime","label","labelPred"],
                                    [SYMBOL,SYMBOL,DATE,TIME,DOUBLE,DOUBLE])
            db.createPartitionedTable(schemaTb, "{self.resultTB}", partitionColumns=`TradeDate, 
                sortColumns=`ModelName`symbol`TradeTime`TradeDate, keepDuplicates=LAST)
            """)

    def getData(self, selectMethod: str, labelName: str,
                startPeriod: int, endPeriod: int, factorList: list = None) -> pd.DataFrame:
        """
        返回一张宽表: symbol,TradeDate,TradeTime,label,factorList
        :param selectMethod: 选择方式str
        :param labelName: 标签名称str
        :param startPeriod: 开始的period int
        :param endPeriod: 结束的period int
        :param factorList: Optional, 不填则根据[startPeriod,endPeriod]区间的所有factorList进行选择
        :return:
        """
        if not factorList:
            factorList = []

        if startPeriod == endPeriod:
            # 说明只需要取一个Period的factor
            data = self.session.run(rf"""
                // 获取时间频率
                freq = "{self.freq.lower()}"
                
                // 获取这个period对应的TradeDate & TradeTime
                DF = select * from objByName("{self.periodDF}")
                idx = find(DF[`period],{startPeriod})
                targetDate = DF[`TradeDate][idx]
                targetTime = DF[`TradeTime][idx]
            
                // 获取特征列名称
                featureList = exec featureName from  
                        loadTable("{self.selectDB}","{self.selectTB}") 
                        where selectMethod == "{selectMethod}" and TradeDate == targetDate and TradeTime == targetTime
                
                if (size({factorList})>0){{
                    featureList = {factorList}
                }}
                
                if (size(featureList) == 0){{
                    print("period:{startPeriod}没有找到对应的特征,请重新设置")
                }}
                
                // 从因子数据库&标签数据库中获取数据
                if (freq=="d"){{    // 日频
                    // 因子数据
                    factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}") 
                                where factor in featureList and date == targetDate 
                                pivot by symbol, date as TradeDate, factor
                    update factor_df set TradeTime = 15:00:00.000
                    // 标签数据
                    label_df = select symbol,TradeDate,label from loadTable("{self.labelDB}","{self.labelTB}") 
                                where labelName == "{labelName}" and TradeDate == targetDate and TradeTime == targetTime
                    matchingCols = ["symbol","TradeDate"]            
                }}else{{    // 分钟频
                    // 因子数据
                    factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}")
                                where factor in featureList and date == targetDate and time == targetTime
                                pivot by symbol, date as TradeDate, time as TradeTime, factor
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
                DF = select * from objByName("{self.periodDF}")
                startIdx = find(DF[`period],{startPeriod})
                endIdx = find(DF[`period],{endPeriod})
                startDate = DF[`TradeDate][startIdx]
                endDate = DF[`TradeDate][endIdx]
                startTime = DF[`TradeTime][startIdx]
                endTime = DF[`TradeTime][endIdx]
                startTimeStamp = concatDateTime(startDate,startTime)
                endTimeStamp = concatDateTime(endDate,endTime)
                
                // 获取特征列名称
                featureList = exec featureName  
                        from loadTable("{self.selectDB}","{self.selectTB}") 
                        where selectMethod == "{selectMethod}" 
                        and (concatDateTime(TradeDate,TradeTime) between startTimeStamp and endTimeStamp)
                
                if (size({factorList}) > 0){{
                    featureList = {factorList}
                }}
                
                if (size(featureList) == 0){{
                    print("period:{startPeriod}~{endPeriod}没有找到对应的特征,请重新设置")
                }}

                // 从因子数据库&标签数据库中获取数据
                if (freq=="d"){{    // 日频
                    // 因子数据
                    factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}") 
                                where factor in featureList and (date between startDate and endDate) 
                                pivot by symbol, date as TradeDate, factor
                    update factor_df set TradeTime = 15:00:00.000
                    // 标签数据
                    label_df = select symbol,TradeDate,label from loadTable("{self.labelDB}","{self.labelTB}") 
                                where labelName == "{labelName}" and (TradeDate between startDate and endDate)
                    matchingCols = ["symbol","TradeDate"]            
                }}else{{    // 分钟频
                    // 因子数据
                    factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}")
                                where factor in featureList and (concatDateTime(date,time) between startTimeStamp and endTimeStamp)
                                pivot by symbol, date as TradeDate, time as TradeTime, factor
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
        model_cfg = self.model_cfg[modelName]
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

    def run(self, startDate: pd.Timestamp, startTime: int, endDate: pd.Timestamp, endTime:int):
        """
        策略运行的主体函数
        startPeriod & endPeriod 的调度系统
        """
        # 准备
        startDate = pd.Timestamp(startDate).strftime("%Y.%m.%d")
        endDate = pd.Timestamp(endDate).strftime("%Y.%m.%d")
        startTime = str(startTime).zfill(4)
        startTime = f"{startTime[:2]}:{startTime[2:]}:00.000"
        endTime = str(endTime).zfill(4)
        endTime = f"{endTime[:2]}:{endTime[2:]}:00.000"
        for model in self.model_list:
            init_path(path_dir=rf"{self.savePath}/{model}")
        resultAppender = ddb.PartitionedTableAppender(dbPath=self.resultDB,
                                     tableName=self.resultTB,
                                     partitionColName="TradeDate",
                                     dbConnectionPool=self.pool)  # 写入数据的appender
        factorAppender = ddb.TableAppender(dbPath=self.factorDB,
                                            tableName=self.factorTB,
                                           ddbSession=self.session)

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
            trainData = data[trainRows, ]
            testData = data[testRows, ]
            return trainData, testData
        }}
        """)
        # 确定最小的period -> select 有时会舍弃一部分开头的数据
        minPeriod = self.session.run(f"""
        t = exec min(concatDateTime(TradeDate,TradeTime)) 
            from loadTable("{self.selectDB}","{self.selectTB}")
        // 查询最小时间戳对应着的period int
        DF = select *, concatDateTime(TradeDate,TradeTime) as TimeStamp 
             from objByName("{self.periodDF}")
        idx = find(DF[`TimeStamp], t);
        minPeriod = DF[`period][idx]; 
        
        // 返回select有数据对应的最小的Period
        minPeriod
        """)
        periodDF = self.session.run(f"""
        select * from objByName("{self.periodDF}")
        """)    # 直接把所有数据取至内存

        for idx,row in tqdm.tqdm(periodDF.iterrows(), total=periodDF.shape[0], desc="Model BackTesting"):
            currentDate, currentTime, currentPeriod = row["MaxDate"], row["MaxTime"], row["period"]
            currentTimeStamp = pd.Timestamp.combine(currentDate.date(), currentTime.time())
            if currentPeriod<=minPeriod:    # 至少留一期历史的
                continue

            # 解析当前对应的period，若没有则跳过
            trainData = self.getData(selectMethod=self.selectMethod,
                                     labelName=self.labelName,
                                     startPeriod=currentPeriod-self.callBackPeriod,
                                     endPeriod=currentPeriod-1)
            factorList = [i for i in list(trainData.columns)
                          if i not in ["symbol","TradeDate","TradeTime","label"]]
            predData = self.getData(selectMethod=self.selectMethod,
                                    labelName=self.labelName,
                                    startPeriod=currentPeriod,
                                    endPeriod=currentPeriod,
                                    factorList=factorList   # 这里与训练集的特征保持一致
                                    )
            x, y = trainData[factorList], trainData["label"]
            train_x, eval_x, train_y, eval_y = train_test_split(x, y,
                                                        test_size=self.splitTrain,
                                                        random_state=self.seed)
            pred_x, real_y = predData[factorList], predData["label"]
            # ---train---
            for modelName in self.model_list:
                if modelName in self.earlyStopModel and self.earlyStopping: # 启用早停机制
                    bestModel = self.train(train_x, train_y, modelName=modelName, cv=self.cv,
                               evalSet=[(eval_x, eval_y)])
                else:   # 不启用早停机制
                    bestModel = self.train(x, y, modelName=modelName, cv=self.cv, evalSet=None)
                # 保存模型
                save_model(bestModel,
                           save_path=rf"{self.savePath}/{modelName}",
                           file_name=str(currentTimeStamp.strftime("%Y-%m-%d_%H-%M-%S")),
                           target_format="bin")
            # ---pred---
            for modelName in self.model_list:
                bestModel = load_model(load_path=rf"{self.savePath}/{modelName}",
                           file_name=str(currentTimeStamp.strftime("%Y-%m-%d_%H-%M-%S")),
                           target_format="bin")
                res = predData[["symbol","TradeDate","TradeTime","label"]]
                res["TradeDate"] = currentDate
                res["TradeTime"] = currentTime
                res["labelPred"] = bestModel.predict(pred_x)
                res.insert(0, "ModelName", [modelName]*res.shape[0])
                # 保存至数据库
                resultAppender.append(res)

                if self.toFactorDB:
                    res["ModelName"] = res["ModelName"].add_prefix(self.factorPrefix)   # 添加前缀
                    if self.freq.lower() == "d":
                        factorAppender.append(res[["symbol","TradeDate","ModelName","label"]])
                    else:
                        factorAppender.append(res[["symbol","TradeDate","TradeTime","ModelName","label"]])

if __name__ == "__main__":
    session = ddb.session()
    session.connect("172.16.0.184",8001,"maxim","dyJmoc-tiznem-1figgu")
    pool=ddb.DBConnectionPool("172.16.0.184",8001,10,"maxim","dyJmoc-tiznem-1figgu")
    from model.Label import get_DayLabel
    from model.Select import get_DayFeature
    from model.Dnn import CustomDNN,get_DNN
    from model.Resnet import CustomResNet,get_RESNET

    with open(r".\config\factorPool_cfg.json5","r",encoding='utf-8') as f:
        factor_cfg = json5.load(f)
    with open(r".\config\model_cfg.json5","r",encoding="utf-8") as f:
        model_cfg = json5.load(f)
    M = ModelBackTest(session, pool,
                      startDate="20200101", endDate="20250430",
                      factorDB="dfs://Dayfactor",
                      factorTB="pt", freq="D", toFactorDB=True, factorPrefix="ShioInter_",
                      factorList=["shio",
                            "shio_avg20",
                            "shio_std20",
                            "shioStrong",
                            "shioStrong_avg20",
                            "shioStrong_std20",
                            "shioWeak",
                            "shioWeak_avg20",
                            "shioWeak_std20",
                            "interDayReturn",
                            "interDayReturn_avg20",
                            "interDayReturn_std20"],
                      model_cfg=model_cfg, cv=5, nJobs=-1, callBackPeriod=1, earlyStopping=True,
                      modelList=["lightgbm","xgboost"],
                      selfModel={"dnn": [CustomDNN, get_DNN],
                                "resnet": [CustomResNet, get_RESNET]},
                      savePath=r"D:\DolphinDB\Project\FactorModel\model",
                      labelDB="dfs://DayLabel", labelTB="pt", labelName="ret5D",
                      selectDB="dfs://Select",selectTB="Select20250920", selectMethod="ret5D",
                      resultDB="dfs://Model",resultTB="Model20250920",
                      selectFunc=get_DayFeature,
                      labelFunc=get_DayLabel)
    M.get_factorList(inplace=True)
    M.get_periodList()
    # print(M.factor_list)
    # M.init_labelDB(dropDB=False,dropTB=True)        # 创建预测标签数据库
    # M.init_selectDB(dropDB=False,dropTB=True)       # 创建因子选择数据库
    # M.init_resultDB(dropDB=False,dropTB=True)       # 创建模型训练结果保存数据库
    # M.add_labelData()   # 添加标签数据库
    # M.add_selectData()  # 添加选择数据库
    M.run(M.startDate, 1500, M.endDate, 1500)
