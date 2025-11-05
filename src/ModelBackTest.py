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

class BasicModelBackTest:   # 基础训练类
    def __init__(self, session: ddb.session, pool:ddb.DBConnectionPool,
                 factorDB: str, factorTB: str, factorList: str, freq: str, toFactorDB: bool,  # 模型频率(D/M)
                 model_cfg: Dict, savePath: str,  # 本地模型保存路径: savePath/{modelName}
                 selectDB: str, selectTB: str, selectMethod: str, selectFunc,
                 # 每期模型的特征: symbol tradeDate tradeTime featureName
                 labelDB: str, labelTB: str, labelName: str, labelFunc,  # 每期模型的标签
                 resultDB: str, resultTB: str,  # 每期模型的结果
                 seed: int = 42, nJobs: int = -1, cv: int = 5,  # K折交叉验证
                 earlyStopping: bool = False,  # 是否对能够使用早停的模型进行早停
                 selfModel: Dict = None,
                 modelList: list = None,
                 factorPrefix: str = ""  # toFactorDB == True时生效, 保存至因子数据库的因子为factorPrefix+ModelName
                 ):
        """
        :param session: DolphinDB session
        :param pool: DolphinDB Connection Pool
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
        :param selfModel: 自定义模型配置项
        :param modelList: 所有使用的模型list
        :param factorPrefix: 保存至因子数据库的合成因子前缀, 完整因子名为factorPrefix+ModelName
        """

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
        self.cache = {} # 此时有标签的数据
        # # TODO: 这里换成异步接口MultiThreadedTableWriter, 同步分区上传会减慢整体运行速度
        self.resultAppender = None
        self.factorAppender = None
        # 结果部分
        self.selectFunc = selectFunc
        self.labelFunc = labelFunc

    def uploadResultData(self, data: pd.DataFrame):
        if not self.resultAppender:
            # self.resultAppender = ddb.PartitionedTableAppender(dbPath=self.resultDB,
            #                                                    tableName=self.resultTB,
            #                                                    partitionColName="TradeDate",
            #                                                    dbConnectionPool=self.pool)  # 写入数据的appender
            self.resultAppender = ddb.MultithreadedTableWriter(host=session.host, port=session.port,
                                                                  userId=session.userid, password=session.password,
                                                                  dbPath=self.resultDB, tableName=self.resultTB,
                                                                  partitionCol="TradeDate", threadCount=1)
        for index, row in data.iterrows():
            self.resultAppender.insert(*row.values)

        # self.resultAppender.append(data)

    def uploadFactorData(self, data: pd.DataFrame):
        if not self.factorAppender:
            # self.factorAppender = ddb.TableAppender(dbPath=self.factorDB,
            #                                         tableName=self.factorTB,
            #                                         ddbSession=self.session)
            self.factorAppender = ddb.MultithreadedTableWriter(host=session.host, port=session.port,
                                                                  userId=session.userid, password=session.password,
                                                                  dbPath=self.factorDB, tableName=self.factorTB,
                                                                  partitionCol="date", threadCount=1)
        for index,row in data.iterrows():
            self.factorAppender.insert(*row.values)

    def get_factorList(self, inplace: bool = True) -> Dict:
        """
        根据传入的factorDB设置factor_list, 并返回
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
        ModelName symbol TradeDate TradeTime label labelPred
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
                startDate: pd.Timestamp, startTime: int, endDate: pd.Timestamp, endTime: int,
                factorList: list = None, isPred: bool = False) -> pd.DataFrame:
        """
        @ 如果是WFA方式的话，那么预测集在取数的时候是永远永远用不到缓存的, 用到缓存说明取数方式有问题
        新增: self.cache = {"selectMethod$labelName": {pd.Timestamp: data}} 表示缓存,
        从而使得根据缓存中的数据动态本次增量查询的SQL长度
        返回一张宽表: symbol,TradeDate,TradeTime,label,factorList
        :param selectMethod: 选择方式str
        :param labelName: 标签名称str
        :param startDate: 开始日期
        :param startTime: 开始时间
        :param endDate: 结束日期
        :param endTime: 结束时间
        :param factorList: Optional, 不填则根据[startPeriod,endPeriod]区间的所有factorList进行选择
        :param isPred: 取出来的是否是预测数据集
        :return:
        """
        if not factorList:
            factorList = []
        startTime = str(startTime).zfill(4)
        endTime = str(endTime).zfill(4)
        oriStartDate = startDate  # 记录开始时间, 考虑一种情况, 本来只需要取20200115~20200201的数据,但是现在数据集里面只有~20200101的数据,此时startDate会变成20200101,最后会取出20200101~20200201的数据,相当于多取了
        cacheKey = f"{selectMethod}${labelName}"
        state = 1   # 0: 表示不用跑SQL 1: 表示需要增量跑 SQL
        if cacheKey in self.cache.keys():
            cacheDateList = list(self.cache[cacheKey].keys())
            maxCacheDate = max(cacheDateList)
            if maxCacheDate > endDate:  # 说明不用跑SQL了
                state = 0
            else:
                startDate = maxCacheDate  # 修改开始时间为最大时间
        else:
            self.cache[cacheKey] = {}

        returnNoLabel = False   # 是否返回无标签数据
        dataDict = {}
        # TODO: 应该先取的是label-> MaxDate & MaxTime 对应的 TradeDate & TradeTime 之后, 再去取因子
        if state == 1:
            if startDate == endDate and startTime == endTime: # 说明取数的人是知道他在干什么的, 用tradeDate+tradeTime匹配即可
                # 说明只需要取一个Period的factor
                data: pd.DataFrame = self.session.run(rf"""
                    // 配置项
                    freq = "{self.freq.lower()}"
                    targetDate = {startDate.strftime("%Y.%m.%d")}
                    targetTime = {str(startTime[:2])+":"+str(startTime[2:])+":00.000"}
                    
                    // 获取特征列名称
                    featureList = exec distinct featureName from  
                            loadTable("{self.selectDB}","{self.selectTB}") 
                            where selectMethod == "{selectMethod}" and TradeDate == targetDate and TradeTime == targetTime
                    
                    if (size({factorList})>0){{
                        featureList = {factorList}
                    }}
                    
                    if (size(featureList) == 0){{
                        print("date-time:{startDate}-{startTime}没有找到对应的特征,请重新设置")
                    }}
                    
                    // 从因子数据库&标签数据库中获取数据
                    if (freq=="d"){{    // 日频
                        // 因子数据
                        factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}") 
                                    where factor in featureList and date == targetDate 
                                    pivot by symbol,date as TradeDate, factor
                        // 标签数据
                        label_df = select symbol,TradeDate,label from loadTable("{self.labelDB}","{self.labelTB}") 
                                    where labelName == "{labelName}" and TradeDate == targetDate
                        matchingCols = ["symbol","TradeDate"]            
                    }}else{{    // 分钟频
                        // 因子数据
                        factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}")
                                    where factor in featureList and date == targetDate and time == targetTime
                                    pivot by symbol, date as TradeDate, time as TradeTime, factor
                        // 标签数据
                        label_df = select symbol,TradeDate,TradeTime,label 
                                    from loadTable("{self.labelDB}","{self.labelTB}") 
                                    where labelName == "{labelName}" and TradeDate == targetDate and TradeTime == targetTime
                        matchingCols = ["symbol","TradeDate","TradeTime"]
                    }}
                    if ({int(isPred)} == 1){{  // 预测集不需要考虑标签
                        result = select * from lj(factor_df, label_df, matchingCols);
                    }}else{{
                        result = select * from lj(label_df, factor_df, matchingCols);
                    }}
                    result
                """, disableDecimal=True)
                return data     # 只查一个的不走缓存

            else:   # 说明取数的人大概率知道里面有未来数据，需要剔除
                # 说明需要取一个区间的数据
                dataDict: Dict[str, pd.DataFrame] = self.session.run(rf"""
                    // 配置项
                    freq = "{self.freq.lower()}"
                    startDate = {startDate.strftime("%Y.%m.%d")}
                    startTime = {str(startTime[:2])+":"+str(startTime[2:])+":00.000"}
                    endDate = {endDate.strftime("%Y.%m.%d")}
                    endTime = {str(endTime[:2])+":"+str(endTime[2:])+":00.000"}
                    startTimeStamp = concatDateTime(startDate, startTime);
                    endTimeStamp = concatDateTime(endDate, endTime);
                    
                    // 获取特征列名称
                    featureList = exec distinct featureName  
                            from loadTable("{self.selectDB}","{self.selectTB}") 
                            where selectMethod == "{selectMethod}" 
                            and (concatDateTime(TradeDate,TradeTime) between startTimeStamp and endTimeStamp)
                    
                    if (size({factorList}) > 0){{
                        featureList = {factorList}
                    }}
                    
                    if (size(featureList) == 0){{
                        print("timestamp: "+string(startTimeStamp)+"~"+string(endTimeStamp)+" 没有找到对应的特征,请重新设置")
                    }}
    
                    // 从因子数据库&标签数据库中获取数据
                    if (freq=="d"){{    // 日频
                        // 标签数据(多取一列MaxDate用于存入对应日期的Cache)
                        label_df = select symbol,TradeDate,MaxDate,label from loadTable("{self.labelDB}","{self.labelTB}") 
                                    where labelName == "{labelName}" and (MaxDate between startDate and endDate) 
                        if ({int(isPred)} == 1){{
                            minDate = startDate
                            maxDate = endDate
                        }}else{{
                            dateList = sort(exec distinct TradeDate from label_df)
                            minDate = first(dateList)
                            maxDate = last(dateList)
                        }}
                        // 因子数据
                        factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}") 
                                    where factor in featureList and (date between minDate and endDate) 
                                    pivot by symbol, date as TradeDate, factor
                        update factor_df set TradeTime = 15:00:00.000                                   
                        matchingCols = ["symbol","TradeDate"]            
                    }}else{{    // 分钟频
                        // 标签数据(多取一列MaxDate用于存入对应日期的Cache)
                        label_df = select symbol,TradeDate,TradeTime,MaxDate,label from loadTable("{self.labelDB}","{self.labelTB}") 
                                    where labelName == "{labelName}" and (concatDateTime(MaxDate,MaxTime) between startTimeStamp and endTimeStamp)
                        if ({int(isPred)} == 1){{
                            minTimestamp = startTimeStamp
                            maxTimestamp = endTimeStamp
                        }}else{{
                            timeList = sort(exec distinct concat(TradeDate,TradeTime) from label_df)
                            minTimestamp = first(timeList)
                            maxTimestamp = last(timeList)
                        }}
                        // 因子数据
                        factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}")
                                    where factor in featureList and (concatDateTime(date,time) between minTimestamp and maxTimestamp)
                                    pivot by symbol, date as TradeDate, time as TradeTime, factor
                        matchingCols = ["symbol","TradeDate","TradeTime"]
                    }}
                    if ({int(isPred)} == 1){{  // 预测集不需要考虑标签
                        data = select * from lj(factor_df, label_df, matchingCols);
                    }}else{{
                        data = select * from lj(label_df, factor_df, matchingCols);
                    }}
                    result = dict(STRING, ANY)
                    result["hasLabel"] = select * from data where not isNull(label);
                    data0 = select * from data where isNull(label)
                    if (rows(data0) != 0){{
                        result["noLabel"] = select * from data where isNull(label);
                    }}
                    result
                """, disableDecimal=True)

            # 只将有Label的数据写入缓存,防止污染缓存数据
            cacheData = dataDict["hasLabel"]
            for data in set(cacheData["MaxDate"]):
                self.cache[cacheKey][data] = cacheData[cacheData["MaxDate"] == data].reset_index(drop=True)
            if isPred and "noLabel" in dataDict.keys(): # 只有预测数据需要返回无标签数据
                returnNoLabel = True

        # 缓存里的数据都有标签
        result = pd.concat([self.cache[cacheKey][date]
                    for date in self.cache[cacheKey].keys() if oriStartDate <= date <= endDate],
                    axis=0, ignore_index=True)
        if not returnNoLabel:
            return result
        return pd.concat([result, dataDict["noLabel"]])

    def train(self, x: any, y: any, modelName: str, cv:int = 5, evalSet: List[Tuple] = None) -> BaseEstimator:
        """
        Sklearn 网格搜索最优模型参数
        :param x: 特征数据
        :param y: 标签数据
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
                            cv=cv, error_score='raise',
                            n_jobs=self.nJobs)
        # 是否早停
        if modelName in self.earlyStopModel and evalSet is not None:
            grid.fit(x, y, eval_set=evalSet)
        else:
            grid.fit(x, y)

        # 返回最佳模型
        best_params, best_scores = grid.best_params_, grid.best_score_
        if modelName not in self.basicModel:    # 说明不是基础模型, 是用户自定义模型, 需要调用getter方法初始化模型
            bestModel = self.modelFunc[modelName][1](input_dim, **best_params)
        else:
            bestModel = self.modelFunc[modelName](**best_params)

        return bestModel.fit(x, y)

class ModelBackTest(BasicModelBackTest):
    def __init__(self, session: ddb.session, pool: ddb.DBConnectionPool, periodDict: dict,
                 factorDB: str, factorTB: str, factorList: str, freq: str, toFactorDB: bool,  # 模型频率(D/M)
                 model_cfg: Dict, savePath: str,  # 本地模型保存路径: savePath/{modelName}
                 selectDB: str, selectTB: str, selectMethod: str, selectFunc,
                 # 每期模型的特征: symbol tradeDate tradeTime featureName
                 labelDB: str, labelTB: str, labelName: str, labelFunc,  # 每期模型的标签
                 resultDB: str, resultTB: str,  # 每期模型的结果
                 seed: int = 42, nJobs: int = -1, cv: int = 5,  # K折交叉验证
                 earlyStopping: bool = False,  # 是否对能够使用早停的模型进行早停
                 selfModel: Dict = None,
                 modelList: list = None,
                 factorPrefix: str = ""):
        super().__init__(session, pool, factorDB, factorTB, factorList, freq, toFactorDB, model_cfg, savePath, selectDB,
                         selectTB, selectMethod, selectFunc, labelDB, labelTB, labelName, labelFunc, resultDB, resultTB,
                         seed, nJobs, cv, earlyStopping, selfModel, modelList, factorPrefix)
        self.periodDict = {int(k): v for k,v in periodDict.items()}
        self.periodList = sorted([int(i) for i in self.periodDict.keys()])
        # 规范格式为2020.01.01
        for k, v in self.periodDict.items():
            for i in self.periodDict[k].keys():
                l = self.periodDict[k][i]
                self.periodDict[k][i] = [pd.Timestamp(l[0]),l[1], pd.Timestamp(l[2]),l[3]]

    def run(self):
        """
        多因子ML&DL模型运行的主体函数
        """
        # 准备
        for model in self.model_list:
            init_path(path_dir=rf"{self.savePath}/{model}")

        for period in tqdm.tqdm(self.periodList, total=len(self.periodList), desc="Model BackTesting"):
            train_period = self.periodDict.get(period).get("train")
            test_period = self.periodDict.get(period).get("test")
            pred_period = self.periodDict.get(period).get("pred")
            trainStartDate, trainStartTime, trainEndDate, trainEndTime = train_period
            testStartDate, testStartTime, testEndDate, testEndTime = test_period
            predStartDate, predStartTime, predEndDate, predEndTime = pred_period
            trainData = self.getData(selectMethod=self.selectMethod,
                                     labelName=self.labelName,
                                     startDate=trainStartDate,startTime=trainStartTime,
                                     endDate=trainEndDate,endTime=trainEndTime
                                     )
            trainData = trainData[~trainData["label"].isna()].reset_index(drop=True)
            print("trainData getted")
            testData = self.getData(selectMethod=self.selectMethod,
                                     labelName=self.labelName,
                                     startDate=testStartDate,startTime=testStartTime,
                                     endDate=testEndDate,endTime=testEndTime
                                     )
            testData = testData[~testData["label"].isna()].reset_index(drop=True)
            print("testData getted")
            factorList = [i for i in list(set(trainData.columns).intersection(set(testData.columns)))
                            if i not in ["symbol","TradeDate","MaxDate","TradeTime","label"]]
            predData = self.getData(selectMethod=self.selectMethod,
                                    labelName=self.labelName,
                                    startDate=predStartDate, startTime=predStartTime,
                                    endDate=predEndDate, endTime=predEndTime,
                                    factorList=factorList,   # 这里与训练集的特征保持一致
                                    isPred=True
                                    )
            print("predData getted")
            train_x, train_y = trainData[factorList], trainData["label"]
            eval_x, eval_y = testData[factorList], testData["label"]
            combined_x = None
            combined_y = None
            print("train:", trainData["TradeDate"].min(), trainData["TradeDate"].max())
            pred_x = predData[factorList]
            print("pred:", predData["TradeDate"].min(), predData["TradeDate"].max())
            # ---train---
            for modelName in self.model_list:
                if modelName in self.earlyStopModel and self.earlyStopping: # 启用早停机制
                    bestModel = self.train(train_x, train_y, modelName=modelName, cv=self.cv,
                               evalSet=[(eval_x, eval_y)])
                else:   # 不启用早停机制
                    if not combined_x:
                        combined_x = pd.concat([train_x, eval_x], axis=0, ignore_index=True)
                        combined_y = pd.concat([train_y, eval_y], axis=0, ignore_index=True)
                    bestModel = self.train(combined_x, combined_y, modelName=modelName, cv=self.cv, evalSet=None)
                # 保存模型
                save_model(bestModel,
                           save_path=rf"{self.savePath}/{modelName}",
                           file_name=str(testEndDate.strftime("%Y%m%d"))+"_"+str(testEndTime).zfill(4),
                           target_format="bin")
            # ---pred---
            for modelName in self.model_list:
                bestModel = load_model(load_path=rf"{self.savePath}/{modelName}",
                           file_name=str(testEndDate.strftime("%Y%m%d"))+"_"+str(testEndTime).zfill(4),
                           target_format="bin")
                res = predData[["symbol","TradeDate","TradeTime","label"]]
                res["labelPred"] = bestModel.predict(pred_x)
                res.insert(0, "ModelName", [modelName]*res.shape[0])
                # 保存至数据库
                self.uploadResultData(res)

                if self.toFactorDB:
                    res["ModelName"] = self.factorPrefix+res["ModelName"]   # 添加前缀
                    if self.freq.lower() == "d":
                        self.uploadFactorData(res[["symbol","TradeDate","ModelName","labelPred"]])

                    else:
                        self.uploadFactorData(res[["symbol","TradeDate","TradeTime","ModelName","labelPred"]])

if __name__ == "__main__":
    session = ddb.session(enableASYNC=False)  # 允许异步上传数据
    session.connect("172.16.0.184",8001,"maxim","dyJmoc-tiznem-1figgu")
    pool=ddb.DBConnectionPool("172.16.0.184",8001,10,"maxim","dyJmoc-tiznem-1figgu")
    from model.Label import get_DayLabel
    from model.Select import get_DayFeature
    from model.Dnn import CustomDNN,get_DNN
    from model.Resnet import CustomResNet,get_RESNET

    with open(r".\config\model_cfg.json5","r",encoding="utf-8") as f:
        model_cfg = json5.load(f)
    with open(r".\config\period_cfg.json5","r",encoding="utf-8") as f:
        period_cfg = json5.load(f)
    config_dict = dict(
        session=session, pool=pool,
        factorDB="dfs://Dayfactor", factorTB="pt", freq="D", toFactorDB=True,
        factorPrefix="Test_",
        factorList=[
            "shio",
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
        model_cfg=model_cfg, cv=5, nJobs=-1, earlyStopping=True,
        modelList=["lightgbm", "xgboost"],
        selfModel={"dnn": [CustomDNN, get_DNN],
                   "resnet": [CustomResNet, get_RESNET]},
        savePath=r"D:\DolphinDB\Project\FactorModel\model",
        labelDB="dfs://DayLabel", labelTB="pt", labelName="ret5D",
        selectDB="dfs://Select", selectTB="Select20250920", selectMethod="ret5D",
        resultDB="dfs://Model", resultTB="Model20250920",
        selectFunc=get_DayFeature,
        labelFunc=get_DayLabel
    )

    # 回测方式
    M = ModelBackTest(**config_dict, periodDict=period_cfg)
    # M.init_labelDB(dropDB=False,dropTB=True)        # 创建预测标签数据库
    # M.init_selectDB(dropDB=False,dropTB=True)       # 创建因子选择数据库
    M.init_resultDB(dropDB=False,dropTB=True)       # 创建模型训练结果保存数据库
    # M.add_labelData()   # 添加标签数据库
    # M.add_selectData()  # 添加选择数据库
    M.run()

