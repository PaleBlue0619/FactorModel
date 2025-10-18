# 0. 项目介绍
FactorModel支持自定义ML&DL模型训练，并行训练实现参数寻优，适用于复杂交易信号的生成。同时ML&DL合成的因子支持自动写入指定的因子数据库，从而与FactorEva无缝衔接。

# 1. 项目架构
src <br>
-ModelBackTest.py FactorModel模块 <br>
-utils.py 工具模块 <br>
config<br>
-model_cfg.json5 # 模型配置文件(固定参数+网格搜索参数)<br>

    // 树模型
    "gbdt": {
      "default_params": {
          "criterion": "friedman_mse",
          "random_state": 42
      },
      "grid_params": {
          "max_depth": [null,5,10],
          "min_samples_split": [2,3,4],
          "min_samples_leaf": [1,3,5],
          "max_features": [1,2]
      }
      },...

model <br>
 Label.py, # 向标签数据库中添加数据的自定义函数<br>
 Select.py, # 向选择数据库中添加每期因子选择（作为训练的X）的自定义函数<br>
 Dnn.py, # 自定义DNN模型(这里为Demo)<br>
 Resnet.py # 自定义ResNet模型(这里为Demo)<br>

# 2. 函数说明
# 2.1 ModelBackTest类
```python
class ModelBackTest:
    def __init__(self, session: ddb.session, pool:ddb.DBConnectionPool, startDate: str, endDate: str, factorDB: str, factorTB: str, factorList: str, freq: str, toFactorDB: bool, # 模型频率(D/M)
    model_cfg: Dict, savePath: str,  # 本地模型保存路径: savePath/{modelName}
    selectDB: str, selectTB: str, selectMethod: str, selectFunc, # 每期模型的特征: symbol tradeDate tradeTime featureName
    labelDB: str, labelTB: str, labelName: str, labelFunc,   # 每期模型的标签
    resultDB: str, resultTB: str,  # 每期模型的结果
    seed: int = 42, nJobs: int = -1, cv: int = 5, # K折交叉验证
    earlyStopping: bool = False,   # 是否对能够使用早停的模型进行早停
    callBackPeriod: int = 1,   # 利用过去K期数据进行训练
    splitTrain: float = 0.8,   # 训练集划分比例
    selfModel: Dict = None,   # 自定义模型<模型名称:接口函数列表>
    modelList: list = None,
    factorPrefix: str = ""     # toFactorDB == True时生效, 保存至因子数据库的因子为: factorPrefix+ModelName
    ):
```
session: DolphinDB session <br>
pool: DolphinDB Connection Pool <br>
:param startDate: 回测开始日期 <br>
endDate: 回测结束日期 <br>
factorDB: 因子数据库名称 <br>
factorTB: 因子数据表名称 <br>
factorList: 所有候选的因子池 <br>
freq: 因子频率->目前只支持同频训练+预测 <br>
toFactorDB: 是否将合成后的因子保存回FactorDB <br>
model_cfg: 模型参数字典 <br>
savePath: 模型保存路径, 实际路径为savePath/modelName/xxx.bin <br>
selectDB: 因子(X)选择数据库 <br>
selectTB: 因子(X)选择数据表 <br>
selectMethod: 特征选择方式 <br>
selectFunc: 特征选择函数 <br>
labelDB: 标签(Y)选择数据库 <br>
labelTB: 标签(Y)选择数据表 <br>
labelName: 标签构造名称 <br>
labelFunc: 标签构造函数 <br>
resultDB: 结果数据库 <br>
resultTB: 结果数据表 <br>
seed: 随机种子 <br>
nJobs: 训练多个模型的并行度 <br>
cv: K折交叉验证 <br>
earlyStopping: 是否对于能够早停的模型进行早停 <br>
callBackPeriod: 回看期:利用[t-callBackPeriod,t-1]的样本进行训练 <br>
splitTrain: 训练集-测试集划分比例 <br>
selfModel: 自定义模型配置项 <br>
modelList: 所有使用的模型list <br>
factorPrefix: 保存至因子数据库的合成因子前缀, 完整因子名为factorPrefix+ModelName <br>

# 2.2 主要函数
```python
def get_factorList(self): -> list
    """
    根据传入的factorDB设置factor_list, 并返回
    """
```
inplace: 是否覆盖当前设置的factor_list<br>
```python
def get_periodList(self):
    """
    返回一个共享变量->TradeDate TradeTime period
    最终格式: symbol TradeDate TradeTime period label factor
    共享变量: TradeDate TradeTime MaxDate MaxTime period
    """
```
```python
def init_labelDB(self, dropDB: bool = False, dropTB: bool = False):
    """初始化标签数据库"""
```
dropDB: 是否删除数据库 <br>
dropTB: 是否删除表 <br>
```python
def init_selectDB(self, dropDB: bool = False, dropTB: bool = False):
    """初始化特征选择数据库"""
```
dropDB: 是否删除数据库 <br>
dropTB: 是否删除表 <br>
```python
def init_resultDB(self, dropDB: bool = False, dropTB: bool = False):
    """初始化结果数据库"""
```
dropDB: 是否删除数据库 <br>
dropTB: 是否删除表 <br>
```python
def train(self, x: any, y: any, modelName: str, cv:int = 5, evalSet: List[Tuple] = None) -> sklearn.BaseEstimator
```
x: 特征数据 <br>
y: 标签数据 <br>
model_cfg: 模型待选参数 <br>
modelName: 模型名称 <br>
modelFunc: 模型函数方法 <br>
cv: k折交叉验证 <br>
evalSet: 验证集 <br>
```python
def run(self, startDate: pd.Timestamp, startTime: int, endDate: pd.Timestamp, endTime:int)
```
startDate: 开始日期 <br>
startTime: 开始时间 <br>
endDate: 结束日期 <br>
endTime: 结束时间 <br>
> 1. 准备相关appender+创建路径 <br>
> 2. 确定回测区间+创建共享变量以指定边界 <br>
> 3. for loop <br>
> 3.1 通过getData方法获取数据 <br>
> 3.2 并行训练模型,寻找最优参数,将最优参数生成模型 <br>
> 3.3 保存最优模型至本地文件夹 <br>
> 3.4 利用最优参数模型进行预测,得到合成因子(预测收益率) <br>
> 3.5 将真实收益率与合成因子(预测收益率)插入至结果数据库 <br>
> 3.6 [可选]将合成因子插入因子数据库 <br>