import dolphindb as ddb
import pandas as pd
from src.ModelBackTest import ModelBackTest

def get_DayFeature(self: ModelBackTest):
    """
    symbol TradeDate TradeTime method featureName period

    :param self:
    :return:
    """
    # 选出每期的特征
    self.session.run("""
        // 读取因子数据
        factor_df = select 
        
        
        // 读取label数据
        
    """)
    return
