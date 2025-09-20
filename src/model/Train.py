import pandas as pd
import dolphindb as ddb
from src.ModelBackTest import ModelBackTest

def to_dataframe(self: ModelBackTest, ) -> pd.DataFrame:
    """
    训练唯一的接口
    """
    self.session.run("""
    
    
    """)

def to_dolphindb(self: ModelBackTest, data: pd.DataFrame):
    """
    回测结果保存至DolphinDB
    """
