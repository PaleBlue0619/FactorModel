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
    self.session.run(f"""
        // 读取因子数据
        factor_df = select symbol,date as TradeDate,factor,value 
                    from loadTable("{self.factorDB}","{self.factorTB}")
                    where factor in {self.factor_list}; 
        
        // 读取标签数据
        label_df = select symbol, MaxDate as TradeDate, labelName, label 
                    from loadTable("{self.labelDB}","{self.labelTB}")
                    where labelName == "ret5D"

        // left join 
        result_df = lj(factor_df, label_df, `symbol`TradeDate)  // symbol,TradeDate,factor,value,labelName,label
        
        // 添加period
        distinct_time_list = sort(exec distinct(TradeDate) from result_df)
        distinct_time_dict = dict(distinct_time_list, 1..size(distinct_time_list))
        result_df[`period] = distinct_time_dict[result_df[`TradeDate]]
        
        // 筛选每期的优秀因子+存入selectDB中
        result_df = select 0 as selected, corr(rank(label),rank(value)) as IC 
                    from result_df 
                    group by TradeDate,factor,labelName 
        update result_df set IC_avg = abs(mavg(IC,5)) context by factor,labelName
        update result_df set selected = 1 
                            where IC_avg >= percentile(IC_avg, 0.5, "midpoint") 
                            context by TradeDate,labelName
        result_df = select TradeDate,15:00:00.000 as TradeTime,labelName,factor as featureName 
                    from result_df where selected == 1;
        loadTable("{self.selectDB}","{self.selectTB}").append!(result_df);
    """)

