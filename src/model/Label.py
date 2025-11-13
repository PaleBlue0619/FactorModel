from src.ModelBackTest import *

def get_DayLabel(self: BasicModelBackTest, k: int = 5):
    """
    统一的日频标签生成函数
    -> return table(
        ["symbol","tradeDate","tradeTime","method","label","maxDate","maxTime"]
    )
    """
    barDBName = "dfs://DayKDB"
    barTBName = "o_tushare_a_stock_daily"
    idxDBName = "dfs://DayKDBOfIndex"
    idxTBName = "DayFreq1000"
    symbolCol = "ts_code"
    dateCol = "trade_date"
    openCol = "open"
    closeCol = "close"

    self.session.run(f"""
    index_df = select symbol,TradeDate from loadTable("{idxDBName}", "{idxTBName}")
                order by symbol,TradeDate
    symbol_list = exec symbol from (select count(*) from loadTable("{idxDBName}","{idxTBName}")
                group by symbol)
    label_df = select {symbolCol} as symbol, 
                      {dateCol} as TradeDate,
                      {openCol} as open,
                      {closeCol} as close
                      from loadTable("{barDBName}", "{barTBName}")
                      where {symbolCol} in symbol_list
                      order by {dateCol},{symbolCol}
    label_df = select symbol, 
                      TradeDate, 
                      15:00:00.000 as TradeTime,
                      "ret{k}D" as labelName,
                      nullFill((move(close,-{k})-open)\open,0) as label,
                      nullFill(move(TradeDate,-{k+1}).date(),temporalDeltas(TradeDate,"XSHG")) as MaxDate,
                      15:00:00.000 as MaxTime
                from label_df 
                context by symbol
    label_df = select * from lj(index_df,label_df,`TradeDate`symbol) where not isNull(labelName)
    loadTable("{self.labelDB}","{self.labelTB}").append!(label_df)
    """)
