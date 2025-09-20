from src.ModelBackTest import ModelBackTest

def get_DayLabel(self: ModelBackTest):
    """
    统一的日频标签生成函数
    -> return table(
        ["symbol","tradeDate","tradeTime","method","label"]
    )
    """
    kdbName = "dfs://DayKDBOfIndex"
    ktbName = "DayFreq500"
    symbolCol = "symbol"
    dateCol = "TradeDate"
    openCol = "open"
    closeCol = "close"

    self.session.run(f"""
    label_df = select {symbolCol} as symbol, 
                      {dateCol} as TradeDate,
                      {openCol} as open,
                      {closeCol} as close
                      from loadTable("{kdbName}", "{ktbName}")
                      order by TradeDate,symbol
                // where TradeDate between 2017.01.01 and 2024.01.01
    label_df = select symbol, 
                      TradeDate, 
                      15:00:00.000 as TradeTime,
                      "ret5D" as method,
                      nullFill((move(close,-5)-open)\open,0) as label,
                      nullFill(move(TradeDate,-5).date(),temporalDeltas(TradeDate,"XSHG")) as MaxDate,
                      nullFill(move(TradeTime,-5).time(),max(TradeTime)) as MaxTime
                from label_df 
                context by symbol
    loadTable("{self.labelDB}","{self.labelTB}").append!(label_df)
    """)

def get_MinLabel(self: ModelBackTest):
    """
       统一的分钟频标签生成函数
       -> return table(
           ["symbol","tradeDate","tradeTime","method","label"]
       )
       """
    kdbName = "dfs://MinKDBOfIndex"
    ktbName = "MinFreq500"
    symbolCol = "symbol"
    dateCol = "TradeDate"
    timeCol = "TradeTime"
    openCol = "open"
    closeCol = "close"

    self.session.run(f"""
       label_df = select {symbolCol} as symbol, 
                         {dateCol} as TradeDate,
                         {timeCol} as TradeTime,
                         {openCol} as open,
                         {closeCol} as close
                         from loadTable("{kdbName}", "{ktbName}")
                         order by TradeDate, TradeTime, symbol
                   // where TradeDate between 2017.01.01 and 2024.01.01
       label_df = select symbol, 
                         TradeDate, 
                         TradeTime,
                         "ret240M" as method,
                         nullFill((move(close,-240)-open)\open,0) as label,
                         nullFill(move(TradeDate,-240).date(),temporalDeltas(TradeDate,"XSHG")) as MaxDate,
                         nullFill(move(TradeTime,-240).time(),max(TradeTime)) as MaxTime
                   from label_df 
                   context by symbol
       loadTable("{self.labelDB}","{self.labelTB}").append!(label_df)
       """)

