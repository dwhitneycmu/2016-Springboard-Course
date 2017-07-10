from capstoneProjectFunctions import *
from part1_Functions import *
from part2_Functions import *
from part3_Functions import *

##############
# SCRIPT DATA
##############
# Load data
reloadFromScratch=False;
if reloadFromScratch:
    # Chose loading parameters
    startDate = datetime.datetime(2008, 4, 1);
    endDate   = datetime.datetime(2016, 4, 1);
    stockPriceType  = 'Close'; # stock metric - can be either [Open, High, Low, Close, Volume, Adj Close]

    # Get an ordered list of stocks (order: NASDAQ index, FAANG stocks, and other Top 100 NASDAQ stocks)
    FAANGGStocks  = ['AAPL','AMZN','FB','GOOGL','GOOG','GILD','NFLX'];
    NASDAQ_Top100 = ['FOXA', 'FOX', 'ATVI', 'ADBE', 'AKAM', 'ALXN', 'GOOGL', 'GOOG',
                   'AMZN', 'AAL', 'AMGN', 'ADI', 'AAPL', 'AMAT', 'ADSK', 'ADP', 'BIDU',
                   'BBBY', 'BIIB', 'BMRN', 'AVGO', 'CSX', 'CA', 'CELG', 'CERN', 'CHTR',
                   'CHKP', 'CSCO', 'CTXS', 'CTSH', 'CMCSA', 'COST', 'CTRP', 'DISCA',
                   'DISCK', 'DISH', 'DLTR', 'EBAY', 'EA', 'ENDP', 'EXPE', 'ESRX', 'FB',
                   'FAST', 'FISV', 'GILD', 'GOOGL','GOOG', 'HSIC', 'ILMN', 'INCY', 'INTC', 
                   'INTU', 'ISRG', 'JD', 'KHC', 'LRCX', 'LBTYA', 'LBTYK', 'QVCA', 'LMCK',
                   'LMCA', 'LVNTA', 'LLTC', 'MAR', 'MAT', 'MXIM', 'MU', 'MSFT', 'MDLZ',
                   'MNST', 'MYL', 'NTAP', 'NFLX', 'NCLH', 'NVDA', 'NXPI', 'ORLY',
                   'PCAR', 'PAYX', 'PYPL', 'QCOM', 'REGN', 'ROST', 'SNDK', 'SBAC',
                   'STX', 'SIRI', 'SWKS', 'SBUX', 'SRCL', 'SYMC', 'TMUS', 'TSLA',
                   'TXN', 'PCLN', 'TSCO', 'TRIP', 'ULTA', 'VRSK', 'VRTX', 'VIAB',
                   'VOD', 'WBA', 'WDC', 'WFM', 'XLNX', 'YHOO'];
    listOfStocks = FAANGGStocks;
    listOfStocks.insert(0,'^IXIC');
    for currentStock in NASDAQ_Top100:
        if currentStock not in listOfStocks:
            listOfStocks.append(currentStock);

    # Then append data for all other NASDAQ stocks
    stockMiscData = pd.DataFrame.from_csv(os.getcwd()+'\\NASDAQList.csv')
    NASDAQ_All = stockMiscData.index;
    for currentStock in NASDAQ_All:
        if currentStock not in listOfStocks:
                listOfStocks.append(currentStock);        
            
    # Read raw data for all stocks
    stockData = loadData(startDate,endDate,listOfStocks,stockPriceType,zScoreData=False) 

    # Average data across a week
    selectedIntervalBinning = 1;
    dateIntervals   = computeTimeBins(startDate,endDate,timeIntervalType='W',selectedIntervalBinning=selectedIntervalBinning);
    binnedStockData, binnedCorrelations = binStockData(dateIntervals,stockData.columns,stockData);   
    
    # Compute local correlations (within +/- two weeks around datapoint) 
    localCorrelationData=localCorrelation(binnedStockData.copy().apply(signal.medfilt,kernel_size=5),windowSize=4)
    
    # Now compute local minima (10). First we find the indexed locations of the minima (minimal spacing is 1 month), sort the indexed locations, and then convert the indexed locations to time-stamps
    correlationExtrema=findLocalExtrema(localCorrelationData,numberOfExtrema=10,minimumSpacing=6);
    correlationExtrema=correlationExtrema.apply(np.sort)
    correlationExtrema=correlationExtrema.apply(lambda index,y: y[index],y=localCorrelationData.index)

    # write all data to file
    stockData.to_csv(os.getcwd()+'\\stockData_NoZScore.csv');
    binnedStockData.to_csv(os.getcwd()+'\\BinnedData_NoZScore.csv');
    binnedCorrelations.to_csv(os.getcwd()+'\\BinnedCorrelations_NoZScore.csv');
    localCorrelationData.to_csv(os.getcwd()+'\\LocalCorrelationData_NoZScore.csv');
    correlationExtrema.to_csv(os.getcwd()+'\\CorrelationExtrema_NoZScore.csv');    
else:
    stockData = pd.DataFrame.from_csv(os.getcwd()+'\\stockData_NoZScore.csv');
    
# Project Parameters
selectedTime    = range(758,2016); #range(154,416) (for binnedStocks) or range(758,2016) (for stockData)
stockData       = stockData.iloc[selectedTime,:]; #Get data for the last five years
stockData       = preProcessStockData(stockData,zScoreData=True,eliminateStocksLessThanAYear=True) # Z-Score data (or use percent difference relative to starting day), and then eliminate stocks which have less than 1 year of data
zeroShiftedData = zeroStockData(stockData.apply(signal.medfilt,kernel_size=3).copy()); # Compute zero-shifted data, in which the starting value for each stock is zero

# Will compute figures for Part 1 of the capstone project
computeCapstoneFigures_Part1 = True;
if computeCapstoneFigures_Part1:    
    # PLOT 1: Show basic relationshop to stocks
    print "Computing Figure 1 - Relationship of FAANG stocks to NASDAQ"
    FAANGStocks = range(0,4)+range(5,8);
    plottingFunctionCorrelation(zeroShiftedData.iloc[:,FAANGStocks].copy())
    
    # PLOT 2: Correlation of stocks to the NASDAQ across time (excludes missing data). We can see see that both FAANG and other Top 100 NASDAQ stocks are more correlated with the NASDAQ than other NASDAQ stocks
    print "Computing Figure 2 - Computing correlations between all stocks and NASDAQ"
    correlationValues = computeCorrelationsWithNASDAQ(stockData)
    
    # Plot 3: Relationship to market cap data and slope
    print "Computing Figure 3a - Computing relationship of NASDAQ correlations with market cap data"
    compareMarketCapDataToNASDAQCorrelation(stockData,correlationValues)
    
    # Plot 4: Show relationship of market data (but using slopes)
    print "Computing Figure 3b - Computing relationship of NASDAQ correlations with slope"
    print "Computing Figure 4  - Computing slope for all stocks and NASDAQ"
    fig=showScatterPlot(zeroShiftedData,yLabel='Normalized value',xLabel='',xLimits=[zeroShiftedData.index[0],zeroShiftedData.index[-1]],yLimits=[-1.5,5]);
    fig.set_tight_layout(True);
    slopeValues=computeSlopeWithNASDAQCorrelation(zeroShiftedData,correlationValues)
    
    # Plot 5 and 6: Show breakdown vs sector
    print "Computing Figure 5 and 6 - Computing how sectors compare to the NASDAQ"
    computeSectorRelationship(zeroShiftedData,correlationValues)
    
    # Plot 7: Show K-Means cluster
    print "Computing Figure 7 - Computing K-Means clusters"
    computeKMeansClusters(zeroShiftedData)
    
    # Plot 8: Show breakdown vs sector
    print "Computing Figure 8 - Compute correlation quintiles"
    computeCorrelationQuintileRelationship(zeroShiftedData,correlationValues,slopeValues)
    
# Will compute figures for Part 2 of the capstone project
computeCapstoneFigures_Part2 = True;
if computeCapstoneFigures_Part2:
    # Load data needed for this section
    selectedTime         = range(154,416); 
    binnedStockData      = pd.DataFrame.from_csv(os.getcwd()+'\\BinnedData_NoZScore.csv').iloc[selectedTime,:];
    binnedStockData      = zeroStockData(preProcessStockData(binnedStockData,zScoreData=True,eliminateStocksLessThanAYear=True)); # Pre-process data like unbinned stock data
    binnedCorrelations   = pd.DataFrame.from_csv(os.getcwd()+'\\BinnedCorrelations_NoZScore.csv').iloc[selectedTime,:];
    localCorrelationData = pd.DataFrame.from_csv(os.getcwd()+'\\LocalCorrelationData_NoZScore.csv').iloc[selectedTime,:];
    correlationExtrema   = pd.DataFrame.from_csv(os.getcwd()+'\\CorrelationExtrema_NoZScore.csv');   
        
    # Plot 9: Show example of a time-varying correlation
    print "Computing Figure 9 - Show example time-varying correlations"
    showTimeVaryingCorrelations(binnedStockData,localCorrelationData,correlationExtrema,selectedStock='AAPL');
    
    # Plot 10: Show summary statistics of time-varying correlations
    print "Computing Figure 10 - Show summary statistics for time-varying correlations"
    computeSummaryStatsForTimeVaryingCorrelation(localCorrelationData);
    
    # Plot 11: Show a simple unweighted running average
    print "Computing Figure 11 - Show unweighted average"
    correlationData = predictMarketFromSelectedStocks(zeroShiftedData)
    
    # Plot 12: Autocorrelations of stock value show hysteresis
    print "Computing Figure 12 - Autocorrelations of stock data"
    showAutoCorrelations(zeroShiftedData)
    
    # Plot 13: Use Kalman filter to demonstrate that hysteresis can be exploited
    print "Computing Figure 13 - Running Kalman filter"
    runKalmanFilter_ForOptimization(binnedStockData, binnedCorrelations, stockSet='NASDAQ')

# Will compute figures for Part 3 of the capstone project
computeCapstoneFigures_Part3 = True;
if computeCapstoneFigures_Part3:
    # Plot 14: Predicting NASDAQ with different weighted, running weighted-averages of NASDAQ stocks
    print "Computing Figure 14 - Predicting NASDAQ with different weighted, running weighted-averages of NASDAQ stocks"
    computeLinearModel(stockData,stockSets=['FAANGStocks'],usePreviousWeights=False,forecastingOn=False,useKalmanFilter=False)    
    
    # Plot 15: Compute correlations between Top 100 stocks
    print "Computing Figure 15 - Computing correlations between stocks"
    computeBinnedCorrelationsBetweenStocks(stockData)
    
    # Plot 16: Compute regression coefficients of FAANG stocks (relative to predicting the NASDAQ)
    print "Computing Figure 16 - Computing regression coefficients"
    computeRegressionCoefficients(stockData,stockSet='FAANGStocks')
        
    # Plot 17,18,19: Predicting NASDAQ with different weighted, running weighted-averages of NASDAQ stocks
    print "Computing Figure 17, 18, and 19 - Forecasting the NASDAQ with different weighted, running weighted-averages of NASDAQ stocks"
    computeLinearModel(stockData,stockSets=['FAANGStocks','Top100StocksExcluding'],usePreviousWeights=True,forecastingOn=True,useKalmanFilter=False)