#from matplotlib.pyplot import *
#import random
#import time
#import math

# PACKAGES AND FUNCTIONS USED TO PERFORM SUBSEQUENT ANALYSIS (SKIP TO NEXT SECTION FOR FIGURES):
import datetime
import matplotlib as plt
from matplotlib import cm
import numpy as np
import os 
import pandas as pd
import pandas_datareader.data as web
import scipy.stats as stats
import scipy.signal as signal
import sklearn
import sklearn.linear_model
import urllib
#%matplotlib inline
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.facecolor']='white'
plt.rcParams['figure.facecolor']='white'
plt.rcParams['font.family'] = 'Arial';#'DejaVu Sans';
plt.rcParams['pdf.fonttype'] = 42
plt.rc('font', serif='Arial') 
plt.rc('text', usetex='false')#, dvipnghack ='True', hinting='native') 
plt.rc('mathtext', fontset='custom'); #valid strings are [u'custom', u'stix', u'cm', u'stixsans']

# Define global variables
NUMBER_OF_STOCK_TYPES = 6;     # Number of different stock types to plot (1: Just NASDAQ, 2: Adds FAANG, 3: Adds Top-100, 4: Adds other NASDAQ, 5: Adds top 20% performing of other NASDAQ stocks, 6: Adds bottom 80% performing of other NASDAQ stocks) 
MAXIMUM_RETRIES       = 5;   # Maximum allowable retries to connect to Google / Yahoo Finance for an individual query
USE_YAHOO             = False; # The original project data was acquired using the Yahoo Finance API, which has been discontinued by Yahoo as of 5/2017. However, similar datasets can be acquired through Google Finance / NASDAQ.com
FONT_SIZE_OBJECTS     = 8;     # Font size for text labels within plots

################# HELPER FUNCTIONS #################
def addDifferencesLabels(xMatrix,yMatrix,pMatrix,ax,params=None):
    """"Helper function to add labels to bar plot for significance test"""
    
    # Default parameters
    if params == None:
        params = {'yAxisRescalingFactor':8,'lineWidth':5}
    
    # Get the current plot axis size in points
    fig=ax.get_figure();
    figSizeInPts = fig.get_size_inches()/0.0139;
    axSizeInPts  = figSizeInPts*ax.get_position().size;

    # Setup the valid statistical comparisons between groups that we'll plot (no duplicates)
    comparison  = []; 
    for offset in range(1,len(xMatrix)):
        for a in range(0,len(xMatrix)):
            b = a+offset;
            if b < len(xMatrix):
                x = (xMatrix[a]+xMatrix[b])/2.;
                y = max(yMatrix[a], yMatrix[b]);
                comparison.append([x,y,pMatrix[a,b],xMatrix[a],xMatrix[b]])     
                
    # Automatically determine new clipping limits for current axis
    scalingFactor = (abs(np.diff(ax.get_ylim()))/axSizeInPts[1]); # Converts units from points to actual y-measurement units
    maxAnnotationValue = np.array(comparison)[:,1].max() \
                        +params['lineWidth']*params['yAxisRescalingFactor']*scalingFactor*abs(xMatrix[-1]-xMatrix[0]);
    if ax.get_ylim()[1]<maxAnnotationValue:
        ax.set_ylim([ax.get_ylim()[0],maxAnnotationValue]) 
    scalingFactor = (abs(np.diff(ax.get_ylim()))/axSizeInPts[1]); # UPDATE scaling factor to reflect change in y-axis
    
    # Add the statistical comparisons to the current axis
    props = {'connectionstyle':'bar,fraction={},armA={},armB={}'.format(0,params['lineWidth'],params['lineWidth']), \
             'arrowstyle':'-','shrinkA':0,'shrinkB':0,'lw':1,'patchA':None}
    for (x,y,p,x1,x2) in comparison:
        if p<0.0001:
            p = 'p<0.0001';
        else:
            p = 'p={}'.format(p);
        yLoc   = y+4*params['lineWidth']*scalingFactor*abs(x1-x2);
        offset = 1*params['lineWidth']*scalingFactor; #0.8
        ax.annotate('', xy=(x1,yLoc), xytext=(x2,yLoc), arrowprops=props);
        ax.annotate(p, xy=(x,yLoc+offset), ha='center',fontsize=FONT_SIZE_OBJECTS)     
        
def binaryDiagonalSquareMatrix(matrixShape,invertMatrix=False):
    """ returns a 2d-square matrix where the diagonalized elements are true """
    binaryMatrix = np.zeros([matrixShape,matrixShape],dtype='bool');
    for i in range(matrixShape):
        binaryMatrix[i,i] = 1;
    if invertMatrix:
        binaryMatrix = 1-binaryMatrix;
    return binaryMatrix;
    
def cat(axis,x,y):
    """ Concatenate two arrays.
    cat[DIM,A,B] concatenates the arrays A and B along the dimension DIM."""
    
    # Get original dimensions of array x and y
    xDim = list(x.shape);
    yDim = list(y.shape);
    numDims = len(xDim);
    
    # checks whether one of the arrays is empty
    isXEmpty = x.size==0;
    isYEmpty = x.size==0;
    isAnArrayEmpty = (isXEmpty | isYEmpty);
    if isAnArrayEmpty: # only runs if one of the arrays are empty
        if( x.size == 0):
            z = y.copy();
        elif (y.size == 0):
            z = x.copy();
    else: 
        # Create a new array, z, to concatenate x and y along specified axis dimensions
        zDim = list(xDim);
        zDim[axis] = xDim[axis]+yDim[axis];    
        z = np.zeros(tuple(zDim),dtype=x.dtype)
        
        # Sequentially assign array x and y along the specified dimension of z
        exec('z[{}{}:{}{}]=x.copy();'.format(axis*':,',0,xDim[axis],(numDims-axis-1)*',:'));
        exec('z[{}{}:{}{}]=y.copy();'.format(axis*':,',xDim[axis],'',(numDims-axis-1)*',:'));
    return z;
    
def find(X,K=-1,orderReversed=False):
    """
    find(X,K) returns at most the first K indices corresponding to 
    the nonzero entries of the array X.  K must be a positive integer, 
    but can be of any numeric type. If K=-1 (default), then all values 
    are returned. Typically the first K values are returned, but the 
    order can be reversed so the last K indices are return (if order)
    """
    indices = [i for (i, val) in enumerate(X) if val==True];
    if orderReversed:
        indices = indices[::-1];
    if ((K!=-1) & (len(indices)>K)): 
        indices = indices[:K];  
    return indices;
    
def showScatterPlot(x,y,xTitle='',yTitle='',title='',xLimits=[],yLimits=[],axes=None):
    """ show a regular scatter plot plotting the elements in x versus y """
    if axes==None:
        fig, axes = plt.pyplot.subplots(nrows=1,ncols=1,figsize=(5,4)); 
    else:
        fig = axes.get_figure();
    
    # Drop NaNs
    selectedValues = ~(np.isnan(x) | np.isnan(y));
    x = x[selectedValues];
    y = y[selectedValues];
    
    # Autogenerate x-limits and y-limits
    if len(xLimits)==0:
        xLimits = [np.min(x),np.max(x)];
    if len(yLimits)==0:
        yLimits = [np.min(y),np.max(y)];
    
    # Compute a first order linear fit and correlation value
    P = np.polyfit(x,y,1)
    fig.suptitle('Relationship of {} vs. {}: r = {}'.format(xTitle,yTitle,np.corrcoef(x,y)[0,1]));
    
    # Plot data
    axes.plot(x,y,'.k');
    axes.plot(np.sort(x),np.polyval(P,np.sort(x)),'-r',linewidth=3)
    axes.set_title(title)
    axes.set_xlabel(xTitle)
    axes.set_ylabel(yTitle)
    axes.set_xlim(xLimits);
    axes.set_ylim(yLimits)
    return fig;

################# STOCK DATA CLASS #################
class stockDataObject():        
    def __init__(self,startDate=None, endDate=None, listOfStocks=None, stockPriceType=None, stockData=None):
        """ Initialize a stock object, where we either automatically connect with Yahoo Finance to update the stock data, 
            or populate the object with a previously instantiated stockData Pandas object"""
        
        # Define (and optionally initialize) object variables
        if startDate==None:
            startDate = self.getDefaultParameters('startDate');
        if endDate==None:
            endDate = self.getDefaultParameters('endDate');
        if listOfStocks==None:
            listOfStocks = self.getDefaultParameters('listOfStocks');
        if stockPriceType==None:
            stockPriceType = self.getDefaultParameters('stockPriceType');
        if isinstance(stockData, pd.DataFrame):
            self.stockData = stockData;
        else:
            self.readStockData(startDate, endDate, listOfStocks, stockPriceType);
            
        # Store references to object variables
        self.startDate      = startDate;
        self.endDate        = endDate;
        self.listOfStocks   = listOfStocks;
        self.stockPriceType = stockPriceType;
        self.numberOfStocks = len(self.listOfStocks);
        self.selectedStocks = range(self.numberOfStocks);

    def getDefaultParameters(self,parameterName):
        """ Returns default parameters for the stock data object. ParameterName can be:
        'startDate','endDate','stockPriceType', or 'listOfStocks'"""
        
        if parameterName=='startDate':
            parameter = datetime.datetime(2011, 1, 1);
        elif parameterName=='endDate':
            parameter = datetime.datetime(2017, 1, 1);
        elif parameterName=='stockPriceType':
            parameter = 'Close'; # stock metric - can be either [Open, High, Low, Close, Volume, Adj Close]
        elif parameterName=='listOfStocks':
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
            useDownloadedListOfNASDAQStocks = True;
            if useDownloadedListOfNASDAQStocks:
                stockMiscData = pd.DataFrame.from_csv(os.getcwd()+'\\NASDAQList.csv')
                NASDAQ_All = stockMiscData.index;
            else:
                NASDAQ_All = self.getListOfNASDAQStocks();
            for currentStock in NASDAQ_All:
                if currentStock not in listOfStocks:
                        listOfStocks.append(currentStock); 
            parameter = listOfStocks;
        return parameter;
        
    def saveData(self,saveDirectory=os.getcwd()):
        """ save important data to csv files in the specified directory  """
        self.stockData.to_csv(saveDirectory+'\\stockData_NoZScore.csv');

    def loadData(self,saveDirectory=os.getcwd()):
        """ load important data from csv files in the specified directory  """
        self.stockData            = pd.DataFrame.from_csv(saveDirectory+'\\stockData_NoZScore.csv');
            
    def readStockData(self, startDate, endDate, listOfStocks, zScoreData=False, useYahoo=USE_YAHOO):
        """    
        Connects to Yahoo Finance to read-in adjusted stock data, stores the stock data in a pandas 
        DataFrame (saved in the stockData object), and returns a reference to the pandas DataFrame. 
        
        startDate and endDate - datetime objects defining the range of dates to search for stock information   
        listOfStocks    - defines the list of stocks to read-in. first stock in array is our reference stock/index
        """
        
        # Loop through different stock name and add them to the stockData object
        self.stockData = pd.DataFrame();
        for stockIndex,stock in enumerate(listOfStocks[:]):
            # Use the built-in web reader of pandas to get financial data for specified stocks.
            # The result data structure is a pandas dataFrame object with 6 columns: 
            # [Open, High, Low, Close, Volumn, Adj Close]
            print 'Reading {} - {} of {}'.format(stock,stockIndex+1,len(listOfStocks))
            numberOfRetries = 0;            
            while numberOfRetries < MAXIMUM_RETRIES:
                try:
                    if useYahoo:
                        allHistoricalStockData = web.DataReader(stock, 'yahoo', startDate, endDate);
                    else:
                        allHistoricalStockData = self.googleReader(stock,startDate,endDate);
                    numberOfRetries = MAXIMUM_RETRIES;
                except:
                    numberOfRetries +=1;      
                    print "Error - Will retry (Attempt {} of {})".format(numberOfRetries,MAXIMUM_RETRIES)
                    if numberOfRetries == MAXIMUM_RETRIES:
                        print "Skipped stock"
                        break; # skip stock
                    else:
                        continue;
                selectedStockData = allHistoricalStockData['Adj Close'];

                # Select the chosen column of data (specified by stockPriceType) and import
                # the data into the stockData object. Resulting data structure columns are 
                # defined by the names list contained in listOfStocks
                self.stockData[stock] = pd.Series(selectedStockData); 
        return self.stockData;
                
    def googleReader(self,stock,startDate,endDate):
        """ Connect with the Google API to download stock data as a CSV file (['Open', 'High', 'Low', 'Close', 'Volume','Adj Close']),
        and then convert the raw data into a pandas DataFrame object.
        
        stock - stock symbol to get data from 
        startDate and endDate - datetime objects defining the range of dates to search for stock information  
        
        IMPORTANT NOTE: The fields 'Close' and 'Adj Close' are the same here, because the Google API already adjusts the closing values
        """
        
        isNASDAQ = stock=='^IXIC'; # Because of how Google Finance works, we have to use a slower API implementation to get data for the NASDAQ
        if isNASDAQ: 
            # Connect with Google API to download stock data as a CSV file
            chunkSize=200
            csv = [];
            for i in range(0,(endDate-startDate).days,chunkSize):
                csv = csv+urllib.urlopen('https://www.google.com/finance/historical?cid=13756934&startdate={}&enddate={}&num={}&ei=XW5QWZH0HZLCeP6rusAO&start={}'.format(startDate.strftime('%b %d, %Y'),endDate.strftime('%b %d, %Y'),chunkSize,i)).readlines();            
            validEntries = [i for i,string in enumerate(csv) if string.startswith('<td class="lm">')];
            
            # Extract and save relevant data into a pandas object
            timeStamps = [];
            historicalStockData = pd.DataFrame(np.zeros([len(validEntries),6]),columns=['Open', 'High', 'Low', 'Close', 'Volume','Adj Close'])
            for i,index in enumerate(validEntries[::-1]): # Entries are reversed, so we read them in reversed order
                # Get timestamp
                timeStamp = csv[index].split('>')[1][:-1];
                timeStamps.append(datetime.datetime.strptime(timeStamp,'%b %d, %Y'));
                
                # Save Open, High, Low, Close, Volume, Adj. Close
                metrics = np.zeros([6]);
                for j in range(5):
                    try:
                        metrics[j] = float(csv[index+1+j].split('>')[1][:-1].replace(',',''));
                    except:
                        metrics[j] = np.nan; 
                metrics[5] = metrics[3]; # Close is already the Adj. Close for Google
                historicalStockData.iloc[i,:] = metrics;
            historicalStockData.index=timeStamps;
        else:
            # Connect with Google API to download stock data as a CSV file
            csv = urllib.urlopen('http://www.google.com/finance/historical?q={}&startdate={}&enddate={}&output=csv'.format(stock,startDate.strftime('%b %d, %Y'),endDate.strftime('%b %d, %Y'))).readlines()
            csv.reverse(); # Raw data is reversed with the most recent data is first, so we need to reverse the list
            csv = csv[:-1]; # Drop the last element in the list, which contains only header information
            
            # Convert data into a pandas object
            timeStamps = [];
            historicalStockData = pd.DataFrame(np.zeros([len(csv),6]),columns=['Open', 'High', 'Low', 'Close', 'Volume','Adj Close'])
            for i,dailyInfo in enumerate(csv): # We ignore the last element because it is just the metric labels
                [timeStamp,Open,High,Low,Close,Volume] = dailyInfo.rstrip().split(',')
                for j,val in enumerate([Open,High,Low,Close,Volume,Close]):
                    if val=='-': #indicates data is not availability
                        historicalStockData.iloc[i,j] = np.nan; 
                    else:
                        historicalStockData.iloc[i,j] = float(val);
                timeStamps.append(datetime.datetime.strptime(timeStamp,'%d-%b-%y')) 
            historicalStockData.index=timeStamps;
        
        return historicalStockData;

    def getListOfNASDAQStocks(self):
        """ Connects with NASDAQ.com to read-in a current list of NASDAQ stocks"""
        csv = urllib.urlopen('http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download').readlines();
        stockSymbols = [];
        for j,entry in enumerate(csv[1:]):
            parsedEntry = entry.rstrip().split('","');
            stockSymbols.append(parsedEntry[0][1:]);  
        return stockSymbols;
        
    def readMarketCapData(self,stockList=None,useYahoo=USE_YAHOO, isVerbose=False):
        """Connects either to Yahoo Finance or NASDAQ.com to read-in current market cap data for the stock list. 
        This function will return a pandas DataFrame reference, and stores the actual data within the stockData object.
        """
        if stockList==None:
            stockList = self.listOfStocks; #If a list of stocks are not supplied, then 
        
        # Either directly connects with Yahoo Finance, or acquires the data directly from the NASDAQ.com website
        if useYahoo: #Note: This function will throw an error after pandas.io.data has been removed
            from pandas.io.data import get_quote_yahoo
            from pandas.io.data import _yahoo_codes
            _yahoo_codes.update({'MarketCap' : 'j1'})
        else:
            csv = urllib.urlopen('http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download').readlines();
            stockSymbols = [];
            marketCapData = pd.Series(np.zeros([len(csv)-1]))
            for j,entry in enumerate(csv[1:]):
                parsedEntry = entry.rstrip().split('","');
                try:
                    marketCapData.iloc[j] = float(parsedEntry[3]);
                except:
                    marketCapData.iloc[j] = np.nan;
                stockSymbols.append(parsedEntry[0][1:]);
            marketCapData.index=stockSymbols;
        
        # Loop through and determine market value for each stock present in the stock list
        self.marketCapData = pd.Series(0,index=stockList);
        numberOfStocks = len(stockList)
        for i,stock in enumerate(stockList):
            if isVerbose:
                print "Reading Market cap data for {} ({} of {})".format(stock,i,numberOfStocks)
            
            if useYahoo:
                numberOfRetries = 0;
                while numberOfRetries < MAXIMUM_RETRIES:
                    try:
                        stockInfo = get_quote_yahoo(stock);
                        marketCapString = stockInfo.iloc[0,0];
                        if marketCapString == 'N/A':
                            marketCap = np.nan;
                        elif isinstance(marketCapString, unicode):
                            marketCap = np.float(marketCapString[:(len(marketCapString)-1)]);
                            if marketCapString[-1]=='B': #'Billion'
                                marketCap = marketCap*1E9;
                            elif marketCapString[-1]=='M': #'Million'
                                marketCap = marketCap*1E6;
                        self.marketCapData[stock]=marketCap;
                        numberOfRetries = MAXIMUM_RETRIES;
                    except:
                        numberOfRetries +=1;      
                        print "Error - Will retry (Attempt {} of {})".format(numberOfRetries,MAXIMUM_RETRIES)
            else:
                try:
                    marketCap = marketCapData[stock];
                except:
                    marketCap = np.nan;
            self.marketCapData[stock]=marketCap;
        return self.marketCapData;
        
    def computeTimeBins(self,startDate=None,endDate=None,timeIntervalType='W',selectedIntervalBinning=4):
        """    
        Function determines the time epochs (default is 4-week bins) used later to bin down the stock data temporally.
        This function will return a pandas-based datetime reference, and stores the actual data within the stockData object.
        
        startDate and endDate   - datetime objects defining the range of dates to search for stock information   
        timeIntervalType        - time unit to performing binning: 'D' (days), 'W' (weeks), 'MS' (months), and 'AS' (years)
        selectedIntervalBinning - how many time units to bin across
        """
        if startDate == None:
            startDate = self.startDate;
        if endDate == None:
            endDate = self.endDate;    
        
        dateIntervals = pd.date_range(start=startDate,end=endDate,freq=timeIntervalType);
        self.dateIntervals = dateIntervals[0::selectedIntervalBinning];    
        return self.dateIntervals;

    def binStockData(self,dateIntervals=None,listOfStocks=[],stockData=None):
        """  
        For the time bins defined by the computeTimeBins function, this function computes and returns:
        a.) temporally-downsampled stock data object (returning the mean value across data contained within each time bin)
        b.) the Pearson's correlation coefficient of each stock with the initial reference stock.
        
        dateIntervals - interval of time to bin stock data
        listOfStocks  - defines stocks to find information. first stock in array is our reference stock/index
        stockData     - pandas data structure containing the stock data (pandas data frame array)
        """
        if isinstance(dateIntervals,pd.DatetimeIndex)==False:
            try:
                dateIntervals = self.dateIntervals;
            except:
                dateIntervals = self.computeTimeBins();
        if len(listOfStocks) == 0:
            listOfStocks = self.listOfStocks;
        if isinstance(stockData, pd.DataFrame)==False:
            stockData = self.stockData;    
        
        # Let's loop through stocks and bin the data down into smaller chunks (also we'll compute local correlations of the stocks)
        self.binnedStockData    = pd.DataFrame(0,index=dateIntervals[:-1],columns=listOfStocks,dtype='float64');
        self.binnedCorrelations = pd.DataFrame(0,index=dateIntervals[:-1],columns=listOfStocks,dtype='float64');
        for binningDateIndex in range(len(dateIntervals)-1):
            # get stock data during current binning interval
            binningStartDate  = dateIntervals[binningDateIndex+0];
            binningEndDate    = dateIntervals[binningDateIndex+1];
            selectedStockData = stockData[(stockData.index>=binningStartDate) & (stockData.index<binningEndDate)];
            
            # save the mean value of the stock data during the current binning interval
            self.binnedStockData.iloc[binningDateIndex,:] = selectedStockData.mean();
            
            # compute local correlations between stocks during this interval
            r = selectedStockData.corr();
            self.binnedCorrelations.iloc[binningDateIndex,:] = r.iloc[0,:];
        return self.binnedStockData, self.binnedCorrelations;
    
    def preProcessStockData(self,stockData=None,zScoreData=True,eliminateStocksLessThanAYear=True):
        """Converts raw stock data into a normalized form (either z-score or a relative change), 
        and then eliminates any stocks with less than a year of data.
        
        zScoreData - either normalizes the data by z-scoring (true), or converting to relative percent difference (false)
        eliminateStocksLessThanAYear - eliminates any NASDAQ stock that was not present for at least one business year
        """
        if isinstance(stockData, pd.DataFrame)==False:
            stockData = self.stockData; # Uses the default stockData pandas DataFrame
        else:
            self.stockData = stockData; # Overwrites the current stockData pandas DataFrame object with the specified input object
        
        # Z-Score data (or use percent difference relative to starting day)
        if zScoreData:
            stockData = (stockData - stockData.mean())/stockData.std();
            stockData = self.zeroShiftStockData(stockData); # While we z-score, we also want to shift the start value to 0, since we are still ultimately interested in any growth.
        else: 
            for stock in stockData.columns:
                firstNonNaNValue = find(stockData[stock].apply(np.isnan)==False,1);
                if len(firstNonNaNValue)==0:
                    offset = 0;
                else:
                    offset = stockData[stock][firstNonNaNValue].values;
                stockData[stock] = 100*(stockData[stock]-offset)/offset;
        
        # Eliminate stocks where we have less than 1-year of data (or 52 weeks)
        numberOfDaysPerIndex=(stockData.index[1]-stockData.index[0]).days;
        if numberOfDaysPerIndex>1: # Takes into account that there are only 5 business days on average
            numberOfDaysPerIndex = int(numberOfDaysPerIndex*5.0/7.0); 
        if eliminateStocksLessThanAYear:
            # Determine stocks that have been present less than one business year (252 days)
            firstValue = np.zeros(stockData.shape[1])
            lastValue  = np.zeros(stockData.shape[1])
            for stockIndex in range(stockData.shape[1]):
                nonNaNs = find(np.isnan(stockData.iloc[:,stockIndex].values)==0);
                if len(nonNaNs)>0:
                    firstValue[stockIndex]=nonNaNs[0];
                    lastValue[stockIndex]=nonNaNs[-1];
            stockThreshold=firstValue<(stockData.shape[0]-(252)/numberOfDaysPerIndex);
            
            # Update stock object list by removing any stocks that were present on the NASDAQ less than a year
            self.stockData = stockData.iloc[:,stockThreshold];
            self.listOfStocks   = self.stockData.columns;
            self.numberOfStocks = len(self.listOfStocks);
        return stockData;        
        
    def zeroShiftStockData(self,stockData=None,standardizeData=False):
        """Shifts all stock values, such that the first non-NaN value is zero
        
        stockData - pandas data structure containing the stock data (pandas data frame array)
        standardizeData - a boolean flag that normalizes the stock data 
        """
        if isinstance(stockData, pd.DataFrame)==False:
            stockData = self.stockData;
            
        # Loops through each stock, determines the first non-NaN value and subtracts this value 
        # from the entire trace (ensuring the initial value is zero). 
        for stock in stockData.columns:
            firstNonNaNValue=find(stockData[stock].apply(np.isnan)==False,1);
            if len(firstNonNaNValue)==0:
                offset=0;
            else:
                offset=stockData[stock][firstNonNaNValue].values;
            stockData[stock]=stockData[stock]-offset;

        # Weights the stock data trace, so that changes in stock price are normalized (by the standard deviation) 
        if standardizeData:
            weightingFactor = (stockData.shape[0]-stockData.apply(np.isnan).apply(np.sum))/stockData.shape[0];
            stockData = weightingFactor*stockData/stockData.std();

        # Eliminates NaNs from the trace
        stockData = stockData.iloc[:,:]-stockData.iloc[0,:].fillna(value=0);
        return stockData;

    def computeDayToDayDifferences(self,stockData=None):
        """ Compute day-to-day fluctuations in stock price by approximating the first temporal derivative"""
        if isinstance(stockData, pd.DataFrame)==False:
            stockData = self.stockData;
        self.stockData_dayToDayDifference = stockData.diff(axis=0);
        return self.stockData_dayToDayDifference;
        
    def computeSlope(self,stockData=None,fittingPackage='numpy'):
        """ computes and returns the rate of change (or slope) in historical stock prices
        using linear regression (change per year)
        
        stockData - pandas data structure containing the stock data (pandas data frame array)
        fittingPackage - uses either 'numpy' or 'sklearn' to compute the slope
        """
        if isinstance(stockData, pd.DataFrame)==False:
            stockData = self.stockData;
            
        self.slopeData = pd.Series(0,index=stockData.columns,dtype='float');
        for stock in stockData.columns:
            validData=stockData[stock].apply(np.isnan).values==False;
            if np.max(validData)==True:
                t = stockData[stock].index-stockData[stock].index[0];
                t = t.days / 365.0;
                if fittingPackage == 'numpy':
                    p = np.polyfit(t[validData],stockData[stock][validData].values,1);
                    self.slopeData[stock] = p[0];
                elif fittingPackage == 'sklearn':
                    regressionModel = sklearn.linear_model.LinearRegression(fit_intercept=False);
                    x = t[validData];
                    x = x.reshape([x.size,1]);
                    y = stockData[stock][validData].values;
                    y = y.reshape([y.size,1]);
                    regressionModel.fit(x,y);
                    self.slopeData[stock] = regressionModel.coef_*regressionModel.score(x,y);
            else:
                self.slopeData[stock] = np.nan;
        return self.slopeData;
        
    def computePortfolioVariance(self,stockData=None,weights=None):
        """ compute portfolio variance (taken from https://en.wikipedia.org/wiki/Modern_portfolio_theory)"""
        if isinstance(stockData, pd.DataFrame)==False:
            stockData = self.stockData;        
        if weights==None:
            weights=np.ones((stockData.shape[1],1));

        sigma = stockData.std(axis=0);
        weightedSigma = weights*sigma.values.reshape([sigma.size,1]);
        rho = stockData.corr();
        portfolioVariance = np.nansum(weightedSigma**2)+np.nansum(np.dot(weightedSigma,weightedSigma.T)*rho*binaryDiagonalSquareMatrix(rho.shape[0],invertMatrix=True));
        return portfolioVariance;
        
    def computePortfolioExpectedReturn(self,stockData=None,weights=None):
        """ compute portfolio expected return (taken from https://en.wikipedia.org/wiki/Modern_portfolio_theory)"""
        if isinstance(stockData, pd.DataFrame)==False:
            stockData = self.stockData;
        if weights==None:
            weights=np.ones(stockData.shape[1]);

        expectedReturn = (stockData.iloc[-2:-1,:].values-stockData.iloc[0:1,:].values);
        return np.nansum(weights.flatten()*expectedReturn.flatten());

    def computeCorrelationsWithMarket(self,stockData=None):   
        """ compute pairwise correlations between each stock and the market (i.e. NASDAQ) """    
        if isinstance(stockData, pd.DataFrame)==False:
            stockData = self.stockData;
        
        # First loop through each stock and compute correlation with NASDAQ
        referenceStock     = stockData.columns[0];
        referenceStockData = stockData[referenceStock];
        self.correlationWithMarket = pd.Series(0,index=stockData.columns,dtype='float64')
        for stock in stockData.columns:
            selectedStockData = stockData[stock];
            selectedIndex     = np.isnan(selectedStockData.values)==False;
            self.correlationWithMarket[stock] = np.corrcoef(referenceStockData.values[selectedIndex],selectedStockData.values[selectedIndex])[0,1];
        return self.correlationWithMarket;
        
    def trainLinearRegressionModel(self, stockData=None, clf=None, startDate=None, endDate=None, dateIntervals=None, usePCA=True): #dimensionReduction='PCA'
        """ Function iteratively generates weighting coefficients for a basic linear regression model 
        using small chunks of time for the selected stock data.
    
        stockData - stock data used for linear fits (dependent and independent variables)
        clf       - linear model for regressions
        startDate and endDate - datetime objects defining the range of dates to search for stock information   
        dateIntervals - The binned time intervals used for the linear regression
        usePCA    - denotes whether to use basic linear regression (false), or to instead perform regression on the principal components (true)
        """
        
        # Define defaults
        if isinstance(stockData, pd.DataFrame)==False:
            stockData = self.stockData;
        if clf == None:    
            from sklearn import linear_model
            clf = linear_model.Ridge(alphas=[0.1,0,1,10],fit_intercept=False,positive=True);
        if startDate == None:
            startDate = self.startDate;
        if endDate == None:
            endDate = self.endDate;  
        if isinstance(dateIntervals,pd.DatetimeIndex)==False:
            try:
                dateIntervals = self.dateIntervals;
            except:
                dateIntervals = self.computeTimeBins()
    
        # Setup input data for linear models
        X = stockData.iloc[:,1:].copy();
        X = X.fillna(value=0); # Determine if there are any NaNs included in the dataset and set them to zero
        y = stockData.iloc[:,0:1].copy(); 
        
        # Optional: Use dimension reduction (like PCA, ICA, or NMF)
        if usePCA: # dimensionReduction != 'None':
            #print("WARNING: Using PCA for regression")
            if X.shape[1]>X.shape[0]:
                inputMatrix = np.zeros([np.max(X.shape),np.max(X.shape)]);
                inputMatrix[:X.values.shape[0],:X.values.shape[1]] = X.values;
            else:
                inputMatrix = X;
            
            # Now perform PCA (using SVD) on the features
            #if dimensionReduction == 'PCA':
            U, s, Vt = np.linalg.svd(inputMatrix,full_matrices=False)
            S = np.diag(s);
            transformMatrix  = np.linalg.inv(np.dot(S, Vt));
                                            
            reduceData=True;
            if reduceData:
                explainedVariance = 0.5; #0.1
                selectedFeatures = np.cumsum(s/np.sum(s))<=explainedVariance;
                #selectedFeatures[selectedFeatures==True] = 0;
                #selectedFeatures[0:3]=1                
                print("Selected {} features".format(np.sum(selectedFeatures)))
            else:
                selectedFeatures = np.ones([s.size],dtype='bool');
            
            # Reshape X and y matrices
            yOriginal = y.copy();
            dates = stockData.index;
            while dates.size<U.shape[0]:
                dates = dates.insert(dates.size,dates[-1]+datetime.timedelta(1))
            #selectedFeatures[0]=False;
            X = pd.DataFrame(U[:,selectedFeatures],index=dates); #selectedFeatures
            y = pd.DataFrame(np.zeros([np.max(X.shape)]),index=dates);
            y.iloc[0:yOriginal.size,0:1] = yOriginal.values;
        else:
            0; #print("WARNING: Not using PCA for regression")
    
        # Loop through each binning interval and determine the performance of a linear model (fit to a training interval) on the test intervals
        trainTestCoeffs = pd.DataFrame(0,index=dateIntervals[:-1],columns=stockData.columns[1:],dtype='float64');
        for currentTrainingDateIndex in range(dateIntervals.size-1): # we skip the last interval
            # 1.) Get data for the current training interval, and train the model on this data
            trainingStartDate = dateIntervals[currentTrainingDateIndex+0];
            trainingEndDate   = dateIntervals[currentTrainingDateIndex+1];
            X_train = X[(X.index>=trainingStartDate) & (X.index<trainingEndDate)];
            y_train = y[(y.index>=trainingStartDate) & (y.index<trainingEndDate)];
            clf.fit(X_train,y_train);
    
            #print("Training model on data between {} and {}".format(trainingStartDate,trainingEndDate));
            
            # 2.) Store the coefficients for the trained linear model
            if usePCA: #if dimensionReduction == 'PCA':
                regressionCoeffs = np.zeros([1,transformMatrix.shape[0]]);
                regressionCoeffs[:,0:clf.coef_.size] = clf.coef_;
                trainTestCoeffs.loc[trainingStartDate,:] = np.dot(regressionCoeffs, transformMatrix.T);
            else:
                trainTestCoeffs.loc[trainingStartDate,:] = clf.coef_;
        return trainTestCoeffs;
        
    def computeRegressionCoefficients(self,stockData=None,stockSet='FAANGStocks',clf = sklearn.linear_model.RidgeCV(fit_intercept=False,cv=5),binningIntervals=[4],skipCorrelations=True,usePCA=False,timeIntervalType='W'):
        """ Compute summary statistics for regression fits (using the stockSet specified). Ridge regression with cross-validation is used as the default."""

        # Get and make a partial copy of the stockData DataFrame object (only copying the specified stock set)
        if isinstance(stockData, pd.DataFrame)==False:
            stockData = self.stockData;
        sequenceOfStocks  = self.getSequenceOfStocks(stockSet);
        stockData         = stockData.iloc[:,sequenceOfStocks].copy(); 
                
        # Bin the data further and compute regression fits
        for index,binningInterval in enumerate(binningIntervals):        
            # Compute correlation coefficients within binning intervals
            dateIntervals       = self.computeTimeBins(timeIntervalType=timeIntervalType,selectedIntervalBinning=binningInterval);
            if skipCorrelations:
                correlationCoeffs = [];
            else:
                binnedStockData, binnedCorrelations = self.binStockData(dateIntervals,stockData.columns,stockData);  
                correlationCoeffs   = (binnedCorrelations.iloc[:,1:]+1)/2;
                normalizationFactor = (correlationCoeffs.sum(axis=1)/correlationCoeffs.shape[1]);
                normalizationFactor = pd.DataFrame(np.tile(normalizationFactor,[correlationCoeffs.shape[1],1]).T,index=correlationCoeffs.index,columns=correlationCoeffs.columns);
                correlationCoeffs   = correlationCoeffs.divide(normalizationFactor);
            
            # Compute regression coefficients within binning intervals
            modelCoeffs = self.trainLinearRegressionModel(stockData, clf, self.startDate, self.endDate, dateIntervals, usePCA=usePCA);
            if index == 0:
                regressionCoeffs = modelCoeffs 
                corrCoeffs       = correlationCoeffs
            else:
                regressionCoeffs.append(modelCoeffs);
                corrCoeffs.append(correlationCoeffs);
        return regressionCoeffs, correlationCoeffs;   

    def compareMarketCapDataToMarketCorrelation(self,marketCapData=None,correlationValues=None):
        """ computes the relationship of a stock's market cap value with the stock's correlation to the market (i.e. NASDAQ)"""
    
        # Get market cap data
        if isinstance(marketCapData, pd.Series)==False:
            if(hasattr(self, 'marketCapData')==False):
                marketCapData = self.readMarketCapData();
            else:
                marketCapData = self.marketCapData;
        if isinstance(correlationValues, pd.Series)==False:
            if(hasattr(self, 'correlationValues')==False):
                correlationValues = self.computeCorrelationsWithMarket();
            else:
                correlationValues = self.correlationWithMarket;
    
        # Construct a comparison of market cap value (normalized or unnormalized) vs correlation with NASDAQ
        marketData = pd.DataFrame(index=correlationValues.index);  
        marketData['CorrelationWithNASDAQ']=correlationValues;
        marketData['MarketCap']=marketCapData;
        marketData['MarketCap_Log10']=marketCapData.apply(np.log10);
        marketData['MarketCap_Log10'][np.isfinite(marketData['MarketCap_Log10'])==False]=np.nan;
        marketData['MarketCapNormalized']=marketCapData; #Computes market cap by rank
        
        # Drop data with NaNs, and renormalize the normalized market cap metric (taking into account the dropped entries)
        hasNaNs = np.sum(np.isnan(marketCapData));
        sortedIndexValues=np.argsort(marketCapData.fillna(value=-1).values);
        for index in range(sortedIndexValues.size):
            if(index<hasNaNs):
                marketData.iloc[sortedIndexValues[index],3]=np.nan;
            else:
                marketData.iloc[sortedIndexValues[index],3]=(index-hasNaNs)/(0.0+sortedIndexValues.size-hasNaNs);
        marketData=marketData.dropna();
        
        # Display correlations between groups
        print(marketData.corr())
    
        # Plot relationships
        fig = showScatterPlot(marketData['MarketCap_Log10'],marketData['CorrelationWithNASDAQ'], \
                              xTitle='Market Cap Value (Log-10)',yTitle='Correlation with NASDAQ',title='', \
                              xLimits=[6,12],yLimits=[-1,1])
        fig.set_tight_layout(True);        
        
    def summarizeRelationshipWithNASDAQ(self,selectedStocks=[],dataArray=None,slopeData=None,correlationWithNASDAQ=None,stockLabels=[],figSize=(10,8),title='',yLabel='Normalized units'):
        """Show a summary plot highlighting the relationship between the selected stock list and the NASDAQ"""
        
        # Setup basic plotting parameters
        if isinstance(dataArray, pd.DataFrame) == False:
            dataArray = self.stockData;
        if len(selectedStocks) == 0:
            selectedStocks = self.getSequenceOfStocks('FAANGStocks');
        dataArray = dataArray.iloc[:,selectedStocks];
        numberOfStocks = dataArray.shape[1];
        lineWidths    = 2*np.ones([numberOfStocks,1]);
        lineWidths[0] = 10;
        LUT       = cat(0,np.zeros([1,4]),0.75*cm.rainbow(np.linspace(0,1,numberOfStocks-1)));
        LUT[:,-1] = 1;
    
        # Top Panel: Scatter plot of historical data for selected stocks and NASDAQ
        h = plt.pyplot.figure(figsize=figSize); 
        h.suptitle(title)
        axis = plt.pyplot.subplot(2,1,1);
        for i in range(numberOfStocks):
            dataArray.iloc[:,i].plot(ax=axis,color=LUT[i,:],linewidth=lineWidths[i])
        axis.set_xlim([self.startDate,self.endDate])
        for xLabels in axis.xaxis.get_majorticklabels():
            xLabels.set_rotation(0);
            xLabels.set_horizontalalignment('center')
        axis.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        axis.get_legend().set_visible(False);
        axis.set_ylabel(yLabel);
        
        # Bottom Left Panel: Show slopes of selected stocks and NASDAQ
        axis = plt.pyplot.subplot(2,2,3);
        if len(stockLabels)==0:
            xLabels = dataArray.columns.values;
        else:
            xLabels = stockLabels;
        x = np.linspace(0,len(xLabels)-1,len(xLabels))-0.4;
        if isinstance(slopeData, pd.Series)==False:
            slopeData = self.computeSlope(dataArray);
        for i in range(0,len(x)):
          plt.pyplot.bar(x[i]-0.4,slopeData.iloc[i],color=LUT[i,:])
        axis.set_xticks(x);
        axis.set_xticklabels(xLabels,rotation=45,horizontalalignment='center');
        axis.set_ylabel('Average annual growth rate');
        axis.set_xlim([x.min()-0.8,x.max()+0.8]);
        
        # Bottom Right Panel: Show correlations of selected stocks with NASDAQ
        axis=plt.pyplot.subplot(2,2,4);
        if isinstance(correlationWithNASDAQ, pd.Series)==False:
            correlationWithNASDAQ = self.computeCorrelationsWithMarket(self.computeDayToDayDifferences(stockData=dataArray));
        for i in range(1,len(x)):
          plt.pyplot.bar(x[i]-0.4,correlationWithNASDAQ.iloc[i],color=LUT[i,:])
        axis.set_xticks(x);
        axis.set_xticklabels(xLabels,rotation=45,horizontalalignment='center');
        axis.set_ylabel('Correlation with NASDAQ');
        axis.set_xlim([x[1:].min()-0.8,x[1:].max()+0.8]);
        h.set_tight_layout(True);
        
    def showHistoricalCharts(self,dataArray=None,xLabel='',yLabel='',title='',xLimits=[],yLimits=[],axes=None,skipMarket=False,figSize=(10,4)):
        """ Historical chart of the input data vs. time:
        Computes and plots the mean historical chart for each stock group (as a solid line) and
        95th confidence intervals (shaded areas). Divides the stock data into a different number of
        groups defined by the getStockTypeInfo function.
        
        dataArray - pandas dataFrame containing the input data
        xLabel    - x-axis label
        yLabel    - y-axis label
        title     - plot title
        xLimits   - x-axis limits 
        yLimits   - y-axis limits
        axes      - optional axes handle defining where to plot the chart data. Else a new figure is generated
        """
        if isinstance(dataArray, pd.DataFrame)==False:
            dataArray = self.stockData;
        if axes==None:
            fig, axes = plt.pyplot.subplots(nrows=1,ncols=1,figsize=figSize); 
        else:
            fig = axes.get_figure();
            
        # Loop through each stock group, and plot the mean trace (and 95th confidence intervals) across time
        for stockType in range(skipMarket,NUMBER_OF_STOCK_TYPES):
            selectedStocks,setName,lineType,lineColor = self.getStockTypeInfo(stockType);
    
            # Compute means and 95th confidence intervals
            x = dataArray.index;
            n = len(selectedStocks) - np.sum(np.isnan(dataArray.values[:,selectedStocks]),axis=1);
            t = stats.t.ppf(1-0.05/2.0, n-1);
            y = dataArray.values[:,selectedStocks];
            yMean = np.nanmean(y,axis=1);
            yStd  = np.nanstd(y,axis=1)
            error = t*yStd/np.sqrt(n)
              
            # Plot the mean and 95th confidence intervals
            axes.plot(x,yMean,lineType,label=setName,color=lineColor)
            axes.fill_between(x, yMean-error, yMean+error, alpha=0.25, edgecolor=lineColor, facecolor=lineColor)
        axes.legend(loc='upper left',ncol=2,framealpha=0,borderpad=None,fontsize=FONT_SIZE_OBJECTS*5/4.);
        if len(xLimits)==2:
            axes.set_xlim(xLimits[0],xLimits[1]);
        if len(yLimits)==2:
            axes.set_ylim(yLimits[0],yLimits[1]);
        axes.set_xlabel(xLabel);
        axes.set_ylabel(yLabel);
        axes.set_title(title);
        #return fig;
    
    def showCumulativeProbabilityPlot(self,dataArray,xLabel='',title='',xLimits=[],yLimits=[0,1],axes=None,skipMarket=True,figSize=(5,4)):
        """Show cumulative probability plot:
        Computes and plots a cumulative probability trace for each stock group (as a solid line). 
        Divides the stock data into a different number of groups as defined by the getStockTypeInfo function.
        
        dataArray - pandas dataSeries containing the input data
        xLabel    - x-axis label
        title     - plot title
        xLimits   - x-axis limits 
        yLimits   - y-axis limits. Default is [0,1]
        axes      - optional axes handle defining where to plot the chart data. Else a new figure is generated
        """
        if axes==None:
            fig, axes = plt.pyplot.subplots(nrows=1,ncols=1,figsize=figSize); 
        else:
            fig = axes.get_figure();
            
        for stockType in range(skipMarket,NUMBER_OF_STOCK_TYPES):
            selectedStocks,setName,lineType,lineColor = self.getStockTypeInfo(stockType);
            x = np.sort(dataArray.values[selectedStocks],axis=0);
            x = x[np.isnan(x)==False];
            y = np.linspace(0,1,len(x));
            axes.plot(x,y,lineType,label=setName,color=lineColor,linewidth=3)
        axes.legend(loc='best',framealpha=0,borderpad=None)
        axes.get_legend().set_visible(False);
        if xLimits != None:
            axes.set_xlim(xLimits[0],xLimits[1]);
        axes.set_ylim(yLimits[0],yLimits[1]);
        axes.set_xlabel(xLabel)
        axes.set_ylabel('Cumulative probability')
        axes.set_title(title)
        #return fig;   
    
    def showBarPlot(self,dataArray,xLabel='',title='',xLimits=[],yLimits=[0,1],axes=None,skipMarket=True,figSize=(5,4)):
        """Show bar plot:
        Computes and plots a bar plot for each stock group (Mean +/- SEM). 
        Divides the stock data into a different number of groups as defined by the getStockTypeInfo function.
        
        dataArray - pandas dataSeries containing the input data
        xLabel    - x-axis label
        title     - plot title
        xLimits   - x-axis limits 
        yLimits   - y-axis limits. Default is [0,1]
        axes      - optional axes handle defining where to plot the chart data. Else a new figure is generated
        """
        if axes==None:
            fig, axes = plt.pyplot.subplots(nrows=1,ncols=1,figsize=figSize); 
        else:
            fig = axes.get_figure();
            
        barList=[];
        for stockType in range(skipMarket,NUMBER_OF_STOCK_TYPES):
            selectedStocks,setName,lineType,lineColor = self.getStockTypeInfo(stockType);
            shortName = self.getShortName(stockType)
            y = dataArray.values[selectedStocks]
            y = y[np.isnan(y)==False];
            axes.bar(stockType-0.4,np.mean(y),label=setName,color=lineColor)
            axes.errorbar(stockType, np.mean(y), yerr=np.std(y)/np.sqrt(len(y)),color='k');
            barList.append(shortName);            
            print("{} - {} +/- {} (SEM)".format(setName,np.mean(y),np.std(y)/np.sqrt(len(y))));
        axes.set_xticks(range(skipMarket,NUMBER_OF_STOCK_TYPES))  
        axes.set_xticklabels(barList)
        axes.set_xlim(0.5,NUMBER_OF_STOCK_TYPES-.5);
        axes.set_ylabel(xLabel);
        #return fig; 
        
    def computeSignificance(self,dataArray):
        # Compute whether there are significant differences between groups. 
        # Significance is assessed with a rank-sum test (i.e. Mann Whitney U test)
        
        significanceMatrix = np.zeros([NUMBER_OF_STOCK_TYPES-1,NUMBER_OF_STOCK_TYPES-1]);
        for currentGroupA in range(1,NUMBER_OF_STOCK_TYPES):
            output=self.getStockTypeInfo(currentGroupA); 
            selectedStocksA=output[0];
            for currentGroupB in range(1,NUMBER_OF_STOCK_TYPES):
                output=self.getStockTypeInfo(currentGroupB); 
                selectedStocksB=output[0];
                statTest=stats.ranksums(dataArray[selectedStocksA],dataArray[selectedStocksB]);
                significanceMatrix[currentGroupA-1,currentGroupB-1] = statTest.pvalue;
        return significanceMatrix;
        
    def showCumulativeProbabilityAndBarPlot(self,dataArray,xLabel='',title='',xLimits=None,yLimits=[0,1],skipMarket=True,addStatsToPlot=True,figSize=(10,4)):
        """Show cumulative probability and bar plot for input pandas DataSeries object, plus compute statistical tests for group distances"""
            
        # Show bar and cumulative probability plots
        fig, (ax1,ax2) = plt.pyplot.subplots(nrows=1,ncols=2,figsize=figSize)
        self.showCumulativeProbabilityPlot(dataArray,xLabel,title,xLimits,yLimits,ax1)
        self.showBarPlot(dataArray,xLabel,title,xLimits,yLimits,ax2,skipMarket=skipMarket)
        fig.set_tight_layout(True);
        
        # Compute statistics to compare groups
        significanceMatrix = self.computeSignificance(dataArray);
        significanceMatrix = np.round(significanceMatrix,4);
        np.set_printoptions(precision=4,suppress=True)
        groupLabels = [self.getShortName(i) for i in range(1,NUMBER_OF_STOCK_TYPES)]
        for index,text in enumerate(groupLabels): 
            if len(text.split('\n'))==2:
                groupLabels[index] = text.split('\n')[1][1:-1]; # Abbreviates strings with the \n character
        significanceTable  = pd.DataFrame(significanceMatrix,index=groupLabels,columns=groupLabels)
        print("--------------------------------------------------------")
        print("Testing significance of differences between stock groups (Mann Whitney U Test): \n{}".format(significanceTable))
        if addStatsToPlot:
            xMatrix = range(1,NUMBER_OF_STOCK_TYPES);
            yMatrix = [];
            for currentGroup in xMatrix:
                output=self.getStockTypeInfo(currentGroup); 
                selectedStocks=output[0];
                yMatrix.append(np.nanmean(dataArray[selectedStocks])+np.nanstd(dataArray[selectedStocks])/np.sqrt(len(selectedStocks)));
            addDifferencesLabels(xMatrix,yMatrix,significanceMatrix,ax2,params=None)
        #return fig;
        
    def getSequenceOfStocks(self,stockSet):
        """get sequence of stock based on label"""
        if(  stockSet=='NASDAQ'):
            sequenceOfStocks = [0];
        elif(stockSet=='FAANGStocks'):
            sequenceOfStocks = range(1,4)+range(5,8); 
        elif(stockSet=='Top100Stocks'):
            sequenceOfStocks = range(1,4)+range(5,103);
        elif(stockSet=='AllStocks'):
            sequenceOfStocks = range(1,4)+range(5,self.stockData.shape[1]); 
        elif(stockSet=='Top100StocksExcluding'):
            sequenceOfStocks = range(8,103);
        elif(stockSet=='AllStocksExcluding'):
            sequenceOfStocks = range(103,self.stockData.shape[1]); 
        elif(stockSet=='Top20PercentExcluding'):
            try:
                slopeData            = pd.Series.from_csv(os.getcwd()+'\stockSlopeValues.csv');
                slopeData.iloc[:103] = np.nan; # Eliminating FAANG and Top-100 Stocks
                sequenceOfStocks     = find(slopeData.values >= np.nanpercentile(slopeData,80));
            except:
                print "Failed to get Top 20% performing stocks. Defaulting to all stocks (excluding Top 100)";
                sequenceOfStocks = range(103,self.stockData.shape[1]);
        elif(stockSet=='Bottom80PercentExcluding'):
            try:
                slopeData            = pd.Series.from_csv(os.getcwd()+'\stockSlopeValues.csv');
                slopeData.iloc[:103] = np.nan; # Eliminating FAANG and Top-100 Stocks
                sequenceOfStocks     = find(slopeData.values < np.nanpercentile(slopeData,80));
            except:
                print "Failed to get Top 20% performing stocks. Defaulting to all stocks (excluding Top 100)";
                sequenceOfStocks = range(103,self.stockData.shape[1]);
        else:
            sequenceOfStocks = range(107,255);
        sequenceOfStocks.insert(0,0);
        return sequenceOfStocks;
        
    def getStockTypeInfo(self,stockType):
        """ returns information about what stocks are FANG, Top 100 NASDAQ, all NASDAQ, and NASDAQ Index"""
        LUT = 0.75*cm.brg(np.linspace(0,1,3)); #cm.rainbow
        LUT[:,-1] = 1;
        if stockType == 1: #FAANG stocks
            selectedStocks = self.getSequenceOfStocks('FAANGStocks')[1:];
            setName   = 'FAANG stocks';
            lineType  = '-';
            lineColor = LUT[1,:]; #'r';
        elif stockType == 2: # Other Top 100 NASDAQ stocks
            selectedStocks = self.getSequenceOfStocks('Top100StocksExcluding')[1:];
            setName   = 'Other Top 100 NASDAQ stocks'
            lineType  = '-';
            lineColor = LUT[0,:]; #'b';
        elif stockType == 3: # All other NASDAQ stocks
            selectedStocks = self.getSequenceOfStocks('AllStocksExcluding')[1:];
            setName   = 'Other NASDAQ stocks (All)';
            lineType  = '-';
            lineColor = LUT[2,:]; #'g';
        elif stockType == 5: # Top 20% Performing Stocks (Excluding Top 100)
            selectedStocks = self.getSequenceOfStocks('Top20PercentExcluding')[1:];
            setName   = 'Other NASDAQ stocks (Top 20%)';
            lineType  = '-'; #'-';
            lineColor = 0.75*np.array([1,0.75,0,1]); 
        elif stockType == 4: # Bottom 80% Performing Stocks (Excluding Top 100)
            selectedStocks = self.getSequenceOfStocks('Bottom80PercentExcluding')[1:];
            setName   = 'Other NASDAQ stocks (Bottom 80%)';
            lineType  = '-'; #'-';
            lineColor = 0.75*np.array([1,0,0.75,1]); 
        else: # NASDAQ
            selectedStocks = [0];
            setName   = 'NASDAQ';
            lineType  = '-';
            lineColor = 'k';
        return selectedStocks,setName,lineType,lineColor;

    def getShortName(self,stockType):
        """return short name for stock type. useful for bar plot labels"""
        if stockType == 1: #FAANG stocks
            shortName = 'FAANG';
        elif stockType == 2: # Other Top 100 NASDAQ stocks
            shortName = 'Top 100';
        elif stockType == 3: # All other NASDAQ stocks
            shortName = 'Other';
        elif stockType == 5: # All other NASDAQ stocks (Top 20%)
            shortName = 'Other\n(Top 20%)';  
        elif stockType == 4: # All other NASDAQ stocks (Bottom 80%)
            shortName = 'Other\n(Bottom 80%)';  
        else: # NASDAQ
            shortName = 'NASDAQ';
        return shortName;