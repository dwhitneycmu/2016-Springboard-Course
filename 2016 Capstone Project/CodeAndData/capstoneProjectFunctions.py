# PACKAGES AND FUNCTIONS USED TO PERFORM SUBSEQUENT ANALYSIS (SKIP TO NEXT SECTION FOR FIGURES):
import datetime
import matplotlib
import matplotlib as plt
from matplotlib import cm
from matplotlib.pyplot import *
import math
import numpy as np
import os 
import pandas as pd
import pandas_datareader.data as web
from pandas.io.data import get_quote_yahoo
from pandas.io.data import _yahoo_codes
import random
import scipy.stats as stats
import scipy.signal as signal
import sklearn
from sklearn import linear_model
import time
#%matplotlib inline
#import matplotlib as mpl
#plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.facecolor']='white'
plt.rcParams['figure.facecolor']='white'
from mpl_toolkits.axes_grid1 import make_axes_locatable

def loadData(startDate, endDate, listOfStocks, stockPriceType, zScoreData=True):
    """    
    function either connects to yahoo.com to read stock data (defined by listOfStocks) from scratch or reads the file from a .csv file
    
    startDate and endDate - datetime objects defining the range of dates to search for stock information   
    listOfStocks    - defines stocks to find information. first stock in array is our reference stock/index
    stockPriceType  - stock metric - can be either [Open, High, Low, Close, Volume, Adj Close]
    zScoreData      - denotes whether to perform normalization on stock data (via z-scoring).
    """
    
    # Loop through different stock name and add them to the stockData object
    stockData       = pd.DataFrame();
    for stock in listOfStocks[:]:
        # Use the built-in web reader of pandas to get financial data for specified stocks.
        # The result data structure is a pandas dataFrame object with 6 columns: 
        # [Open, High, Low, Close, Volumn, Adj Close]
        print 'Reading {}'.format(stock)
        numberOfRetries = 0;
        maximumRetries  = 100;
        
        while numberOfRetries < maximumRetries:
            try:
                allHistoricalStockData = web.DataReader(stock, 'yahoo', startDate, endDate);
                numberOfRetries = maximumRetries;
            except:
                numberOfRetries +=1;      
                print "Error - Will retry (Attempt {} of {})".format(numberOfRetries,maximumRetries)
                if numberOfRetries == maximumRetries:
                    print "Skipped stock"
                    continue; # skip stock
        selectedStockData = allHistoricalStockData[stockPriceType];  
        
        # Next automatically acquire data for stock splits, and then alter the original data
        # so that stock splits are taken into account (i.e. multiply the stock price by the stock split ratio)
        currentMiscInfo  = web.DataReader(stock, 'yahoo-actions', startDate, endDate);
        if currentMiscInfo.size>0:
            stockSplitEvents = currentMiscInfo[currentMiscInfo['action']=='SPLIT'];
            for currentStockSplit in stockSplitEvents.index:
                selectedStockDates = selectedStockData.index >=currentStockSplit; # only corrects for data with timestamps that are equal to or greater than the stock split date.
                selectedStockData[selectedStockDates] = selectedStockData[selectedStockDates]*(1.0/stockSplitEvents.value[currentStockSplit]);
        
        # Select the chosen column of data (specified by stockPriceType) and import
        # the data into the stockData object. Resulting data structure columns are 
        # defined by the names list contained in listOfStocks
        stockData[stock] = pd.Series(selectedStockData);
    
    # Now because of the great variation in stock price, let's normalize stock data from each column (via z-scoring)
    if zScoreData:
        stockData=(stockData - stockData.mean())/stockData.std()
    return stockData;
 
def getMarketCapData(stockList):
    """returns current market cap data for the stock list"""

    _yahoo_codes.update({'MarketCap' : 'j1'})
    marketCapData  = pd.Series(0,index=stockList);
    numberOfStocks = len(stockList)
    for i,stock in enumerate(stockList):
        print "Reading Market cap data for {} ({} of {})".format(stock,i,numberOfStocks)
        numberOfRetries = 0;
        maximumRetries  = 100;
        
        while numberOfRetries < maximumRetries:
            try:
                stockInfo = get_quote_yahoo(stock);
                marketCapString = stockInfo.iloc[0,0];
                if marketCapString == 'N/A':
                    marketCap = np.nan;
                elif isinstance(marketCapString, unicode):
                    marketCap = np.float(marketCapString[:(len(marketCapString)-1)]);
                    if marketCapString[-1]=='B':
                        marketCap = marketCap*1E9;
                    elif marketCapString[-1]=='M':
                        marketCap = marketCap*1E6;
                marketCapData[stock]=marketCap;
                numberOfRetries = maximumRetries;
            except:
                numberOfRetries +=1;      
                print "Error - Will retry (Attempt {} of {})".format(numberOfRetries,maximumRetries)
        marketCapData[stock]=marketCap;
    return marketCapData;
    
def computeTimeBins(startDate,endDate,timeIntervalType='W',selectedIntervalBinning=4):
    """    
    function determines time bins for dates.
    
    startDate and endDate   - datetime objects defining the range of dates to search for stock information   
    timeIntervalType        - time unit to performing binning: 'D' (days), 'W' (weeks), 'MS' (months), and 'AS' (years)
    selectedIntervalBinning - how many time units to bin across
    """
    
    dateIntervals = pd.date_range(start=startDate,end=endDate,freq=timeIntervalType);
    dateIntervals = dateIntervals[0::selectedIntervalBinning];    
    return dateIntervals;

def binStockData(dateIntervals,listOfStocks,stockData):
    """    
    function bins stock data across time bins, and also returns the correlations of each stock with the reference stock (within the binned time frame).
    
    dateIntervals - interval of time to bin stock data
    listOfStocks  - defines stocks to find information. first stock in array is our reference stock/index
    stockData     - data structure containing the stock data (pandas data frame array)
    """
    
    # Let's loop through stocks and bin the data down into smaller chunks (also we'll compute local correlations of the stocks)
    binnedStockData    = pd.DataFrame(0,index=dateIntervals[:-1],columns=listOfStocks,dtype='float64');
    binnedCorrelations = pd.DataFrame(0,index=dateIntervals[:-1],columns=listOfStocks,dtype='float64');
    for binningDateIndex in range(len(dateIntervals)-1):
        # get stock data during current binning interval
        binningStartDate  = dateIntervals[binningDateIndex+0];
        binningEndDate    = dateIntervals[binningDateIndex+1];
        selectedStockData = stockData[(stockData.index>=binningStartDate) & (stockData.index<binningEndDate)];
        
        # save the mean value of the stock data during the current binning interval
        binnedStockData.iloc[binningDateIndex,:] = selectedStockData.mean();
        
        # compute local correlations between stocks during this interval
        r = selectedStockData.corr();
        binnedCorrelations.iloc[binningDateIndex,:] = r.iloc[0,:];
    return binnedStockData, binnedCorrelations;
    
def zeroStockData(stockData):
    """Set the first valid stock value to zero and shift all stock values appropriately"""
    for stock in stockData.columns:
        firstNonNaNValue=find(stockData[stock].apply(np.isnan)==False,1);
        if len(firstNonNaNValue)==0:
            offset=0;
        else:
            offset=stockData[stock][firstNonNaNValue].values;
        stockData[stock]=stockData[stock]-offset;
    weightingFactor=(stockData.shape[0]-stockData.apply(np.isnan).apply(np.sum))/stockData.shape[0];
    stockData = weightingFactor*stockData/stockData.std();
    stockData = stockData.iloc[:,:]-stockData.iloc[0,:].fillna(value=0);
    return stockData;
    
def showScatterPlot(dataArray,xLabel='',yLabel='',title='',xLimits=[],yLimits=[],axes=None):
    """ show scatter plot"""
    if axes==None:
        fig, axes = plt.pyplot.subplots(nrows=1,ncols=1,figsize=(10,4)); 
    else:
        fig = axes.get_figure();
        
    for stockType in range(4):
        selectedStocks,setName,lineType,lineColor = getStockTypeInfo(stockType);

        x = dataArray.index;
        n = len(selectedStocks) - np.sum(np.isnan(dataArray.values[:,selectedStocks]),axis=1);
        t = stats.t.ppf(1-0.05/2.0, n-1);
        y = dataArray.values[:,selectedStocks];
        yMean = np.nanmean(y,axis=1);
        yStd  = np.nanstd(y,axis=1)
        error = t*yStd/np.sqrt(n)
                
        axes.plot(x,yMean,lineType,label=setName,color=lineColor)
        axes.fill_between(x, yMean-error, yMean+error, alpha=0.25, edgecolor=lineColor, facecolor=lineColor)
    #axes.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axes.legend(loc='upper left',ncol=2,framealpha=0,borderpad=None);
    axes.set_xlim(xLimits[0],xLimits[1]);
    axes.set_ylim(yLimits[0],yLimits[1]);
    axes.set_xlabel(xLabel);
    axes.set_ylabel(yLabel);
    axes.set_title(title);
    return fig;
    
def showScatterPlot_TwoProperties(x,y,xTitle='',yTitle='',title='',xLimits=[],yLimits=[],axes=None):
    """ show a regular scatter plot plotting two properties versus each other """
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
    if len(xLimits)==0:
        yLimits = [np.min(y),np.max(y)];
    
    # Compute a first order linear fit and correlation value
    P = np.polyfit(x,y,1)
    print 'Relationship of {} vs. {}: r = {}'.format(xTitle,yTitle,np.corrcoef(x,y)[0,1]);
    
    # Plot data
    axes.plot(x,y,'.k');
    axes.plot(np.sort(x),np.polyval(P,np.sort(x)),'-r',linewidth=3)
    axes.set_title(title)
    axes.set_xlabel(xTitle)
    axes.set_ylabel(yTitle)
    axes.set_xlim(xLimits);
    axes.set_ylim(yLimits)
    return fig;
    
def showCumulativeProbabilityPlot(dataArray,xLabel='',title='',xLimits=[],yLimits=[0,1],axes=None):
    """Show cumulative probability plot"""
    if axes==None:
        fig, axes = plt.pyplot.subplots(nrows=1,ncols=1,figsize=(5,4)); 
    else:
        fig = axes.get_figure();
        
    for stockType in range(1,4):
        selectedStocks,setName,lineType,lineColor = getStockTypeInfo(stockType);
        x = np.sort(dataArray.values[selectedStocks],axis=0);
        x = x[np.isnan(x)==False];
        y = np.linspace(0,1,len(x));
        axes.plot(x,y,lineType,label=setName,color=lineColor,lineWidth=3)
    axes.legend(loc='best',framealpha=0,borderpad=None)
    axes.get_legend().set_visible(False);
    if xLimits != None:
        axes.set_xlim(xLimits[0],xLimits[1]);
    axes.set_ylim(yLimits[0],yLimits[1]);
    axes.set_xlabel(xLabel)
    axes.set_ylabel('Cumulative probability')
    axes.set_title(title)
    return fig;   
    
def showBarPlot(dataArray,xLabel='',title='',xLimits=[],yLimits=[0,1],axes=None):
    """Show bar plot"""
    if axes==None:
        fig, axes = plt.pyplot.subplots(nrows=1,ncols=1,figsize=(5,4)); 
    else:
        fig = axes.get_figure();
        
    barList=[];
    for stockType in range(1,4):
        selectedStocks,setName,lineType,lineColor = getStockTypeInfo(stockType);
        shortName = getShortName(stockType)
        x = np.sort(dataArray.values[selectedStocks],axis=0)
        x = x[np.isnan(x)==False];
        y = np.linspace(0,1,len(x));
        axes.bar(stockType-0.4,np.mean(x),label=setName,color=lineColor)
        axes.errorbar(stockType, np.mean(x), yerr=np.std(x)/np.sqrt(len(y)),color='k');
        barList.append(shortName);
        
        print "{} - {} +/- {} (SEM)".format(setName,np.mean(x),np.std(x)/np.sqrt(len(y)));
    axes.set_xticks(range(1,4))  
    axes.set_xticklabels(barList)
    axes.set_xlim(0.5,3.5);
    axes.set_ylim(0,1); #0.75
    axes.set_ylabel(xLabel);
    return fig; 
    
def showCumulativeProbabilityAndBarPlot(dataArray,xLabel='',title='',xLimits=None,yLimits=[0,1]):
    """Show cumulative probability and bar plot, plus compute statistical tests for group distances"""

    fig, (ax1,ax2) = plt.pyplot.subplots(nrows=1,ncols=2,figsize=(10,4))
    showCumulativeProbabilityPlot(dataArray,xLabel,title,xLimits,yLimits,ax1)
    showBarPlot(dataArray,xLabel,title,xLimits,yLimits,ax2)
    fig.set_tight_layout(True);
    
    # Compute statistics to compare groups
    significanceTest=np.zeros([3,3]);
    for currentGroupA in range(1,4):
        output=getStockTypeInfo(currentGroupA); 
        selectedStocksA=output[0];
        for currentGroupB in range(1,4):
            output=getStockTypeInfo(currentGroupB); 
            selectedStocksB=output[0];
            statTest=stats.ranksums(dataArray[selectedStocksA],dataArray[selectedStocksB]);
            significanceTest[currentGroupA-1,currentGroupB-1]=statTest.pvalue;
    np.set_printoptions(precision=4,suppress=True)
    print "Testing significance of differences between stock groups (Mann Whitney): \n{}".format(significanceTest)    
    return fig;
    
def predictMarketFromSelectedStocks(dataArray,xLabel='Number of Stocks (included in model)',yLabel='Correlation between model and NASDAQ',title='Model Summary:',yLimits=[0,1]):
    """Simplistic linear model of summing up X number of stocks and taking average"""
    numberOfStocksInSet = [1,2,5,10,20,50,100,500,3000];
    numberOfIterations  = 250;
    
    fig=plt.figure.Figure();
    axes=fig.add_axes();    
    plt.pylab.clf()
    plt.pylab.hold(1);
    for stockType in range(1,4):
        selectedStocks,setName,lineType,lineColor = getStockTypeInfo(stockType);
        
        x = range(len(numberOfStocksInSet));
        correlationValues = pd.DataFrame(0.0,index=range(numberOfIterations),columns=numberOfStocksInSet);
        for currentNumberOfStockSet in numberOfStocksInSet:
            for currentBootstrapIteration in range(numberOfIterations):
                random.seed(currentBootstrapIteration);
                selectedBootstrapSet = map(lambda x: np.round(random.uniform(selectedStocks[0],selectedStocks[-1])), range(currentNumberOfStockSet))
                
                modelTrace  = dataArray.iloc[:,selectedBootstrapSet].mean(axis=1);
                marketTrace = dataArray.iloc[:,0];
                r = np.corrcoef(modelTrace.fillna(0.0),marketTrace.fillna(0.0))
                correlationValues[currentNumberOfStockSet][currentBootstrapIteration] = r[0,1];
                
        n = numberOfIterations;
        t = stats.t.ppf(1-0.05/2.0, n-1);
        y = correlationValues.values;
        yMean = np.nanmean(y,axis=0);
        yStd  = np.nanstd(y,axis=0)
        error = t*yStd/np.sqrt(n)
                
        plt.pylab.plot(x,yMean,lineType,label=setName,color=lineColor)
        plt.pylab.fill_between(x, yMean-error, yMean+error, alpha=0.25, edgecolor=lineColor, facecolor=lineColor)
    plt.pylab.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.pylab.xticks(x,numberOfStocksInSet)
    plt.pylab.xlim(0,len(numberOfStocksInSet));
    plt.pylab.ylim(yLimits[0],yLimits[1]);
    plt.pylab.xlabel(xLabel)
    plt.pylab.ylabel(yLabel)
    plt.pylab.title(title)
    plt.pylab.show();
    return fig;
    
def trainLinearRegressionModel(stockData, clf, startDate, endDate, dateIntervals):
    """    
    Function iteratively generates coefficients for a linear model using small chunks of time for the selected stock data.

    stockData - stock data used for linear fits (dependent and independent variables)
    clf - linear model for regressions
    startDate and endDate - datetime objects defining the range of dates to search for stock information   
    dateIntervals - The binned time intervals used for the linear regression
    """

    # Setup input data for linear models
    X = stockData.iloc[:,1:].copy();
    X = X.fillna(value=0) # Determine if there are any NaNs included in the dataset and set them to zero
    y = stockData.iloc[:,0:1].copy(); 

    # Loop through each binning interval and determine the performance of a linear model (fit to a training interval) on the test intervals
    trainTestCoeffs     = pd.DataFrame(0,index=dateIntervals[:-1],columns=X.columns,dtype='float64');
    for currentTrainingDateIndex in range(dateIntervals.size-1): # we skip the last interval
        # 1.) Get data for the current training interval, and train the model on this data
        trainingStartDate = dateIntervals[currentTrainingDateIndex+0];
        trainingEndDate   = dateIntervals[currentTrainingDateIndex+1];
        X_train = X[(X.index>=trainingStartDate) & (X.index<trainingEndDate)];
        y_train = y[(y.index>=trainingStartDate) & (y.index<trainingEndDate)];
        clf.fit(X_train,y_train);
        
        # 2.) Store the coefficients for the trained linear model
        trainTestCoeffs.loc[trainingStartDate,:]= clf.coef_;
    return trainTestCoeffs;
    
def computeSlope(data):
    """ compute slope using linear regression (change per year)"""
    slope = pd.Series(0,index=data.columns,dtype='float');
    for stock in data.columns:
        validData=data[stock].apply(np.isnan).values==False;
        if np.max(validData)==True:
            t = data[stock].index-data[stock].index[0];
            t = t.days / 365.0;
            p = np.polyfit(t[validData],data[stock][validData].values,1);
            slope[stock] = p[0];
        else:
            slope[stock] = np.nan;
    return slope;
    
def autocorr(x):
    # normalized 1d-autocorrelation function
    result = np.correlate(x, x, mode = 'full')
    result = result / np.max(result)
    return result[result.size/2:]
    
def localCorrelation(testTraces,windowSize=4):
    """ compute correlation between all time-series elements with the first time-series, but within a local window size (default is +/- 5 elements around each time point)"""
    correlationTraces = testTraces.copy();
    for t in range(testTraces.index.size):
        # Determine a valid window size
        desiredTimeWindow = np.linspace(t-windowSize,t+windowSize,2*windowSize+1);
        selectedWindowElements = np.multiply(desiredTimeWindow>=0,desiredTimeWindow<testTraces.index.size);
        actualTimeWindow = [val for (i, val) in enumerate(desiredTimeWindow) if selectedWindowElements[i]==True];
            
        # compute local correlation across all of the windowed time-series traces
        for stockIndex in range(testTraces.columns.size):
            r=np.corrcoef(testTraces.iloc[actualTimeWindow,stockIndex],testTraces.iloc[actualTimeWindow,0]);
            correlationTraces.iloc[t,stockIndex] = r[1,0];
    return correlationTraces;
        
def findLocalExtrema(testTraces,numberOfExtrema=5,minimumSpacing=5,computeMaxima=False):
    """ Find points where the correlation of stocks become most decorrelated with the market (i.e. compute minima). 
        If the computeMaxima flag is set to True, then it looks for points where correlation is high"""
    
    extrema = pd.DataFrame(-numberOfExtrema-1,index=range(numberOfExtrema),columns=testTraces.columns);
    for (stockIndex,stock) in enumerate(testTraces.columns):
        # Get stock time-series and current local minima
        stockTrace  = testTraces[stock];
        localExtrema = extrema[stock];
        
        # Sort stock time-series in ascending order, returning the sorted indices
        sortedIndex = np.argsort(stockTrace.values)
        if computeMaxima:
            sortedIndex = sortedIndex[::-1]; # reverse direction of this array to compute it in descending order
        
        # Iterate through the indices and finding extrema that are minimally seperated by a certain distance from other extrema.
        n = 0;
        for index in sortedIndex:
            distanceFromExtrema        = np.abs(index-localExtrema);
            minimalDistanceFromExtrema = np.min(distanceFromExtrema);
            
            if(minimalDistanceFromExtrema >= minimumSpacing):
                localExtrema[n] = index;
                n += 1;
            if n>(numberOfExtrema-1):
                break;
    return extrema;

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

def indices(a, func):
    """returns indices of data that are true for the function"""
    return [i for (i, val) in enumerate(a) if func(val)];
    
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
    
def preProcessStockData(stockData,zScoreData=True,eliminateStocksLessThanAYear=True):
    """Converts raw stock data to z-score or percent difference, and then eliminates any stocks with less than a year of data"""
    
    # Z-Score data (or use percent difference relative to starting day)
    if zScoreData:
        stockData=(stockData - stockData.mean())/stockData.std();
    else: 
        for stock in stockData.columns:
            firstNonNaNValue=find(stockData[stock].apply(np.isnan)==False,1);
            if len(firstNonNaNValue)==0:
                offset=0;
            else:
                offset=stockData[stock][firstNonNaNValue].values;
            stockData[stock]=(stockData[stock]-offset)/offset;
        stockData = stockData.iloc[:,:]-stockData.iloc[0,:].fillna(value=0);
    
    # Eliminate stocks where we have less than 1-year of data (52weeks)
    numberOfDaysPerIndex=(stockData.index[1]-stockData.index[0]).days;
    if numberOfDaysPerIndex>1: # Takes into account that there are only 5 business days on average
        numberOfDaysPerIndex = int(numberOfDaysPerIndex*5.0/7.0); 
    if eliminateStocksLessThanAYear:
        firstValue = np.zeros(stockData.shape[1])
        lastValue  = np.zeros(stockData.shape[1])
        for stockIndex in range(stockData.shape[1]):
            nonNaNs = find(np.isnan(stockData.iloc[:,stockIndex].values)==0);
            if len(nonNaNs)>0:
                firstValue[stockIndex]=nonNaNs[0];
                lastValue[stockIndex]=nonNaNs[-1];
        stockThreshold=firstValue<(stockData.shape[0]-252/numberOfDaysPerIndex); 
        stockData = stockData.iloc[:,stockThreshold];
    return stockData;
    
def getSequenceOfStocks(stockSet):
    """get sequence of stock based on label"""
    if(  stockSet=='NASDAQ'):
        sequenceOfStocks = [0,0];
    elif(stockSet=='FAANGStocks'):
        sequenceOfStocks = range(0,4)+range(5,8);
    elif(stockSet=='Top100Stocks'):
        sequenceOfStocks = range(0,4)+range(5,107);
    elif(stockSet=='AllStocks'):
        sequenceOfStocks = range(0,4)+range(5,3014);
    elif(stockSet=='Top100StocksExcluding'):
        sequenceOfStocks = range(8,107);
        sequenceOfStocks.insert(0,0);
    elif(stockSet=='AllStocksExcluding'):
        sequenceOfStocks = range(107,3014);
        sequenceOfStocks.insert(0,0);
    else:
        sequenceOfStocks = range(107,255);
        sequenceOfStocks.insert(0,0);
    return sequenceOfStocks;
    
def getStockTypeInfo(stockType):
    """ returns information about what stocks are FANG, Top 100 NASDAQ, all NASDAQ, and NASDAQ Index"""
    LUT = 0.75*cm.brg(np.linspace(0,1,3)); #cm.rainbow
    LUT[:,-1] = 1;
    if stockType == 1: #FAANG stocks
        selectedStocks = range(1,8);
        setName   = 'FAANG stocks';
        lineType  = '--';
        lineColor = LUT[1,:]; #'r';
    elif stockType == 2: # Other Top 100 NASDAQ stocks
        selectedStocks = range(8,107);
        setName   = 'Other Top 100 NASDAQ stocks'
        lineType  = '--';
        lineColor = LUT[0,:]; #'b';
    elif stockType == 3: # All other NASDAQ stocks
        selectedStocks = range(107,2789); #2905 #3013
        setName   = 'Other NASDAQ stocks';
        lineType  = '--';
        lineColor = LUT[2,:]; #'g';
    else: # NASDAQ
        selectedStocks = [0];
        setName   = 'NASDAQ';
        lineType  = '-';
        lineColor = 'k';
    return selectedStocks,setName,lineType,lineColor;
    
def getShortName(stockType):
    """return short name for stock type. useful for bar plot labels"""
    if stockType == 1: #FAANG stocks
        shortName = 'FAANG';
    elif stockType == 2: # Other Top 100 NASDAQ stocks
        shortName = 'Top 100';
    elif stockType == 3: # All other NASDAQ stocks
        shortName = 'Other';
    else: # NASDAQ
        shortName = 'NASDAQ';
    return shortName;