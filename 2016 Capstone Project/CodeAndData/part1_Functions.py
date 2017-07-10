from capstoneProjectFunctions import *
import sklearn.cluster as cluster

def plottingFunctionCorrelation(plottingData):
    """Plot basic relationship of FAANG stocks to NASDAQ"""
        
    # FIGURE 1: Show basic correlations
    numberOfStocks = plottingData.shape[1];
    lineWidths=2*np.ones([numberOfStocks,1]);
    lineWidths[0]=10;
    LUT=cat(0,np.zeros([1,4]),0.75*cm.rainbow(np.linspace(0,1,numberOfStocks-1)));
    LUT[:,-1]=1
    #LUT=LUT[:-1,:]

    # Scatter plot of historical data for FAANG stocks and NASDAQ
    h=figure(figsize=(10,8)); 
    axis=subplot(2,1,1);
    for i in range(numberOfStocks):
        plottingData.iloc[:,i].plot(ax=axis,color=LUT[i,:],linewidth=lineWidths[i])
    for xLabels in axis.xaxis.get_majorticklabels():
        xLabels.set_rotation(45);
        xLabels.set_horizontalalignment('center')
    axis.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axis.get_legend().set_visible(False);
    axis.set_ylabel('Normalized value');
    
    # Show slopes of FAANG stocks and NASDAQ
    axis=subplot(2,2,3);
    xLabels = plottingData.columns.values;
    x = np.linspace(0,len(xLabels)-1,len(xLabels))-0.4;
    slope = computeSlope(plottingData);
    for i in range(0,len(x)):
      bar(x[i]-0.4,slope.iloc[i],color=LUT[i,:])
    axis.set_xticks(x);
    axis.set_xticklabels(xLabels,rotation=45,horizontalalignment='center');
    axis.set_ylabel('Slope (change per year)');
    axis.set_xlim([x.min()-0.8,x.max()+0.8]);
    
    # Show correlations of FAANG stocks with NASDAQ
    axis=subplot(2,2,4);
    r = plottingData.corr();
    for i in range(1,len(x)):
      bar(x[i]-0.4,r.iloc[0,i],color=LUT[i,:])
    axis.set_xticks(x);
    axis.set_xticklabels(xLabels,rotation=45,horizontalalignment='center');
    axis.set_ylabel('Correlation with NASDAQ');
    axis.set_xlim([x[1:].min()-0.8,x[1:].max()+0.8]);
    h.set_tight_layout(True);
   
def computeCorrelationsWithNASDAQ(stockData):   
    """ compute pairwise correlations between each stock and the NASDAQ """    
    
    # First loop through each stock and compute correlation with NASDAQ
    referenceStock     = stockData.columns[0];
    referenceStockData = stockData[referenceStock];
    correlationValues  = pd.Series(0,index=stockData.columns,dtype='float64')
    for stock in stockData.columns:
        selectedStockData  = stockData[stock];
        selectedIndex = selectedStockData.apply(math.isnan);
        r = np.corrcoef(referenceStockData[selectedIndex==False],selectedStockData[selectedIndex==False]);
        correlationValues[stock] = r[0,1];

    # Show summary plot
    fig = showCumulativeProbabilityAndBarPlot(correlationValues,xLabel='Correlation with NASDAQ',xLimits=[-1,1])
    fig.set_tight_layout(True);

    return correlationValues;

def compareMarketCapDataToNASDAQCorrelation(stockData,correlationValues):
    """ computes the relationship of a stock's market cap value with the stock's correlation to the NASDAQ"""

    # Get market cap data
    loadMarketCapData = False;
    if(loadMarketCapData):
        # log onto yahoo.com to get market cap data
        marketCapData = getMarketCapData(stockData.columns)
        marketCapData.to_csv(os.getcwd()+'\\MarketCapData_NASDAQ.csv');
    else:
        marketCapData = pd.Series.from_csv(os.getcwd()+'\\MarketCapData_NASDAQ.csv');
        marketCapData = marketCapData[stockData.columns]

    # Construct a comparison of market cap value (normalized or unnormalized) vs correlation with NASDAQ
    marketData = pd.DataFrame(index=correlationValues.index);  
    marketData['CorrelationWithNASDAQ']=correlationValues;
    marketData['MarketCap']=marketCapData;
    marketData['MarketCap_Log10']=marketCapData.apply(np.log10);
    marketData['MarketCapNormalized']=marketCapData; #Computes market cap by rank
    hasNaNs = np.sum(np.isnan(marketCapData));
    sortedIndexValues=np.argsort(marketCapData.fillna(value=-1).values);
    for index in range(sortedIndexValues.size):
        if(index<hasNaNs):
            marketData.iloc[sortedIndexValues[index],3]=np.nan;
        else:
            marketData.iloc[sortedIndexValues[index],3]=(index-hasNaNs)/(0.0+sortedIndexValues.size-hasNaNs);
    marketData=marketData.dropna();
    
    # Display correlations between groups
    print marketData.corr()

    # Plot relationships
    fig = showScatterPlot_TwoProperties(marketData['MarketCap_Log10'],marketData['CorrelationWithNASDAQ'], \
                                        xTitle='Market Cap Value (Log-10)',yTitle='Correlation with NASDAQ',title='', \
                                        xLimits=[6,12],yLimits=[-1,1])
    fig.set_tight_layout(True);

def computeSlopeWithNASDAQCorrelation(stockData,correlationValues):
    """ computes the relationship of a stock's growth consistency (i.e. slope) with the stock's correlation to the NASDAQ"""
    
    # Compute slope of the stock data
    slopeValues=computeSlope(stockData);
    
    # Show summary plot and scatter plot
    showCumulativeProbabilityAndBarPlot(slopeValues,xLabel='Slope (change per year)',xLimits=[-1,1])   
    fig = showScatterPlot_TwoProperties(slopeValues.values,correlationValues.values, \
                                        xTitle='Slope (change per year)',yTitle='Correlation with NASDAQ',title='', \
                                        xLimits=[-0.75,0.75],yLimits=[-1,1])
    fig.set_tight_layout(True);  
    return slopeValues
                          
def computeSectorRelationship(stockData,correlationValues):
    """ compares how individual sectors compare to the NASDAQ """

    # Determine sector breakdown
    miscInfo = pd.DataFrame.from_csv('C:\Users\David\Desktop\Springboard Course\RawData\NASDAQList.csv');
    sectorBreakdown = pd.Series('Other',index=stockData.columns,dtype='object')
    for stock in sectorBreakdown.index:
        if stock in miscInfo.index:
            currentSector = miscInfo['Sector'][stock];
            if currentSector!='n/a':
                sectorBreakdown[stock]=currentSector;

    # Determine unique sectors
    useAutomaticDetection = False;
    if useAutomaticDetection:
        sectorListType = [];
        for stock in sectorBreakdown.index:
            if sectorBreakdown[stock] not in sectorListType:
                sectorListType.append(sectorBreakdown[stock])
    else:
        sectorListType = ['Basic Industries', 'Capital Goods', 'Consumer Durables',
                          'Consumer Non-Durables', 'Consumer Services', 'Energy', 'Finance',
                          'Health Care', 'Miscellaneous', 'Public Utilities', 'Technology',
                          'Transportation', 'Other'];
    
    # Determine sector makeup of each stock group using pie charts
    LUT = cat(0,1.0*cm.rainbow(np.linspace(0,1,np.ceil(len(sectorListType)/2.0))), \
                0.5*cm.rainbow(np.linspace(0,1,np.ceil(len(sectorListType)/2.0)))); 
    LUT[:,-1] = 1;
    LUT=LUT[:len(sectorListType),:]
    #fig  = matplotlib.pyplot.figure(figsize=(10,4));
    #axes = [matplotlib.pyplot.subplot2grid((1,5), (0,i), colspan=1) for i in range(3)];
    fig  = matplotlib.pyplot.figure(figsize=(10,4));
    axes = [matplotlib.pyplot.subplot2grid((2,3), (0,i), colspan=1) for i in range(3)];
    #fig, axes = plt.pyplot.subplots(nrows=1,ncols=4,figsize=(10,4))
    for stockType in range(1,4):
        axis=axes[stockType-1];
        selectedStocks,setName,lineType,lineColor = getStockTypeInfo(stockType);
        shortName=getShortName(stockType);
        
        numberOfStocks = 0.0+len(selectedStocks);
        sectorMakeup   = pd.Series(0.0,index=sectorListType[:])
        for currentSector in sectorListType[:]:
            sectorMakeup[currentSector]=len(find(sectorBreakdown[selectedStocks]==currentSector))/numberOfStocks
    
        currentSectorList=sectorListType[:];
        for index in find(sectorMakeup==0.0):
            currentSectorList[index]='';
        patches, texts = axis.pie(sectorMakeup,colors=LUT,startangle=90)
        axis.set_title(shortName);
        axis.set_aspect('equal')
    #axis.legend(patches, currentSectorList, ncol=2, loc='center left', bbox_to_anchor=(1.0, 0.5),fontsize='small')
    axis.legend(patches, currentSectorList, ncol=3, loc='center right', bbox_to_anchor=(1.0, -0.5))    
    fig.set_tight_layout(True);
    
    # scatter plot of the stock data
    sectorListType.append('NASDAQ')  
    fig  = matplotlib.pyplot.figure(figsize=(10,8));
    axis = matplotlib.pyplot.subplot2grid((2,2), (0,0), colspan=2);
    yLabel='Normalized value';
    xLabel='Time elapsed (years)';
    xLimits=[stockData.index[0],stockData.index[-1]]
    yLimits=[-3,3];
    for index,currentSector in enumerate(sectorListType):
        if currentSector == 'NASDAQ':
            selectedStocks = [0];
            currentColor = 'k';
        else:
            selectedStocks = find(sectorBreakdown==currentSector);
            currentColor   = LUT[index,:];
        
        x = stockData.index;
        n = len(selectedStocks) - np.sum(np.isnan(stockData.values[:,selectedStocks]),axis=1);
        t = stats.t.ppf(1-0.05/2.0, n-1);
        y = stockData.values[:,selectedStocks];
        yMean = np.nanmean(y,axis=1);
        yStd  = np.nanstd(y,axis=1)
        error = t*yStd/np.sqrt(n)
                
        axis.plot(x,yMean,label=currentSector,color=currentColor,lineWidth=2)
        axis.fill_between(x, yMean-error, yMean+error, alpha=0.1, edgecolor=currentColor, facecolor=currentColor)
    legend = axis.legend(loc='center left',ncol=5, bbox_to_anchor=(1.0, 0.5));
    legend.remove()
    axis.set_xlim(xLimits[0],xLimits[1]);
    axis.set_ylim(yLimits[0],yLimits[1]);
    axis.set_ylabel(yLabel)

    # Cumulative probability plot
    axis = matplotlib.pyplot.subplot2grid((2,2), (1,0), colspan=1);
    for index,currentSector in enumerate(sectorListType[:-1]):
        selectedStocks = find(sectorBreakdown==currentSector);
        currentColor   = LUT[index,:];
        x = np.sort(correlationValues.values[selectedStocks],axis=0)
        y = np.linspace(0,1,len(selectedStocks));
        axis.plot(x,y,lineType,label=currentSector,color=currentColor,lineWidth=3)
    axis.legend(loc='best',framealpha=0,borderpad=None)
    axis.get_legend().set_visible(False);
    axis.set_xlim(-1.0,1.0);
    axis.set_ylim(0.0,1.0);
    axis.set_xlabel('Correlation with NASDAQ')
    axis.set_ylabel('Cumulative probability')

    # Bar plot
    axis = matplotlib.pyplot.subplot2grid((2,2), (1,1), colspan=1);
    barList=[];
    for index,currentSector in enumerate(sectorListType[:-1]):
        selectedStocks = find(sectorBreakdown==currentSector);
        currentColor   = LUT[index,:];
        x = np.sort(correlationValues.values[selectedStocks],axis=0)
        x = x[np.isnan(x)==False];
        y = np.linspace(0,1,len(selectedStocks));
        axis.bar(index-0.4,np.mean(x),label=currentSector,color=currentColor)
        axis.errorbar(index, np.mean(x), yerr=np.std(x)/np.sqrt(len(y)),color='k');
        barList.append(currentSector);
    axis.set_xticks(np.linspace(0,index,index+1))  
    axis.set_xticklabels(barList,rotation=90,horizontalalignment='center')
    axis.set_xlim(-0.5,12.5);
    axis.set_ylim(-0.5,1);
    axis.set_ylabel('Correlation with NASDAQ');
    fig.set_tight_layout(True);
    
def computeCorrelationQuintileRelationship(stockData,correlationValues,slopeValues):
    """ compares how correlation quintiles compare to the NASDAQ """

    # Break stock data into quintiles based on correlation with the NASDAQ
    quintileListType        = ['0-20%', '21-40%', '41-60%','61-80%','81-100%'];
    correlationQuintiles    = pd.Series(-1,index=stockData.columns);
    correlationValuesSorted = correlationValues.copy();
    correlationValuesSorted = correlationValuesSorted.fillna(1).sort_values();
    correlationThresholds   = correlationValuesSorted[np.round(np.linspace(0,correlationValues.size-1,6)).astype(int)].values;
    correlationThresholds[-1] = correlationValuesSorted['^IXIC']
    for i in range(5):
        correlationQuintiles[(correlationValuesSorted>=correlationThresholds[i])&(correlationValuesSorted<correlationThresholds[i+1])]=i;
    
    # Determine line properties for each quintiles
    lineType = '--';
    LUT = 0.75*cm.rainbow(np.linspace(0,1,5))
    LUT[:,-1] = 1;
    
    # scatter plot of the stock data
    quintileListType.append('NASDAQ')  
    fig  = matplotlib.pyplot.figure(figsize=(10,8));
    axis = matplotlib.pyplot.subplot2grid((2,2), (0,0), colspan=2);
    yLabel='Normalized value';
    xLabel='Time elapsed (years)';
    xLimits=[stockData.index[0],stockData.index[-1]]
    yLimits=[-3,4];
    for index,currentSector in enumerate(quintileListType):
        if currentSector == 'NASDAQ':
            selectedStocks = [0];
            currentColor = 'k';
        else:
            selectedStocks = find(correlationQuintiles==index);
            currentColor   = LUT[index,:];
        
        x = stockData.index;
        n = len(selectedStocks) - np.sum(np.isnan(stockData.values[:,selectedStocks]),axis=1);
        t = stats.t.ppf(1-0.05/2.0, n-1);
        y = stockData.values[:,selectedStocks];
        yMean = np.nanmean(y,axis=1);
        yStd  = np.nanstd(y,axis=1)
        error = t*yStd/np.sqrt(n)
                
        axis.plot(x,yMean,label=currentSector,color=currentColor,lineWidth=2)
        axis.fill_between(x, yMean-error, yMean+error, alpha=0.1, edgecolor=currentColor, facecolor=currentColor)
    legend = axis.legend(loc='center left',ncol=3, bbox_to_anchor=(0.0, 0.85));
    #legend.remove()
    axis.set_xlim(xLimits[0],xLimits[1]);
    axis.set_ylim(yLimits[0],yLimits[1]);
    axis.set_ylabel(yLabel)

    # Cumulative probability plot of slope
    axis = matplotlib.pyplot.subplot2grid((2,2), (1,0), colspan=1);
    for index,currentQuintile in enumerate(quintileListType[:-1]):
        selectedStocks = find(correlationQuintiles==index);
        currentColor   = LUT[index,:];
        x = np.sort(slopeValues.values[selectedStocks],axis=0)
        y = np.linspace(0,1,len(selectedStocks));
        axis.plot(x,y,lineType,label=currentQuintile,color=currentColor,lineWidth=3)
    axis.legend(loc='best',framealpha=0,borderpad=None)
    axis.get_legend().set_visible(False);
    axis.set_xlim(-1.0,1.0);
    axis.set_ylim(0.0,1.0);
    axis.set_xlabel('Slope (change per year)')
    axis.set_ylabel('Cumulative probability')

    # Bar plot
    axis = matplotlib.pyplot.subplot2grid((2,2), (1,1), colspan=1);
    barList=[];
    for index,currentQuintile in enumerate(quintileListType[:-1]):
        selectedStocks = find(correlationQuintiles==index);
        currentColor   = LUT[index,:];
        x = np.sort(slopeValues.values[selectedStocks],axis=0)
        x = x[np.isnan(x)==False];
        y = np.linspace(0,1,len(selectedStocks));
        axis.bar(index-0.4,np.mean(x),label=currentQuintile,color=currentColor)
        axis.errorbar(index, np.mean(x), yerr=np.std(x)/np.sqrt(len(y)),color='k');
        barList.append(currentQuintile);
    axis.set_xticks(np.linspace(0,index,index+1))  
    axis.set_xticklabels(barList,rotation=90,horizontalalignment='center')
    axis.set_xlim(-0.5,4.5);
    axis.set_ylim(-1,1);
    axis.set_ylabel('Slope (change per year)');
    fig.set_tight_layout(True);
    
def getSortedIndices(originalList,sortedRanking):
    """ returns the list of each stock's membership to a cluster (but now using the sorted ranking)"""
    sortedList = np.copy(originalList);
    numberOfClusters = np.max(sortedRanking)+1;
    for currentCluster in range(numberOfClusters):
        sortedList[find(originalList==sortedRanking[currentCluster])]=currentCluster;
    return sortedList;

def computeKMeansClusters(stockData):
    """Segregate the stock groups based on K-Means clustering. Note: Principal components for each can be used instead of the actual stock trace"""
    
    # First perform PCA via SVD on the stock data
    M = stockData.iloc[:,:].fillna(value=0).values;
    U, s, Vt = np.linalg.svd(M,full_matrices=False)
    V = Vt.T;
    
    # Compute k-means clusters on the stock data, but we can use weights from principal components instead of the actual stock trace
    clusterBasedOnPCA = False;
    numberOfClusters = 5; 
    randomSeed=2;
    kmeans = cluster.KMeans(n_clusters=numberOfClusters,random_state=randomSeed,n_init=100);
    LUT = .5*cm.rainbow(np.linspace(1,0,numberOfClusters))
    LUT[:,-1] = 1;
    if(clusterBasedOnPCA):    
        # Determine the number of PCs explaining 70% of the dataset's variance
        selectedPrincipleComponents=find(np.cumsum(s/np.sum(s))<=0.7)
        print 'The first {} PCs explain 70% variance'.format(len(selectedPrincipleComponents));
        
        clusteringIndex=kmeans.fit_predict(V[:,selectedPrincipleComponents]);
    else:
        clusteringIndex=kmeans.fit_predict(M.T);
    
    # Now we'll sort the clusters by number of stocks explained (but first will always contain NASDAQ)
    NASDAQCluster = clusteringIndex[0];
    stocksPerCluster = np.zeros([numberOfClusters,1]);
    for currentCluster in range(numberOfClusters):
        stocksPerCluster[currentCluster] = len(find(clusteringIndex==currentCluster));
    stocksPerCluster = clusteringIndex.size+1-stocksPerCluster; # ensures that largest elements are sorted first
    stocksPerCluster[NASDAQCluster]=-1; # ensures that the first cluster is nasdaq
    sortedStockIndices = stocksPerCluster.argsort(axis=0);
    clusteringIndex = getSortedIndices(clusteringIndex,sortedStockIndices);    
    NASDAQCluster = clusteringIndex[0]; # updates cluster with NASDAQ  
        
    # Show clustering analysis
    h=matplotlib.pyplot.figure(figsize=(10,8));
    axis=matplotlib.pyplot.subplot(2,2,3);
    plot(np.sort(clusteringIndex[:]),np.linspace(0,1,len(clusteringIndex)));
    axis.set_xlabel('K-Means Cluster #');
    axis.set_ylabel('Cumulative probability')
    axis.set_title('NASDAQ cluster is {}'.format(NASDAQCluster))
    axis.set_xlim([-0.1,numberOfClusters-1+0.1])
    axis=matplotlib.pyplot.subplot(2,2,4)
    xTickLabels=[];
    for stockType in range(1,4):
        selectedStocks,setName,lineType,lineColor = getStockTypeInfo(stockType);
        shortName=getShortName(stockType);
        proportionOfStocksInNASDAQCluster = (np.sum(clusteringIndex[selectedStocks]==NASDAQCluster)+0.0)/len(selectedStocks);
        bar(stockType-0.4,proportionOfStocksInNASDAQCluster,color=lineColor,label=shortName)
        xTickLabels.append(shortName)
    axis.set_xlim([0.5,3.5]);
    axis.set_ylim([0.0,1.0]);
    axis.set_xticks(range(1,4))
    axis.set_xticklabels(xTickLabels);
    axis.set_ylabel('Percent in NASDAQ Cluster');
    axis.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axis.get_legend().set_visible(False);
    h.set_tight_layout(True);
    
    # Show the stock traces for the clusters  
    axis = matplotlib.pyplot.subplot(2,1,1);
    numberOfComponentsToDisplay = numberOfClusters;
    for currentComponent in range(numberOfComponentsToDisplay):
        currentColor   = LUT[currentComponent,:];
        selectedStocks = find(clusteringIndex==currentComponent)
        x = stockData.index;
        n = len(selectedStocks) - np.sum(np.isnan(stockData.values[:,selectedStocks]),axis=1);
        t = stats.t.ppf(1-0.05/2.0, n-1);
        y = stockData.values[:,selectedStocks];
        yMean = np.nanmean(y,axis=1);
        yStd  = np.nanstd(y,axis=1)
        error = t*yStd/np.sqrt(n)    
    
        plot(x,yMean,label='Cluster #'+str(currentComponent),color=currentColor,lineWidth=2)  
        fill_between(x, yMean-error, yMean+error, alpha=0.1, edgecolor=currentColor, facecolor=currentColor)    
    axis.set_ylim([-5,5])   
    legend = axis.legend(loc='center left',ncol=3, bbox_to_anchor=(0.0, 0.85));
    #legend.remove()
    axis.set_ylabel('Normalized value');