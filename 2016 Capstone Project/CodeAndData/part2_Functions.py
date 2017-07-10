from capstoneProjectFunctions import *
import scipy
import cPickle
np.set_printoptions(suppress=True)

def showTimeVaryingCorrelations(binnedStockData,localCorrelationData,correlationExtrema,selectedStock='AAPL'):
    """Show time-varying correlations of an example stock against the NASDAQ (with correlations of NASDAQ to Top 100 NASDAQ stocks as reference)"""
    
    # Demonstrate that correlations are time-varying.
    selectedStockIndex  = binnedStockData.columns.get_loc(selectedStock);
    FAANG_Trace         = pd.DataFrame(binnedStockData.iloc[:,1:8  ].median(axis=1),index=binnedStockData.index     ,columns=['FAANG Average'  ]);
    Top100_Trace        = pd.DataFrame(binnedStockData.iloc[:,1:107].median(axis=1),index=binnedStockData.index     ,columns=['Top 100 Average']);
    FAANG_Correlations  = pd.DataFrame(localCorrelationData.iloc[:,1:8  ].median(axis=1),index=binnedStockData.index,columns=['FAANG Average'  ]);
    Top100_Correlations = pd.DataFrame(localCorrelationData.iloc[:,1:107].median(axis=1),index=binnedStockData.index,columns=['Top 100 Average']);

    # Show historical data
    fig, (ax1,ax2) = matplotlib.pyplot.subplots(nrows=2,ncols=1,figsize=(10,8))
    binnedStockData.iloc[:,0].plot(ax=ax1,style='k',lw=10);
    binnedStockData.iloc[:,selectedStockIndex].plot(ax=ax1,style='m',lw=2);
    Top100_Trace.plot(ax=ax1,style='b',lw=2)
    ax1.legend(loc='upper left',ncol=3)
    ax1.set_xticklabels([])
    ax1.set_ylabel('Normalized value');
    #ax1.set_title('Historical time-series of Index/Stock')
    ax1.set_ylim(bottom=-1,top=4)
    
    # Show time-varying correlations
    #FAANG_Correlations.plot(ax=ax2,style='r')
    localCorrelationData.iloc[:,selectedStockIndex].plot(ax=ax2,style='m',lw=2);
    Top100_Correlations.plot(ax=ax2,style='b',lw=2)
    ax2.set_ylabel('Correlation');
    #ax2.set_title('Time-varying correlation between Index/Stock')
    ax2.set_ylim(bottom=-1,top=1)
    ax2.get_legend().set_visible(False);
    for timeStamp in correlationExtrema[selectedStock]:
        datetimeTimeStamp = datetime.datetime.strptime(timeStamp,'%Y-%m-%d');
        selectedTimeRange = (binnedStockData.index>=(datetimeTimeStamp-datetime.timedelta(7*2))) & (binnedStockData.index<(datetimeTimeStamp+datetime.timedelta(7*2)))
        ax1.fill_between(binnedStockData.index, -5, 5, where=selectedTimeRange, facecolor='black', alpha=0.25, interpolate=True)
        ax2.fill_between(binnedStockData.index, -5, 5, where=selectedTimeRange, facecolor='black', alpha=0.25, interpolate=True)
    print('Correlation Extrema: \n{}'.format(correlationExtrema[selectedStock]))
    
def computeSummaryStatsForTimeVaryingCorrelation(localCorrelationData):
    """Compute summary plot for short time-varying correlations (like 4-weeks)"""
    fig=figure(figsize=(10,8));        
    for i in [1,2]:
        if(i==1): # Mean correlation
            dataArray   = localCorrelationData.mean(axis=0);
            xLabel      = 'Mean correlation';
            yLabelSplit = 'Mean correlation'; #'Mean \n correlation';
        else: # Coefficient of variation for the correlations      
            dataArray   = localCorrelationData.std(axis=0);# / np.abs(localCorrelationData.mean(axis=0));
            xLabel      = 'Variation of correlations';
            yLabelSplit = 'Variation of correlations'; #'Variation of \n correlations'
    
        # Cumulative probability plot
        axes=subplot(2,2,1+2*(i-1));
        for stockType in range(1,4):
            selectedStocks,setName,lineType,lineColor = getStockTypeInfo(stockType);
            x = np.sort(dataArray.values[selectedStocks],axis=0);
            x = x[np.isnan(x)==False];
            y = np.linspace(0,1,len(x));
            plot(x,y,lineType,label=setName,color=lineColor,lineWidth=3)
            #axes.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        axes.legend(loc='best',framealpha=0,borderpad=None)
        axes.get_legend().set_visible(False);
        axes.set_ylim(0,1);
        axes.set_yticks(np.linspace(0,1,5))
        if i==1:
            axes.set_xlim(-0.25,1); #0.75
        else:
            #axes.set_xlim(0,1);
            axes.set_xlim(0.4,0.8);
        axes.set_xlabel(xLabel)
        for xLabels in axes.xaxis.get_majorticklabels():
            xLabels.set_rotation(45);
            xLabels.set_horizontalalignment('center')
        axes.set_ylabel('Cumulative probability') #'Cumulative\nprobability'
        #axes.set_title(title)
        
        # Bar plot
        axes2=subplot(2,2,2+2*(i-1));
        barList=[];
        for stockType in range(1,4):
            selectedStocks,setName,lineType,lineColor = getStockTypeInfo(stockType);
            shortName = getShortName(stockType)
            x = np.sort(dataArray.values[selectedStocks],axis=0)
            x = x[np.isnan(x)==False];
            y = np.linspace(0,1,len(x));
            bar(stockType-0.4,np.mean(x),label=setName,color=lineColor)
            errorbar(stockType, np.mean(x), yerr=np.std(x)/np.sqrt(len(y)),color='k');
            barList.append(shortName);
            
            print "{} - {} +/- {} (SEM)".format(setName,np.mean(x),np.std(x)/np.sqrt(len(y)));
        axes2.set_xticks(range(1,4))  
        axes2.set_xticklabels(barList)
        axes2.set_xlim(0.5,3.5);
        if i==1:
            axes2.set_ylim(0,0.75); #0.75
            axes2.set_yticks(np.linspace(0,0.75,4))
        else:
            axes2.set_ylim(0,15);
            axes2.set_yticks(np.linspace(0,15,4))
            axes2.set_ylim(0,0.75);
            axes2.set_yticks(np.linspace(0,0.75,4))
        axes2.set_ylabel(yLabelSplit);   
        
        # Compute significance
        significanceTest=np.zeros([3,3]);
        for currentGroupA in range(1,4):
            output=getStockTypeInfo(currentGroupA); 
            selectedStocksA=output[0];
            for currentGroupB in range(1,4):
                output=getStockTypeInfo(currentGroupB); 
                selectedStocksB=output[0];
                statTest=scipy.stats.ranksums(dataArray[selectedStocksA],dataArray[selectedStocksB]);
                significanceTest[currentGroupA-1,currentGroupB-1]=statTest.pvalue;
        print significanceTest
    fig.set_tight_layout(True);

def predictMarketFromSelectedStocks(dataArray,xLabel='Number of Stocks (included in model)',yLabel='Correlation between model and NASDAQ',title='Model Summary:',yLimits=[0,1]):
    """ 5-Year correlations and time-varying correlations (4-Weeks) of bootstrapped running averages """
    numberOfStocksInSet = [1,2,5,10,20,50,100,250,500,3000];
    numberOfIterations  = 100;
    
    fig, axes = matplotlib.pyplot.subplots(nrows=1,ncols=2,figsize=(10,4))
    t0=datetime.datetime.now();
    correlationData = [];
    for ii,use4WeekAverage in enumerate([False,True]):
        for stockType in range(1,4):
            selectedStocks,setName,lineType,lineColor = getStockTypeInfo(stockType);
            
            x = range(len(numberOfStocksInSet));
            correlationValues = pd.DataFrame(0.0,index=range(numberOfIterations),columns=numberOfStocksInSet);
            for currentNumberOfStockSet in numberOfStocksInSet:
                for currentBootstrapIteration in range(numberOfIterations):
                    random.seed(currentBootstrapIteration);
                    selectedBootstrapSet = map(lambda x: np.round(random.uniform(selectedStocks[0],selectedStocks[-1])), range(currentNumberOfStockSet))
                    
                    if use4WeekAverage:
                        rTot = 0;
                        intervalSize=(5*4);
                        weekIntervals=range(0,dataArray.shape[0],intervalSize);
                        if (weekIntervals[-1]+intervalSize)>=dataArray.shape[0]:
                            weekIntervals = weekIntervals[:-1];
                        for weekIndex in weekIntervals:
                            #print weekIndex
                            modelTrace  = dataArray.iloc[range(0+weekIndex,intervalSize+weekIndex),selectedBootstrapSet].mean(axis=1);
                            marketTrace = dataArray.iloc[range(0+weekIndex,intervalSize+weekIndex),0];
                            r = np.corrcoef(modelTrace.fillna(0.0),marketTrace.fillna(0.0));
                            rTot = rTot+r[0,1]/len(weekIntervals);
                    else:
                        modelTrace  = dataArray.iloc[:,selectedBootstrapSet].mean(axis=1);
                        marketTrace = dataArray.iloc[:,0];
                        r = np.corrcoef(modelTrace.fillna(0.0),marketTrace.fillna(0.0));
                        rTot = r[0,1];
                    correlationValues[currentNumberOfStockSet][currentBootstrapIteration] = rTot;
                t1=datetime.datetime.now();
                timeElapsed=t1-t0;
                print "Stock Type {} - Number of Stocks included #{}: Time elapsed is {}s".format(stockType,currentNumberOfStockSet,timeElapsed.total_seconds())
            correlationData.append(correlationValues);
            
            n = numberOfIterations;
            t = stats.t.ppf(1-0.05/2.0, n-1);
            y = correlationValues.values;
            yMean = np.nanmean(y,axis=0);
            yStd  = np.nanstd(y,axis=0)
            error = t*yStd/np.sqrt(n)
                    
            axes[ii].plot(x,yMean,lineType,label=setName,color=lineColor)
            axes[ii].fill_between(x, yMean-error, yMean+error, alpha=0.25, edgecolor=lineColor, facecolor=lineColor)
        axes[ii].legend(loc='lower right')
        axes[ii].set_xticks(x,numberOfStocksInSet)
        axes[ii].set_xticklabels(numberOfStocksInSet)
        axes[ii].set_xlim(0,len(numberOfStocksInSet));
        axes[ii].set_ylim(yLimits[0],yLimits[1]);
        axes[ii].set_xlabel(xLabel)
        axes[ii].set_ylabel(yLabel)
        axes[ii].set_title(title)
    fig.set_tight_layout(True);
    cPickle.dump(correlationData, open(os.getcwd()+'simpleLinearModel_4WeekRunningAverage.p', 'wb')) 
    return correlationData;
    
def showAutoCorrelations(stockData):
    """ Compute autocorrelations of stock value and day-to-day differences"""
    
    # Autocorrelation of stock value shows correlations over time
    fig, axes = matplotlib.pyplot.subplots(nrows=1,ncols=2,figsize=(10,4))
    autoCorrelationData=stockData.copy();
    autoCorrelationData=autoCorrelationData.apply(signal.medfilt,kernel_size=7);
    autoCorrelationData=autoCorrelationData.fillna(value=0).apply(autocorr)
    autoCorrelationData.index = ((autoCorrelationData.index[:]-autoCorrelationData.index[0]).days)/7.0;
    plottingData = autoCorrelationData;
    #showScatterPlot(plottingData,yLabel='Autocorrelation',xLabel='Time elapsed (weeks)',xLimits=[plottingData.index[0],plottingData.index[-1]],yLimits=[-0.5,1],axes=axes[0])
    showScatterPlot(plottingData,yLabel='Autocorrelation',xLabel='Time elapsed (weeks)',xLimits=[plottingData.index[0],plottingData.index[-1]],yLimits=[-0.5,1],axes=axes[0])
    leg=axes[0].legend();
    leg.remove();
    
    # Autocorrelation of the time derivative of stock values - Unlike raw stock value, we find little predictive behavior over time
    autoCorrelationData = stockData.copy().diff().fillna(value=0).apply(autocorr);
    autoCorrelationData.index = ((autoCorrelationData.index[:]-autoCorrelationData.index[0]).days)/7.0;
    plottingData = autoCorrelationData;
    #showScatterPlot(plottingData,yLabel='Autocorrelation',xLabel='Time elapsed (weeks)',xLimits=[plottingData.index[0],plottingData.index[-1]],yLimits=[-0.25,1],axes=axes[1])
    showScatterPlot(plottingData,yLabel='Autocorrelation',xLabel='Time elapsed (weeks)',xLimits=[plottingData.index[0],3],yLimits=[-0.25,1],axes=axes[1])
    leg=axes[1].legend(ncol=1, loc='upper right',fontsize = 'medium'); # bbox_to_anchor=(1.0, -0.5));
    fig.set_tight_layout(True);
    
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
        X_train = X[(X.index>trainingStartDate) & (X.index<trainingEndDate)];
        y_train = y[(y.index>trainingStartDate) & (y.index<trainingEndDate)];
        clf.fit(X_train,y_train);
        
        # 2.) Store the coefficients for the trained linear model
        trainTestCoeffs.loc[trainingStartDate,:]= clf.coef_;
    return trainTestCoeffs;
    
def runKalmanFilter_ForOptimization(binnedStockData, binnedCorrelations, stockSet='NASDAQ'):
    """ Tries different values of Q- and R- values for the Kalman Filter. 
    
    PARAMETERS:
    binnedStockData    - Binned stock data
    binnedCorrelations - Correlations of each stock with the NASDAQ (within the binned time frame)
    stockSet           - Stock set to analyze: NASDAQ, FAANGStocks, Top100Stocks, AllStocks, Top100StocksExcluding (Excludes FAANG), and AllStocksExcluding (Excludes Top 100)
    """
    
    # Load data
    sequenceOfStocks = getSequenceOfStocks(stockSet);
    binnedStockData = binnedStockData.iloc[:,sequenceOfStocks].copy();  
            
    # Loop through different Q- and R- values for the Kalman filter
    t0 = time.time(); # Start timer
    ratioValues    = [10**(n) for n in range(-3,6,1)]; #-3,9,1
    errorInModel   = np.zeros([len(ratioValues),len(ratioValues)]);                                  
    rModelToData   = np.zeros([len(ratioValues),len(ratioValues)]);
    modelEstimates = [];
    for i,ratioValueQ in enumerate(ratioValues):
        for j,ratioValueR in enumerate(ratioValues):
            print "Q=10^{:1.0f} / R=10^{:1.0f} - Time elapsed is {:1.1f}s".format(np.log10(ratioValueQ),np.log10(ratioValueR),time.time()-t0)        
            
            inputData=binnedStockData.copy();
            inputData=inputData.fillna(value=0);
            m = inputData.shape[0];
            Q = ratioValueQ;     # Estimated error in process (or bias towards the internal model). 
            R = ratioValueR;     # Estimated error in measurements (or bias towards measurements). 
            if len(inputData.shape)==1: # Initial state estimate
                initial_state_estimate  = inputData.values[0]; 
            else:
                initial_state_estimate  = inputData.values[0,0];
            initial_prob_estimate   = 1  # Initial probability estimate
            kalman_gain     = pd.Series(0, index=inputData.index);
            prob_estimates  = pd.Series(0, index=inputData.index);
            state_estimates = pd.Series(0, index=inputData.index);
            for currentTimeStamp in inputData.index:    
                # Get measurement and control data
                rawMeasurementData = inputData[inputData.index == currentTimeStamp];
                rawMeasurementData = rawMeasurementData.fillna(value=0);
                rawMeasurementData = rawMeasurementData.iloc[:,1:];
               
                measurementType='unweighted';
                if   measurementType=='unweighted':
                    measurementData    = rawMeasurementData.mean(axis=1);#-initial_state_estimate;
                elif measurementType=='correlationWeighting':
                    stockWeighting     = binnedCorrelations[binnedCorrelations.index == currentTimeStamp];
                    stockWeighting     = (stockWeighting.iloc[:,1:]+1)/2;
                    netStockWeighting  = stockWeighting.sum(axis=1).values[0];
                    measurementData    = stockWeighting*rawMeasurementData/netStockWeighting;
                    measurementData    = measurementData.sum(axis=1);
                elif measurementType=='linearRegressionWeighting':
                    stockWeighting     = regressionCoeffs[binnedCorrelations.index == currentTimeStamp];
                    netStockWeighting  = stockWeighting.sum(axis=1).values[0];
                    measurementData    = stockWeighting*rawMeasurementData/netStockWeighting;
                    measurementData    = measurementData.sum(axis=1);
                
                # Prediction step of Kalman filter
                currentTimeStampIndex = state_estimates.index == currentTimeStamp;
                if currentTimeStamp==inputData.index[0]:
                    state_estimates[currentTimeStampIndex] = initial_state_estimate;
                    prob_estimates[currentTimeStampIndex]  = initial_prob_estimate + Q;
                else:
                    previousTimeStampIndex = state_estimates.index == previousTimeStamp;
                    state_estimates[currentTimeStampIndex] = state_estimates[previousTimeStampIndex].values;
                    prob_estimates[currentTimeStampIndex]  =  prob_estimates[previousTimeStampIndex].values + Q;
                
                # Update step of Kalman filter
                previousTimeStamp = currentTimeStamp;
                innovation        = measurementData - state_estimates[currentTimeStampIndex];
                kalman_gain[currentTimeStampIndex]     = prob_estimates[currentTimeStampIndex] / (prob_estimates[currentTimeStampIndex] + R);
                state_estimates[currentTimeStampIndex] = state_estimates[currentTimeStampIndex] + kalman_gain[currentTimeStampIndex]*innovation;
                prob_estimates[currentTimeStampIndex]  = (1-kalman_gain[currentTimeStampIndex])*prob_estimates[currentTimeStampIndex];
            # Compute MSE for state estimate of kalman filter and the actual data
            MSE = np.mean(np.sqrt((binnedStockData.iloc[:,0]-state_estimates)**2));
            errorInModel[i,j]=MSE;
            
            # Compute correlation between model and state estimates
            r = np.corrcoef(binnedStockData.iloc[:,0],state_estimates);
            rModelToData[i,j] = r[0,1];
            
            # Append to list
            modelEstimates.append([i,j,ratioValueQ,ratioValueR,state_estimates.copy(),prob_estimates.copy(),MSE,r])
    
    # Show examples of Kalman filter
    f, (ax1,ax2)  = plt.pyplot.subplots(nrows=2,ncols=1,figsize=(10,8))
    #f = plt.pyplot.Figure();
    currentAxis = plt.pyplot.subplot(2,1,1); #ax1
    selectedSteps = range(0,len(ratioValues),1);
    RIndex = [i for i,ratioValue in enumerate(ratioValues) if ratioValue==1E3];
    R      = RIndex[0];
    scalarMap = cm.ScalarMappable(cmap='spectral');      # Setup LUT
    LUT = scalarMap.to_rgba(range(1+len(ratioValues)));
    LUT = LUT[:,:-1];
    for Q in selectedSteps:
        data = modelEstimates[R+Q*len(ratioValues)][4];
        data.plot(ax=currentAxis,color=LUT[Q+1,:],label='Model (Q = 1E{:1.0f})'.format(np.log10(ratioValues[Q])))
    binnedStockData.iloc[:,0].plot(ax=currentAxis, color=LUT[0,:],label='Actual data')
    for xLabels in currentAxis.xaxis.get_majorticklabels():
        xLabels.set_rotation(45);
        xLabels.set_horizontalalignment('center')
    currentAxis.set_ylabel('Normalized value')
    #currentAxis.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    currentAxis.legend(title='Legend',loc='center left', bbox_to_anchor=(0.1, -0.8)) #bbox_to_anchor=(-0.05, -0.82)
    currentAxis.set_title('Comparing performance of Kalman filter model (R = 1E{:1.0f})'.format(np.log10(ratioValues[R])))
    currentAxis.set_ylim([-1,3])
    
    # Show relationship of Q and R parameters
    currentAxis=plt.pyplot.subplot(2,2,4);#ax2;
    plotTicks  = range(0,len(ratioValues),2);
    plotLabels = ['1E{:1.0g}'.format(np.log10(val)) for i,val in enumerate(ratioValues) if i in plotTicks];
    im=currentAxis.imshow(errorInModel,cmap=cm.spectral,origin='lower',interpolation='bicubic',clim=(0.0, 1.0))
    currentAxis.set_xticks(plotTicks)
    currentAxis.set_yticks(plotTicks)
    currentAxis.set_xticklabels(plotLabels)
    currentAxis.set_yticklabels(plotLabels)
    currentAxis.set_ylabel('Process Noise Covariance (Q)')
    currentAxis.set_xlabel('Measurement Noise Covariance (R)')
    for xLabels in currentAxis.xaxis.get_majorticklabels():
        xLabels.set_rotation(45);
        xLabels.set_horizontalalignment('center')
    currentAxis.set_title('MSE of Kalman Filter fit')
    divider = make_axes_locatable(currentAxis)
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    plt.pyplot.colorbar(im, cax=cax)  
    f.set_tight_layout(True);