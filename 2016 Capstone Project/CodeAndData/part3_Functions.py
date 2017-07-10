from capstoneProjectFunctions import *

def runKalmanFilter(inputData, deltaData, Q=1E-3, R=1E-3):
    """Overview: A stripped down-version of the Kalman filter
    
    Parameters:
    inputData - Data structure containing the stock data (pandas data frame array)
    deltaData - Control model input (i.e. such as day-to-day differences of the input data)
    Q - Estimated error in process (or bias towards the internal model).                                                                                                                                                                                                                               
    R - Estimated error in measurements (or bias towards measurements)
    """
        
    # Setup our model data and parameters, as well as ouput variables
    inputData=inputData.fillna(value=0).copy();
    deltaData=deltaData.fillna(value=0).copy();
    
    initial_state_estimate = inputData[0]; 
    initial_prob_estimate  = R  # Initial probability estimate
    kalman_gain     = pd.Series(0.0, index=inputData.index);
    prob_estimates  = pd.Series(0.0, index=inputData.index);
    state_estimates = pd.Series(0.0, index=inputData.index);
    
    # Loop through each time-step and recalculate our state estimate
    for timeStampIndex,timeStamp in enumerate(inputData.index):    
        # Get measurement and control data
        measurementData  = inputData[timeStampIndex];
        
        # Prediction step of Kalman filter
        if timeStampIndex == 0:
            state_estimates[timeStamp] = initial_state_estimate;
            prob_estimates[timeStamp]  = initial_prob_estimate + Q;
        else:
            previousTimeStamp = inputData.index[timeStampIndex-1];
            controlModelData  = deltaData[(timeStampIndex-1):timeStampIndex];
            state_estimates[timeStampIndex] = state_estimates[previousTimeStamp] + controlModelData.values[0];
            prob_estimates[timeStampIndex]  =  prob_estimates[previousTimeStamp] + Q;
        
        # Update step of Kalman filter
        innovation = measurementData - state_estimates[timeStampIndex];
        kalman_gain[timeStampIndex]     = prob_estimates[timeStampIndex] / (prob_estimates[timeStampIndex] + R);
        state_estimates[timeStampIndex] = state_estimates[timeStampIndex] + kalman_gain[timeStampIndex]*innovation;
        prob_estimates[timeStampIndex]  = (1-kalman_gain[timeStampIndex])*prob_estimates[timeStampIndex];
    return state_estimates;#, prob_estimates;

def computeGoodnessOfFitAndMSE(yModel,yActual,dateIntervals):
    """ computes goodness of fit and MSE in each time bin. Then returns the mean 
        goodness-of-fit and MSE across all bins, plus 95th percent confidence intervals """
     
    # Loop through each time bin and compute the goodness of fit and MSE for the model
    numberOfTimeBins  = len(dateIntervals);
    individualFittingStats = np.zeros([numberOfTimeBins-1,2]);
    for timeIndex in range(numberOfTimeBins-1):
        selectedTimes = (yModel.index>=dateIntervals[timeIndex])&(yModel.index<dateIntervals[timeIndex+1])
        individualFittingStats[timeIndex,0] = np.corrcoef(yModel[selectedTimes],yActual[selectedTimes])[1,0]**2
        individualFittingStats[timeIndex,1] = np.mean((yModel[selectedTimes]-yActual[selectedTimes])**2);
    
    # Summarize across all time bins, and return the mean and 95th confidence intervals
    summaryFittingStats = np.zeros([3,2]);
    n = numberOfTimeBins-1;
    t = stats.t.ppf(1-0.05/2.0, n-1);
    for stat in range(2):
        y = individualFittingStats[:,stat];      
        useNormalStatistics = False;
        if useNormalStatistics:
            yMean   = np.nanmean(y,axis=0);
            yStd    = np.nanstd(y,axis=0);
            error   = t*yStd/np.sqrt(n);
            summaryFittingStats[0,stat]=yMean;
            summaryFittingStats[1,stat]=yMean-error;
            summaryFittingStats[2,stat]=yMean+error;
        else:
            summaryFittingStats[0,stat]=np.percentile(y,50);
            summaryFittingStats[1,stat]=np.percentile(y,25);
            summaryFittingStats[2,stat]=np.percentile(y,75);
    return summaryFittingStats

def applyModelTransform(modelCoefficients,stockData,dateIntervals,forecast=False):
    """ apply model coefficients generated for a specific time period to stock data """

    # Clip data between periods that we actually have data to analyze
    transformedData  = stockData[(stockData.index>=dateIntervals[int(forecast)])&(stockData.index<=dateIntervals[-1])].copy();
    transformedData  = transformedData.iloc[:,1:];
    
    # Loop through each time bin and apply model coefficients to data
    numberOfTimeBins = len(dateIntervals);
    if numberOfTimeBins == 1:
        if forecast==True:
            transformedData = 0*transformedData; # this is an error condition. we should never evaluate
        else:
            transformedData = transformedData*pd.Series(modelCoefficients.values[0,:],index=modelCoefficients.columns);
    else:
        for timeIndex in range(int(forecast),numberOfTimeBins-1):
            if timeIndex == 0: #first entry
                selectedTimes = transformedData.index<dateIntervals[timeIndex+1];
            elif timeIndex == (numberOfTimeBins-1): # last entry
                selectedTimes = transformedData.index>=dateIntervals[timeIndex];
            else:   
                selectedTimes = (transformedData.index>=dateIntervals[timeIndex]) & (transformedData.index<dateIntervals[timeIndex+1]);
            
            transformedData[selectedTimes] = transformedData[selectedTimes]*modelCoefficients.iloc[timeIndex-int(forecast),:];      
    return transformedData;
    
    
def computeInterStockCorrelations(dateIntervals,listOfStocks,stockData):
    """ function computes correlations between stocks
    
    dateIntervals - interval of time to bin stock data
    listOfStocks  - defines stocks to find information. first stock in array is our reference stock/index
    stockData     - data structure containing the stock data (pandas data frame array)
    """

    # Let's loop through stocks and bin the data down into smaller chunks (also we'll compute local correlations of the stocks)
    binnedCorrelations_FAANG  = np.array([]);
    binnedCorrelations_Top100 = np.array([]);
    for binningDateIndex in range(len(dateIntervals)-1):
        # get stock data during current binning interval
        binningStartDate  = dateIntervals[binningDateIndex+0];
        binningEndDate    = dateIntervals[binningDateIndex+1];
        selectedStockData = stockData[(stockData.index>=binningStartDate) & (stockData.index<binningEndDate)];
        
        # compute local correlations between stocks within each group during this interval
        for stockType in range(2):
            if stockType == 0:
                selectedStocks = range(1,8);
            else:
                selectedStocks = range(8,107);
            correlationValues = selectedStockData.iloc[:,selectedStocks].corr();
            correlationValues = correlationValues.fillna(1).values;
            correlationValues = correlationValues[correlationValues!=1]
            correlationValues = correlationValues.reshape([correlationValues.size,1])
            if stockType == 0:
                binnedCorrelations_FAANG = cat(0,binnedCorrelations_FAANG,correlationValues)
            else:
                binnedCorrelations_Top100 = cat(0,binnedCorrelations_Top100,correlationValues)
    return binnedCorrelations_FAANG,binnedCorrelations_Top100;

def computeBinnedCorrelationsBetweenStocks(stockData):
    """ Computes time-varying, inter-correlations between Top 100 stocks """    
    
    # Only take the FAANG stocks
    sequenceOfStocks = range(0,107);
    stockData = stockData.iloc[:,sequenceOfStocks].copy(); 
        
    # Bin the data into different weekly intervals. Then compute the time-varying correlations of different stocks with each other.    
    binningIntervals = [1,2,4,8,12,24,48,96,260]; #8,12,24,48,96,120,260];
    interCorrelationStates = np.zeros([len(binningIntervals),2,3]);
    for i,binningInterval in enumerate(binningIntervals):
        # Compute the binned correlations
        dateIntervals   = computeTimeBins(startDate=stockData.index[0],endDate=stockData.index[-1],timeIntervalType='W',selectedIntervalBinning=binningInterval);
        interCorrelations = computeInterStockCorrelations(dateIntervals,stockData.columns,stockData);
        
        # Compute summary statistics
        for stockType in range(2):
            n = interCorrelations[stockType].shape[0]
            t = stats.t.ppf(1-0.05/2.0, n-1);
            #y = interCorrelations.values[:,stockType];
            y = interCorrelations[stockType];        
            
            useNormalStatistics = False;
            if useNormalStatistics:
                yMean   = np.nanmean(y,axis=0);
                yStd    = np.nanstd(y,axis=0);
                error   = t*yStd/np.sqrt(n);
                interCorrelationStates[i,stockType,0]=yMean;
                interCorrelationStates[i,stockType,1]=yMean-error;
                interCorrelationStates[i,stockType,2]=yMean+error;
            else:
                interCorrelationStates[i,stockType,0]=np.percentile(y,50);
                interCorrelationStates[i,stockType,1]=np.percentile(y,25);
                interCorrelationStates[i,stockType,2]=np.percentile(y,75);
          
    # Show scatter plot
    f, (ax)  = matplotlib.pyplot.subplots(nrows=1,ncols=1,figsize=(5,4))
    for stockType in range(2):
        selectedStocks,setName,lineType,lineColor = getStockTypeInfo(stockType+1);
        
        x = binningIntervals;
        y = interCorrelationStates[:,stockType,0];
        yNegativeError = interCorrelationStates[:,stockType,1];
        yPositiveError = interCorrelationStates[:,stockType,2];
        ax.semilogx(x,y,lineType,label=setName,color=lineColor)
        ax.fill_between(x, yNegativeError, yPositiveError, alpha=0.25, edgecolor=lineColor, facecolor=lineColor)
    ax.set_xticks(binningIntervals)
    ax.set_xlim([1,260])
    ax.set_ylim([-1,1])
    ax.set_ylabel('Median correlations between stocks')
    ax.set_xlabel('Time interval (weeks)')
    ax.legend(loc='lower right',fontsize = 'medium')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #for xLabels in ax.xaxis.get_majorticklabels():
    #    xLabels.set_rotation(45);
    #    xLabels.set_horizontalalignment('center')
    f.set_tight_layout(True);

def computeLinearModel(inputData,stockSets=['FAANGStocks','Top100StocksExcluding'],usePreviousWeights=False, forecastingOn=False,useKalmanFilter=False):
    """Overview: Computes three classes of linear models to created a weighted running average of stocks to compare against the NASDAQ
    
    Parameters:
    inputData          - Stock data used for linear fits (dependent and independent variables)
    stockSets          - A list of stock types to analyze: NASDAQ, FAANGStocks, Top100Stocks, AllStocks, Top100StocksExcluding (Excludes FAANG), and AllStocksExcluding (Excludes Top 100)
    usePreviousWeights - Denotes whether to use previous time-bin (in weeks) to use for correlations
    forecastingOn      - Denotes whether to forecast current NASDAQ's measurements from stock values in the past
    useKalmanFilter    - Tries to implement a Kalman filter on top of the regression analysis (experimental)
    """
  
    modelSets = [];
    for stockSet in stockSets:
        # Get stock data
        sequenceOfStocks  = getSequenceOfStocks(stockSet);
        stockData         = inputData.iloc[:,sequenceOfStocks].copy(); 
        originalStockData = stockData.copy(); # Used for forecasting

        # Bin the data further
        f    = matplotlib.pyplot.figure(figsize=(10,8));
        ax1  = matplotlib.pyplot.subplot2grid((3,10), (0,0), colspan=10);
        ax2  = matplotlib.pyplot.subplot2grid((3,10), (1,0), colspan=7);
        ax2b = matplotlib.pyplot.subplot2grid((3,10), (1,7), colspan=3);
        ax3  = matplotlib.pyplot.subplot2grid((3,10), (2,0), colspan=7);
        ax3b = matplotlib.pyplot.subplot2grid((3,10), (2,7), colspan=3);
        if forecastingOn:
            daysToLookBack = 1*5; # how many days to looks back
            if daysToLookBack>0:
                stockData.iloc[daysToLookBack:,0:]  = stockData.iloc[0:-daysToLookBack,0:].values;
                stockData.iloc[0:daysToLookBack,0:] = 0;    
        if useKalmanFilter:
            Q = 1E-3; R = 1E1;
        listOfStocks     = stockData.columns;
        startDate        = datetime.datetime(2011, 4, 1); #2008
        endDate          = datetime.datetime(2016, 4, 1);
        dateTickLabels   = range(2012,2017);
        dateTicks        = [datetime.datetime(year, 1, 1) for year in dateTickLabels];
        binningIntervals = [1,2,4,8,12,24,48]; #8,12,24,48,96,120,260];
        if usePreviousWeights==False:
            binningIntervals.append(96);
            #binningIntervals.append(260);
        modelErrors = np.zeros([len(binningIntervals),3+int(useKalmanFilter),3,2]);
        for i,binningInterval in enumerate(binningIntervals):
            dateIntervals   = computeTimeBins(startDate,endDate,timeIntervalType='W',selectedIntervalBinning=binningInterval);
            if forecastingOn:
                actualData = originalStockData[(stockData.index>=dateIntervals[int(usePreviousWeights)]) & (stockData.index<dateIntervals[-1])][listOfStocks[0]];
            else:
                actualData =         stockData[(stockData.index>=dateIntervals[int(usePreviousWeights)]) & (stockData.index<dateIntervals[-1])][listOfStocks[0]];    
        
            # compute correlation coefficients within binning intervals
            binnedStockData, binnedCorrelations = binStockData(dateIntervals,listOfStocks,stockData);  
            correlationCoeffs   = (binnedCorrelations.iloc[:,1:]+1)/2;
            normalizationFactor = (correlationCoeffs.sum(axis=1)/correlationCoeffs.shape[1]);
            normalizationFactor = pd.DataFrame(np.tile(normalizationFactor,[correlationCoeffs.shape[1],1]).T,index=correlationCoeffs.index,columns=correlationCoeffs.columns);
            correlationCoeffs   = correlationCoeffs.divide(normalizationFactor);
            
            # Compute regression coefficients within binning intervals
            #clf = sklearn.linear_model.ElasticNet(alpha=1E-13/len(sequenceOfStocks),l1_ratio=0.0,fit_intercept=False,positive=True); #ElasticNet
            clf = sklearn.linear_model.LassoLars(alpha=1E-13/len(sequenceOfStocks),fit_intercept=False,positive=True,max_iter=5000,normalize=True,fit_path=False); #ElasticNet, 1E-3 for LassoLars (FAANG), alpha=1E-3/len(sequenceOfStocks)
            regressionCoeffs = trainLinearRegressionModel(stockData, clf, startDate, endDate, dateIntervals) #alpha=1E-1 for Top 100, #0.001 or 0.0001
            
            # Setup a pandas array for unweighted coefficients within binning intervals  
            unweightedCoeffs  = pd.DataFrame(1./correlationCoeffs.columns.shape[0],correlationCoeffs.index,correlationCoeffs.columns)
            
            # Compute model predictions for the NASDAQ trace (using the model coefficients)
            if useKalmanFilter:
                deltaData         = stockData.diff().fillna(0).copy();
                modelNone         = runKalmanFilter(actualData,0.0*actualData                                                                                , Q, R);
                modelUnweighted   = runKalmanFilter(actualData,applyModelTransform(unweightedCoeffs ,deltaData,dateIntervals,usePreviousWeights).sum(axis=1) , Q, R);
                modelCorrelations = runKalmanFilter(actualData,applyModelTransform(correlationCoeffs,deltaData,dateIntervals,usePreviousWeights).mean(axis=1), Q, R);
                modelRegression   = runKalmanFilter(actualData,applyModelTransform(regressionCoeffs ,deltaData,dateIntervals,usePreviousWeights).sum(axis=1) , Q, R);
            else:
                modelUnweighted   = applyModelTransform(unweightedCoeffs ,stockData,dateIntervals,usePreviousWeights).sum(axis=1)
                modelCorrelations = applyModelTransform(correlationCoeffs,stockData,dateIntervals,usePreviousWeights).mean(axis=1)
                modelRegression   = applyModelTransform(regressionCoeffs ,stockData,dateIntervals,usePreviousWeights).sum(axis=1)
            
            # Compute the model error
            modelError_Unweighted   = computeGoodnessOfFitAndMSE(modelUnweighted  ,actualData,dateIntervals[int(usePreviousWeights):]);
            modelError_Correlations = computeGoodnessOfFitAndMSE(modelCorrelations,actualData,dateIntervals[int(usePreviousWeights):]);
            modelError_Regression   = computeGoodnessOfFitAndMSE(modelRegression  ,actualData,dateIntervals[int(usePreviousWeights):]);
            modelErrors[i,0,:,:] = modelError_Unweighted;
            modelErrors[i,1,:,:] = modelError_Correlations;
            modelErrors[i,2,:,:] = modelError_Regression;
            if useKalmanFilter:
                modelError_None = computeGoodnessOfFitAndMSE(modelNone,actualData,dateIntervals[int(usePreviousWeights):]);
                modelErrors[i,3,:,:] = modelError_None;
            
            # Plot traces
            if binningInterval == 4:
                actualData.plot(lw=3,  ax=ax1, style='k', label='NASDAQ')
                if useKalmanFilter:
                    modelNone.plot(    ax=ax1, style='y', label='None')
                modelUnweighted.plot(  ax=ax1, style='r', label='Unweighted')
                modelCorrelations.plot(ax=ax1, style='b', label='Correlation')
                modelRegression.plot(  ax=ax1, style='g', label='Regression')
        ax1.set_xticks(dateTicks)
        ax1.set_xticklabels(dateTickLabels)
        ax1.set_ylabel('Normalized value')
        ax1.set_ylim([-3,3])
        leg = ax1.legend(loc='lower right',ncol=2+int(useKalmanFilter))
        for xLabels in ax1.xaxis.get_majorticklabels():
                xLabels.set_rotation(0);
                xLabels.set_horizontalalignment('center')
        #ltext  = leg.get_texts()  # all the text.Text instance in the legend
        #matplotlib.pyplot.setp(ltext, fontsize='small')    # the legend text fontsize 
                
        # Now compute summary statistics for evaluating whether models correctly predicted the NASDAQ (using Goodness-of-fit and MSE)
        for statIndex,statType in enumerate(['Goodness-of-Fit','MSE']):
            if statType=='Goodness-of-Fit':
                ax  = ax2;
                axb = ax2b;
                ax.set_ylim([0.0,1])
            elif(statType == 'MSE'):
                ax  = ax3;
                axb = ax3b;
                ax.set_ylim([0,0.15])
            
            modelTypes = ['Regression','Correlation','Unweighted','None']
            nModels    = len(modelTypes)-1+int(useKalmanFilter);
            for modelType in modelTypes[:nModels]:
                if(modelType == 'Unweighted'):
                    color = 'r';
                    modelIndex = 0;
                elif(modelType == 'Correlation'):
                    color = 'b';
                    modelIndex = 1;
                elif(modelType == 'Regression'):
                    color = 'g';
                    modelIndex = 2;
                elif(modelType == 'None'):
                    color = 'y';
                    modelIndex = 3;
                plotStyle = 'o--'+color;
                
                yMean          = modelErrors[:,modelIndex,0,statIndex];
                yNegativeError = modelErrors[:,modelIndex,1,statIndex];
                yPositiveError = modelErrors[:,modelIndex,2,statIndex];
                yMeanRelative  = modelErrors[:,modelIndex,0,statIndex] - modelErrors[:,0,0,statIndex];
                ax.semilogx(binningIntervals,yMean,plotStyle,label=modelType); 
                ax.fill_between(binningIntervals, yNegativeError, yPositiveError, alpha=0.25, edgecolor=color, facecolor=color)
                axb.bar(modelIndex-0.4,np.mean(yMeanRelative),label=modelType,color=color)
                axb.errorbar(modelIndex, np.mean(yMeanRelative), yerr=np.std(yMeanRelative)/np.sqrt(len(yMeanRelative)),color='k');
            ax.set_xticks(binningIntervals)
            ax.set_xlim([binningIntervals[0],binningIntervals[-1]])
            ax.set_ylabel(statType+'\n of model')
            ax.set_xlabel('Time interval to compute model weights (weeks)')
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axb.set_ylabel('$\Delta$ with \n unweighted \n average');
            axb.set_xticks(range(nModels))
            axb.set_xticklabels(modelTypes[:nModels][::-1])
            modelTypes[:(len(modelTypes)-1+int(useKalmanFilter))]
            for xLabels in axb.xaxis.get_majorticklabels():
                xLabels.set_rotation(30);
                xLabels.set_horizontalalignment('center')
                #xLabels.set_visible(bool(statIndex))
        ax3b.set_ylim([-0.1,0.01])
        ax2b.set_ylim([-0.25,0.25])
        #    if statType == 'Goodness-of-Fit':
        #        ax.legend(loc='best')
        f.set_tight_layout(True);
        
        # Now copy data structure for current stock set. We'll reuse when comparing FAANG vs Top 100
        modelSets.append(modelErrors.copy())
        
    # Summary plot to compare stock groups (FAANG vs Top 100)
    if len(stockSets)>1:
        fig2, axes = matplotlib.pyplot.subplots(nrows=1,ncols=2,figsize=(10,4));
        for statIndex,statType in enumerate(['Goodness-of-Fit','MSE']):
            if statType=='Goodness-of-Fit':
                ax  = axes[0];
            elif(statType == 'MSE'):
                ax  = axes[1];
            
            modelTypes = ['Regression','Correlation','Unweighted','None']
            nModels    = len(modelTypes)-1+int(useKalmanFilter);
            for modelType in modelTypes[:nModels]:
                if(modelType == 'Unweighted'):
                    color = 'r';
                    modelIndex = 0;
                elif(modelType == 'Correlation'):
                    color = 'b';
                    modelIndex = 1;
                elif(modelType == 'Regression'):
                    color = 'g';
                    modelIndex = 2;
                elif(modelType == 'None'):
                    color = 'y';
                    modelIndex = 3;
                plotStyle = 'o--'+color;
                
                yMeanRelative  = modelSets[0][:,modelIndex,0,statIndex]-modelSets[1][:,modelIndex,0,statIndex];
                ax.bar(modelIndex-0.4,np.mean(yMeanRelative),label=modelType,color=color)
                ax.errorbar(modelIndex, np.mean(yMeanRelative), yerr=np.std(yMeanRelative)/np.sqrt(len(yMeanRelative)),color='k');
            ax.set_ylabel('$\Delta$'+statType);
            ax.set_title('FAANG vs other Top 100')
            ax.set_xticks(range(nModels))
            ax.set_xticklabels(modelTypes[:nModels][::-1])
            modelTypes[:(len(modelTypes)-1+int(useKalmanFilter))]
            for xLabels in ax.xaxis.get_majorticklabels():
                xLabels.set_rotation(30);
                xLabels.set_horizontalalignment('center')
        axes[0].set_ylim([-0.3,0.3])
        axes[1].set_ylim([0,0.1])
        fig2.set_tight_layout(True);
    
        # Show stats
        statTest = np.ones([2,nModels],dtype='float64')
        for statIndex,statType in enumerate(['Goodness-of-Fit','MSE']):
            for modelIndex in range(nModels):
                h,statTest[statIndex,modelIndex] = stats.ranksums(modelSets[0][:,modelIndex,0,statIndex], \
                                                                  modelSets[1][:,modelIndex,0,statIndex]);
        print statTest
        
def computeRegressionCoefficients(stockData,stockSet='FAANGStocks'):
    """ Compute summary statistics for regression fits (using the stockSet specified)"""

    # Get stock data
    sequenceOfStocks  = getSequenceOfStocks(stockSet);
    stockData         = stockData.iloc[:,sequenceOfStocks].copy(); 
            
    # Bin the data further and compute regression fits
    startDate        = datetime.datetime(2011, 4, 1); #2008
    endDate          = datetime.datetime(2016, 4, 1);
    dateTickLabels   = range(2012,2017);
    dateTicks        = [datetime.datetime(year, 1, 1) for year in dateTickLabels];
    binningIntervals = [1,2,4,8,12,24,48]; #[1,2,4,8,12,24,48]
    for binningInterval in binningIntervals:        
        # compute correlation coefficients within binning intervals
        dateIntervals       = computeTimeBins(startDate,endDate,timeIntervalType='W',selectedIntervalBinning=binningInterval);
        binnedStockData, binnedCorrelations = binStockData(dateIntervals,stockData.columns,stockData);  
        correlationCoeffs   = (binnedCorrelations.iloc[:,1:]+1)/2;
        normalizationFactor = (correlationCoeffs.sum(axis=1)/correlationCoeffs.shape[1]);
        normalizationFactor = pd.DataFrame(np.tile(normalizationFactor,[correlationCoeffs.shape[1],1]).T,index=correlationCoeffs.index,columns=correlationCoeffs.columns);
        correlationCoeffs   = correlationCoeffs.divide(normalizationFactor);
        
        # Compute regression coefficients within binning intervals
        #clf              = sklearn.linear_model.ElasticNet(alpha=1E-13/len(sequenceOfStocks),l1_ratio=1.0,fit_intercept=False,positive=True);
        #clf              = sklearn.linear_model.LassoLarsIC(fit_intercept=False,positive=True); #ElasticNet, 1E-3 for LassoLars (FAANG), alpha=1E-3/len(sequenceOfStocks)
        clf              = sklearn.linear_model.LassoLars(alpha=1E-13/len(sequenceOfStocks),fit_intercept=False,positive=True,max_iter=5000,normalize=True,fit_path=False); #ElasticNet, 1E-3 for LassoLars (FAANG), alpha=1E-3/len(sequenceOfStocks)
        modelCoeffs      = trainLinearRegressionModel(stockData, clf, startDate, endDate, dateIntervals);
        if binningInterval == 1:
            regressionCoeffs = modelCoeffs 
            corrCoeffs       = correlationCoeffs
        else:
            regressionCoeffs = regressionCoeffs.append(modelCoeffs);
            corrCoeffs       = corrCoeffs.append(correlationCoeffs);
        
    # Compute statistics to compare groups
    statTest = np.zeros([regressionCoeffs.columns.size, regressionCoeffs.columns.size],dtype='float64')
    for A,currentStockA in enumerate(regressionCoeffs.columns):
        for B,currentStockB in enumerate(regressionCoeffs.columns):
            h,statTest[A,B] = stats.ranksums(regressionCoeffs[currentStockA],regressionCoeffs[currentStockB])
    np.set_printoptions(precision=4,suppress=True)
    statTest
    
    # Compute statistics to compare groups
    nullMean = regressionCoeffs.sum(axis=1)/regressionCoeffs.columns.size;
    nullMean = pd.DataFrame(np.tile(nullMean,[regressionCoeffs.shape[1],1]).T,index=regressionCoeffs.index,columns=regressionCoeffs.columns);
    nullHypothesis = regressionCoeffs - regressionCoeffs.mean(axis=0) + nullMean;
    statTest = np.zeros([regressionCoeffs.columns.size, 1],dtype='float64')
    for i,stock in enumerate(regressionCoeffs.columns):
        h,statTest[i] = stats.ranksums(regressionCoeffs[stock],nullHypothesis[stock])
    np.set_printoptions(precision=8,suppress=True)
    statTest*(1+np.argsort(statTest,axis=0));
    print "Binning Interval = {}, Significance = {}".format(binningInterval,statTest.T)
    
    # Compute box plots of summary plots
    medianCoeffsSortedLabels = regressionCoeffs.mean(axis=0).sort_values(ascending=False).index;
    if(len(sequenceOfStocks)>30):
        medianCoeffsSortedLabels = medianCoeffsSortedLabels[:30]
    regressionCoeffsSelected = regressionCoeffs.loc[:,medianCoeffsSortedLabels].copy();
    fig = matplotlib.pyplot.figure(figsize=(10,8))
    ax  = matplotlib.pyplot.subplot2grid((2,3), (0,0), colspan=2);
    ax2 = matplotlib.pyplot.subplot2grid((2,3), (1,0), colspan=2);
    regressionCoeffsSelected.sort().apply(signal.medfilt,kernel_size=9).plot(ax=ax,cmap='jet')
    dateTickLabels = range(2012,2017);
    dateTicks      = [datetime.datetime(year, 1, 1) for year in dateTickLabels];
    ax.set_xticks(dateTicks)
    ax.set_xticklabels(dateTickLabels)
    ax.set_ylabel('Coefficient weight')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=2, fontsize=12)
    
    regressionCoeffsSelected.boxplot(ax=ax2)
    ax2.set_ylabel('Coefficient weight')
    for xLabels in ax2.xaxis.get_majorticklabels():
        xLabels.set_rotation(45);
        xLabels.set_horizontalalignment('center')
        
    x = corrCoeffs.values.reshape(corrCoeffs.size)
    y = regressionCoeffs.values.reshape(regressionCoeffs.size)
    mask = np.invert(np.isnan(x) | np.isnan(y))
    x = x[mask];
    y = y[mask];
    P = np.polyfit(x,y,1)
    ax3 = matplotlib.pyplot.subplot2grid((2,3), (1,2), colspan=1);
    ax3.plot(x,y,'ok')
    ax3.plot(np.sort(x),np.polyval(P,np.sort(x)),'-r',linewidth=3)
    ax3.set_ylabel('Regression coefficient');
    ax3.set_xlabel('Correlation coefficient')
    print "r={}".format(np.corrcoef(x,y)[0,1])
    fig.set_tight_layout(True);