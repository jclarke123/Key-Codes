#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
These messy sets of functions are the ones that make the samples used in extrapolation
There is a hierarchy here - we must always have a FARMER LP sample cut up appropriately (e.g. FARMER LP, exept just 0.5<z<1)
We then make corresponding mask sample first. Then, we do the voronoi weighting to make the corresponding voronoi weighting map for future interp
Then, we do the CANDELS interp (if wanted)
Then, we fit a schechter curve to the combined sample of the above (have to find on your own, will make a function later for that)
Once we have that schechter function, we use all of the previous samples and hand them in for the final additional sample
For each of these, we have "previous extrapolated..." stuff. This is a relic from where each interpolated part overlapped. Don't need now, but will keep as "legacy"
We also have main sequence stuff generated here as well
And condensing sample!


THIS IS AN UPDATED VERSION FOR THE CHANGES WE MDE in 2025, MAKING THE EXTRAPOLATED SAMPLES FAR MORE ACCURATELY
E.g. following mass functions from COSMOS 2025.



"""
###import the relevant functions
import FluxMapCreation as FMC
import numpy as np
from astropy.table import Table
from tqdm import tqdm


#the schechter curve we use. Log form, consistent with previous files
def LogSchechterCurve(logL,logphi0,a,logLc):
    """
    

    Parameters
    ----------
    logL : TYPE
        DESCRIPTION.
    logphi0 : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    logLc : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return logphi0-a*(logL-logLc)-(np.power(10,logL)/np.power(10,logLc))*np.log10(2.71)


#flipping an array around, just in case we need to. Store in table format so I can make a fits file
def flipArray(Data,BaselineArr):
    """
    

    Parameters
    ----------
    Data : TYPE
        DESCRIPTION.
    BaselineArr : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    flippedArray=[]
    for i in tqdm(range(len(Data[0]))):
        dummy=[] 
        for j in range(len(Data)):
            dummy.append(Data[j][i]) 
        flippedArray.append(dummy)
    return Table(flippedArray)

#the way to find coordinates from a weigthting map - basically, the uppermost digit gives the slice, the modulo gives x and y. We return coords in the format of "COSMOS units", i.e. 0.15 arcsec, with the mask
def findXYZ(SelectionMap,OneDProbDistr,VoronoiNumpixels,VoronoiNumslices,OriginalMapParams,StartingRedshift,EndingRedshift):
    """
    

    Parameters
    ----------
    SelectionMap : TYPE
        DESCRIPTION.
    OneDProbDistr : TYPE
        DESCRIPTION.
    VoronoiNumpixels : TYPE
        DESCRIPTION.
    VoronoiNumslices : TYPE
        DESCRIPTION.
    OriginalMapParams : TYPE
        DESCRIPTION.
    StartingRedshift : TYPE
        DESCRIPTION.
    EndingRedshift : TYPE
        DESCRIPTION.

    Returns
    -------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.

    """
    zxy=np.random.choice(SelectionMap,1,p=OneDProbDistr)
    z=StartingRedshift+(EndingRedshift-StartingRedshift)*(np.floor(zxy/(VoronoiNumpixels**2))+np.random.rand())/VoronoiNumslices
    xy=zxy%(VoronoiNumpixels**2)
    x,y=(int(xy/VoronoiNumpixels)+np.random.rand())*(OriginalMapParams[1]/VoronoiNumpixels)+OriginalMapParams[3],((xy%VoronoiNumpixels)[0]+np.random.rand())*(OriginalMapParams[2]/VoronoiNumpixels)+OriginalMapParams[5]
    return x,y,z




##########################
#TODO
#This is basically just the mask we get from the basic sample. We use pixel params to decide where the galaxies are
#We will need to do this using different masks when looking at z>6.3 or z<6.3
#The frequency of the mask we use doesn't actually matter, as long as we use the appropriate pixel scaling factor
#WE MUST HAVE A "TRIMMED DOWN" SAMPLE FOR THIS TO WORK
def extrapolateSampleViaMask(MainFilename,TemplateMaskFilename,ExtrapolatedMaskFilename,StartingRedshift,EndingRedshift,OriginalMapParams,PixelScalingFactor,indexRedshift="",indexXpix="",indexYpix="",indexMass="",indexSFR="",indexLIR="",indexOIII=""): #here StartingLimitingRedshift is the redshift we start generating from, PreviousFilename is what we load from
        


    #we take the original sample, and also the mask. We find each pixel with no signal
    Data=FMC.loadSample(MainFilename)
    MaskMap=np.load(TemplateMaskFilename,allow_pickle=True)
    MaskPixels=[]
    for i in range(len(MaskMap)):
        for j in range(len(MaskMap[0])):
            if MaskMap[j][i]!=0: MaskPixels.append([i,j])
    #using this we find the relative areas of these. Then, 
    MaskArea=len(MaskPixels)*(PixelScalingFactor**2) 
    TotalMapArea=len(MaskMap)*len(MaskMap[0])*(PixelScalingFactor**2) 
    MasklessMapArea=TotalMapArea-MaskArea
    MaskDensity=0
    MapDensity=len(Data)/MasklessMapArea
    NewMaskData=[]
    
    
    Masses=[]
    SFRs=[]
    LIRs=[]
    OIIIs=[]
    for i in range(len(Data)):
        
        if not np.isnan(float(Data[i][indexMass])) and not np.isnan(float(Data[i][indexSFR]))  and not np.isnan(float(Data[i][indexLIR]))  and not np.isnan(float(Data[i][indexOIII]))  and OriginalMapParams[3]<Data[i][3] and OriginalMapParams[4]>Data[i][3] and OriginalMapParams[5]<Data[i][4] and OriginalMapParams[6]>Data[i][4]:
            
            Masses.append(float(Data[i][indexMass]))
            SFRs.append(float(Data[i][indexSFR]))
            LIRs.append(float(Data[i][indexLIR]))
            OIIIs.append(float(Data[i][indexOIII]))
    
    #don't want to draw on the very highest galaxies! if use highest
    LimIndex=-5 #don't use the very highest 5
    if len(Masses)<20: #if very few
        LimIndex=-1
    
    MaxMass=sorted(Masses)[LimIndex]
    MaxSFR=sorted(SFRs)[LimIndex]
    MaxLIR=sorted(LIRs)[LimIndex]
    MaxOIII=sorted(OIIIs)[LimIndex]
    
    print(MaxMass,MaxSFR,MaxLIR,MaxOIII)
    
    #whilst the mask density hasn't matched the map density. We add galaxies in a random, proportional way, that should line up with the previous sample
    Failurecount=0
    while MaskDensity<MapDensity:
        RandomSelectionInteger=np.random.randint(0,len(MaskPixels)-1)
        x,y=(MaskPixels[RandomSelectionInteger][0]+np.random.rand())*PixelScalingFactor+OriginalMapParams[3],(MaskPixels[RandomSelectionInteger][1]+np.random.rand())*PixelScalingFactor+OriginalMapParams[5]
        
        Invalid=True
        while Invalid:
            RandomGalaxyInteger=np.random.randint(0,len(Data)-1)
            #print(Data[RandomGalaxyInteger][indexMass],MaxMass,Data[RandomGalaxyInteger][indexSFR],MaxSFR,Data[RandomGalaxyInteger][indexLIR],MaxLIR,Data[RandomGalaxyInteger][indexLIR],MaxOIII)
            if Data[RandomGalaxyInteger][indexMass]<MaxMass and Data[RandomGalaxyInteger][indexSFR]<MaxSFR and Data[RandomGalaxyInteger][indexLIR]<MaxLIR  and Data[RandomGalaxyInteger][indexOIII]<MaxOIII and OriginalMapParams[3]<Data[RandomGalaxyInteger][3] and OriginalMapParams[4]>Data[RandomGalaxyInteger][3] and OriginalMapParams[5]<Data[RandomGalaxyInteger][4] and OriginalMapParams[6]>Data[RandomGalaxyInteger][4]: 
                Invalid=False
            else:
                Failurecount=Failurecount+1
            #     print("a")
            # print("b")
        NewMaskData.append(Data[RandomGalaxyInteger]) 
        NewMaskData[-1][indexXpix],NewMaskData[-1][indexYpix]=x,y 
        NewMaskData[-1][indexRedshift]=StartingRedshift+np.random.rand()*(EndingRedshift-StartingRedshift) #to randomise the redshift
        MaskDensity=MaskDensity+(1/MaskArea)
    print("Num successes: "+str(len(NewMaskData)))
    print("Failure count: "+str(Failurecount))

    #then the save this new data. Turns it into a table
    t = flipArray(NewMaskData,Data[0]) 
    t.write(ExtrapolatedMaskFilename, format='fits',overwrite=True) #saves it as a new file
    print("")


##########################
#TODO
#When I have a given sample (sample+mask) and we want to make a voronoi map from it, we need to use this
#In this case we can weight by log_mass or log_LIR, user choice
#The voronoi map we used is ALWAYS normalised to 10 slices, 100x100 pixels. Have hardcoded
def calcVoronoiMap(MainFilename,MaskFilename,VoronoiFilename,StartingRedshift,EndingRedshift,OriginalMapParams,WeightOrders,indexRedshift="",indexXpix="",indexYpix="",indexMass="",indexSFR="",indexLIR="",indexOIII="",WeightingMethod="Mass"):
    """
    

    Parameters
    ----------
    MainFilename : TYPE
        DESCRIPTION.
    MaskFilename : TYPE
        DESCRIPTION.
    VoronoiFilename : TYPE
        DESCRIPTION.
    StartingRedshift : TYPE
        DESCRIPTION.
    EndingRedshift : TYPE
        DESCRIPTION.
    OriginalMapParams : TYPE
        DESCRIPTION.
    WeightOrders : TYPE
        DESCRIPTION.
    indexRedshift : TYPE, optional
        DESCRIPTION. The default is "".
    indexXpix : TYPE, optional
        DESCRIPTION. The default is "".
    indexYpix : TYPE, optional
        DESCRIPTION. The default is "".
    indexMass : TYPE, optional
        DESCRIPTION. The default is "".
    indexSFR : TYPE, optional
        DESCRIPTION. The default is "".
    indexLIR : TYPE, optional
        DESCRIPTION. The default is "".
    indexOIII : TYPE, optional
        DESCRIPTION. The default is "".
    WeightingMethod : TYPE, optional
        DESCRIPTION. The default is "Mass".

    Returns
    -------
    None.

    """
    
    
    
    Data=FMC.loadSample(MainFilename)
    MaskData=FMC.loadSample(MaskFilename)
    log_mass,log_LIR,zpix,xpix,ypix=[],[],[],[],[]
    #hardcoded the limits for now
    Numslices,Numpixels=10,100
    
    ########if sample behaves, can concanetate sample using
    #Data=np.asarray(Data,MaskData)
    for i in tqdm(range(len(Data))):
        if not np.isnan(Data[i][indexMass]) and not np.isnan(Data[i][indexSFR]) and not np.isnan(Data[i][indexLIR]) and not np.isnan(Data[i][indexOIII]):
            log_mass.append(Data[i][indexMass])
            log_LIR.append(np.log10(Data[i][indexLIR]))
            xpix.append(round((Data[i][indexXpix]-OriginalMapParams[3])/(OriginalMapParams[1]/Numpixels)))
            ypix.append(round((Data[i][indexYpix]-OriginalMapParams[5])/(OriginalMapParams[2]/Numpixels)))
            zpix.append(np.floor(Numslices*(EndingRedshift-Data[i][indexRedshift])/(EndingRedshift-StartingRedshift)))
    for j in tqdm(range(len(MaskData))):
        if not np.isnan(MaskData[j][indexMass]) and not np.isnan(Data[i][indexSFR]) and not np.isnan(MaskData[j][indexLIR]) and not np.isnan(MaskData[j][indexOIII]):
            log_mass.append(MaskData[j][indexMass])
            log_LIR.append(np.log10(MaskData[j][indexLIR]))
            xpix.append(round((MaskData[j][indexXpix]-OriginalMapParams[3])/(OriginalMapParams[1]/Numpixels)))
            ypix.append(round((MaskData[j][indexYpix]-OriginalMapParams[5])/(OriginalMapParams[2]/Numpixels)))
            zpix.append(np.floor(Numslices*(EndingRedshift-MaskData[j][indexRedshift])/(EndingRedshift-StartingRedshift)))
    if WeightingMethod=="Number": IndivWeight=np.ones(len(log_mass)) #normalised weights, which we multiply by weightorders
    elif WeightingMethod=="Mass": IndivWeight=log_mass/np.max(log_mass)
    WeightingMap=np.zeros((Numslices,Numpixels,Numpixels))+WeightOrders[0] #i.e. we have a base weighting
    centre,outer,v_outer=WeightOrders[1],WeightOrders[2],WeightOrders[3]
    #for each galaxy that we use, we assign a given weighting, spread around multiple pickles
    for i in tqdm(range(len(IndivWeight))):
        z,x,y=int(zpix[i]),xpix[i],ypix[i]
        if not(x>=Numpixels) and not(y>=Numpixels):  
            #we set the weight of the central pixel
            WeightingMap[z][y][x]=WeightingMap[z][y][x]+centre*IndivWeight[i]
            #we set the weight of surrounding pixels (ignoring ones that would lie outside the map)
            if x+1<Numpixels: WeightingMap[z][y][x+1]=WeightingMap[z][y][x+1]+outer*IndivWeight[i] #we need to set everything to a conditional for the surrounding pixels
            if x-1>=0: WeightingMap[z][y][x-1]=WeightingMap[z][y][x-1]+outer*IndivWeight[i] #just in case this is a non-existent pixels
            if y+1<Numpixels: WeightingMap[z][y+1][x]=WeightingMap[z][y+1][x]+outer*IndivWeight[i]
            if y-2>=0: WeightingMap[z][y-1][x]=WeightingMap[z][y-1][x]+outer*IndivWeight[i]
            #we set the weight of pixels further out
            if x+2<Numpixels: WeightingMap[z][y][x+2]=WeightingMap[z][y][x+2]+v_outer*IndivWeight[i]
            if x-2>=0: WeightingMap[z][y][x-2]=WeightingMap[z][y][x-2]+v_outer*IndivWeight[i]
            if y+2<Numpixels: WeightingMap[z][y+2][x]=WeightingMap[z][y+2][x]+v_outer*IndivWeight[i]
            if y-2>=0: WeightingMap[z][y-2][x]=WeightingMap[z][y-2][x]+v_outer*IndivWeight[i]
            if x+1<Numpixels and y+1<Numpixels: WeightingMap[z][y+1][x+1]=WeightingMap[z][y+1][x+1]+v_outer*IndivWeight[i]
            if x+1<Numpixels and y-1>=0:WeightingMap[z][y-1][x+1]=WeightingMap[z][y-1][x+1]+v_outer*IndivWeight[i]
            if x-1>=0 and y+1<Numpixels:WeightingMap[z][y+1][x-1]=WeightingMap[z][y+1][x-1]+v_outer*IndivWeight[i]
            if x-1>=0 and y-1>=0:WeightingMap[z][y-1][x-1]=WeightingMap[z][y-1][x-1]+v_outer*IndivWeight[i]
    np.save(VoronoiFilename,np.asarray(WeightingMap),allow_pickle=True)

# calcVoronoiMap("Basic Samples/New Interp Method Old Bands/COSMOS_FARMER_SIMPLIFIED_z~3,42-3,87_MagU26.fits", "Basic Samples/New Interp Method Old Bands/COSMOS_FARMER_SIMPLIFIED_z~3,42-3,87_MagU26_MASK.fits", "Basic Samples/New Interp Method Old Bands/COSMOS_FARMER_SIMPLIFIED_z~3,42-3,87_MagU26_VORONOI_MASS.npy", [45000,29000,29000,7000,36000,7600,36600], "Mass", [1,6,4,2], 3.42, 3.87)

###############################
#TODO
#Once we have the original file, the mask data, and the voronoi data, we start with the candels stuff
def extrapolateSampleViaCANDELs(MainFilename,MaskFilename,VoronoiFilename,CANDELsFilename,CANDELsFactor,StartingRedshift,EndingRedshift,OriginalMapParams,MassLim="",indexRedshift="",indexXpix="",indexYpix="",indexMass="",indexSFR="",indexLIR="",indexOIII=""): # Here we do this seperately for each filename. We do it seperately for the OG sample, for the Mask, for the
        


    #get all the data we need
    OriginalData=FMC.loadSample(MainFilename)
    MaskData=FMC.loadSample(MaskFilename)
    #Data=np.append(DataOrig,DataMask)
    Data=[]
    for i in range(len(OriginalData)):
        Data.append(OriginalData[i])
    for j in range(len(MaskData)):
        Data.append(MaskData[j])
    
    PerCounter=0
    #Now the voronoi parameters. Hard code these maps for now. We have our mask selection
    VoronoiNumslices,VoronoiNumpixels=10,100
    VoronoiMap=np.load(VoronoiFilename,allow_pickle=True)
    OneDProbDistr=VoronoiMap.flatten()/np.sum(VoronoiMap)
    SelectionMap=np.arange(0,VoronoiNumslices*(VoronoiNumpixels**2),1)
    
    
    ###set limiter on the CANDELS
    Masses=[]
    SFRs=[]
    LIRs=[]
    OIIIs=[]
    for i in range(len(Data)):
        if not np.isnan(float(Data[i][indexMass])) and not np.isnan(float(Data[i][indexSFR])) and not np.isnan(float(Data[i][indexLIR]))  and not np.isnan(float(Data[i][indexOIII])) and OriginalMapParams[3]<Data[i][3] and OriginalMapParams[4]>Data[i][3] and OriginalMapParams[5]<Data[i][4] and OriginalMapParams[6]>Data[i][4]:
            
            Masses.append(float(Data[i][indexMass]))
            SFRs.append(float(Data[i][indexSFR]))
            LIRs.append(float(Data[i][indexLIR]))
            OIIIs.append(float(Data[i][indexOIII]))
    
    #don't want to draw on the very highest galaxies
    LimIndex=-5 #don't use the very highest 5
    if len(Masses)<20: #if very few, ignore that
        LimIndex=-1
    
    MaxMass=sorted(Masses)[LimIndex]
    MaxSFR=sorted(SFRs)[LimIndex]
    MaxLIR=sorted(LIRs)[LimIndex]
    MaxOIII=sorted(OIIIs)[LimIndex]
    
    print(MaxMass,MaxSFR,MaxLIR,MaxOIII)
    
    
    
    #again legacy stuff, if we need to import new stuff
    NewData=[]

    Failurecount=0
    #Here, we use the "PerCounter" method. For each galaxy above the knee in our given band, we use PerCounter to determine how many additional galaxies we have
    #e.g. CANDELsFactor of 0.75 means you extrapolate an additional 1/3 galaxies. Once this gets to above 1, we generate an additional galaxy, taking all properties as appropriate
    #for x,y,z, randomise within slice
    PerCounter=0
    
    if CANDELsFactor==1:
        NewData.append(Data[0])
        x,y,z=findXYZ(SelectionMap,OneDProbDistr,VoronoiNumpixels,VoronoiNumslices,OriginalMapParams,StartingRedshift,EndingRedshift)

        NewData[-1][indexXpix],NewData[-1][indexYpix],NewData[-1][indexRedshift]=x,y,z
    for j in tqdm(range(len(Data))):
        if Data[j][indexMass]>MassLim:
            #also have this limit
            if Data[j][indexMass]<MaxMass and Data[j][indexSFR]<MaxSFR and Data[j][indexLIR]<MaxLIR and Data[j][indexOIII]<MaxOIII  and OriginalMapParams[3]<Data[j][3] and OriginalMapParams[4]>Data[j][3] and OriginalMapParams[5]<Data[j][4] and OriginalMapParams[6]>Data[j][4]:
                PerCounter=PerCounter+(1-CANDELsFactor)/CANDELsFactor
                while PerCounter>1:
                    NewData.append(Data[j])
                    
                    x,y,z=findXYZ(SelectionMap,OneDProbDistr,VoronoiNumpixels,VoronoiNumslices,OriginalMapParams,StartingRedshift,EndingRedshift)
    
                    NewData[-1][indexXpix],NewData[-1][indexYpix],NewData[-1][indexRedshift]=x,y,z
                    PerCounter=PerCounter-1
            else:
                Failurecount=Failurecount+1
                # print(MaxMass,MaxSFR,MaxLIR,MaxOIII)
                # print(Data[j][indexMass],Data[j][indexSFR],Data[j][indexLIR],Data[j][indexLIR])
    print("Num successes: "+str(len(NewData)))
    print("Failure count: "+str(Failurecount))
    #once we have all this data, we flip it and save!
    t = flipArray(NewData,Data[0]) 
    t.write(CANDELsFilename, format='fits',overwrite=True) 
    print("")
    

##################################
#TODO


def ALTextrapolateSampleViaMassF(MainFilename,MaskFilename,VoronoiFilename,CANDELSFilename,MassFFilename,StartingRedshift,EndingRedshift,OriginalMapParams,SchechterParams,Volume,Dex,MinMass,GalaxyMSParams,MaxMass=0,MassIndex=300,SFRIndex=304,RedshiftIndex=269,LIRIndex=342,OIIIIndex=349,SFRDex=0.5):
    #import data, make into a big list
    OriginalData=FMC.loadSample(MainFilename)
    MaskData=FMC.loadSample(MaskFilename)
    CANDELSData=FMC.loadSample(CANDELSFilename)
    #Data=np.append(OriginalData,[MaskData,CANDELSData])
    Data=[]
    for i in range(len(OriginalData)):
        Data.append(OriginalData[i])
    for j in range(len(MaskData)):
        Data.append(MaskData[j])
    for k in range(len(CANDELSData)):
        Data.append(CANDELSData[k])

    log_mass,indexes,DataToDrawFrom=[],[],[]
    counter=0 #use counter instead of i or anything like that, as is the relevant index for our new constructed array
    #get the data to draw from, focusing on the masses
    for i in tqdm(range(len(Data))):
        if not np.isnan(Data[i][MassIndex]) and OriginalMapParams[3]<Data[i][3] and OriginalMapParams[4]>Data[i][3] and OriginalMapParams[5]<Data[i][4] and OriginalMapParams[6]>Data[i][4]:
            log_mass.append(Data[i][MassIndex])
            indexes.append(counter)
            DataToDrawFrom.append(Data[i])
            counter=counter+1
    #again, if we want an unbound maximum, we don't set this optional paramerer
    if MaxMass==0:MaxMass=np.max(log_mass)

    #legacy data, as before
    NewData=[]


    #Now we have the masses, we need to find the mass intervals we calc the schechter function on, and the corresponding number of galaxies we compare against
    MassIntervals=np.arange(MinMass,MaxMass+Dex,Dex) 
    MassMidpoints=(MassIntervals[1:]+MassIntervals[:-1])/2
    ModelMassFunction=np.power(10,LogSchechterCurve(MassMidpoints,SchechterParams[0],SchechterParams[1],SchechterParams[2]))
    ModelNumbers=ModelMassFunction*Dex*Volume
    print(MassMidpoints)
    print(ModelMassFunction)
    print(ModelNumbers)
    #set up voronoi parameters for comparison
    VoronoiNumslices,VoronoiNumpixels=10,100
    VoronoiMap=np.load(VoronoiFilename,allow_pickle=True)
    OneDProbDistr=VoronoiMap.flatten()/np.sum(VoronoiMap)
    SelectionMap=np.arange(0,VoronoiNumslices*(VoronoiNumpixels**2),1)
    
    #Right: For each mass interval, we find the existing galaxies that lie within it, and record their indexes. This gives us the number of galaxies we need to generate
    #Make galaxies in a very similar way, position accordingly, etc. The problem is, if there is no currently existing galaxies in the band, we can't generate!
    #To compensate, we take a random galaxy, change its mass, AND change its SFR according to a main sequence. We need the projected galaxy main sequence for this
        
    
    #####this vesion is moe probabilitistic
    NumberToMake=0
    for i in tqdm(range(len(MassMidpoints))):
        CurrentNumber=[]
        for j in range(len(log_mass)):
            if log_mass[j]>MassIntervals[i] and log_mass[j]<MassIntervals[i+1]:
                CurrentNumber.append(indexes[j])   
        if np.floor(ModelNumbers[i])-len(CurrentNumber)>0:
            NumberToMake=NumberToMake+np.floor(ModelNumbers[i])-len(CurrentNumber)
            
    
    ##########don't need mass limiter on these
    
    
    NewData=np.zeros((int(NumberToMake),int(len(Data[0]))))
    counter=0
    for i in range(len(MassMidpoints)):
        CurrentNumber=[]
        for j in range(len(log_mass)):
            if log_mass[j]>MassIntervals[i] and log_mass[j]<MassIntervals[i+1]:
                CurrentNumber.append(indexes[j]) 
        if np.floor(ModelNumbers[i])-len(CurrentNumber)>0:
            SliceNumberToMake=np.floor(ModelNumbers[i])-len(CurrentNumber)
        else:
            SliceNumberToMake=0
        print(SliceNumberToMake)
        if SliceNumberToMake>0:
            RatioProb=len(CurrentNumber)/SliceNumberToMake #prob of drawing from an existing galaxy, can go over 1
            for j in tqdm(range(int(SliceNumberToMake))):
            
                if np.random.rand(1)[0]<RatioProb: #here, rand is from 0 to 1. More existing galaxies exist, more likely to draw from an existing one
                    RandomIndex=np.random.choice(CurrentNumber)
                    NewData[counter]=DataToDrawFrom[RandomIndex]
                    
                else:#i.e. no existing galaxies, need to invent some
                    NewData[counter]=DataToDrawFrom[np.random.randint(0,len(DataToDrawFrom)-1)] #a random galaxy for other properties
                    NewData[counter][MassIndex]=MassMidpoints[i]+(np.random.rand()-0.5)*Dex #mass
                    NewData[counter][SFRIndex]=GalaxyMSParams[0]*NewData[counter][MassIndex]+GalaxyMSParams[1]+(np.random.rand()-0.5)*SFRDex #sfr
                    if NewData[counter][SFRIndex]>2: 
                        LIRDexFactor=2
                    else: 
                        LIRDexFactor=4
                        
                    ###we keep these fixed, what we want to do
                    NewData[counter][LIRIndex]=10**(0.7323759*NewData[counter][SFRIndex] + 10.01490057+(np.random.rand()-0.75)*LIRDexFactor) #these relations for these values are FIXED
                    NewData[counter][OIIIIndex]=10**(0.67183032*NewData[counter][SFRIndex]+ 7.66488542 +(np.random.rand()-0.75)*2) #don't duplicate as with the other examples, as these are more volatile
                    

                    #NEED TO CHANGE LIR; OIII
                x,y,z=findXYZ(SelectionMap,OneDProbDistr,VoronoiNumpixels,VoronoiNumslices,OriginalMapParams,StartingRedshift,EndingRedshift)
    
                NewData[counter][3],NewData[counter][4],NewData[counter][RedshiftIndex]=x,y,z
                
                ###print(NewData[counter][SFRIndex],NewData[counter][LIRIndex],NewData[counter][OIIIIndex])
                counter=counter+1
                NumberToMake=NumberToMake-1
                
        print(NumberToMake)
        print(counter)
        print(NumberToMake+counter)
    
    

    print(len(NewData))
    print(len(NewData[0]))
    
    
    
    #flip array and save!
    t = flipArray(NewData,Data[0]) 
    t.write(MassFFilename, format='fits',overwrite=True) 
    



