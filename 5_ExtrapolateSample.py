"""
These sets of functions are the ones that make the samples used in extrapolation
There is a hierarchy here - we must always have a FARMER LP sample cut up appropriately (e.g. FARMER LP, exept just 0.5<z<1)
We then make corresponding mask sample first. Then, we do the voronoi weighting to make the corresponding voronoi weighting map for future interp
Then, we do the CANDELS interp (if wanted)
Then, we fit a schechter curve to the combined sample of the above. This function is NOT included here
Once we have that schechter function, we use all of the previous samples and hand them in for the final additional sample

"""
###import the relevant functions
import CANDELSextrapolation as CE
import numpy as np
from astropy.table import Table
from tqdm import tqdm

def LogSchechterCurve(logL,logphi0,a,logLc):
    """
    Returns a Schechter curve, in log format, used for fitting, or for finding the "expected" numbers of galaxies

    Parameters
    ----------
    logL : Float
        Input luminosity, in log form
    logphi0 : Float
        Parameterisation.
    a : Float
        Parameterises gradient of curve
    logLc : Float
        Location of "knee" of Schechter curve 

    Returns
    -------
    Float
        Number density for the given input luminosity

    """
    return logphi0-a*(logL-logLc)-(np.power(10,logL)/np.power(10,logLc))*np.log10(2.71)


def flipArray(Data):
    """
    Flips a 2D array and saves it as a Table, to save in a fits file. Is required due to how we filter and condense the fits files from the originals, to ensure columns are correct way around 

    Parameters
    ----------
    Data : Numpy Array or List
        Input Array

    Returns
    -------
    Table
        Fits table we can then save

    """

    flippedArray=np.flip(np.rot90(Data,axes=(1,0)),axis=1)
    return Table(flippedArray)


def findXYZ(SelectionMap,OneDProbDistr,OriginalMapParams,StartingRedshift,EndingRedshift):
    """
     the way to find coordinates from a voronoi weighting map - basically, the uppermost digit gives the slice, the modulo gives x and y. 
     We return coords in the format of "COSMOS units", i.e. 0.15 arcsec, with the mask, so these coordinates match the original  

    Parameters
    ----------
    SelectionMap : Numpy array
        Array of size of the map, where each voxel contains its own coordinates (stored as an integer, in modulo form)
    OneDProbDistr : Numpy Array
        Array of same size as above, where each value contains the normalised probability of a new galaxy lying there
    OriginalMapParams : List
        a list of 7 values describing COSMOS. The first is the size of the cube in its arbitrary pixel units (of 0.15 arcsec). The second and third are the desired cube 
        dimensions, in those same units, for x and y. The fourth and fifth are the lower and upper limits, in the same pixel units, for x (sixth and seventh are for y) 
    Starting/EndingRedshift : floats
        The redshift range we are making galaxies for


    Returns
    -------
    x,y,z : floats
        Pixels (in COSMOS units), and redshift, of the new galaxy

    """
    SliceNumber=len(SelectionMap)
    PixelNumber=len(SelectionMap[0])
    #these selection maps store own coordinates in their pixels. We draw them out, and derive the x and y coordinates from there
    zxy=np.random.choice(SelectionMap,1,p=OneDProbDistr)
    z=StartingRedshift+(EndingRedshift-StartingRedshift)*(zxy//(PixelNumber**2)+np.random.rand())/SliceNumber #we must derive the actual redshift for this
    xy=zxy%(PixelNumber**2)
    #x and y are in arbitrary pixel units
    x,y=(int(xy/PixelNumber)+np.random.rand())*(OriginalMapParams[1]/PixelNumber)+OriginalMapParams[3],((xy%PixelNumber)[0]+np.random.rand())*(OriginalMapParams[2]/PixelNumber)+OriginalMapParams[5]
    return x,y,z


def extrapolateSampleViaMask(Filename_Base,TemplateFilename_Mask,ExtrapolatedFilename_Mask,StartingRedshift,EndingRedshift,OriginalMapParams,PixelScalingFactor,
                             NameParams={"LIR":"LIR","log_mass":"logMass","log_SFR":"logSFR","Redshift":"z","OIII":"OIII","x_orig":"X","y_orig":"Y","FLAG":"FLAG","Redshift_err":"z68LL"}):
    """
    This is basically just the mask we get from the basic sample. We use pixel params to decide where the galaxies are
    We will need to do this using different masks when looking at z>6.3 or z<6.3
    The frequency of the mask we use doesn't actually matter, as long as we use the appropriate pixel scaling factor

    Parameters
    ----------
    Filename_Base : string
        Name of the base COSMOS2020 sample for the z~0.5 interval
    TemplateFilename_Mask : string
        Name of the file that determines the locations of the tomography that are masked out within COSMOS2020. This is determined separately, and is not included in this code sample.
    ExtrapolatedFilename_Mask : string
        Name of the file that we will save the "stellar mask" sample to
    Starting/EndingRedshift : floats
        The redshift range we are making galaxies for
    OriginalMapParams : List
        a list of 7 values describing COSMOS. The first is the size of the cube in its arbitrary pixel units (of 0.15 arcsec). The second and third are the desired cube 
        dimensions, in those same units, for x and y. The fourth and fifth are the lower and upper limits, in the same pixel units, for x (sixth and seventh are for y) 
    PixelScalingFactor: float
        The size of the beam in the aforementioned arbitrary pixel units (e.g. 37.2 arcsec -> 248). This is used to convert the galaxy locations from the pixel units into the pixels we use for our maps
    NamesParams: dict, optional
        dictionary used to find the names of the columns we take galaxy data from. The default is {"LIR":"LIR","log_mass":"logMass","log_SFR":"logSFR","Redshift":"z","OIII":"OIII","x_orig":"X","y_orig":"Y","FLAG":"FLAG","Redshift_err":"z68LL"}.

    Returns
    -------
    None.

    """
    #hardcoded for COSMOS2020, the names for which we draw colums from (using the dict NameParams)
    nameRedshift,nameX,nameY=NameParams["Redshift"],NameParams["x_orig"],NameParams["y_orig"]


    #we take the original sample, and also the mask. We find each pixel with no signal
    Data=CE.loadSample(Filename_Base)
    MaskMap=np.load(TemplateFilename_Mask,allow_pickle=True)
    MaskPixels=[]
    MaskPixels=np.where(MaskMap!=0)
    #using this we find the relative areas of these.
    MaskArea=len(MaskPixels)*(PixelScalingFactor**2) 
    TotalMapArea=len(MaskMap)*len(MaskMap[0])*(PixelScalingFactor**2) 
    MasklessMapArea=TotalMapArea-MaskArea
    MapDensity=len(Data)/MasklessMapArea #how many galaxies per arbitrary pixel²
    
    #we then want to add galaxies into the mask region until these are the same. We do not know how many we need to add in this 2D array, so we append to a list
    MaskDensity=0
    NewMaskData=[] 

    while MaskDensity<MapDensity:
        #choose a random galaxy, determine a random location in the mask
        RandomSelectionInteger=np.random.randint(0,len(MaskPixels)-1)
        x,y=(MaskPixels[RandomSelectionInteger][0]+np.random.rand())*PixelScalingFactor+OriginalMapParams[3],(MaskPixels[RandomSelectionInteger][1]+np.random.rand())*PixelScalingFactor+OriginalMapParams[5]
        #choose a random galaxy from the original sample
        RandomGalaxyInteger=np.random.randint(0,len(Data)-1)
        #then save
        NewMaskData.append(Data[RandomGalaxyInteger]) 
        #we exchange the locations for ones from the mask
        NewMaskData[-1][nameX],NewMaskData[-1][nameY]=x,y 
        NewMaskData[-1][nameRedshift]=StartingRedshift+np.random.rand()*(EndingRedshift-StartingRedshift) #to randomise the redshift accordingly
        MaskDensity+=1/MaskArea #increment the counter. We do this in density units
    print("Num extra galaxies: "+str(len(NewMaskData)))
    #then the save this new data. Turns it into a table. Need to flip around to get columns right
    t = flipArray(NewMaskData) 
    t.write(ExtrapolatedFilename_Mask, format='fits',overwrite=True) #saves it as a new file
    #We add column names later (not included here)
    print("")


def calcVoronoiMap(Filename_Base,Filename_Mask,Filename_Voronoi,StartingRedshift,EndingRedshift,OriginalMapParams,WeightOrders,
                   NameParams={"LIR":"LIR","log_mass":"logMass","log_SFR":"logSFR","Redshift":"z","OIII":"OIII","x_orig":"X","y_orig":"Y","FLAG":"FLAG","Redshift_err":"z68LL"},
                   WeightingMethod="Mass"):
    """
    When I have a given sample (sample+mask) and we want to make a voronoi map from it, we need to use this
    In this case we can weight by log_mass or log_LIR, user choice
    The voronoi map we used is ALWAYS normalised to 10 slices, 100x100 pixels. Have hardcoded

    Parameters
    ----------
    Filename_Base : string
        Name of the base COSMOS2020 sample for the z~0.5 interval
    Filename_Mask : string
        Name of the stellar mask COSMOS2020 sample for the z~0.5 interval, determined above
    Filename_Voronoi : string
        Name of the file we save the voronoi tesselation from this
    Starting/EndingRedshift : floats
        The redshift range we are making galaxies for
    OriginalMapParams : List
        a list of 7 values describing COSMOS. The first is the size of the cube in its arbitrary pixel units (of 0.15 arcsec). The second and third are the desired cube 
        dimensions, in those same units, for x and y. The fourth and fifth are the lower and upper limits, in the same pixel units, for x (sixth and seventh are for y) 
    WeightOrders : list
        Four integers which determine how we weight each galaxy. The first is the baseline weighting, the second-fourth are the weights we give to the central and surrounding voxels
        in the voronoi weighting map
    NamesParams: dict, optional
        dictionary used to find the names of the columns we take galaxy data from. The default is {"LIR":"LIR","log_mass":"logMass","log_SFR":"logSFR","Redshift":"z","OIII":"OIII","x_orig":"X","y_orig":"Y","FLAG":"FLAG","Redshift_err":"z68LL"}.
    WeightingMethod : string, optional
        Whether we weight the galaxies by stellar mass, LIR, or give no weight to galaxies. This provides structure to later extrapolation. The default is "".
    

    Returns
    -------
    None.

    """
    #hardcoded for COSMOS2020, the names for which we draw colums from (using the dict NameParams)
    nameLIR,nameLogMass,nameRedshift,nameX,nameY=NameParams["LIR"],NameParams["log_mass"],NameParams["Redshift"],NameParams["x_orig"],NameParams["y_orig"] 
    
    
    BaseData=CE.loadSample(Filename_Base)
    MaskData=CE.loadSample(Filename_Mask)
    #hardcoded the size of the voronoi array. Want to keep arbitary, which we can transform to a given redshift segment. Hence, pixel size is not that of beamFWHM, using "standard" format
    VoronoiSliceNumber,VoronoiPixelNumber=10,100
    
    #We load relevant galaxy data we use to construct the voronoi tesselation from. Note, x,y,zpix are fit to the 10x100x100
    log_mass=np.asarray(list(BaseData[nameLogMass])+list(MaskData[nameLogMass]))
    log_LIR=np.log10(np.asarray(list(BaseData[nameLIR])+list(MaskData[nameLIR])))
    xpix_vor=(np.asarray(list(BaseData[nameX])+list(MaskData[nameX]))-OriginalMapParams[3])/(OriginalMapParams[1]/VoronoiPixelNumber)
    ypix_vor=(np.asarray(list(BaseData[nameY])+list(MaskData[nameY]))-OriginalMapParams[5])/(OriginalMapParams[2]/VoronoiPixelNumber)
    zpix_vor=VoronoiSliceNumber*(EndingRedshift-np.asarray(list(BaseData[nameRedshift])+list(MaskData[nameRedshift])))//(EndingRedshift-StartingRedshift)
    
    #then, for each of the galaxies, we make the weight we assign to each galaxy, for the voronoi tesselation. These are all normalised - 
    #"Ones" gives each galaxy equal weight, Mass normalises by log mass, LIR normalises by log LIR
    if WeightingMethod=="Number": IndivWeight=np.ones(len(log_mass)) #normalised weights, which we multiply by weightorders
    elif WeightingMethod=="Mass": IndivWeight=log_mass/np.max(log_mass)
    elif WeightingMethod=="LIR": IndivWeight=log_LIR/np.max(log_LIR)
    
    #Now, we construct the weighting. This is based on WeightingOrders - the first index is the "base" weight, below:
    WeightingMap=np.zeros((VoronoiSliceNumber,VoronoiPixelNumber,VoronoiPixelNumber))+WeightOrders[0]
    #the latter three assign weights in a 3x3 array around the central voxel
    centre,outer,v_outer=WeightOrders[1],WeightOrders[2],WeightOrders[3]
    #for each galaxy that we use, we assign a given weighting, spread around multiple pixels, being careful around edges
    for i in tqdm(range(len(IndivWeight))):
        z,x,y=int(zpix_vor[i]),xpix_vor[i],ypix_vor[i]
        if not(x>=VoronoiPixelNumber) and not(y>=VoronoiPixelNumber):  
            #we set the weight of the central pixel
            WeightingMap[z][y][x]=WeightingMap[z][y][x]+centre*IndivWeight[i]
            #we set the weight of surrounding pixels (ignoring ones that would lie outside the map)
            if x+1<VoronoiPixelNumber: WeightingMap[z][y][x+1]=WeightingMap[z][y][x+1]+outer*IndivWeight[i] #we need to set everything to a conditional for the surrounding pixels
            if x-1>=0: WeightingMap[z][y][x-1]=WeightingMap[z][y][x-1]+outer*IndivWeight[i] #just in case this is a non-existent pixel
            if y+1<VoronoiPixelNumber: WeightingMap[z][y+1][x]=WeightingMap[z][y+1][x]+outer*IndivWeight[i]
            if y-2>=0: WeightingMap[z][y-1][x]=WeightingMap[z][y-1][x]+outer*IndivWeight[i]
            #we set the weight of pixels further out
            if x+2<VoronoiPixelNumber: WeightingMap[z][y][x+2]=WeightingMap[z][y][x+2]+v_outer*IndivWeight[i]
            if x-2>=0: WeightingMap[z][y][x-2]=WeightingMap[z][y][x-2]+v_outer*IndivWeight[i]
            if y+2<VoronoiPixelNumber: WeightingMap[z][y+2][x]=WeightingMap[z][y+2][x]+v_outer*IndivWeight[i]
            if y-2>=0: WeightingMap[z][y-2][x]=WeightingMap[z][y-2][x]+v_outer*IndivWeight[i]
            if x+1<VoronoiPixelNumber and y+1<VoronoiPixelNumber: WeightingMap[z][y+1][x+1]=WeightingMap[z][y+1][x+1]+v_outer*IndivWeight[i]
            if x+1<VoronoiPixelNumber and y-1>=0:WeightingMap[z][y-1][x+1]=WeightingMap[z][y-1][x+1]+v_outer*IndivWeight[i]
            if x-1>=0 and y+1<VoronoiPixelNumber:WeightingMap[z][y+1][x-1]=WeightingMap[z][y+1][x-1]+v_outer*IndivWeight[i]
            if x-1>=0 and y-1>=0:WeightingMap[z][y-1][x-1]=WeightingMap[z][y-1][x-1]+v_outer*IndivWeight[i]
    #we then save the 10x100x100 voronoi map
    np.save(Filename_Voronoi,np.asarray(WeightingMap),allow_pickle=True)


def extrapolateSampleViaCANDELs(Filename_Base,Filename_Mask,Filename_Voronoi,Filename_CANDELS,CANDELSFactor,StartingRedshift,EndingRedshift,OriginalMapParams,
                                MinMassLim=0,
                                NameParams={"LIR":"LIR","log_mass":"logMass","log_SFR":"logSFR","Redshift":"z","OIII":"OIII","x_orig":"X","y_orig":"Y","FLAG":"FLAG","Redshift_err":"z68LL"}):
    """
    
    Once we have the original file, the mask data, and the voronoi data, we start with the CANDELS extrapolation
    Using the CANDELSratio (determined in the previous file) to determine the number of galaxies, voronoi as a basis for the galaxy locations,
    and drawing from the existing files for galaxy data, we make the new file
    Parameters
    ----------
    Filename_Base : string
        Name of the base COSMOS2020 sample for the z~0.5 interval
    Filename_Mask : string
        Name of the stellar mask COSMOS2020 sample for the z~0.5 interval
    Filename_Voronoi : string
        Name of the file we load the voronoi tesselation from
    Filename_CANDELS : string
        Name of the file we save the CANDELS extrapolation to
    CANDELSFactor : float
        The ratio of CANDELS to COSMOS2020 galaxies used in this extrapolation, determined in the previous file
    Starting/EndingRedshift : floats
        The redshift range we are making galaxies for
    OriginalMapParams : List
        a list of 7 values describing COSMOS. The first is the size of the cube in its arbitrary pixel units (of 0.15 arcsec). The second and third are the desired cube 
        dimensions, in those same units, for x and y. The fourth and fifth are the lower and upper limits, in the same pixel units, for x (sixth and seventh are for y) 
    MinMassLim : float, optional
        The minimum mass we extrapolate down to, following the mass completeness equations of Weaver+22,23 determined earlier. The default is "".
    NamesParams: dict, optional
        dictionary used to find the names of the columns we take galaxy data from. The default is {"LIR":"LIR","log_mass":"logMass","log_SFR":"logSFR","Redshift":"z","OIII":"OIII","x_orig":"X","y_orig":"Y","FLAG":"FLAG","Redshift_err":"z68LL"}.

    Returns
    -------
    None.

    """    
    #get names for indexes to draw data
    nameLogMass,nameRedshift,nameX,nameY=NameParams["log_mass"],NameParams["Redshift"],NameParams["x_orig"],NameParams["y_orig"]

    #get all the data we need
    BaseData=CE.loadSample(Filename_Base)
    MaskData=CE.loadSample(Filename_Mask)
    #We need to concanetate the total data as random galaxies to draw from
    ExistingData=np.asarray(list(BaseData)+list(MaskData))
        
    #Now the voronoi parameters. Hard code these maps for now. We have our mask selection
    VoronoiMap=np.load(Filename_Voronoi,allow_pickle=True)
    OneDProbDistr=VoronoiMap.flatten()/np.sum(VoronoiMap)
    VoronoiSliceNumber,VoronoiPixelNumber=10,100
    SelectionMap=np.arange(0,VoronoiSliceNumber*(VoronoiPixelNumber**2),1)
       
    #When adding galaxies, again we use the same method to store data. Don't know exact number of galaxies we need, so append to a list
    CANDELSData=[]

    #if CANDELSFactor=1, i.e. all galaxies in CANDELS are contained within COSMOS2020, so there is no need for extrapolation, we just add a small dummy galaxy and end it there
    if CANDELSFactor==1:
        CANDELSData.append(ExistingData[np.random.randint(0,len(ExistingData)-1)])
        x,y,z=findXYZ(SelectionMap,OneDProbDistr,OriginalMapParams,StartingRedshift,EndingRedshift)

        CANDELSData[-1][nameX],CANDELSData[-1][nameY],CANDELSData[-1][nameRedshift]=x,y,z
    else:
        #The case where we actually extrapolate
        #Here, we use the "PerCounter" method. For each galaxy above the knee in our given band, we use PerCounter to determine how many additional galaxies we have
        #e.g. CANDELsFactor of 0.75 means you extrapolate an additional 1/3 galaxies. Once this gets to above 1, we generate an additional galaxy, taking all properties as appropriate
        #to stop the same galaxies being re-made multiple times, we randomise the index to draw from first
        randomisedIndexes=np.random.shuffle(np.arange(0,len(ExistingData),1))
        PerCounter=0
        for Index in randomisedIndexes:
            #make sure galaxy is above the knee
            if ExistingData[Index][nameLogMass]>MinMassLim:
                #and are within the alloted range
                if OriginalMapParams[3]<ExistingData[Index][nameX] and OriginalMapParams[4]>ExistingData[Index][nameX] and OriginalMapParams[5]<ExistingData[Index][nameY] and OriginalMapParams[6]>ExistingData[Index][nameY]:
                    #this increments PerCounter, to the number of galaxies we need
                    PerCounter=PerCounter+(1-CANDELSFactor)/CANDELSFactor
                    while PerCounter>1: #duplicate the galaxies until we have made enough
                        CANDELSData.append(ExistingData[Index])
                        #for x,y,z, randomise within slice via voronoi
                        x,y,z=findXYZ(SelectionMap,OneDProbDistr,OriginalMapParams,StartingRedshift,EndingRedshift)
                        CANDELSData[-1][nameX],CANDELSData[-1][nameY],CANDELSData[-1][nameRedshift]=x,y,z
                        PerCounter=PerCounter-1
    print("Num successes: "+str(len(CANDELSData)))
    #once we have all this data, we flip it and save!
    t = flipArray(CANDELSData) 
    t.write(Filename_CANDELS, format='fits',overwrite=True) 
    #implement column names for later
    print("")
    

def extrapolateSampleViaMassF(Filename_Base,Filename_Mask,Filename_Voronoi,Filename_CANDELS,Filename_Mass,StartingRedshift,EndingRedshift,OriginalMapParams,SchechterParams,Volume,Dex,MinMass,MainSequenceParams,
                              MinVal=10,
                              NameParams={"LIR":"LIR","log_mass":"logMass","log_SFR":"logSFR","Redshift":"z","OIII":"OIII","x_orig":"X","y_orig":"Y","FLAG":"FLAG","Redshift_err":"z68LL"}):
    """
    
    As above, but for extrapolating by the mass function. We need to have determined the Schechter fit from the data beforehand
    Parameters
    ----------
    
    Filename_Base : string
        Name of the base COSMOS2020 sample for the z~0.5 interval
    Filename_Mask : string
        Name of the stellar mask COSMOS2020 sample for the z~0.5 interval
    Filename_Voronoi : string
        Name of the file we load the voronoi tesselation from
    Filename_CANDELS : string
        Name of the CANDELS COSMOS2020 sample for the z~0.5 interval
    Filename_Mass : string
        Name of the file we save the mass function extrapolation to
 
    Starting/EndingRedshift : floats
        The redshift range we are making galaxies for
    OriginalMapParams : List
        a list of 7 values describing COSMOS. The first is the size of the cube in its arbitrary pixel units (of 0.15 arcsec). The second and third are the desired cube 
        dimensions, in those same units, for x and y. The fourth and fifth are the lower and upper limits, in the same pixel units, for x (sixth and seventh are for y) 
    SchechterParams : list
        List of floats that represent the fitting to existing data, determined earlier (not included here - used scipy.curve_fit to existing data)
    Volume : float
        Volume of area covered by redshift range, map on-sky, in Mpc. Use to determine numbers to extrapolate
    Dex : float
        Dex interval used in the schechter function/extrapolation
    MinMass : float
        The minimum mass we extrapolate down to (typically use logmass=8, below that point we do not expect much extrapolaton)
    MainSequenceParams : list
        The two values determining the mass function for the given redshift range
    NameParams : dict, optional
        dictionary used to find the names of the columns we take galaxy data from. The default is {"LIR":"LIR","log_mass":"logMass","log_SFR":"logSFR","Redshift":"z","OIII":"OIII","x_orig":"X","y_orig":"Y","FLAG":"FLAG","Redshift_err":"z68LL"}.
    MinVal: int, optional
        Minimum number of existing galaxies in a band to safely extrapolate from the original sample
    Returns
    -------
    None.

    """
    #get names for indexes to draw data
    nameLogMass,nameLogSFR,nameRedshift,nameX,nameY=NameParams["log_mass"],NameParams["log_SFR"],NameParams["Redshift"],NameParams["x_orig"],NameParams["y_orig"]
    
    
    #import data, make into a big list as before
    BaseData=CE.loadSample(Filename_Base)
    MaskData=CE.loadSample(Filename_Mask)
    CANDELSData=CE.loadSample(Filename_CANDELS)
    ExistingData=np.asarray(list(BaseData)+list(MaskData)+list(CANDELSData))
    
    #set up voronoi parameters for comparison
    VoronoiSliceNumber,VoronoiPixelNumber=10,100
    VoronoiMap=np.load(Filename_Voronoi,allow_pickle=True)
    OneDProbDistr=VoronoiMap.flatten()/np.sum(VoronoiMap)
    SelectionMap=np.arange(0,VoronoiSliceNumber*(VoronoiPixelNumber**2),1)
    
    #We then need to add galaxies within the range. Need to determine the intervals we apply the chechter params to
    Masses=ExistingData[nameLogMass]
    MaxMass=np.max(Masses)

    #Now we have the masses, we need to find the mass intervals we calc the schechter function on, and the corresponding number of galaxies we compare against
    MassIntervals=np.arange(MinMass,MaxMass+Dex,Dex) 
    MassMidpoints=(MassIntervals[1:]+MassIntervals[:-1])/2
    #this is crucial - how many galaxires already exist within the existing sample
    ModelMassFunction=np.power(10,LogSchechterCurve(MassMidpoints,SchechterParams[0],SchechterParams[1],SchechterParams[2])) 
    ModelNumbers=ModelMassFunction*Dex*Volume
    
    #For each mass interval, we find the existing galaxies that lie within it, and record their indexes. This gives us the number of galaxies we need to generate
    #Make galaxies in a very similar way, position accordingly, etc. The problem is, if there is no currently existing galaxies in the band, we can't generate!
    #To compensate, we take a random galaxy, change its mass, AND change its SFR according to a main sequence. We need the projected galaxy main sequence for this
    #for each mass section, we determine how many galaxies we need to make, considering the existing galaxies within the fit
    NewData=[]
    for i in range(len(MassMidpoints)):
        NumberToMake=0
        ExistingNumber=0
        Indexes=[]#
        DataToDrawFrom=[]
        for j in range(len(Masses)):
            if Masses[j]>MassIntervals[i] and Masses[j]<MassIntervals[i+1]:
                ExistingNumber+=1 
                DataToDrawFrom.append(ExistingData[j])
                Indexes.append(j) #index of the galaxies we can draw from
        if np.floor(ModelNumbers[i])-len(ExistingNumber)>0: #do not want this to be negative
            #i.e. how many we still need to make to fit the schecter curve
            NumberToMake=np.floor(ModelNumbers[i])-ExistingNumber
        
        #i.e. if we need to make any, we randomly raw
        if NumberToMake>0:
            for j in tqdm(range(int(NumberToMake))):
                if ExistingNumber>MinVal: #if we have enough galaxies to safely draw from
                    RandomIndex=np.random.choice(Indexes)
                    NewData.append(DataToDrawFrom[RandomIndex])
                else: #if too few source galaxies to draw from, need to generate key parameters from main sequence, just to be representative/safe
                    RandomIndex=np.random.randint(0,len(ExistingData))
                    NewData.append(ExistingData[RandomIndex])
                    #but edit mass and SFR
                    NewData[-1][nameLogMass]=MassIntervals[i]+np.random.rand()*Dex
                    NewData[-1][nameLogSFR]=MainSequenceParams[0]+NewData[-1][nameLogMass]*MainSequenceParams[1]
                    
                x,y,z=findXYZ(SelectionMap,OneDProbDistr,OriginalMapParams,StartingRedshift,EndingRedshift)
                NewData[-1][nameX],NewData[-1][nameY],NewData[-1][nameRedshift]=x,y,z
                
    #after doing for each slice, flip array and save!
    t = flipArray(NewData) 
    t.write(Filename_Mass, format='fits',overwrite=True) 
    
