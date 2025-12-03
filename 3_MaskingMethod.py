import numpy as np
from tqdm import tqdm
from astropy.convolution import convolve
import os
import PSAndCC_Creation as PSC
import CANDELSextrapolation as CE
import FluxMapCreation as FMC

    
    
def makeGaussxBinMasksCOSMOS(cosmo,Samplefilenames,RestFrequency,StartFrequency,EndFrequency,PixelScalingFactor,SliceNumber,sigma=1,miniarrlen=7,MainSequence="" ,ComplMaskname="",BrightMaskname="",MapSpreadParameter=3,FreqSpreadParameter=3,OriginalMapParams=[45000,29000,29000,7000,36000,7600,36600],NameParams=""):
    """
    Making a binary mask multiplied with a gaussian mask, apodizing the edges. We make one for "complete" and one for "bright" data, for all galaxies above mass cutoff or just the bright ones
    Based on Van Cuyck+2023, Karoumpis+2024
    Parameters
    ----------
    cosmo : astropy cosmology object
        The cosmology we use for determining e.g. luminositiy distances
    Samplefilenames : list
        Strings containing all the source files we draw from
    RestFrequency : float
        Frequency of the line we are making a mask for
    Start/EndFrequency : float
        Frequencies of the cube we make masks for
    sigma : float, optional
        The size of the "binary" part of the mask, as a fraction of the beam sigma (where 1 non-subgridded voxel is 1 beam FWHM). The default is 1.
    miniarrlen : int, optional
        Parameter used to determine array for the masking. We want this to be as small as possible, whilst containing all the mask info. The default is 7.
    MainSequence : string, optional
        File name for the COSMOS2020 main sequence information. The default is "".
    Compl/BrightMaskname : string, optional
        Names to save masks, including all relevant galaxies or just those above the main sequence ("Complete or Bright"). The default is "".
    Map/FreqSpreadParameter : int, optional
        If we are convolving a map with the beams, these integers are the subgridding (i.e. MSP=3 means a 3x3 subgrid). The default is 3.
    OriginalMapParams : list, optional
        a list of 7 values describing COSMOS. The first is the size of the cube in its arbitrary pixel units (of 0.15 arcsec). The second and third are the desired cube 
        dimensions, in those same units, for x and y. The fourth and fifth are the lower and upper limits, in the same pixel units, for x (sixth and seventh are for y)  The default is [45000,29000,29000,7000,36000,7600,36600].
    NamesParams: dict, optional
        dictionary used to find the names of the columns we take galaxy data from


    Returns
    -------
    None.

    """
    
    #Get some base properties of the tomography we are making masks for, in a similar way to map making
    StartingRedshift=(RestFrequency/EndFrequency)-1
    EndingRedshift=(RestFrequency/StartFrequency)-1
    MapLength=int(round(OriginalMapParams[1]/PixelScalingFactor))
    TrueMapLength=MapLength*MapSpreadParameter #accounting for the subgridding we do
    TrueMapDepth=SliceNumber*FreqSpreadParameter
    
    #we will need to do apply a mask to all relevant galaxies. This is a binary mask which we convolve with a gaussian, and then apply to all relevant locations to save time
    #So first, we need to make the binary mask, of much smaller dimensions than the map. Done by the methods of Van Cuyck+23
    BinaryMiniArr=np.ones((miniarrlen,miniarrlen,miniarrlen))
    midpoint=miniarrlen//2
    #inefficient but small array, only do once
    for i in range(len(BinaryMiniArr)):
        for j in range(len(BinaryMiniArr[0])):
            for k in range(len(BinaryMiniArr[0][0])):
                if ((i-midpoint)**2+(j-midpoint)**2+(k-midpoint)**2)**0.5<MapSpreadParameter*sigma/2.355:
                    BinaryMiniArr[i][j][k]=0 #locations within the sigma range are set to 0
    
    #we then need to set up the relevant arrays, including those we later convolve. Based on the methodology of Karoumpis+24
    OnesArray=np.ones((TrueMapDepth,TrueMapLength,TrueMapLength), dtype=np.float32) #a unitary array used in matrix calculations for the binary masks
    ComplB=np.ones((TrueMapDepth,TrueMapLength,TrueMapLength), dtype=np.float32)
    BrightB=np.ones((TrueMapDepth,TrueMapLength,TrueMapLength), dtype=np.float32)
    GaussKern=FMC.gaussian_kernel_2d(MapSpreadParameter, 13)
    LorntKern=FMC.lorentzian_kernel_1d(FreqSpreadParameter, 13)
    
    
    #We need the main sequence data for determining the bright/complete masks. We use average redshift for this
    MSArray=np.load(MainSequence)
    #we also form the redshift bands, required to form the completeness threshold
    z_bands_lower=np.arange(0,8.5,0.5) #redshift
    z_bands_upper=z_bands_lower+0.5
    z_bands_midpoints=(z_bands_upper+z_bands_lower)/2
    CompletenessArray=0-3.23e7*(1 + z_bands_midpoints) + 7.83e7*(1 + z_bands_midpoints)**2 #Msol
    
    #We know need all indexes that we aim to mask. We do not know the length of these arrays, including using multiple files, hence use list and append the relevant data
    IndexesComplete=[]
    IndexesBright=[]
    
    #We then start applying the binary mask to each galaxy
    #We load each sample separately
    for Samplefilename in Samplefilenames:
        #load their data
        Data=CE.loadSample(Samplefilename)
        VariableList,_=FMC.fetchVariablesViaNamesCOSMOS(Data,StartingRedshift,EndingRedshift,OriginalMapParams,PixelScalingFactor,SliceNumber,RestFrequency,NameParams,MapSpreadParameter=MapSpreadParameter,FreqSpreadParameter=FreqSpreadParameter)
        xpix,ypix,zpix=VariableList["xpix"],VariableList["ypix"],VariableList["zpix"]
        log_mass=VariableList["log_mass"]
        log_SFR=VariableList["log_SFR"]
        Redshift=VariableList["Redshift"]
    
        ######################now, we make the mask
        
        #we need the correct main sequence parameters. This is the FIRST index where the below condition is met
        ValidIndex=np.where(MSArray[:][0]>(StartingRedshift+EndingRedshift)/2)[0]
        a,b=MSArray[ValidIndex][3],MSArray[ValidIndex][2]

        #we then find all relevant galaxies. Search through all of them, with valid locations, and find the appropriate completeness threshold.
        #Then select galaxies, calculate the relevant main sequence, then append the relevant location data
        #Inefficient, but making it more efficient has less clairtiy
        for i in range(len(log_mass)):
            if ypix[i]<TrueMapLength and xpix[i]<TrueMapLength and ypix[i]>=0 and xpix[i]>=0:
                for ii in range(len(z_bands_lower)): #different bands have different completeness thresholds, need to select the correct one
                    if Redshift[i]>=z_bands_lower[ii] and  Redshift[i]<=z_bands_upper[ii]:
                        CompletenessThreshold=CompletenessArray[ii]
                if log_mass[i]>CompletenessThreshold:
                    dMS=log_SFR[i]-FMC.galMainSeq(log_mass[i],a,b)
                    #all galaxies above threshold in complete
                    IndexesComplete.append((int(zpix[i]),int(ypix[i]),int(xpix[i])))
                    #only a few galaxies are "brught
                    if dMS>0:
                        IndexesBright.append((int(zpix[i]),int(ypix[i]),int(xpix[i])))
    
    #Now that we have all the indexes, we can now apply the binary mask we prepared earlier
    #Trying to convolve the array simply runs into issues around the edges, so we need to do this manually
    #For each index: We look through each part of the mini binary array. If THAT offset lies within the cube, we then mask the relevant bit of the mask
    #for complete
    for ii in tqdm(range(len(IndexesComplete))):            
        for i in range(len(BinaryMiniArr)):
            if i-midpoint+IndexesComplete[ii][0]>=0 and i-midpoint+IndexesComplete[ii][0]<=TrueMapDepth-1:
                for j in range(len(BinaryMiniArr[0])):
                    if j-midpoint+IndexesComplete[ii][1]>=0 and j-midpoint+IndexesComplete[ii][1]<=TrueMapLength-1:
                        for k in range(len(BinaryMiniArr[0][0])):
                            if k-midpoint+IndexesComplete[ii][2]>=0 and k-midpoint+IndexesComplete[ii][2]<=TrueMapLength-1:
                                if ComplB[i-midpoint+IndexesComplete[ii][0]][j-midpoint+IndexesComplete[ii][1]][k-midpoint+IndexesComplete[ii][2]]!=0: #if already masked, don't want to undo!
                                    ComplB[i-midpoint+IndexesComplete[ii][0]][j-midpoint+IndexesComplete[ii][1]][k-midpoint+IndexesComplete[ii][2]]=BinaryMiniArr[i][j][k]
        
        
    #for bright
    for ii in tqdm(range(len(IndexesBright))):
        for i in range(len(BinaryMiniArr)):
            if i-midpoint+IndexesBright[ii][0]>=0 and i-midpoint+IndexesBright[ii][0]<=TrueMapDepth-1:
                for j in range(len(BinaryMiniArr[0])):
                    if j-midpoint+IndexesBright[ii][1]>=0 and j-midpoint+IndexesBright[ii][1]<=TrueMapLength-1:
                        for k in range(len(BinaryMiniArr[0][0])):
                            if k-midpoint+IndexesBright[ii][2]>=0 and k-midpoint+IndexesBright[ii][2]<=TrueMapLength-1:
                                if BrightB[i-midpoint+IndexesBright[ii][0]][j-midpoint+IndexesBright[ii][1]][k-midpoint+IndexesBright[ii][2]]!=0:
                                    BrightB[i-midpoint+IndexesBright[ii][0]][j-midpoint+IndexesBright[ii][1]][k-midpoint+IndexesBright[ii][2]]=BinaryMiniArr[i][j][k]
            
    #Now, we have made the binary masked arrays. We now need the gaussian component, from Karoumpis+2024
    #first, we need an inverted form of the binary. We convolve it with the gaussian and lorentzian components of the beam, then subtract it from 1 and convolve with the binary mask
    #for complete
    ComplOneMinusGauss=1-ComplB
    for jj in tqdm(range(len(ComplOneMinusGauss))):
        ComplOneMinusGauss[jj]=convolve(ComplOneMinusGauss[jj], GaussKern)
    ComplOneMinusGauss = np.apply_along_axis(lambda spectrum: convolve(spectrum, LorntKern, boundary='extend'),
                                            axis=0, arr=ComplOneMinusGauss)    
    ComplMaskArray=ComplB*(OnesArray-ComplOneMinusGauss) #after convolving

    #for bright
    BrightOneMinusGauss=1-BrightB
    for jj in tqdm(range(len(BrightOneMinusGauss))):
        BrightOneMinusGauss[jj]=convolve(BrightOneMinusGauss[jj], GaussKern)
    BrightOneMinusGauss = np.apply_along_axis(lambda spectrum: convolve(spectrum, LorntKern, boundary='extend'),
                                            axis=0, arr=BrightOneMinusGauss)
    BrightMaskArray=BrightB*(OnesArray-BrightOneMinusGauss)
    
    #as a final check, we trim any values that are outside the 0-1 range
    ComplMaskArray=np.where(ComplMaskArray<0,0,ComplMaskArray)
    ComplMaskArray=np.where(ComplMaskArray>1,1,ComplMaskArray)   
    
    BrightMaskArray=np.where(BrightMaskArray<0,0,BrightMaskArray)
    BrightMaskArray=np.where(BrightMaskArray>1,1,BrightMaskArray)   


    #we then save
    np.save(ComplMaskname, ComplMaskArray ,allow_pickle=True)
    np.save(BrightMaskname, BrightMaskArray ,allow_pickle=True)
    


def fetchSamplefile(StartFrequency,EndFrequency,RestFrequency,MaskType):
    """
    Not included for as includes details about internal file structure. In summary, this function calculates the redshift range of a given line, considering its rest and observed frequency.
    It then fetches the appropriate catalogue file(s), taking into account whether it is a "Base", "Stellar Mask", "CANDELS extrapolated", or "Mass extrapolated".
    This is used below

    Parameters
    ----------
    Start/EndFrequency : float
        Frequencies of the cube we make masks for
    RestFrequency : float
        Frequency of the line we are making a mask for
    MaskType : string
        Type of source file we draw from to make masks for

    Returns
    -------
    Files : list
        List of filenames of all source files

    """
    Files=""
    
    return Files


def makingMasks(cosmo,FoldernameBase,FoldernameMask,FoldernameCANDELS,StartFrequency,sigma,MainSequence):
    """
    The function we run to make masks for all files. For a cosmology and frequency, as well as desired mask size, we run the above multiple times
    This is custom for FYST/CCAT, will need to edit for other frequencies
    Parameters
    ----------
    cosmo : astropy cosmology object
        The cosmology we use for determining e.g. luminositiy distances
    FoldernameBase/Mask/CANDELS : string
        Names of the folders we save the masks to
    StartFrequency : int
        frequency of the cube we are making the mask for
    sigma : float
        size of the mask in terms of beam sigma
    MainSequence : string
        Filename of main sequence file
    
    Returns
    -------
    None.

    """
    #hardcoding the names we use for file loading
    NameParams={"LIR":"LIR","log_mass":"logMass","log_SFR":"logSFR","Redshift":"z","OIII":"OIII","x_orig":"X","y_orig":"Y","FLAG":"FLAG","Redshift_err":"z68LL"}

    ##for the files we use, we want to save with a unique identifier. To be safe, the part that denotes the sigma used should replace the decimal.
    sigmalabel=str(sigma).replace(".",",")
 
    #These are hardcoded for FYST - all relevant CO transitions, pixel sizes, etc
    if StartFrequency==390: 
        COTransitionLabels=[ "CO4_3","CO5_4",  "CO6_5", "CO7_6",  "CO8_7","CO9_8"]
        Transitions=[ 4,5, 6, 7, 8,9]
        PixelScalingFactor=33/0.15
        SliceNumber=7 
        FetchFreqMod=28
    elif StartFrequency==330:  
        COTransitionLabels=[ "CO4_3","CO5_4",  "CO6_5", "CO7_6",  "CO8_7","CO9_8"]
        Transitions=[ 4,5, 6, 7, 8,9]
        PixelScalingFactor=round(37/0.15)
        SliceNumber=11
        FetchFreqMod=40
    elif StartFrequency==260: 
        COTransitionLabels=[ "CO3_2","CO4_3","CO5_4",  "CO6_5","CO7_6",   "CO8_7","CO9_8"]
        Transitions=[ 3, 4,5,6,7, 8,9]
        PixelScalingFactor=48/0.15
        SliceNumber=14
        FetchFreqMod=40
    elif StartFrequency==205:  
        COTransitionLabels=[ "CO3_2","CO4_3","CO5_4",  "CO6_5","CO7_6",   "CO8_7","CO9_8"]
        Transitions=[ 3, 4,5,6,7, 8,9]
        PixelScalingFactor=round(58/0.15)
        SliceNumber=18   
        FetchFreqMod=40
    
    #The final frequency is also given
    EndFrequency=StartFrequency+FetchFreqMod
    
    #as are the rest frequencies of each line
    COfreqs=115.27*np.asarray(Transitions)

    #for each transition, we fetch files and save in a consistent format. We do this for the base sample, the stellar mask sample, and the CANDELS extrapolation sample
    for i in range(len(COTransitionLabels)):
        FirstMaskname_Base=FoldernameBase+"GAUSSMASK_"+COTransitionLabels[i]+"_"+str(StartFrequency)+"GHz_COMPLETEMASK_BASE_sigma="+sigmalabel+".npy"
        SecondMaskname_Base=FoldernameBase+"GAUSSMASK_"+COTransitionLabels[i]+"_"+str(StartFrequency)+"GHz_BRIGHTMASK_BASE_sigma="+sigmalabel+".npy"
        BasesamplefilenameBase=fetchSamplefile(StartFrequency,EndFrequency,COfreqs[i],"Base")
        makeGaussxBinMasksCOSMOS(cosmo,BasesamplefilenameBase,COfreqs[i],StartFrequency,EndFrequency,PixelScalingFactor,SliceNumber,sigma=sigma,miniarrlen=7,MainSequence=MainSequence ,FirstMaskname=FirstMaskname_Base,SecondMaskname=SecondMaskname_Base,MapSpreadParameter=3,FreqSpreadParameter=3,OriginalMapParams=[45000,29000,29000,7000,36000,7600,36600],NameParams=NameParams)

        FirstMaskname_SM=FoldernameMask+"GAUSSMASK_"+COTransitionLabels[i]+"_"+str(StartFrequency)+"GHz_COMPLETEMASK_STELLARMASK_sigma="+sigmalabel+".npy"
        SecondMaskname_SM=FoldernameMask+"GAUSSMASK_"+COTransitionLabels[i]+"_"+str(StartFrequency)+"GHz_BRIGHTMASK_STELLARMASK_sigma="+sigmalabel+".npy"
        BasesamplefilenameMask=fetchSamplefile(StartFrequency,EndFrequency,COfreqs[i],"Mask")
        makeGaussxBinMasksCOSMOS(cosmo,BasesamplefilenameMask,COfreqs[i],StartFrequency,EndFrequency,PixelScalingFactor,SliceNumber,sigma=sigma,miniarrlen=7,MainSequence=MainSequence ,FirstMaskname=FirstMaskname_SM,SecondMaskname=SecondMaskname_SM,MapSpreadParameter=3,FreqSpreadParameter=3,OriginalMapParams=[45000,29000,29000,7000,36000,7600,36600],NameParams=NameParams)

        FirstMaskname_CANDELS=FoldernameCANDELS+"GAUSSMASK_"+COTransitionLabels[i]+"_"+str(StartFrequency)+"GHz_COMPLETEMASK_CANDELS_sigma="+sigmalabel+".npy"
        SecondMaskname_CANDELS=FoldernameCANDELS+"GAUSSMASK_"+COTransitionLabels[i]+"_"+str(StartFrequency)+"GHz_BRIGHTMASK_CANDELS_sigma="+sigmalabel+".npy"
        BasesamplefilenameCANDELS=fetchSamplefile(StartFrequency,EndFrequency,COfreqs[i],"CANDELS")
        makeGaussxBinMasksCOSMOS(cosmo,BasesamplefilenameCANDELS,COfreqs[i],StartFrequency,EndFrequency,PixelScalingFactor,SliceNumber,sigma=sigma,miniarrlen=7,MainSequence=MainSequence ,FirstMaskname=FirstMaskname_CANDELS,SecondMaskname=SecondMaskname_CANDELS,MapSpreadParameter=3,FreqSpreadParameter=3,OriginalMapParams=[45000,29000,29000,7000,36000,7600,36600],NameParams=NameParams)

    


def sumMasks(FoldernameBase,FoldernameMask,FoldernameCANDELS,FoldernameBase_Mask,FoldernameBase_Mask_CANDELS):
    """
    The function where we convolve masks with each other from each of the smaller samples
    Sometimes we want to include the CANDELS galaxies, other times we do not. We therefore include cases for both

    Parameters
    ----------
    FoldernameBase/Mask/CANDELS/Base_Mask/Base_Mask_CANDELS : strings
        Folders where the masks are stored, the ones existing and the ones to be multiplied together
        The latter two are the standard mask, and the mask also including CANDELS galaxies

    Returns
    -------
    None.

    """
    #We list all relevant files from the relevant folders. 
    Basefiles=os.listdir(FoldernameBase)
    Maskfiles=os.listdir(FoldernameMask)
    CANDELSfiles=os.listdir(FoldernameCANDELS)
    #to quickly, automatically map, we see if the filenames are the same aside from the part denoting what sample it is from. If so, from our above naming scheme, we know it is fair to convolve the masks with each other
    for i in range(len(Basefiles)):
        Prefix=Basefiles[i].replace("MASK_BASE","MASK_MASK")
        for j in range(len(Maskfiles)):
            if Prefix in Maskfiles[j]:
                Base=np.load(FoldernameBase+Basefiles[i])
                Mask=np.load(FoldernameMask+Maskfiles[j])
                Base_Maskfile=Basefiles[i].replace("MASK_BASE","MASK_COMPLETE")
                np.save(FoldernameBase_Mask+Base_Maskfile,Base*Mask) #just base+mask
                
                #we check CANDELS as well
                PrefixCANDELS=Prefix.replace("MASK_STELLARMASK","MASK_CANDELS")
                Base_Mask_CANDELSfile=Basefiles[i].replace("MASK_BASE","MASK_COMPLETE+COMPLETE")
                for k in range(len(CANDELSfiles)):
                    if PrefixCANDELS in CANDELSfiles[k]:
                        CANDELS=np.load(FoldernameCANDELS+CANDELSfiles[k]) #need to solve this
                        np.save(FoldernameBase_Mask_CANDELS+Base_Mask_CANDELSfile,Base*Mask*CANDELS)
                    

def applyMasksCO(cosmo,InitialCOMapfile,StoreCOMapSuffix,MaskSuffix,MasksFoldername,StoreCOMapFoldername,StartFrequency,RemoveMaps=True):      
    """
    We take the masks derived from the above, and systematically apply them to the CO cubes to determine an "optimal masking order"
    This is then applied to other lines in turn
    This is hardcoded for FYST/CCAT for now

    Parameters
    ----------
    cosmo : astropy cosmology object
        The cosmology we use for determining e.g. luminositiy distances
    InitialCOMapfile : string
        Name of the total CO map file we then apply the masks to
    StoreCOMapSuffix : string
        String of the file name we save to, with REPLACENUMBER, REPLACELINE, and REPLACETYPE as dummy identifiers to be replaced by the order/type of masking
    MaskSuffix : string
        Name of the mask, with REPLACELINE and REPLACETYPE as dummy identifiers as above
    MasksFoldername : string
        Foldername where the masks we used are saved to
    StoreCOMapFoldername : string
        Folder name that we store the maps and spectra into
    StartFrequency : int
        frequency of the cube we are making the mask for
    RemoveMaps : boolean
        Maps can take up a lot of file space, 10s of MB, and we will generate hundreds of these. If this is enabled, we delete those and just leave the spectra. default is "true"
        
    Returns
    -------
    None.

    """
    #hardcoded parameters for FYST
    #"MaskLines" here are what we use for labelling, and to keep track of which masks we have already applied
    #We see two rows which are identical, as we need to record the bright and complete masks
    #We then dummy these values out as we select each mask
    if StartFrequency=="390" or StartFrequency=="330":
        MaskLines=["CO4_3","CO5_4","CO6_5","CO7_6","CO8_7","CO9_8",
                "CO4_3","CO5_4","CO6_5","CO7_6","CO8_7","CO9_8",]
    elif StartFrequency=="260" or StartFrequency=="205":
        MaskLines=["CO3_2","CO4_3","CO5_4","CO6_5","CO7_6","CO8_7","CO9_8",
                "CO3_2","CO4_3","CO5_4","CO6_5","CO7_6","CO8_7","CO9_8",]
    Numbers_For_Indexing=(len(MaskLines)/2)-1 #half length of the above, -1. VERY IMPORTANT FOR INDEXING, as this determines the cutoff index between bright and complete indicators
    
    
    
    #we load the initialmap, and use to determine the dimensions of the masking (need for efficiency). We also determine the name of the file, which we then iterate on
    Currentname=InitialCOMapfile
    InitialMapdata=np.load(Currentname)
    CumulativeMask=np.ones(np.shape(InitialMapdata['FluxMap'])) #use to determine how many voxels are masked
    
    #multiple dummy arrays, storing key details of masking (e.g. efficiency, how much each mask covers) which we save later
    MasterLines=np.zeros(len(MaskLines))
    MasterTypes=np.zeros(len(MaskLines))
    MasterEfficiencies=np.zeros(len(MaskLines))
    MasterPercentageOfMaskedVoxelsIterative=np.zeros(len(MaskLines))
    MasterPercentageOfMaskedVoxelsCumulative=np.zeros(len(MaskLines))
    
    #For each mask we have to apply...
    for i in range(len(MaskLines)): 
        #We load the map from the "current map". This will change throughout
        Mapfile=np.load(Currentname)
        FluxCube=Mapfile['FluxMap']
        #We make the array for efficiencies, which we shall replace
        Efficiencies=np.zeros(len(MaskLines))-10
        #Then we look at all the masks we have to apply
        for j in range(len(MaskLines)):
            #If we already applied a mask, it is changed to "ignore", so we ignore it
            if MaskLines[j]!="IGNORE":
                #depending of where we are in the sequence, we label the mask as bright or complete
                if j>Numbers_For_Indexing: 
                    TypeUsed="BRIGHT"
                else:
                    TypeUsed="COMPLETE"
                #we use that MaskLine and Label in order to find the corresponding mask
                Mask=np.load(MasksFoldername+(MaskSuffix.replace("REPLACELINE",MaskLines[j])).replace("REPLACETYPE",TypeUsed)) * CumulativeMask #need to include previous masks
                #combined with the existing cumulative mask, we calculate the efficiency
                NumMaskedVoxels=(len(Mask)*len(Mask[0])*len(Mask[0][0]))-np.sum(Mask) #this assumes number of masked voxels is "independant" of each other
                ConvolvedFluxCube=FluxCube*Mask
                Eff=(np.std(FluxCube)-np.std(ConvolvedFluxCube))/NumMaskedVoxels
                #we have a control factor for if the mask covers areas that have already been fully masked
                if Eff==0:
                    Efficiencies[j]=-10 #to make it less likely to be picked
                else:
                    Efficiencies[j]=Eff
            else: #we do not want to choose a mask that has already been applied
                Efficiencies[j]=-10000
        
        #having checked each of the existing masks, we choose the one with the highest efficiency 
        maxEff=max(Efficiencies)
        maxIndex=Efficiencies.index(maxEff)
    
        
        #we find the relevant type, which we then use to load the file properly, form a new file, and save it
        if maxIndex>Numbers_For_Indexing: 
            TypeUsed="BRIGHT"
        else:
            TypeUsed="COMPLETE"
            
        CorrectMask=np.load(MasksFoldername+(MaskSuffix.replace("REPLACELINE",MaskLines[maxIndex])).replace("REPLACETYPE",TypeUsed))
        Currentname=StoreCOMapFoldername+((StoreCOMapSuffix.replace("REPLACENUMBER",str(i))).replace("REPLACELINE",MaskLines[maxIndex])).replace("REPLACETYPE",TypeUsed)
        #Keep all other values as the same
        np.savez(Currentname,
                SliceNumber=Mapfile["SliceNumber"],
                StartingRedshift=Mapfile["StartingRedshift"],
                EndingRedshift=Mapfile["EndingRedshift"],
                BeamSizeRad=Mapfile["BeamSizeRad"],
                RestFrequency=Mapfile["RestFrequency"],
                VariableLabel1=Mapfile["VariableLabel1"],
                VariableLabel2=Mapfile["VariableLabel2"],
                OriginalMapParams=Mapfile["OriginalMapParams"],
                ModelCoefficients=Mapfile["ModelCoefficients"],
                FluxMap=FluxCube*CorrectMask)
        
        #We then set up everything for next time - make the cumulative mask, save the percentage of masked voxels
        CumulativeMask=CumulativeMask*CorrectMask
        PercentageOfMaskedVoxelsIterative=(len(CorrectMask)*len(CorrectMask[0])*len(CorrectMask[0][0])-np.sum(CorrectMask))/(len(CorrectMask)*len(CorrectMask[0])*len(CorrectMask[0][0]))
        PercentageOfMaskedFoxelsCumulative=(len(CumulativeMask)*len(CumulativeMask[0])*len(CumulativeMask[0][0])-np.sum(CumulativeMask))/(len(CumulativeMask)*len(CumulativeMask[0])*len(CumulativeMask[0][0]))
        
        #we save these parameters
        MasterLines[i]=MaskLines[maxIndex]
        MasterTypes[i]=TypeUsed
        MasterEfficiencies[i]=maxEff
        MasterPercentageOfMaskedVoxelsIterative[i]=PercentageOfMaskedVoxelsIterative
        MasterPercentageOfMaskedVoxelsCumulative[i]=PercentageOfMaskedFoxelsCumulative
        
        
        #then we set the mask mask to ignore
        MaskLines[maxIndex]="IGNORE"
        
        #We make the spectra, taking into account the volume factor
        PSC.runPSC(cosmo,Currentname,
            Currentname.replace("_Map_","_PS_"),
            ["Three Bins Alt"], 
            AutoCII=True,
            FreqSpreadParameter=3,
            MapSpreadParameter=3,
            VolumeFactor=1-MasterPercentageOfMaskedVoxelsCumulative[i])
        
    #After applying all the masks, we save the master information in the "efficiency order" file
    np.savez(StoreCOMapFoldername+"EfficiencyOrder.npz",MaskLines=MasterLines,Type=MasterTypes,Efficiencies=MasterEfficiencies,
            PercentageOfMaskedVoxelsIterative=MasterPercentageOfMaskedVoxelsIterative,
            PercentageOfMaskedVoxelsCumulative=MasterPercentageOfMaskedVoxelsCumulative,allow_pickle=True)
    

    #removing the maps made, if too many/take up too much space
    if RemoveMaps:
        AllFiles=os.listdir(StoreCOMapFoldername)
        FilesMap=[]
        for AllFile in AllFiles:
            if "_Map_" in AllFile:
                FilesMap.append(AllFile)
        
        for FileMap in FilesMap:
            os.remove(StoreCOMapFoldername+FileMap)

                

def applyMasksCII_Total(cosmo,BaseFile,StoreMapSuffix,MaskSuffix,MasksFoldername,StoreMapFoldername,Maskfile, StartFrequency,RemoveMaps=True):
    """
    Once we have determined the optimal masking order (from above), we apply it to all other relevant cubes

    Parameters
    ----------
    cosmo : astropy cosmology object
        The cosmology we use for determining e.g. luminositiy distances
    BaseFile : string
        Name of the total map file we then apply the masks to
    StoreMapSuffix : string
        String of the file name we save to, with REPLACENUMBER, REPLACELINE, and REPLACETYPE as dummy identifiers to be replaced by the order/type of masking
    MaskSuffix : string
        Name of the mask, with REPLACELINE and REPLACETYPE as dummy identifiers as above
    MasksFoldername : string
        Foldername where the masks we used are saved to
    StoreMapFoldername : string
        Folder name that we store the maps and spectra into
    Maskfile : string
        Filename of optimal masking order
    StartFrequency : int
        frequency of the cube we are making the mask for
    RemoveMaps : boolean
        Maps can take up a lot of file space, 10s of MB, and we will generate hundreds of these. If this is enabled, we delete those and just leave the spectra. default is "true"


    Returns
    -------
    None.

    """
    
    #takethe map
    FluxCube=BaseFile['FluxMap']

    #take relevant parameters
    MaskingOrderLine=Maskfile['MaskLines']
    MaskingOrderType=Maskfile['Type']
    VoxelsMasked=    Maskfile['PercentageOfMaskedVoxelsCumulative']
    
    for i in range(len(MaskingOrderLine)):
        #from the stored info, get the mask label and make the new map from it
        if MaskingOrderType[i]=="COMPLETE":
            TypeUsed="Complete"
        elif MaskingOrderType[i]=="BRIGHT":
            TypeUsed="Bright"
        Mask=np.load(MasksFoldername+"GAUSSMASK_"+MaskingOrderLine[i]+"_"+StartFrequency+"GHz_"+MaskingOrderType[i]+"MASK_"+MaskSuffix+".npy")
        FluxCube=FluxCube*Mask #this is effectively cumulative
        
        #make the new name
        Currentname=StoreMapFoldername+((StoreMapSuffix.replace("REPLACENUMBER",str(i))).replace("REPLACELINE",MaskingOrderLine[i])).replace("REPLACETYPE",TypeUsed)
        
        #make maps and do spectra
        np.savez(Currentname,
                SliceNumber=BaseFile["SliceNumber"],
                StartingRedshift=BaseFile["StartingRedshift"],
                EndingRedshift=BaseFile["EndingRedshift"],
                BeamSizeRad=BaseFile["BeamSizeRad"],
                RestFrequency=BaseFile["RestFrequency"],
                VariableLabel1=BaseFile["VariableLabel1"],
                VariableLabel2=BaseFile["VariableLabel2"],
                OriginalMapParams=BaseFile["OriginalMapParams"],
                ModelCoefficients=BaseFile["ModelCoefficients"],
                FluxMap=FluxCube)
        
        PSC.runPSC(cosmo,Currentname,
            Currentname.replace("_Map_","_PS_"),
                    ["Three Bins Alt"], 
                    AutoCII=True,
                    FreqSpreadParameter=3,
                    MapSpreadParameter=3,
                    VolumeFactor=1-VoxelsMasked[i])
        
     
    #removing the maps made, if take up too much space, just to have spectra
    if RemoveMaps:
        AllFiles=os.listdir(StoreMapFoldername)
        FilesMap=[]
        for AllFile in AllFiles:
            if "_Map_" in AllFile:
                FilesMap.append(AllFile)
        
        for FileMap in FilesMap:
            os.remove(StoreMapFoldername+FileMap)
     
    #removing the maps made
    AllFiles=os.listdir(StoreMapFoldername)
    FilesMap=[]
    for AllFile in AllFiles:
        if "_Map_" in AllFile:
            FilesMap.append(AllFile)
    
    for FileMap in FilesMap:
        os.remove(StoreMapFoldername+FileMap)


