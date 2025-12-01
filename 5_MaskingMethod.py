import numpy as np
import astropy.units as u
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM
from astropy.convolution import convolve
import os
import PSAndCC_Creation as PSC
import CANDELSextrapolation as CE
import FluxMapCreation as FMC

cosmo= FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.73 * u.K, Om0=0.3)


    
    
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
    
    StartingRedshift=(RestFrequency/EndFrequency)-1
    EndingRedshift=(RestFrequency/StartFrequency)-1
    MapLength=int(round(OriginalMapParams[1]/PixelScalingFactor))
    TrueMapLength=MapLength*MapSpreadParameter
    TrueMapDepth=SliceNumber*FreqSpreadParameter
    
    MiniArr=np.ones((miniarrlen,miniarrlen,miniarrlen))
    midpoint=miniarrlen//2
    for i in range(len(MiniArr)):
        for j in range(len(MiniArr[0])):
            for k in range(len(MiniArr[0][0])):
                if ((i-midpoint)**2+(j-midpoint)**2+(k-midpoint)**2)**0.5<MapSpreadParameter*sigma/2.355:
                    MiniArr[i][j][k]=0
    
    
    OnesArray=np.ones((TrueMapDepth,TrueMapLength,TrueMapLength), dtype=np.float32)
    print(np.shape(OnesArray))
    
    ComplB=np.ones((TrueMapDepth,TrueMapLength,TrueMapLength), dtype=np.float32)
    ComplOneMinusB=np.zeros((TrueMapDepth,TrueMapLength,TrueMapLength), dtype=np.float32)
    
    BrightB=np.ones((TrueMapDepth,TrueMapLength,TrueMapLength), dtype=np.float32)
    BrightOneMinusB=np.zeros((TrueMapDepth,TrueMapLength,TrueMapLength), dtype=np.float32)
    
    GaussKern=FMC.gaussian_kernel_2d(MapSpreadParameter, 13)
    LorntKern=FMC.lorentzian_kernel_1d(FreqSpreadParameter, 13)
    
    for iiii in range(len(Samplefilenames)):
        Samplefilename=Samplefilenames[iiii]
        Data=CE.loadSample(Samplefilename)
        VariableList,_=FMC.fetchVariablesViaNamesCOSMOS(Data,StartingRedshift,EndingRedshift,OriginalMapParams,PixelScalingFactor,SliceNumber,RestFrequency,NameParams,MapSpreadParameter=MapSpreadParameter,FreqSpreadParameter=FreqSpreadParameter)
        xpix,ypix,zpix=VariableList["xpix"],VariableList["ypix"],VariableList["zpix"]
        log_mass=VariableList["log_mass"]
        log_SFR=VariableList["log_SFR"]
        Redshift=VariableList["Redshift"]
    
    

    
        #Maybe have sub-lims then? Mlim=5 up to 0.1, Mlim=6 up to 0.2, Mlim=7 to 0.3, MLim=7.5 to 0.4
        #OK, fine! We can try these out!
            
        
        z_bands_lower=np.arange(0,8.5,0.5) #redshift
        z_bands_upper=z_bands_lower+0.5
        z_bands_midpoints=(z_bands_upper+z_bands_lower)/2
        CompletenessArray=0-3.23e7*(1 + z_bands_midpoints) + 7.83e7*(1 + z_bands_midpoints)**2 #Msol
        


        print(CompletenessArray)
        
        ######################now, we make the mask
        MSArray=np.load(MainSequence)
        FoundVals=False
        for i in range(len(MSArray)):
            #print(MSArray[i][0],(StartingRedshift+EndingRedshift)/2)
            if MSArray[i][0]>(StartingRedshift+EndingRedshift)/2 and FoundVals==False:
                a,b=MSArray[i][3],MSArray[i][2]
                FoundVals=True
        if FoundVals==False:
            for i in range(len(MSArray)):
                #print(MSArray[i][0],(StartingRedshift+EndingRedshift)/2)
                if MSArray[i][0]>StartingRedshift and FoundVals==False:
                    a,b=MSArray[i][3],MSArray[i][2]
                    FoundVals=True
        print([a,b]) #check main sequence params
        
        
        IndexesComplete=[]
        IndexesBright=[]
        for i in range(len(log_mass)):
            if ypix[i]<TrueMapLength and xpix[i]<TrueMapLength and ypix[i]>=0 and xpix[i]>=0:
                for ii in range(len(z_bands_lower)):
                    if Redshift[i]>=z_bands_lower[ii] and  Redshift[i]<=z_bands_upper[ii]:
                        CompletenessThreshold=CompletenessArray[ii]
                if log_mass[i]>CompletenessThreshold:
                    dMS=log_SFR[i]-FMC.galMainSeq(log_mass[i],a,b)
                    IndexesComplete.append((int(zpix[i]),int(ypix[i]),int(xpix[i])))
                    if dMS>0:
                        IndexesBright.append((int(zpix[i]),int(ypix[i]),int(xpix[i])))
        #for complete
   
        
        for ii in tqdm(range(len(IndexesComplete))):            
            for i in range(len(MiniArr)):
                if i-midpoint+IndexesComplete[ii][0]>=0 and i-midpoint+IndexesComplete[ii][0]<=TrueMapDepth-1:
                    for j in range(len(MiniArr[0])):
                        if j-midpoint+IndexesComplete[ii][1]>=0 and j-midpoint+IndexesComplete[ii][1]<=TrueMapLength-1:
                            for k in range(len(MiniArr[0][0])):
                                if k-midpoint+IndexesComplete[ii][2]>=0 and k-midpoint+IndexesComplete[ii][2]<=TrueMapLength-1:
                                    if ComplB[i-midpoint+IndexesComplete[ii][0]][j-midpoint+IndexesComplete[ii][1]][k-midpoint+IndexesComplete[ii][2]]!=0:
                                        ComplB[i-midpoint+IndexesComplete[ii][0]][j-midpoint+IndexesComplete[ii][1]][k-midpoint+IndexesComplete[ii][2]]=MiniArr[i][j][k]
                                        ComplOneMinusB[i-midpoint+IndexesComplete[ii][0]][j-midpoint+IndexesComplete[ii][1]][k-midpoint+IndexesComplete[ii][2]]=1-MiniArr[i][j][k]
            
            
        #for bright
        for ii in tqdm(range(len(IndexesBright))):
            
            for i in range(len(MiniArr)):
                if i-midpoint+IndexesBright[ii][0]>=0 and i-midpoint+IndexesBright[ii][0]<=TrueMapDepth-1:
                    for j in range(len(MiniArr[0])):
                        if j-midpoint+IndexesBright[ii][1]>=0 and j-midpoint+IndexesBright[ii][1]<=TrueMapLength-1:
                            for k in range(len(MiniArr[0][0])):
                                if k-midpoint+IndexesBright[ii][2]>=0 and k-midpoint+IndexesBright[ii][2]<=TrueMapLength-1:
                                    if BrightB[i-midpoint+IndexesBright[ii][0]][j-midpoint+IndexesBright[ii][1]][k-midpoint+IndexesBright[ii][2]]!=0:
                                        #print([i-midpoint+IndexesBright[ii][0],j-midpoint+IndexesBright[ii][1],k-midpoint+IndexesBright[ii][2]])
                                        BrightB[i-midpoint+IndexesBright[ii][0]][j-midpoint+IndexesBright[ii][1]][k-midpoint+IndexesBright[ii][2]]=MiniArr[i][j][k]
                                        BrightOneMinusB[i-midpoint+IndexesBright[ii][0]][j-midpoint+IndexesBright[ii][1]][k-midpoint+IndexesBright[ii][2]]=1-MiniArr[i][j][k]
            
            
        
        
        
        
        
        
        
    for jj in tqdm(range(len(ComplOneMinusB))):
        ComplOneMinusB[jj]=convolve(ComplOneMinusB[jj], GaussKern)
    ComplOneMinusB = np.apply_along_axis(lambda spectrum: convolve(spectrum, LorntKern, boundary='extend'),
                                            axis=0, arr=ComplOneMinusB)
    
    print(np.max(ComplOneMinusB))
    
    ComplOneMinusB=ComplOneMinusB/np.max(ComplOneMinusB)
    
    ComplMaskArray=ComplB*(OnesArray-ComplOneMinusB) #after convolving

    
    
    for jj in tqdm(range(len(BrightOneMinusB))):
        BrightOneMinusB[jj]=convolve(BrightOneMinusB[jj], GaussKern)
    BrightOneMinusB = np.apply_along_axis(lambda spectrum: convolve(spectrum, LorntKern, boundary='extend'),
                                            axis=0, arr=BrightOneMinusB)
    print(np.max(BrightOneMinusB))
    BrightOneMinusB=BrightOneMinusB/np.max(BrightOneMinusB)
    

    BrightMaskArray=BrightB*(OnesArray-BrightOneMinusB)
    
    
    ComplMaskArray=np.where(ComplMaskArray<0,0,ComplMaskArray)
    ComplMaskArray=np.where(ComplMaskArray>1,1,ComplMaskArray)   
    
    BrightMaskArray=np.where(BrightMaskArray<0,0,BrightMaskArray)
    BrightMaskArray=np.where(BrightMaskArray>1,1,BrightMaskArray)   



    np.save(ComplMaskname, ComplMaskArray ,allow_pickle=True)
    np.save(BrightMaskname, BrightMaskArray ,allow_pickle=True)
    


def fetchSamplefile(StartFrequency,EndFrequency,RestFrequency,MaskType):
    """
    Not included for as includes details about internal file structure. In summary, this function calculates the redshift range of a given line, considering its rest and observed frequency.
    It then fetches the appropriate catalogue file(s), taking into account whether it is a "Base", "Stellar Mask", "CANDELS extrapolated", or "Mass extrapolated".
    

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


def makingMasks(FoldernameBase,FoldernameMask,FoldernameCANDELS,StartFrequency,sigma,MainSequence):
    """
    The function we run to make masks for all files

    Parameters
    ----------
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
    NameParams={"LIR":"LIR","log_mass":"logMass","log_SFR":"logSFR","Redshift":"z","OIII":"OIII","x_orig":"X","y_orig":"Y","FLAG":"FLAG","Redshift_err":"z68LL"}

        
    sigmalabel=str(sigma).replace(".",",")
 
    
    if StartFrequency==390: 
        COtransitions=[ "CO4_3","CO5_4",  "CO6_5", "CO7_6",  "CO8_7","CO9_8"]
        
        Transitions=[ 4,5, 6, 7, 8,9]
        PixelScalingFactor=33/0.15
        SliceNumber=7 
        FetchFreqMod=28
    elif StartFrequency==330:  
        COtransitions=[ "CO4_3","CO5_4",  "CO6_5", "CO7_6",  "CO8_7","CO9_8"]
        
        Transitions=[ 4,5, 6, 7, 8,9]
        PixelScalingFactor=round(37/0.15)
        SliceNumber=11
        FetchFreqMod=40
    elif StartFrequency==260: 
        COtransitions=[ "CO3_2","CO4_3","CO5_4",  "CO6_5","CO7_6",   "CO8_7","CO9_8"]
        
        Transitions=[ 3, 4,5,6,7, 8,9]
        PixelScalingFactor=48/0.15
        SliceNumber=14
        FetchFreqMod=40
    elif StartFrequency==205:  
        COtransitions=[ "CO3_2","CO4_3","CO5_4",  "CO6_5","CO7_6",   "CO8_7","CO9_8"]
        
        Transitions=[ 3, 4,5,6,7, 8,9]
        PixelScalingFactor=round(58/0.15)
        SliceNumber=18   
        FetchFreqMod=40
    
    
    EndFrequency=StartFrequency+FetchFreqMod
    
    
    
    COfreqs=115.27*np.asarray(Transitions)

    for i in range(len(COtransitions)):
        FirstMaskname_Base=FoldernameBase+"GAUSSMASK_"+COtransitions[i]+"_"+str(StartFrequency)+"GHz_COMPLETEMASK_BASE_sigma="+sigmalabel+".npy"
        SecondMaskname_Base=FoldernameBase+"GAUSSMASK_"+COtransitions[i]+"_"+str(StartFrequency)+"GHz_BRIGHTMASK_BASE_sigma="+sigmalabel+".npy"
        Basesamplefilename=fetchSamplefile(StartFrequency,EndFrequency,COfreqs[i],"Base")
        makeGaussxBinMasksCOSMOS(cosmo,Basesamplefilename,COfreqs[i],StartFrequency,EndFrequency,PixelScalingFactor,SliceNumber,sigma=sigma,miniarrlen=7,MainSequence=MainSequence ,FirstMaskname=FirstMaskname_Base,SecondMaskname=SecondMaskname_Base,MapSpreadParameter=3,FreqSpreadParameter=3,OriginalMapParams=[45000,29000,29000,7000,36000,7600,36600],NameParams=NameParams)

        FirstMaskname_SM=FoldernameMask+"GAUSSMASK_"+COtransitions[i]+"_"+str(StartFrequency)+"GHz_COMPLETEMASK_STELLARMASK_sigma="+sigmalabel+".npy"
        SecondMaskname_SM=FoldernameMask+"GAUSSMASK_"+COtransitions[i]+"_"+str(StartFrequency)+"GHz_BRIGHTMASK_STELLARMASK_sigma="+sigmalabel+".npy"
        Basesamplefilename=fetchSamplefile(StartFrequency,EndFrequency,COfreqs[i],"Mask")
        makeGaussxBinMasksCOSMOS(cosmo,Basesamplefilename,COfreqs[i],StartFrequency,EndFrequency,sigma=sigma,miniarrlen=7,MainSequence=MainSequence ,FirstMaskname=FirstMaskname_SM,SecondMaskname=SecondMaskname_SM,MapSpreadParameter=3,FreqSpreadParameter=3,OriginalMapParams=[45000,29000,29000,7000,36000,7600,36600],NameParams=NameParams)

        FirstMaskname_CANDELS=FoldernameCANDELS+"GAUSSMASK_"+COtransitions[i]+"_"+str(StartFrequency)+"GHz_COMPLETEMASK_CANDELS_sigma="+sigmalabel+".npy"
        SecondMaskname_CANDELS=FoldernameCANDELS+"GAUSSMASK_"+COtransitions[i]+"_"+str(StartFrequency)+"GHz_BRIGHTMASK_CANDELS_sigma="+sigmalabel+".npy"
        Basesamplefilename=fetchSamplefile(StartFrequency,EndFrequency,COfreqs[i],"CANDELS")
        makeGaussxBinMasksCOSMOS(cosmo,Basesamplefilename,COfreqs[i],StartFrequency,EndFrequency,sigma=sigma,miniarrlen=7,MainSequence=MainSequence ,FirstMaskname=FirstMaskname_CANDELS,SecondMaskname=SecondMaskname_CANDELS,MapSpreadParameter=3,FreqSpreadParameter=3,OriginalMapParams=[45000,29000,29000,7000,36000,7600,36600],NameParams=NameParams)

    


def sumMasks(FoldernameBase,FoldernameMask,FoldernameCANDELS,FoldernameBase_Mask,FoldernameBase_Mask_CANDELS):
    """
    The function where we convolve masks with each other from each of the smaller samples
    Sometimes we want to include the CANDELS galaxies, other times we do not

    Parameters
    ----------
    FoldernameBase/Mask/CANDELS/Base_Mask/Base_Mask_CANDELS : strings
        Folders where the masks are stored, the ones existing and the ones to be multiplied together
        The latter two are the standard mask, and the mask also including CANDELS galaxies

    Returns
    -------
    None.

    """
    Basefiles=os.listdir(FoldernameBase)
    Maskfiles=os.listdir(FoldernameMask)
    CANDELSfiles=os.listdir(FoldernameCANDELS)
    
    for i in range(len(Basefiles)):
        Prefix=Basefiles[i].replace("MASK_BASE","MASK_MASK")
        for j in range(len(Maskfiles)):
            if Prefix in Maskfiles[j]:
                Base=np.load(FoldernameBase+Basefiles[i])
                Mask=np.load(FoldernameMask+Maskfiles[j])
                Base_Maskfile=Basefiles[i].replace("MASK_BASE","MASK_COMPLETE")
                np.save(FoldernameBase_Mask+Base_Maskfile,Base*Mask)
                
                
                PrefixCANDELS=Prefix.replace("MASK_STELLARMASK","MASK_CANDELS")
                Base_Mask_CANDELSfile=Basefiles[i].replace("MASK_BASE","MASK_COMPLETE+COMPLETE")
                for k in range(len(CANDELSfiles)):
                    if PrefixCANDELS in CANDELSfiles[k]:
                        CANDELS=np.load(FoldernameCANDELS+CANDELSfiles[k]) #need to solve this

                        np.save(FoldernameBase_Mask_CANDELS+Base_Mask_CANDELSfile,Base*Mask*CANDELS)
                    

def applyMasksCO(InitialCOMapfile,StoreCOMapSuffix,MaskSuffix,MasksFoldername,StoreCOMapFoldername,StartFrequency):      
    """
    We take the masks derived from the above, and systematically apply them to the CO cubes to determine an "optimal masking order"
    This is then applied to other lines in turn

    Parameters
    ----------
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

    Returns
    -------
    None.

    """
    
    if StartFrequency=="390" or StartFrequency=="330":
        Lines=["CO4_3","CO5_4","CO6_5","CO7_6","CO8_7","CO9_8",
                "CO4_3","CO5_4","CO6_5","CO7_6","CO8_7","CO9_8",
                ]
        Numbers=[12,5]
    elif StartFrequency=="260" or StartFrequency=="205":
        Lines=["CO3_2","CO4_3","CO5_4","CO6_5","CO7_6","CO8_7","CO9_8",
                "CO3_2","CO4_3","CO5_4","CO6_5","CO7_6","CO8_7","CO9_8",
                ]
        Numbers=[14,6]
    
    
    
    
    Currentname=InitialCOMapfile
    MapSizeStuff=np.load(Currentname)
    CumulativeMask=np.ones(np.shape(MapSizeStuff['FluxMap'])) #use to determine how many voxels are masked
    
    
    MasterLines=[]
    MasterTypes=[]
    MasterEfficiencies=[]
    MasterPercentageOfMaskedVoxelsIterative=[]
    MasterPercentageOfMaskedVoxelsCumulative=[]
    for i in range(Numbers[0]): #22 or 20
    #for i in range(Numbers[1]+1,Numbers[0]):
        MapFile=np.load(Currentname)
        Map=MapFile['FluxMap']
        Efficiencies=[]
        for j in range(len(Lines)):
            if Lines[j]!="IGNORE":
                if j>Numbers[1]: #10 or 9
                    TypeUsed="BRIGHT"
                else:
                    TypeUsed="COMPLETE"
                
                #TypeUsed="BRIGHT"
                
                Mask=np.load(MasksFoldername+(MaskSuffix.replace("REPLACELINE",Lines[j])).replace("REPLACETYPE",TypeUsed)) * CumulativeMask #need to include previous masks
                NumMaskedVoxels=(len(Mask)*len(Mask[0])*len(Mask[0][0]))-np.sum(Mask) #this assumes number of masked voxels is "independant" of each other
                ConvolvedMap=Map*Mask
                Eff=(np.std(Map)-np.std(ConvolvedMap))/NumMaskedVoxels
                if Eff==0: #if 0 anyway, need a failsafe 
                    Efficiencies.append(-10)
                elif np.isnan(Eff):
                    Efficiencies.append(1e-10)
                else:
                    Efficiencies.append(Eff)
            else:
                Efficiencies.append(0)
        
        
        if np.nansum(Efficiencies)==0: #i.e. have some "blank" masks left, still need to run through
            for j in range(len(Efficiencies)):
                if np.isnan(Efficiencies[j]):
                    Efficiencies[j]=5 #forces them to run through
        
        
        
        NegCheck=True
        for ii in range(len(Efficiencies)):
            if Efficiencies[ii]>0:
                NegCheck=False
        if NegCheck:
            Efficiencies=list(np.abs(Efficiencies))
        
        maxEff=max(Efficiencies)
        maxIndex=Efficiencies.index(maxEff)
    
        
    
        if maxIndex>Numbers[1]: #10 or 9
            TypeUsed="BRIGHT"
        else:
            TypeUsed="COMPLETE"
            
            
        CorrectMask=np.load(MasksFoldername+(MaskSuffix.replace("REPLACELINE",Lines[maxIndex])).replace("REPLACETYPE",TypeUsed))
        Currentname=StoreCOMapFoldername+((StoreCOMapSuffix.replace("REPLACENUMBER",str(i))).replace("REPLACELINE",Lines[maxIndex])).replace("REPLACETYPE",TypeUsed)
    
        
        np.savez(Currentname,
                SliceNumber=MapFile["SliceNumber"],
                StartingRedshift=MapFile["StartingRedshift"],
                EndingRedshift=MapFile["EndingRedshift"],
                BeamSizeRad=MapFile["BeamSizeRad"],
                RestFrequency=MapFile["RestFrequency"],
                VariableLabel1=MapFile["VariableLabel1"],
                VariableLabel2=MapFile["VariableLabel2"],
                OriginalMapParams=MapFile["OriginalMapParams"],
                ModelCoefficients=MapFile["ModelCoefficients"],
                FluxMap=Map*CorrectMask)
        
        
  
        
        
        CumulativeMask=CumulativeMask*CorrectMask
        PercentageOfMaskedVoxelsIterative=(len(CorrectMask)*len(CorrectMask[0])*len(CorrectMask[0][0])-np.sum(CorrectMask))/(len(CorrectMask)*len(CorrectMask[0])*len(CorrectMask[0][0]))
        PercentageOfMaskedFoxelsCumulative=(len(CumulativeMask)*len(CumulativeMask[0])*len(CumulativeMask[0][0])-np.sum(CumulativeMask))/(len(CumulativeMask)*len(CumulativeMask[0])*len(CumulativeMask[0][0]))
        MasterLines.append(Lines[maxIndex])
        MasterTypes.append(TypeUsed)
        MasterEfficiencies.append(maxEff)
        
        
        MasterPercentageOfMaskedVoxelsIterative.append(PercentageOfMaskedVoxelsIterative)
        MasterPercentageOfMaskedVoxelsCumulative.append(PercentageOfMaskedFoxelsCumulative)
        
        
        

        
        Lines[maxIndex]="IGNORE"
        print(1-MasterPercentageOfMaskedVoxelsCumulative[-1])
        
        PSC.runPSC(cosmo,Currentname,
            Currentname.replace("_Map_","_PS_"),
            ["Three Bins Alt"], 
            AutoCII=True,
            FreqSpreadParameter=3,
            MapSpreadParameter=3,
            VolumeFactor=1-MasterPercentageOfMaskedVoxelsCumulative[-1])
        
        
        
       

    np.savez(StoreCOMapFoldername+"Both_Mask_EfficiencyOrder.npz",Lines=MasterLines,Type=MasterTypes,Efficiencies=MasterEfficiencies,
            PercentageOfMaskedVoxelsIterative=MasterPercentageOfMaskedVoxelsIterative,
            PercentageOfMaskedVoxelsCumulative=MasterPercentageOfMaskedVoxelsCumulative,allow_pickle=True)
    
    
    
    #removing the maps made
    
    AllFiles=os.listdir(StoreCOMapFoldername)
    FilesMap=[]
    for AllFile in AllFiles:
        if "_Map_" in AllFile:
            FilesMap.append(AllFile)
    
    for FileMap in FilesMap:
        os.remove(StoreCOMapFoldername+FileMap)

                

def applyMasksCII_Total(BaseFile,StoreMapSuffix,MaskSuffix,MasksFoldername,StoreMapFoldername,MaskFile, StartFrequency):
    """
    Once we have determined the optimal masking order (from above), we apply it to all other relevant cubes

    Parameters
    ----------
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
    MaskFile : string
        Filename of optimal masking order
    StartFrequency : int
        frequency of the cube we are making the mask for



    Returns
    -------
    None.

    """

    Map=BaseFile['FluxMap']

    
    MaskingOrderLine=MaskFile['Lines']
    MaskingOrderType=MaskFile['Type']
    VoxelsMasked=    MaskFile['PercentageOfMaskedVoxelsCumulative']
    
    for i in range(len(MaskingOrderLine)):
        if MaskingOrderType[i]=="COMPLETE":
            TypeUsed="Complete"
        elif MaskingOrderType[i]=="BRIGHT":
            TypeUsed="Bright"
        Mask=np.load(MasksFoldername+"GAUSSMASK_"+MaskingOrderLine[i]+"_"+StartFrequency+"GHz_"+MaskingOrderType[i]+"MASK_"+MaskSuffix+".npy")
        Map=Map*Mask
        

        Currentname=StoreMapFoldername+((StoreMapSuffix.replace("REPLACENUMBER",str(i))).replace("REPLACELINE",MaskingOrderLine[i])).replace("REPLACETYPE",TypeUsed)
        
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
                FluxMap=Map)
        
        PSC.runPSC(cosmo,Currentname,
            Currentname.replace("_Map_","_PS_"),
                    ["Three Bins Alt"], 
                    AutoCII=True,
                    FreqSpreadParameter=3,
                    MapSpreadParameter=3,
                    VolumeFactor=1-VoxelsMasked[i])
        
     
    #removing the maps made
    AllFiles=os.listdir(StoreMapFoldername)
    FilesMap=[]
    for AllFile in AllFiles:
        if "_Map_" in AllFile:
            FilesMap.append(AllFile)
    
    for FileMap in FilesMap:
        os.remove(StoreMapFoldername+FileMap)


