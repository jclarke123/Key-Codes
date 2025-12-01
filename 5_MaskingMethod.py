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


    
    
def makeGaussxBinMasksCOSMOS(cosmo,Samplefilenames,RestFrequency,StartFrequency,EndFrequency,PixelScalingFactor,SliceNumber,sigma=1,miniarrlen=7,MainSequence="" ,FirstMaskname="",SecondMaskname="",MapSpreadParameter=3,FreqSpreadParameter=3,OriginalMapParams=[45000,29000,29000,7000,36000,7600,36600],NameParams="",CompleteType="",ActualBeamSize=True):
    """
    

    Parameters
    ----------
    cosmo : TYPE
        DESCRIPTION.
    Samplefilenames : TYPE
        DESCRIPTION.
    RestFrequency : TYPE
        DESCRIPTION.
    StartFrequency : TYPE
        DESCRIPTION.
    EndFrequency : TYPE
        DESCRIPTION.
    sigma : TYPE, optional
        DESCRIPTION. The default is 1.
    miniarrlen : TYPE, optional
        DESCRIPTION. The default is 7.
    MainSequence : TYPE, optional
        DESCRIPTION. The default is "".
    FirstMaskname : TYPE, optional
        DESCRIPTION. The default is "".
    SecondMaskname : TYPE, optional
        DESCRIPTION. The default is "".
    MapSpreadParameter : TYPE, optional
        DESCRIPTION. The default is 3.
    FreqSpreadParameter : TYPE, optional
        DESCRIPTION. The default is 3.
    OriginalMapParams : TYPE, optional
        DESCRIPTION. The default is [45000,29000,29000,7000,36000,7600,36600].
    NameParams : TYPE, optional
        DESCRIPTION. The default is "".
    CompleteType : TYPE, optional
        DESCRIPTION. The default is "".
    ActualBeamSize : TYPE, optional
        DESCRIPTION. The default is True.

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



    np.save(FirstMaskname, ComplMaskArray ,allow_pickle=True)
    np.save(SecondMaskname, BrightMaskArray ,allow_pickle=True)
    


#TODO

def fetchSamplefile(StartFrequency,EndFrequency,LineFreq,MaskType):
    """
    Not included for as includes details about internal file structure. In summary, this function calculates the redshift range of a given line, considering its rest and observed frequency.
    It then fetches the appropriate catalogue file(s), taking into account whether it is a "Base", "Stellar Mask", "CANDELS extrapolated", or "Mass extrapolated".

    Parameters
    ----------
    StartFrequency : TYPE
        DESCRIPTION.
    EndFrequency : TYPE
        DESCRIPTION.
    LineFreq : TYPE
        DESCRIPTION.
    MaskType : TYPE
        DESCRIPTION.

    Returns
    -------
    Files : TYPE
        DESCRIPTION.

    """
    Files=""
    
    return Files


def makingMasks(SavenameBase,SavenameMask,SavenameCANDELS,StartFrequency,sigma,MainSequence):
    """
    

    Parameters
    ----------
    SavenameBase : TYPE
        DESCRIPTION.
    SavenameMask : TYPE
        DESCRIPTION.
    SavenameCANDELS : TYPE
        DESCRIPTION.
    StartFrequency : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    sigmalabel : TYPE
        DESCRIPTION.
    MainSequence : TYPE
        DESCRIPTION.
    
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
        SliceNumber=7 #this for no spread in 3rd dimension
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
        FirstMaskname_Base=SavenameBase+"GAUSSMASK_"+COtransitions[i]+"_"+str(StartFrequency)+"GHz_COMPLETEMASK_BASE_sigma="+sigmalabel+".npy"
        SecondMaskname_Base=SavenameBase+"GAUSSMASK_"+COtransitions[i]+"_"+str(StartFrequency)+"GHz_BRIGHTMASK_BASE_sigma="+sigmalabel+".npy"
        Basesamplefilename=fetchSamplefile(StartFrequency,EndFrequency,COfreqs[i],"Base")
        makeGaussxBinMasksCOSMOS(cosmo,Basesamplefilename,COfreqs[i],StartFrequency,EndFrequency,PixelScalingFactor,SliceNumber,sigma=sigma,miniarrlen=7,MainSequence=MainSequence ,FirstMaskname=FirstMaskname_Base,SecondMaskname=SecondMaskname_Base,MapSpreadParameter=3,FreqSpreadParameter=3,OriginalMapParams=[45000,29000,29000,7000,36000,7600,36600],NameParams=NameParams,ActualBeamSize=True)

        FirstMaskname_SM=SavenameMask+"GAUSSMASK_"+COtransitions[i]+"_"+str(StartFrequency)+"GHz_COMPLETEMASK_STELLARMASK_sigma="+sigmalabel+".npy"
        SecondMaskname_SM=SavenameMask+"GAUSSMASK_"+COtransitions[i]+"_"+str(StartFrequency)+"GHz_BRIGHTMASK_STELLARMASK_sigma="+sigmalabel+".npy"
        Basesamplefilename=fetchSamplefile(StartFrequency,EndFrequency,COfreqs[i],"Mask")
        makeGaussxBinMasksCOSMOS(cosmo,Basesamplefilename,COfreqs[i],StartFrequency,EndFrequency,sigma=sigma,miniarrlen=7,MainSequence=MainSequence ,FirstMaskname=FirstMaskname_SM,SecondMaskname=SecondMaskname_SM,MapSpreadParameter=3,FreqSpreadParameter=3,OriginalMapParams=[45000,29000,29000,7000,36000,7600,36600],NameParams=NameParams,ActualBeamSize=True)

        FirstMaskname_CANDELS=SavenameCANDELS+"GAUSSMASK_"+COtransitions[i]+"_"+str(StartFrequency)+"GHz_COMPLETEMASK_CANDELS_sigma="+sigmalabel+".npy"
        SecondMaskname_CANDELS=SavenameCANDELS+"GAUSSMASK_"+COtransitions[i]+"_"+str(StartFrequency)+"GHz_BRIGHTMASK_CANDELS_sigma="+sigmalabel+".npy"
        Basesamplefilename=fetchSamplefile(StartFrequency,EndFrequency,COfreqs[i],"CANDELS")
        makeGaussxBinMasksCOSMOS(cosmo,Basesamplefilename,COfreqs[i],StartFrequency,EndFrequency,sigma=sigma,miniarrlen=7,MainSequence=MainSequence ,FirstMaskname=FirstMaskname_CANDELS,SecondMaskname=SecondMaskname_CANDELS,MapSpreadParameter=3,FreqSpreadParameter=3,OriginalMapParams=[45000,29000,29000,7000,36000,7600,36600],NameParams=NameParams,ActualBeamSize=True)

    


def sumMasks(BaseFolder,MaskFolder,CANDELSFolder,Base_MaskFolder,Base_Mask_CANDELSFolder):
    """


    Parameters
    ----------
    BaseFolder : TYPE
        DESCRIPTION.
    MaskFolder : TYPE
        DESCRIPTION.
    CANDELSFolder : TYPE
        DESCRIPTION.
    Base_MaskFolder : TYPE
        DESCRIPTION.
    Base_Mask_CANDELSFolder : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    Basefiles=os.listdir(BaseFolder)
    Maskfiles=os.listdir(MaskFolder)
    CANDELSfiles=os.listdir(CANDELSFolder)
    
    for i in range(len(Basefiles)):
        Prefix=Basefiles[i].replace("MASK_BASE","MASK_MASK")
        for j in range(len(Maskfiles)):
            if Prefix in Maskfiles[j]:
                Base=np.load(BaseFolder+Basefiles[i])
                Mask=np.load(MaskFolder+Maskfiles[j])
                Base_Maskfile=Basefiles[i].replace("MASK_BASE","MASK_COMPLETE")
                np.save(Base_MaskFolder+Base_Maskfile,Base*Mask)
                
                
                PrefixCANDELS=Prefix.replace("MASK_STELLARMASK","MASK_CANDELS")
                Base_Mask_CANDELSfile=Basefiles[i].replace("MASK_BASE","MASK_COMPLETE+COMPLETE")
                for k in range(len(CANDELSfiles)):
                    if PrefixCANDELS in CANDELSfiles[k]:
                        CANDELS=np.load(Base_Mask_CANDELSFolder+CANDELSfiles[k]) #need to solve this

                        np.save(Base_Mask_CANDELSFolder+Base_Mask_CANDELSfile,Base*Mask*CANDELS)
                    
########################################these are for the main 4 ccat samples - 390 to 205GHz. Not touching the lower ones right now
########################################also, these are interactive, with or without extrap. NO WN, THAT'S SEPARATE
#TODO

# for kk in range(len(Sigmas)):
#     for Freq in FreqS:
def applyMasksCO(InitialCOMap,StoreCOMapSuffix,MaskSuffix,MasksFoldername,StoreCOMapFoldername,Freq):      
    """
    

    Parameters
    ----------
    InitialCOMap : TYPE
        DESCRIPTION.
    StoreCOMapSuffix : TYPE
        DESCRIPTION.
    MaskSuffix : TYPE
        DESCRIPTION.
    MasksFoldername : TYPE
        DESCRIPTION.
    StoreCOMapFoldername : TYPE
        DESCRIPTION.
    Freq : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    if Freq=="390" or Freq=="330":
        Lines=["CO4_3","CO5_4","CO6_5","CO7_6","CO8_7","CO9_8",
                "CO4_3","CO5_4","CO6_5","CO7_6","CO8_7","CO9_8",
                ]
        Numbers=[12,5]
    elif Freq=="260" or Freq=="205":
        Lines=["CO3_2","CO4_3","CO5_4","CO6_5","CO7_6","CO8_7","CO9_8",
                "CO3_2","CO4_3","CO5_4","CO6_5","CO7_6","CO8_7","CO9_8",
                ]
        Numbers=[14,6]
    
    
    
    
    Currentname=InitialCOMap
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
        print(Lines)
        print(Efficiencies)
        
        maxEff=max(Efficiencies)
        maxIndex=Efficiencies.index(maxEff)
    
        
        
        print(maxIndex)
        
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

                

# ####TODO

def applyMasksCII_Total(BaseFile,StoreMapSuffix,MaskSuffix,MasksFoldername,StoreMapFoldername,MaskFile, Freq):
    """
    

    Parameters
    ----------
    BaseFile : TYPE
        DESCRIPTION.
    StoreMapSuffix : TYPE
        DESCRIPTION.
    MaskSuffix : TYPE
        DESCRIPTION.
    MasksFoldername : TYPE
        DESCRIPTION.
    StoreMapFoldername : TYPE
        DESCRIPTION.
    MaskFile : TYPE
        DESCRIPTION.
    Freq : TYPE
        DESCRIPTION.

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
        Mask=np.load(MasksFoldername+"GAUSSMASK_"+MaskingOrderLine[i]+"_"+Freq+"GHz_"+MaskingOrderType[i]+"MASK_"+MaskSuffix+".npy")
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


