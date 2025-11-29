#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code involves loading in the existing flux maps to be used in creating power spectra
A 2D or 3D power spectrum can be made from this. the 2D verson is 'converted' to 3D by dividing through by the comoving distance between the ends of the redshift bin
The 2D power spectrum uses my own code. The 3D power spectrum uses code from Chung et al, found here https://github.com/georgestein/limlam_mocker/blob/missy/limlam_mocker/map_to_pspec.py 
(Please credit them if you use this code!)


NOTE: FOR ERROR PROPOGATION, Pn IS HARDCODED FOR FYST/CCAT - YOU WILL HAVE TO CHANGE!

"""

import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
import scipy.ndimage as nd
import astropy.units as u

###Loads in the file, and unpacks the metadata from the flux map. This is all passed into the files
def loadFile(Loadname):
    arr=np.load(Loadname,allow_pickle='True')
    OtherProp,RedshiftProp,FluxMap=arr[0],arr[1],arr[2]
    return OtherProp,RedshiftProp,FluxMap

#we need to remove nans from our final values - do for both arrays!
def removeNans(k,PS):
    nk,nPS=[],[]
    for i in range(len(PS)):
        if not np.isnan(PS[i]):
            nk.append(k[i])
            nPS.append(PS[i])
    return nk,nPS

#the exponential function we integrate for the errors
def expf(x,k,kparmax,sigmaperp,sigmapar):
    return np.exp((sigmaperp**2-sigmapar**2)*(np.min([kparmax,x*k]))**2)

#This function is required to get in the form (x,y,z), instead of (z,x,y) which FluxMapCreation leaves us in, for a 3D map! If you don't need this, either remove from below,
#or apply this function beforehand, as create3DPowerSpectrum does it anyway!
def TrueReshape(FluxMap):

    NewFluxMap=np.flip(np.rot90(FluxMap,axes=(2,0)),axis=2)
    return NewFluxMap
#TODO
#Most of the 3D power spectrum code is very similar to the 2D code, however there are notable differences, such as the 3D k map creation, and the k maximum having another candidate
#One key difference, is that a) the "map" portion MUST be a square for 3D, and b) it must be in the format (z,x,y), as done in FluxMapCreation

def create3DPowerSpectrum(cosmo,kValParameter,FluxCube,MeanRedshift,StartingRedshift,EndingRedshift,PixelLengthMpc,PixelLengthArcsec,IndivLineFreq,
                          MapSpreadParameter,FreqSpreadParameter,VolumeFactor=1): #volumefactor is used if some pixels masked, need to be removed
    print("VolFactor: "+str(VolumeFactor))
    #a lot is similar to above, but we do make some adjustments
    #we now have a z direction! This tells us about how many slices we have
    
    #note that the format we get the map is the wrong way around (we want x,y,z; not z,x,y)! We need to reshape to get the axes in order
    
    print(np.shape(FluxCube))
    VoxelXNum,VoxelYNum,VoxelZNum=len(FluxCube),len(FluxCube[0]),len(FluxCube[0][0]) 
    ComovingZLength=cosmo.comoving_distance(EndingRedshift).value-cosmo.comoving_distance(StartingRedshift).value
    ComovingMapLength=max([VoxelXNum,VoxelYNum])*PixelLengthMpc
    VolOfVoxel=(PixelLengthMpc**2)*(ComovingZLength/VoxelZNum)
    spatialkscale=2*np.pi/ComovingMapLength
    redshiftkscale=2*np.pi/ComovingZLength 

    
    #NEED TO BE CAREFUL WITH THIS! THIS IS NOT Vol of voxel/number of voxels, this is volume of individual element/number of elements. Hence include MSP, FSP
    #Where an element is defined as beamFWHM*beamFWHM*freq slice width. With no spread, I don't need a factor. If spread, I need to undo
    #Have a **2 param for each factor changed. Map spread is **2**2 because of X and Y
    
    ####matter of taste
    VolOfDiscreteElementDivNumElements=(1/VolumeFactor)**2 *(VolOfVoxel/(VoxelXNum*VoxelYNum*VoxelZNum)) ###vol factor **2
    fouriernrm  = VolOfDiscreteElementDivNumElements * np.abs(np.fft.fftshift(np.fft.fftn(FluxCube)))**2 
    

    
    kx        = np.fft.fftshift(np.fft.fftfreq(VoxelXNum,d=1/VoxelXNum)) 
    ky        = np.fft.fftshift(np.fft.fftfreq(VoxelYNum,d=1/VoxelYNum)) 
    kz        = np.fft.fftshift(np.fft.fftfreq(VoxelZNum,d=1/VoxelZNum))

    
    unitlessgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(kx,ky,kz,indexing='ij')))
    unitlessgrid = unitlessgrid.astype(int)
    kgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(spatialkscale*kx,spatialkscale*ky,redshiftkscale*kz,indexing='ij')))
    totalampbins = np.bincount(unitlessgrid.flatten(), fouriernrm.flatten()) 
    totalkbins = np.bincount(unitlessgrid.flatten(), kgrid.flatten()) 
    numforbin = np.bincount(unitlessgrid.flatten())
    NormFTProfile = totalampbins / numforbin
    NormkProfile = totalkbins / numforbin#
    

    #we have an extra dimension to watch out for when setting the max and min limits
    kparmin=2*np.pi/   ComovingZLength
    kperpmin=2*np.pi/  ComovingMapLength 
    kcross=2*np.pi/  (ComovingMapLength**2+ComovingZLength**2)**0.5    #should actually be (comovin map Length**2)*2, as the face size adds root 2 **2

    kparmax=np.pi/  (ComovingZLength/VoxelZNum)/FreqSpreadParameter
    kperpmax=np.pi/   PixelLengthMpc /MapSpreadParameter
    

    #when using spacing, we add two more options. The "three bins" option gives bins the size of 0.3 (name is weird). 10x typical defaults, use as wide bin examples
    if kValParameter[0]=='Linspace':
        kbins=np.linspace(min([kparmin,kperpmin,kcross]),min([kperpmax,max(NormkProfile)]),kValParameter[1])
    elif kValParameter[0]=='Logspace':
        kb=np.linspace(min([np.log10(kparmin),np.log10(kperpmin),np.log10(kcross)]),min([np.log10(kperpmax),max(np.log10(NormkProfile))]),kValParameter[1])
        kbins=np.power(10,kb)
    elif kValParameter[0]=='Fixed':
        kbins=np.linspace(kValParameter[2],kValParameter[3],kValParameter[1])
    elif kValParameter[0]=='Three Bins': #these are if we want to use winder bins. The former goes to lower k models
        kbins=np.asarray([0,0.3,0.6,0.9,1.2,1.5,1.8,2.1])-0.1
    elif kValParameter[0]=='Three Bins Alt': #while the latter has centering around 0
        kbins=np.asarray([0,0.3,0.6,0.9,1.2,1.5,1.8,2.1])-0.05
    elif kValParameter[0]=='Final Bins': #while the latter has centering around 0
        kbins=np.asarray([0.02,0.32,0.62,0.92,1.22,1.52,1.82,2.12])

    PS,bin_edges,binnumber=stats.binned_statistic(NormkProfile,(NormkProfile**3)*NormFTProfile/(2*((np.pi)**2)), statistic='mean', bins=kbins)
    

    dK=-bin_edges[:-1]+bin_edges[1:]
    k=bin_edges[1:]-(dK/2.)


    k,PS=removeNans(k,PS)
    k=np.asarray(k)

    ##########these errors
    StartingFrequency=IndivLineFreq/(1+EndingRedshift)
    if    StartingFrequency<400 and StartingFrequency>380: PnFactor=((2.2e4)**2)  #similar, to turn to volume to beam
    elif  StartingFrequency<340 and StartingFrequency>320: PnFactor=((1.2e4)**2)
    elif  StartingFrequency<270 and StartingFrequency>250: PnFactor=((6.2e3)**2)
    elif  StartingFrequency<215 and StartingFrequency>195: PnFactor=((3.9e3)**2)
    else: PnFactor=((1e4)**2)*VolOfVoxel  #just an exception we can handle. Error will be off, but oh well
    PnDEFAULT=PnFactor*VolOfVoxel * (MapSpreadParameter**2)
    
    ###For auto-spectra, can estimate sensitivity for the DEFAULT noise parameters in this way. You will need to calculated
    nummodesDEFAULT=[]
    WDEFAULT=[]
    #as now 3D, we use the volume of the map
    VolMap=VolOfVoxel*VoxelXNum*VoxelYNum*VoxelZNum *VolumeFactor
    #I think this set of errors is the correct version
    sigfactor=2.355
    sigmaperp=(cosmo.kpc_comoving_per_arcmin(MeanRedshift).value*PixelLengthArcsec/(60*1000))/sigfactor #first bit is in Mpc/arcseconds, second bit is beam FWHM in arcsecs
    sigmapar=(3*10**5)/(cosmo.H(MeanRedshift).value)*((1+MeanRedshift)*(40/VoxelZNum))/(sigfactor*IndivLineFreq/(1+MeanRedshift))
    
    
    for i in range(len(k)):  
        #here, we just use as described, no need to modify. Karoumpis reminds us that this does get weird at high k, so we need a slight correction in those cases
        #also be careful if bins not same size, e.g. logspace
        nummodesDEFAULT.append(np.min([kparmax,k[i]])*k[i]*(k[1]-k[0])*(VolMap)/4/np.pi/np.pi) 
        WDEFAULT.append(np.exp(0-(sigmaperp*k[i])**2)*quad(expf,0,1,args=(k[i],kparmax,sigmaperp,sigmapar))[0])

    return k, PS,PnDEFAULT,nummodesDEFAULT,WDEFAULT



def create3DCrossSpectrum(cosmo,kValParameter,FluxCube1,MeanRedshift1,StartingRedshift1,EndingRedshift1,PixelLengthMpc1,
                          FluxCube2,MeanRedshift2,StartingRedshift2,EndingRedshift2,PixelLengthMpc2,VolUsed="Average",ZoomDim=(20,200,200),
                          MSP=3,FSP=3): 
    #we do the first step seperately for both cubes
    ##FluxCube1=   RescaleArrayGauss(FluxCube1) #WE NEED TO RESCALE TO STANDARD 20x200x200 now
    PixelLengthMpc1=PixelLengthMpc1/(ZoomDim[1]/len(FluxCube1[0]))
    #print(np.sum(FluxCube1),np.shape(FluxCube1))
    FluxCube1=nd.zoom(FluxCube1,(ZoomDim[0]/len(FluxCube1),ZoomDim[1]/len(FluxCube1[0]),ZoomDim[2]/len(FluxCube1[0][0])))
    #print(np.sum(FluxCube1),np.shape(FluxCube1))
    VoxelXNum1,VoxelYNum1,VoxelZNum1=len(FluxCube1[0][0]),len(FluxCube1[0]),len(FluxCube1) 
    FluxCube1=TrueReshape(FluxCube1) 
    
    
    ComovingZLength1=cosmo.comoving_distance(EndingRedshift1).value-cosmo.comoving_distance(StartingRedshift1).value
    ComovingMapLength1=max([VoxelXNum1,VoxelYNum1])*PixelLengthMpc1
    VolOfVoxel1=(PixelLengthMpc1**2)*(ComovingZLength1/VoxelZNum1)
    VolOfDiscreteElementDivNumElements1=(VolOfVoxel1/(VoxelXNum1*VoxelYNum1*VoxelZNum1))
    

    #go again
    PixelLengthMpc2=PixelLengthMpc2/(ZoomDim[1]/len(FluxCube2[0]))
    #print(np.sum(FluxCube2),np.shape(FluxCube2))
    FluxCube2=nd.zoom(FluxCube2,(ZoomDim[0]/len(FluxCube2),ZoomDim[1]/len(FluxCube2[0]),ZoomDim[2]/len(FluxCube2[0][0])))
    #print(np.sum(FluxCube2),np.shape(FluxCube2))
    VoxelXNum2,VoxelYNum2,VoxelZNum2=len(FluxCube2[0][0]),len(FluxCube2[0]),len(FluxCube2) 
    FluxCube2=TrueReshape(FluxCube2) 
    
    ComovingZLength2=cosmo.comoving_distance(EndingRedshift2).value-cosmo.comoving_distance(StartingRedshift2).value
    ComovingMapLength2=max([VoxelXNum2,VoxelYNum2])*PixelLengthMpc2
    VolOfVoxel2=(PixelLengthMpc2**2)*(ComovingZLength2/VoxelZNum2)
    VolOfDiscreteElementDivNumElements2=(VolOfVoxel2/(VoxelXNum2*VoxelYNum2*VoxelZNum2))


    #then average of the physical scales
    #NEED TO BE CAREFUL WITH THIS! THIS IS NOT Vol of voxel/number of voxels, this is number of elements/number of elements
    #Where an element is defined as beamFWHM*beamFWHM*freq slice width. With no spread, I don't need a factor. If spread, I need to undo
    if VolUsed=="Average":
        spatialkscaleUsed=4*np.pi/(ComovingMapLength1+ComovingMapLength2)
        redshiftkscaleUsed=4*np.pi/(ComovingZLength1+ComovingZLength2) 
        VolOfDiscreteElementDivNumElementsUsed=(VolOfDiscreteElementDivNumElements1+VolOfDiscreteElementDivNumElements2)/2
        VoxelXNumUsed,VoxelYNumUsed,VoxelZNumUsed=int((VoxelXNum1+VoxelXNum2)/2),int((VoxelYNum1+VoxelYNum2)/2),int((VoxelZNum1+VoxelZNum2)/2)
    elif VolUsed=="First":
        spatialkscaleUsed=2*np.pi/(ComovingMapLength1)
        redshiftkscaleUsed=2*np.pi/(ComovingZLength1) 
        VolOfDiscreteElementDivNumElementsUsed=VolOfDiscreteElementDivNumElements1
        VoxelXNumUsed,VoxelYNumUsed,VoxelZNumUsed=VoxelXNum1,VoxelYNum1,VoxelZNum1
    elif VolUsed=="Second":
        spatialkscaleUsed=2*np.pi/(ComovingMapLength2)
        redshiftkscaleUsed=2*np.pi/(ComovingZLength2) 
        VolOfDiscreteElementDivNumElementsUsed=VolOfDiscreteElementDivNumElements2
        VoxelXNumUsed,VoxelYNumUsed,VoxelZNumUsed=VoxelXNum2,VoxelYNum2,VoxelZNum2

    #print(VolOfDiscreteElementDivNumElementsUsed)
    ##then we multiply the conj of one by the original of the other. We bin as normal
    #OLD
    fouriernrm  = VolOfDiscreteElementDivNumElementsUsed*np.real(np.fft.fftshift(np.fft.fftn(FluxCube1))*np.conj(np.fft.fftshift(np.fft.fftn(FluxCube2)))) #cross correlation
    #WE ACTUALLY AVERAGE BOTH ORDERS
    #fouriernrm  = VolOfDiscreteElementDivNumElementsUsed*(np.real(np.fft.fftshift(np.fft.fftn(FluxCube1))*np.conj(np.fft.fftshift(np.fft.fftn(FluxCube2))))+np.real(np.fft.fftshift(np.fft.fftn(FluxCube2))*np.conj(np.fft.fftshift(np.fft.fftn(FluxCube1)))))/2
    

    kx        = np.fft.fftshift(np.fft.fftfreq(VoxelXNumUsed,d=1/VoxelXNumUsed)) 
    ky        = np.fft.fftshift(np.fft.fftfreq(VoxelYNumUsed,d=1/VoxelYNumUsed)) 
    kz        = np.fft.fftshift(np.fft.fftfreq(VoxelZNumUsed,d=1/VoxelZNumUsed))

    unitlessgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(kx,ky,kz,indexing='ij')))
    unitlessgrid = unitlessgrid.astype(int)
    kgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(spatialkscaleUsed*kx,spatialkscaleUsed*ky,redshiftkscaleUsed*kz,indexing='ij')))
    totalampbins = np.bincount(unitlessgrid.flatten(), fouriernrm.flatten()) 
    totalkbins = np.bincount(unitlessgrid.flatten(), kgrid.flatten()) 
    numforbin = np.bincount(unitlessgrid.flatten())
    NormFTProfile = totalampbins / numforbin
    NormkProfile = totalkbins / numforbin
    
    #we have an extra dimension to watch out for when setting the max and min limits
    kparmin= redshiftkscaleUsed
    kperpmin= spatialkscaleUsed
    
    ####the average for these values, need to redo
    kcross=4*np.pi/  ((ComovingMapLength1+ComovingMapLength2)**2+(ComovingZLength1+ComovingZLength2)**2)**0.5    
    kparmax=2*np.pi/  ((ComovingZLength1+ComovingZLength2)/VoxelZNumUsed)/FSP
    kperpmax=2*np.pi/   (PixelLengthMpc1+PixelLengthMpc2)/MSP
    
    #when using spacing, we add two more options. The "three bins" option gives bins the size of 0.3 (name is weird). 10x typical defaults, use as wide bin examples
    if kValParameter[0]=='Linspace':
        kbins=np.linspace(min([kparmin,kperpmin,kcross]),min([kperpmax,max(NormkProfile)]),kValParameter[1])
    elif kValParameter[0]=='Logspace':
        kb=np.linspace(min([np.log10(kparmin),np.log10(kperpmin),np.log10(kcross)]),min([np.log10(kperpmax),max(np.log10(NormkProfile))]),kValParameter[1])
        kbins=np.power(10,kb)
    elif kValParameter[0]=='Fixed':
        kbins=np.linspace(kValParameter[2],kValParameter[3],kValParameter[1])
    elif kValParameter[0]=='Three Bins': #these are if we want to use winder bins. The former goes to lower k models
        kbins=np.asarray([0,0.3,0.6,0.9,1.2,1.5,1.8,2.1])-0.1
    elif kValParameter[0]=='Three Bins Alt': #while the latter has centering around 0
        kbins=np.asarray([0,0.3,0.6,0.9,1.2,1.5,1.8,2.1])-0.05

    CCPS,bin_edges,binnumber=stats.binned_statistic(NormkProfile,(NormkProfile**3)*NormFTProfile/(2*((np.pi)**2)), statistic='mean', bins=kbins)
    dK=-bin_edges[:-1]+bin_edges[1:]
    k=bin_edges[1:]-(dK/2.)
    k,CCPS=removeNans(k,CCPS)
    k=np.asarray(k)
    CCPS=np.asarray(CCPS)
    return k, CCPS
















#TODO
###The thing we run. We need the filename to load from, plus a name to save to
#Is fairly self-explanatory. Main thing is dA- distance covered by 1 arcsecond. Multiplying this by PixelLength gives the pixel length in Mpc
#accounting for any spread of course
#Also have "AutoCII" - that is, using the scales of CII (that is, the redshifts) - in cases where we want to stay to a fixed volume scale. Normally false, may need to use when stacking stuff, etc
def runPSC(cosmo,Mapname,PSname,kValParameters,AutoCII=True,VolumeFactor=1,MapSpreadParameter=1,FreqSpreadParameter=1): 
    #The introduction bit is a bit off. Basically, to maintain compatability with previous work, we need to have a whole bunch of different options - e.g. do we use default redshiftprop, or import one in?
    #print(Mapname)
    Mapdata=np.load(Mapname,allow_pickle=True)
    #some old files didn't have signal tracer - for those without, how do we ensure we have signal
    #print(Mapdata.files)
    FluxMap, StartingRedshift, EndingRedshift=Mapdata['FluxMap'],Mapdata['StartingRedshift'],Mapdata['EndingRedshift']
    RestFrequency=Mapdata['RestFrequency']
    print(np.shape(FluxMap))
    
    
    #by default, we normalise all power spectra by the volume covered by [CII]
    if AutoCII:
        print("AUTOCII")
        
        #we update the parameters to fit COSMOS
        StartingRedshift=(1900.537*(1+StartingRedshift)/RestFrequency)-1
        EndingRedshift=(1900.537*(1+EndingRedshift)/RestFrequency)-1
        RestFrequency=1900.537 
        

        
    MeanRedshift=(StartingRedshift+EndingRedshift)/2
    dA=cosmo.kpc_comoving_per_arcmin(MeanRedshift).value/60/1000
    
    BeamSizeArcsec=((Mapdata["BeamSizeRad"]*u.rad).to(u.arcsecond)).value
    
    PixelLengthArcsec=BeamSizeArcsec/MapSpreadParameter #length of pixels in arcseconds
    PixelLengthMpc=PixelLengthArcsec*dA
    k,k3Pk,PnDEFAULT,nummodesDEFAULT,WDEFAULT=create3DPowerSpectrum(cosmo,kValParameters,FluxMap,MeanRedshift,StartingRedshift,EndingRedshift,PixelLengthMpc,PixelLengthArcsec,RestFrequency,MapSpreadParameter,FreqSpreadParameter,VolumeFactor)

    np.savez(PSname,k=k,kCubedPk=k3Pk,nummodesDEFAULT=nummodesDEFAULT,PnDEFAULT=PnDEFAULT,WDEFAULT=WDEFAULT)

 ################################################################
 
 

#TODO
def runCCPSC(cosmo,Filename1,Filename2,CCPSname,kValParameters,StartFreq1,StartFreq2,RestFreq1,RestFreq2,Slices1,Slices2):

    FluxMap1=np.load(Filename1)["FluxMap"]
    FluxMap2=np.load(Filename2)["FluxMap"]


    Dep1=len(FluxMap1)
    Dep2=len(FluxMap2)
    Len1=len(FluxMap1[0])
    Len2=len(FluxMap2[0])

    if StartFreq1==390:
        FreqInt1=28/Dep1
    else:
        FreqInt1=40/Dep1
    if StartFreq2==390:
        FreqInt2=28/Dep2
    else:
        FreqInt2=40/Dep2

    ActualStartFreq1=StartFreq1+FreqInt1*Slices1[0]
    ActualEndFreq1=StartFreq1+FreqInt1*Slices1[1]

    ActualStartFreq2=StartFreq2+FreqInt2*Slices2[0]
    ActualEndFreq2=StartFreq2+FreqInt2*Slices2[1]


    StartingRedshift1=(RestFreq1/ActualEndFreq1)-1 
    EndingRedshift1=(RestFreq1/ActualStartFreq1)-1 
    MeanRedshift1=(StartingRedshift1+EndingRedshift1)/2
    dA=cosmo.kpc_comoving_per_arcmin(MeanRedshift1).value*60/1000 #Mpc/deg
    ComovingMapLength1=1.2*dA
    PixelLengthMpc1=ComovingMapLength1/Len1

    StartingRedshift2=(RestFreq2/ActualEndFreq2)-1 
    EndingRedshift2=(RestFreq2/ActualStartFreq2)-1 
    MeanRedshift2=(StartingRedshift2+EndingRedshift2)/2
    dA=cosmo.kpc_comoving_per_arcmin(MeanRedshift2).value*60/1000 #Mpc/deg
    ComovingMapLength2=1.2*dA
    PixelLengthMpc2=ComovingMapLength2/Len2
    

    NewMapDepth=np.max([Slices1[1]+1-Slices1[0],Slices2[1]+1-Slices2[0]])
    NewMapLength=np.max([Len1,Len2])
    k,Pk=create3DCrossSpectrum(cosmo,kValParameters,FluxMap1[Slices1[0]:Slices1[1]+1],MeanRedshift1,StartingRedshift1,EndingRedshift1,PixelLengthMpc1,
                              FluxMap2[Slices2[0]:Slices2[1]+1],MeanRedshift2,StartingRedshift2,EndingRedshift2,PixelLengthMpc2,VolUsed="Average",
                              ZoomDim=(NewMapDepth,NewMapLength,NewMapLength))
    np.savez(CCPSname,k=k,kCubedPk=Pk)
                        
