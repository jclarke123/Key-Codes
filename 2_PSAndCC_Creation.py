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





def expf(x,k,kparmax,sigmaperp,sigmapar):
    """
    the exponential function we integrate for the error calculation (from Chung+2019). Also see Karoumpis+2021, Clarke+2024 appendices for the full consstruction

    Parameters
    ----------
    x : float
        we integrate between 0 and 1 for this
    k : float
        spatial frequency value of the k bin we are doing this integration for
    kparmax : float
        maxmum spatial frequency value
    sigmaperp/par : float
        uncertainity from sensitivity in perpendicular and parallel directions

    Returns
    -------
    float
        the result of the integral

    """
    return np.exp((sigmaperp**2-sigmapar**2)*(np.min([kparmax,x*k]))**2)


def CubeReshape(FluxCube):
    """
    We make cubes in (z,x,y) format. However, making power spectra is far easier in the (x,y,z) format. This function rotates the 3D cube appropriately

    Parameters
    ----------
    FluxCube : numpy array
        The 3D cube we aim to rotate

    Returns
    -------
    numpy array
        The resulting cube

    """
    return np.flip(np.rot90(FluxCube,axes=(2,0)),axis=2)


def create3DPowerSpectrum(cosmo,kValParameter,FluxCube,MeanRedshift,StartingRedshift,EndingRedshift,PixelLengthMpc,PixelLengthArcsec,IndivLineFreq,
                          MapSpreadParameter,FreqSpreadParameter,VolumeFactor=1):
    """
    The code to create a 3D power spectrum, as well as its sensitivity, from an intensity cube. This is done for a given redshift, calculated assuming all signal comes from one line (typically [CII])

    Parameters
    ----------
    cosmo : astropy cosmology object
        The cosmology we use for determining e.g. luminositiy distances
    kValParameter : list
        First part determines the type of k bins we are using. Any subsequent parts of list determine e.g. boundaries, depending on the type used
    FluxCube : numpy array
        The tomography we are taking the power spectra of 
    Mean/Starting/EndingRedshift : floats
        The redshift range covered by the cube, for the rest frequency of the line we are calibrating for
    PixelLengthMpc : float
        Length of a given pixel in space, in Mpc
    PixelLengthArcsec : float
        Length of a given pixel in space, in arcsec
    IndivLineFreq : float
        The rest frequency of the line we are calibrating for - we need for sensitivity calculations
    Map/FreqSpreadParameter : float
        If we wish to convolve with a beam, we should subgrid the voxels. These are the corresponding parameters for each direction.
    VolumeFactor : float, optional
        Key in calibrating the PS via volume normalisation, assuming there was masking. Must be between 0 and 1 (for masking all vs masking no volume)

    Returns
    -------
    k : numpy array
        array of spatial frequency bins we are calculating the power spectra for 
    PS : numpy array
        array of those values, using the k³P(k)/2pi² normalisation
    PnDEFAULT : numpy array
        array of noise sensitivity for those values, see Chung+2019
    nummodesDEFAULT : numpy array
        Values used in above
    WDEFAULT : numpy array
        Values used in above

    """
    #a lot is similar to above, but we do make some adjustments
    #we now have a z direction! This tells us about how many slices we have
    
    #note that the format we get the map is the wrong way around (we want x,y,z; not z,x,y)! We need to reshape to get the axes in order
    FluxCube=CubeReshape(FluxCube) 
    #We then determine the dimensions, both in voxel number and in Mpc
    VoxelXNum,VoxelYNum,VoxelZNum=len(FluxCube),len(FluxCube[0]),len(FluxCube[0][0]) 
    ComovingZLength=cosmo.comoving_distance(EndingRedshift).value-cosmo.comoving_distance(StartingRedshift).value #Mpc
    ComovingMapLength=max([VoxelXNum,VoxelYNum])*PixelLengthMpc #Mpc
    VolOfVoxel=(PixelLengthMpc**2)*(ComovingZLength/VoxelZNum) #Mpc³
    #from these we can determine the maximum and minimum spatial frequency scales (from beam scalze/map size)
    #note - max is of one beamFWHM
    kparmin=2*np.pi/   ComovingZLength
    kperpmin=2*np.pi/  ComovingMapLength 
    kcross=2*np.pi/  (ComovingMapLength**2+ComovingZLength**2)**0.5
    kparmax=np.pi/  (ComovingZLength/VoxelZNum)/FreqSpreadParameter
    kperpmax=np.pi/   PixelLengthMpc /MapSpreadParameter

    #We need to then find the NORMALISED volume of a voxel per number of voxels, which we use to normalise the FFT of the cube
    #NEED TO BE CAREFUL!!! We divide this by the volume factor**2. This is because the normalisation factor is passed through the squared of the FFT (though we also need it for sensitivity)
    VolOfVoxelPerVoxel=(1/VolumeFactor)**2 *(VolOfVoxel/(VoxelXNum*VoxelYNum*VoxelZNum))
    fouriernrm  = VolOfVoxelPerVoxel * np.abs(np.fft.fftshift(np.fft.fftn(FluxCube)))**2 

    #we then form grids in k space from the number of voxels, which we use to form the k bins
    kx        = np.fft.fftshift(np.fft.fftfreq(VoxelXNum,d=1/VoxelXNum)) 
    ky        = np.fft.fftshift(np.fft.fftfreq(VoxelYNum,d=1/VoxelYNum)) 
    kz        = np.fft.fftshift(np.fft.fftfreq(VoxelZNum,d=1/VoxelZNum))
    unitlessgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(kx,ky,kz,indexing='ij')))
    unitlessgrid = unitlessgrid.astype(int)
    kgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(kperpmin*kx,kperpmin*ky,kparmin*kz,indexing='ij')))
    #we then flatten and normalise to use in the binning
    totalampbins = np.bincount(unitlessgrid.flatten(), fouriernrm.flatten()) 
    totalkbins = np.bincount(unitlessgrid.flatten(), kgrid.flatten()) 
    numforbin = np.bincount(unitlessgrid.flatten())
    NormFTProfile = totalampbins / numforbin
    NormkProfile = totalkbins / numforbin
    
    #Then, we determine the dimensions of the bins we fit into. We have a few options
    if kValParameter[0]=='Linspace': #maximum extent in linear space
        kbins=np.linspace(min([kparmin,kperpmin,kcross]),min([kperpmax,max(NormkProfile)]),kValParameter[1])
    elif kValParameter[0]=='Logspace': #maximum extent in log space
        kb=np.linspace(min([np.log10(kparmin),np.log10(kperpmin),np.log10(kcross)]),min([np.log10(kperpmax),max(np.log10(NormkProfile))]),kValParameter[1])
        kbins=np.power(10,kb)
    elif kValParameter[0]=='Fixed': #fixed values derived from kValParameter
        kbins=np.linspace(kValParameter[2],kValParameter[3],kValParameter[1])
    elif kValParameter[0]=='Three Bins': #these are if we want to use winder bins. The former goes to lower k modes
        kbins=np.asarray([0,0.3,0.6,0.9,1.2,1.5,1.8,2.1])-0.1
    elif kValParameter[0]=='Three Bins Alt': #while the latter has centering around 0
        kbins=np.asarray([0,0.3,0.6,0.9,1.2,1.5,1.8,2.1])-0.05

    #the actual binning is done here
    PS,bin_edges,binnumber=stats.binned_statistic(NormkProfile,(NormkProfile**3)*NormFTProfile/(2*((np.pi)**2)), statistic='mean', bins=kbins)
    #we then find the k around the edges
    dK=-bin_edges[:-1]+bin_edges[1:]
    k=bin_edges[1:]-(dK/2.)

    #We then do sensitivity calculations. Many of these values are hardcoded for COSMOS2020
    StartingFrequency=IndivLineFreq/(1+EndingRedshift)
    if    StartingFrequency<400 and StartingFrequency>380: PnFactor=((2.2e4)**2) 
    elif  StartingFrequency<340 and StartingFrequency>320: PnFactor=((1.2e4)**2)
    elif  StartingFrequency<270 and StartingFrequency>250: PnFactor=((6.2e3)**2)
    elif  StartingFrequency<215 and StartingFrequency>195: PnFactor=((3.9e3)**2)
    else: 
        PnFactor=((1e4)**2)  #just an exception we can handle. Error will be off, but oh well
        print("ERROR IN SENSITIVITY CALCULATION; RE-EVALUATE")
    PnDEFAULT=PnFactor*VolOfVoxel * (MapSpreadParameter**2)
    
    ###For auto-spectra, can estimate sensitivity for the DEFAULT noise parameters in this way. You will need to calculated
    nummodesDEFAULT=np.zeros(len(k))
    WDEFAULT=np.zeros(len(k))
    #we use the volume of the map
    VolMap=VolOfVoxel*VoxelXNum*VoxelYNum*VoxelZNum *VolumeFactor
    sigfactor=2.355 #converts from beamFWHM to beamsigma
    sigmaperp=(cosmo.kpc_comoving_per_arcmin(MeanRedshift).value*PixelLengthArcsec/(60*1000))/sigfactor #first bit is in Mpc/arcseconds, second bit is beam FWHM in arcsecs, so we need to be careful
    sigmapar=(3*10**5)/(cosmo.H(MeanRedshift).value)*((1+MeanRedshift)*(40/VoxelZNum))/(sigfactor*IndivLineFreq/(1+MeanRedshift))
    
    #due to integral we need to treat each k separately
    for i in range(len(k)):  
        nummodesDEFAULT[i]=np.min([kparmax,k[i]])*k[i]*(k[1]-k[0])*(VolMap)/4/np.pi/np.pi
        WDEFAULT[i]=np.exp(0-(sigmaperp*k[i])**2)*quad(expf,0,1,args=(k[i],kparmax,sigmaperp,sigmapar))[0]

    return k, PS,PnDEFAULT,nummodesDEFAULT,WDEFAULT



def create3DCrossSpectrum(cosmo,kValParameter,FluxCube1,MeanRedshift1,StartingRedshift1,EndingRedshift1,PixelLengthMpc1,
                          FluxCube2,MeanRedshift2,StartingRedshift2,EndingRedshift2,PixelLengthMpc2,VolUsed="Average",ZoomDim=(20,200,200),
                          MapSpreadParameter=3,FreqSpreadParameter=3): 
    """
    Similar to above, but for cross-spectra. We need to match the mapsize of the two cubes before we perform the CC
    Sensitivity analysis and separate masking normalisations to be added later

    Parameters
    ----------
    cosmo : astropy cosmology object
        The cosmology we use for determining e.g. luminositiy distances
    kValParameter : list
        First part determines the type of k bins we are using. Any subsequent parts of list determine e.g. boundaries, depending on the type used
    For the two tomographies:
    FluxCube : numpy array
        The tomography we are taking the power spectra of 
    Mean/Starting/EndingRedshift : floats
        The redshift range covered by the cube, for the rest frequency of the line we are calibrating for
    PixelLengthMpc : float
        Length of a given pixel in space, in Mpc
    PixelLengthArcsec : float
        Length of a given pixel in space, in arcsec
    IndivLineFreq : float
        The rest frequency of the line we are calibrating for - we need for sensitivity calculations

    
    Others
    VolUsed : string, optional
        Cubes will cover extremely similar, but not identical volumes - this parameter determines which we use. The default is "Average".
    ZoomDim : list, optional
        the 3D array we interpolate the cubes into the shape of, in order for the cross-correlation to successfully function. The default is (20,200,200).
    Map/FreqSpreadParameter : float, optional
        If we wish to convolve with a beam, we should subgrid the voxels. These are the corresponding parameters for each direction.

    Currently no volume masking normalisation has been implemented

    Returns
    -------
    k : numpy array
        array of spatial frequency bins we are calculating the power spectra for 
    CCPS : numpy array
        array of those values, using the k³P(k)/2pi² normalisation

    """
    #we do the first step seperately, similar to above for both cubes
    #But first, we normalise our dimensions to be of a unified shape
    PixelLengthMpc1=PixelLengthMpc1/(ZoomDim[1]/len(FluxCube1[0]))
    FluxCube1=nd.zoom(FluxCube1,(ZoomDim[0]/len(FluxCube1),ZoomDim[1]/len(FluxCube1[0]),ZoomDim[2]/len(FluxCube1[0][0])))
    VoxelXNum1,VoxelYNum1,VoxelZNum1=len(FluxCube1[0][0]),len(FluxCube1[0]),len(FluxCube1) 
    FluxCube1=CubeReshape(FluxCube1) 
    ComovingZLength1=cosmo.comoving_distance(EndingRedshift1).value-cosmo.comoving_distance(StartingRedshift1).value
    ComovingMapLength1=max([VoxelXNum1,VoxelYNum1])*PixelLengthMpc1
    VolOfVoxel1=(PixelLengthMpc1**2)*(ComovingZLength1/VoxelZNum1)
    VolOfVoxelPerVoxel1=(VolOfVoxel1/(VoxelXNum1*VoxelYNum1*VoxelZNum1))
    

    #for the second cube
    PixelLengthMpc2=PixelLengthMpc2/(ZoomDim[1]/len(FluxCube2[0]))
    FluxCube2=nd.zoom(FluxCube2,(ZoomDim[0]/len(FluxCube2),ZoomDim[1]/len(FluxCube2[0]),ZoomDim[2]/len(FluxCube2[0][0])))
    VoxelXNum2,VoxelYNum2,VoxelZNum2=len(FluxCube2[0][0]),len(FluxCube2[0]),len(FluxCube2) 
    FluxCube2=CubeReshape(FluxCube2) 
    ComovingZLength2=cosmo.comoving_distance(EndingRedshift2).value-cosmo.comoving_distance(StartingRedshift2).value
    ComovingMapLength2=max([VoxelXNum2,VoxelYNum2])*PixelLengthMpc2
    VolOfVoxel2=(PixelLengthMpc2**2)*(ComovingZLength2/VoxelZNum2)
    VolOfVoxelPerVoxel2=(VolOfVoxel2/(VoxelXNum2*VoxelYNum2*VoxelZNum2))


    #then average of the physical scales, or take one or another
    if VolUsed=="Average":
        VolOfVoxelPerVoxel=(VolOfVoxelPerVoxel1+VolOfVoxelPerVoxel2)/2
        VoxelXNumUsed,VoxelYNumUsed,VoxelZNumUsed=int((VoxelXNum1+VoxelXNum2)/2),int((VoxelYNum1+VoxelYNum2)/2),int((VoxelZNum1+VoxelZNum2)/2)
        kperpmin=4*np.pi/(ComovingMapLength1+ComovingMapLength2)
        kparmin=4*np.pi/(ComovingZLength1+ComovingZLength2) 
        kcross=4*np.pi/  ((ComovingMapLength1+ComovingMapLength2)**2+(ComovingZLength1+ComovingZLength2)**2)**0.5    
        kparmax=2*np.pi/  ((ComovingZLength1+ComovingZLength2)/VoxelZNumUsed)/FreqSpreadParameter
        kperpmax=2*np.pi/   (PixelLengthMpc1+PixelLengthMpc2)/MapSpreadParameter
    elif VolUsed=="First":
        VolOfVoxelPerVoxel=VolOfVoxelPerVoxel1
        VoxelXNumUsed,VoxelYNumUsed,VoxelZNumUsed=VoxelXNum1,VoxelYNum1,VoxelZNum1
        kperpmin=2*np.pi/(ComovingMapLength1)
        kparmin=2*np.pi/(ComovingZLength1) 
        kcross=2*np.pi/  (ComovingMapLength1**2+ComovingZLength1**2)**0.5    
        kparmax=np.pi/  ((ComovingZLength1)/VoxelZNumUsed)/FreqSpreadParameter
        kperpmax=np.pi/   (PixelLengthMpc1)/MapSpreadParameter
    elif VolUsed=="Second":
        VolOfVoxelPerVoxel=VolOfVoxelPerVoxel2
        VoxelXNumUsed,VoxelYNumUsed,VoxelZNumUsed=VoxelXNum2,VoxelYNum2,VoxelZNum2
        kperpmin=2*np.pi/(ComovingMapLength2)
        kparmin=2*np.pi/(ComovingZLength2) 
        kcross=2*np.pi/  (ComovingMapLength2**2+ComovingZLength2**2)**0.5    
        kparmax=np.pi/  ((ComovingZLength2)/VoxelZNumUsed)/FreqSpreadParameter
        kperpmax=np.pi/   (PixelLengthMpc2)/MapSpreadParameter

    #then we continue as before. The CC applies here
    fouriernrm  = VolOfVoxelPerVoxel*np.real(np.fft.fftshift(np.fft.fftn(FluxCube1))*np.conj(np.fft.fftshift(np.fft.fftn(FluxCube2)))) #cross correlation    

    kx        = np.fft.fftshift(np.fft.fftfreq(VoxelXNumUsed,d=1/VoxelXNumUsed)) 
    ky        = np.fft.fftshift(np.fft.fftfreq(VoxelYNumUsed,d=1/VoxelYNumUsed)) 
    kz        = np.fft.fftshift(np.fft.fftfreq(VoxelZNumUsed,d=1/VoxelZNumUsed))

    unitlessgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(kx,ky,kz,indexing='ij')))
    unitlessgrid = unitlessgrid.astype(int)
    kgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(kperpmin*kx,kperpmin*ky,kparmin*kz,indexing='ij')))
    totalampbins = np.bincount(unitlessgrid.flatten(), fouriernrm.flatten()) 
    totalkbins = np.bincount(unitlessgrid.flatten(), kgrid.flatten()) 
    numforbin = np.bincount(unitlessgrid.flatten())
    NormFTProfile = totalampbins / numforbin
    NormkProfile = totalkbins / numforbin
    
    #then the binning
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
    return k, CCPS
















#TODO
###The thing we run. We need the filename to load from, plus a name to save to
#Is fairly self-explanatory. Main thing is 
#accounting for any spread of course
#Also have "AutoCII" - that is, using the scales of CII (that is, the redshifts) - in cases where we want to stay to a fixed volume scale. Normally false, may need to use when stacking stuff, etc
def runPSC(cosmo,Mapname,PSname,kValParameters,AutoCII=True,VolumeFactor=1,MapSpreadParameter=1,FreqSpreadParameter=1): 
    """
    Running the above from a cube made by FluxMapCreation, taking the relevant parameters to make the cube from

    Parameters
    ----------

    cosmo : astropy cosmology object
        The cosmology we use for determining e.g. luminositiy distances
    Mapname : string
        File location of the map we make spectra of
    PSname : string
        File location of the spectra we make save to
    kValParameter : list
        First part determines the type of k bins we are using. Any subsequent parts of list determine e.g. boundaries, depending on the type used
    AutoCII : boolean, optional
        Deciding whether we are using the values in the cube file. The default is True.
    Map/FreqSpreadParameter : float
        If we wish to convolve with a beam, we should subgrid the voxels. These are the corresponding parameters for each direction.
    VolumeFactor : float, optional
        Key in calibrating the PS via volume normalisation, Must be between 0 and 1 (for masking all vs masking no volume)


    Returns
    -------
    None.

    """
    #load map and relevant data
    Mapdata=np.load(Mapname,allow_pickle=True)
    FluxCube, StartingRedshift, EndingRedshift=Mapdata['FluxMap'],Mapdata['StartingRedshift'],Mapdata['EndingRedshift']
    RestFrequency=Mapdata['RestFrequency']
    
    #we usually update the parameters to fit as if we are assuming we are observing [CII] signal
    if AutoCII:
        print("AUTOCII")
        
        
        StartingRedshift=(1900.537*(1+StartingRedshift)/RestFrequency)-1
        EndingRedshift=(1900.537*(1+EndingRedshift)/RestFrequency)-1
        RestFrequency=1900.537 
        

        
    MeanRedshift=(StartingRedshift+EndingRedshift)/2
    dA=cosmo.kpc_comoving_per_arcmin(MeanRedshift).value/60/1000 #distance covered by 1 arcsecond in Mpc. Multiplying this by PixelLength gives the pixel length in Mpc
    BeamSizeArcsec=((Mapdata["BeamSizeRad"]*u.rad).to(u.arcsecond)).value
    PixelLengthArcsec=BeamSizeArcsec/MapSpreadParameter #length of pixels in arcseconds
    PixelLengthMpc=PixelLengthArcsec*dA #and in Mpc
    #make and save the cube    
    k,k3Pk,PnDEFAULT,nummodesDEFAULT,WDEFAULT=create3DPowerSpectrum(cosmo,kValParameters,FluxCube,MeanRedshift,StartingRedshift,EndingRedshift,PixelLengthMpc,PixelLengthArcsec,RestFrequency,MapSpreadParameter,FreqSpreadParameter,VolumeFactor)
    np.savez(PSname,k=k,kCubedPk=k3Pk,nummodesDEFAULT=nummodesDEFAULT,PnDEFAULT=PnDEFAULT,WDEFAULT=WDEFAULT)

 ################################################################
 
 

#TODO
def runCCPSC(cosmo,Mapname1,Mapname2,CCPSname,kValParameters,StartFrequency1,StartFrequency2,RestFrequency1,RestFrequency2,Slices1,Slices2):
    """
    Making the cross-spectra. This function is less developed, with no masking parameter implemented yet    
    Note: as the cubes must cover the same volume, these MUST be explicit with the different lines/rest frequencies
    We take a few slices from each cube, to cross-correlate over specific sections
    Parameters
    ----------
    cosmo : astropy cosmology object
        The cosmology we use for determining e.g. luminositiy distances
        
    For each of the two cubes:
    Mapname : string
        File location of the map we make spectra of
    
    StartFrequency : float
        Starting frequency of the cubes you use. Note - this is NOT the start frequency of the section we CC, which we must calcuate

    RestFrequency : float
        Rest frequency of the given line (NOT NORMALISING TO CII BY DEFAULT)

    Slices : list
        Two indexes, representing the sections of the base cube we then cut out

    And then:
    CCPSname : string
        File location of the spectra we make save to
    kValParameters : list
        First part determines the type of k bins we are using. Any subsequent parts of list determine e.g. boundaries, depending on the type used

    Returns
    -------
    None.

    """
    

    #these are the original cubes
    FluxCube1=np.load(Mapname1)["FluxMap"]
    FluxCube2=np.load(Mapname2)["FluxMap"]

    #but we do not CC the original cubes, but segments from that. So we need to adapt from that
    Depth1=len(FluxCube1)
    Depth2=len(FluxCube2)
    Length1=len(FluxCube1[0])
    Length2=len(FluxCube2[0])

    if StartFrequency1==390:
        FreqInterval1=28/Depth1
    else:
        FreqInterval1=40/Depth1
    if StartFrequency2==390:
        FreqInterval2=28/Depth2
    else:
        FreqInterval2=40/Depth2

    #the start/ending points change
    ActualStartFrequency1=StartFrequency1+FreqInterval1*Slices1[0]
    ActualEndFrequency1=StartFrequency1+FreqInterval1*Slices1[1]

    ActualStartFrequency2=StartFrequency2+FreqInterval2*Slices2[0]
    ActualEndFrequency2=StartFrequency2+FreqInterval2*Slices2[1]

    #For each cube:
    StartingRedshift1=(RestFrequency1/ActualEndFrequency1)-1 
    EndingRedshift1=(RestFrequency1/ActualStartFrequency1)-1 
    MeanRedshift1=(StartingRedshift1+EndingRedshift1)/2
    dA=cosmo.kpc_comoving_per_arcmin(MeanRedshift1).value*60/1000 #Mpc/deg
    ComovingMapLength1=1.2*dA #fixed for COSMOS2020
    PixelLengthMpc1=ComovingMapLength1/Length1

    StartingRedshift2=(RestFrequency2/ActualEndFrequency2)-1 
    EndingRedshift2=(RestFrequency2/ActualStartFrequency2)-1 
    MeanRedshift2=(StartingRedshift2+EndingRedshift2)/2
    dA=cosmo.kpc_comoving_per_arcmin(MeanRedshift2).value*60/1000 #Mpc/deg
    ComovingMapLength2=1.2*dA
    PixelLengthMpc2=ComovingMapLength2/Length2
    
    #we need to do cross-correlation over consistent array sizes, so we will need to squash and stretch the cubes
    #the least instrusive form of doing this
    NewMapDepth=np.max([Slices1[1]+1-Slices1[0],Slices2[1]+1-Slices2[0]])
    NewMapLength=np.max([Length1,Length2])
    #the cubes are sliced appropriately
    k,Pk=create3DCrossSpectrum(cosmo,kValParameters,FluxCube1[Slices1[0]:Slices1[1]+1],MeanRedshift1,StartingRedshift1,EndingRedshift1,PixelLengthMpc1,
                              FluxCube2[Slices2[0]:Slices2[1]+1],MeanRedshift2,StartingRedshift2,EndingRedshift2,PixelLengthMpc2,VolUsed="Average",
                              ZoomDim=(NewMapDepth,NewMapLength,NewMapLength))
    np.savez(CCPSname,k=k,kCubedPk=Pk)
