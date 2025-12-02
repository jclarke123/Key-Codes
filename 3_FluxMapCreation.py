"""
This code involves creating the flux maps to be used in creating power spectra, based on catalogue galaxy data
These maps are 3D cubes. Galaxies from a given redshift band are painted on this map.
This painting is converted to flux units of Jy/sr. The painting is either storing flux in single voxels (size=FWHM of beam)
Or you spread the flux over multiple voxels (subgridding the above into 3x3, not hardcoded). This can be purely over the map ("angular"), purely in the z direction ("spectral"), or both ("angular+spectral")

You can then do stats like e.g. the power spectra

The core function you run is runFMC, which calls anything else

This is currently compatible with COSMOS2020, but edits to fetchVariables should make this compatible with any catalogue set
This is also aimed for use with CCAT/FYST, but feel free to use for other instruments

"""

######################Imports the modules used. These are all fairly standard
import numpy as np
import astropy.units as u
from tqdm import tqdm
from astropy.convolution import convolve
import CANDELSextrapolation as CE



def fetchVariablesCOSMOS(Data,StartingRedshift,EndingRedshift,OriginalMapParams,PixelScalingFactor,SliceNumber,RestFrequency,NameParams,MapSpreadParameter=1,FreqSpreadParameter=1):
    """
    For a given input fits file, load values
    You cannot use this for other samples without alterations - "Original Map Params" is designed so we only include valid data away from the edges of COSMOS
    Will return mass, SFR, metallicity calculated from Mannucci et al (currently hardcoded), OIII, LIR, and location data
    This is far quicker than the above function
    
    Parameters
    -------
    Data: numpy.ndarray
        The catalogue data we imported
    StartingRedshift, EndingRedshift: float
        The redshift range we want to draw data from
    OriginalMapParams: list
        a list of 7 values describing COSMOS. The first is the size of the cube in its arbitrary pixel units (of 0.15 arcsec). The second and third are the desired cube 
        dimensions, in those same units, for x and y. The fourth and fifth are the lower and upper limits, in the same pixel units, for x (sixth and seventh are for y) 
    PixelScalingFactor: float
        The size of the beam in the aforementioned arbitrary pixel units (e.g. 37.2 arcsec -> 248). This is used to convert the galaxy locations from the pixel units
        into the pixels we use for our maps, subgridding if necessary via MapSpreadParameter
    SliceNumber: int
        The number of frequency channels of our cube, before subgridding by FreqSpreadParameter. Used to determine the z location of the galaxies redshift
    RestFrequency: float
        The frequency of the line we want to make the cube for. Used to determine z location of the galaxies from redshift
    NamesParams: dict
        dictionary used to find the names of the columns we take galaxy data from
    MapSpreadParameter/FreqSpreadParameter: integers (optional)
        If we are convolving a map with the beams, these integers are the subgridding (i.e. MSP=3 means a 3x3 subgrid). Consequently, x,y,z positions must be updated
    RandRedshift, sigmacap: boolean, float (optional)
        In some cases you may want to simulate randomisation of the redshift - RandRedshift determines if this is the case, sigmacap determines how many standard deviations out from the original are allowed
    Returns
    -------
    dict of numpy arrays
        Returns the dictionary of the core properties used to make models with. Can modify and extend as needed
    """  
    #NameParams is a dictionary of a dictionary, to allow for different column names for the source files
    nameLIR,nameLogMass,nameLogSFR,nameRedshift,nameOIII,nameX,nameY,nameFlag,nameRedshifterr=NameParams["LIR"],NameParams["log_mass"],NameParams["log_SFR"],NameParams["Redshift"],NameParams["OIII"],NameParams["x_orig"],NameParams["y_orig"],NameParams["FLAG"],NameParams["Redshift_err"] 
    
    #check to see if there are any galaxies in the source file (sometimes not at high redshift for stellar mask)
    if len(Data)!=0:
        Failure=False    
            
        #load data we use to verify the correct galaxies
        Flags=Data[nameFlag]
        Redshift=Data[nameRedshift]
        x=Data[nameX]
        y=Data[nameY]
        
        #this ensures all galaxies are within our given range and the given map bounds, that the galaxy is flagged as a galaxy. You will need custom checks for input files,
        #will have to edit this/provide different versions if using different inputs. This JUST works for COSMOS2020
        ValidIndexes=(Redshift>StartingRedshift)*(Redshift<EndingRedshift)*(Flags==0)*(x>=OriginalMapParams[3])*(x<=OriginalMapParams[4])*(y>=OriginalMapParams[5])*(y<=OriginalMapParams[6]) #all the validity stuff from earlier
        
        #for those valid galaxies, we load all relevant data
        Redshift=Redshift[ValidIndexes]
        #we have to recenter the arbitrary pixel scale of COSMOS2020 to originate from 0,0, and 1 pixel to refer to 1/3 FWHM of the beam, requriing "MapSpreadParameter" and PixelScalingFactor
        xpix=np.round((x[ValidIndexes]-OriginalMapParams[3])*MapSpreadParameter/PixelScalingFactor)
        ypix=np.round((y[ValidIndexes]-OriginalMapParams[5])*MapSpreadParameter/PixelScalingFactor)
        log_mass=Data[nameLogMass][ValidIndexes]
        log_SFR=Data[nameLogSFR][ValidIndexes]
        Redshift_err=Data[nameRedshifterr][ValidIndexes]-Redshift
        OIII=Data[nameOIII][ValidIndexes]
        LIR=Data[nameLIR][ValidIndexes]
        #error handling for when we take log of these values later on
        OIII[OIII<0]=0
        LIR[LIR<0]=0
        log_OIII=np.log10(OIII)
        log_LIR=np.log10(LIR)
        #using Mannucci+2010 metallicity calculation for any models. Can edit if desired
        log_met=8.90+0.37*(log_mass-10)-0.14*(log_SFR)-0.19*(log_mass-10)**2+0.12*(log_mass-10)*(log_SFR)-0.054*(log_SFR)**2
        
        #calculate relative z index in the frequency cube based on redshift
        StartFrequency=RestFrequency/(EndingRedshift+1)
        EndFrequency=RestFrequency/(StartingRedshift+1)
        Frequencies=RestFrequency/(Redshift+1)
        zpix=np.floor(FreqSpreadParameter*SliceNumber*(Frequencies-StartFrequency)/(EndFrequency-StartFrequency)) 
        zpix.astype(int)
        
    else:
        #error handling, if no valid galaxies for a particular file. We run this for multiple files, so this may be the case
        Failure=True
        Flags,Redshift,xpix,ypix,log_mass,log_SFR,Redshift_err,log_OIII,log_LIR=[],[],[],[],[],[],[],[],[]
        zpix=[]
        print("No Signal, skipping map creation for this file")
    if len(log_mass)==0:
        Failure=True
        print("No Signal, skipping map creation for this file")
    #save in dictionary
    return {"log_mass":log_mass,"log_SFR":log_SFR,"log_met":log_met,"xpix":xpix,"ypix":ypix,"zpix":zpix,"log_OIII":log_OIII,"log_LIR":log_LIR,"Redshift":Redshift,"Redshift_err":Redshift_err},Failure




def calcLsun(VariableList,VariableLabel1,VariableLabel2,ModelCoefficients):
    """
    For each of the galaxies which we have fetched, we calculate the luminosity in units of log(Lsun)
    The luminosity is calculated using either the 2D quadratic with a linear cross term (using input parameters), or with a custom model  
    We use custom models if VariableLabel1 is a given code name. Else, we move to the more generic models. You can add more custom models if needed
    
    Parameters
    ----------
    VariableList: dict of numpy.ndarrays
        The aforementioned galaxy properties, stored in dict format
    VariableLabel1, 2: string
        What determines the galaxy properties of the model used. There are a number of custom models, determined by VariableLabel1
        Else we assume a 2d quad model, which is governed by ModelCoefficients
    ModelCoefficients: list
        A list of 6 parameters which govern the 2D quadratic. 
        log L=MC[0]+MC[1]*Variable1+MC[2]*Variable1²+MC[3]*Variable2+MC[4]*Variable2²+MC[5]*Variable1*Variable2
    Returns
    -------
    numpy array
        The luminosities from the models
    """   
    
    
    #First, each of the custom models! We check the VariableLabel1, which can run specific models
    if VariableLabel1=='Lagache':
        ModelVar1=VariableList["log_SFR"] 
        ModelVar2=VariableList["Redshift"] 
        LumArray=(7.1-0.07*ModelVar2)+(1.4-0.07*ModelVar2)*ModelVar1
    elif VariableLabel1=='Vallini': 
        ModelVar1=VariableList["log_mass"] 
        ModelVar2=VariableList["log_SFR"]
        #they use a slightly different metallicity construction which we must construct, see below
        solarMet=1
        Met=np.power(10,0.21+0.37*(ModelVar1-10)-0.14*ModelVar2-0.19*((ModelVar1-10)**2)-0.054*(ModelVar2**2)+0.12*(ModelVar1-10)*ModelVar2)
        LumArray=7+1.2*ModelVar2+0.021*np.log10(Met/solarMet)+0.012*np.log10(Met/solarMet)*ModelVar2-0.74*(np.log10(Met/solarMet))**2
    #We can use some luminosities directly, depending on the code. In these cases, we just import directly!
    #for COSMOS 2020, we can do this for OIII Lum. You can rework depending on your needs
    elif VariableLabel1=='OIII':
        LumArray=VariableList["log_OIII"]
    #In all other cases, we use the 'generic' model, which is a quadratic in either 1 or two dimensions
    #we always have to lead the first variable, load relevant data
    else:
        if VariableLabel1=='Log Mass':
            ModelVar1=VariableList["log_mass"]
        elif VariableLabel1=='Log SFR':
            ModelVar1=VariableList["log_SFR"]
        elif VariableLabel1=='Log Met':
            ModelVar1=VariableList["log_met"]
        #we may need values not included earlier, hence these slight alterations
        elif VariableLabel1=='Met':
            ModelVar1=np.power(10,VariableList["log_met"])
        elif VariableLabel1=='Log sSFR':
            ModelVar1=VariableList["log_SFR"]-VariableList["log_mass"]
        elif VariableLabel1=='Log Mass - 10':
            ModelVar1=VariableList["log_mass"]-10
        elif VariableLabel1=='Log LIR':
            ModelVar1=VariableList["log_LIR"]
        #If it is just a 1D model (indicated by "None" for two variables), we can just finish up here, using the generic model.
        if VariableLabel2=='None':
            LumArray=ModelCoefficients[0]+ModelCoefficients[1]*ModelVar1+ModelCoefficients[2]*(ModelVar1**2) 
        #otherwise it's 2D, so we must load the second variable and calculate
        else:
            if VariableLabel2=='Log Mass':
                ModelVar2=VariableList["log_mass"]
            elif VariableLabel2=='Log SFR':
                ModelVar2=VariableList["log_SFR"]
            elif VariableLabel2=='Log Met':
                ModelVar2=VariableList["log_met"]
            elif VariableLabel2=='Met':
                ModelVar2=np.power(10,VariableList["log_met"])
            elif VariableLabel2=='Log sSFR':
                ModelVar2=VariableList["log_SFR"]-VariableList["log_mass"]
            elif VariableLabel2=='Log Mass - 10':
                ModelVar2=VariableList["log_mass"]-10
            elif VariableLabel2=='Log LIR':
                ModelVar2=VariableList["log_LIR"]
            #similar idea, but calc is 2D
            LumArray=ModelCoefficients[0]+ModelCoefficients[1]*ModelVar1+ModelCoefficients[2]*(ModelVar1**2)+ModelCoefficients[3]*ModelVar2+ModelCoefficients[4]*(ModelVar2**2)+ModelCoefficients[5]*ModelVar1*ModelVar2
    #the above can be extended if needed
    
    #We have this to catch any problematic results
    ProblematicIndexes=(LumArray<=0)+(np.isnan(LumArray))+(np.isinf(LumArray))
    LumArray[ProblematicIndexes]=0

    return LumArray


def galMainSeq(x,a,b):
    """
    The function for a galaxy main sequence (i.e. linear line of best fit between logSFR and logMass)
    Needed for LCO propagation

    Parameters
    ----------
    x : np.ndarray
        Input x, i.e. log mass
    a : float
        y-intercept for function
    b : float
        gradient for function

    Returns
    -------
    np.ndarray
        output y, i.e. log SFR

    """
    return a+b*x

def conversionCOSLED(VariableList,CO10LumArray,muCoeff1,muCoeff2,Transition,SLEDFilename,MSFilename,SLEDType="Normal"):
    """
    If we are making a CO map, the previous functions actually determine the CO(1-0) luminosity. This function propagates up an assumed SLED via the main sequences
    These require external files, detailed below
    This follows the methods of Karoumpis et al. (2024)

    Parameters
    ----------
    VariableList : dict of np.ndarray
        The galaxy properties. We specifically need log mass, log SFR, and redshift
    CO10LumArray : np.ndarray
        The CO(1-0) luminosities for the galaxies, calculated using a model as normal (using the above function)
    muCoeff1 : float
        Factors used to calibrate mu, i.e. the SLED parameter, based on a "normal" galaxy. Following Karoumpis et al. 2024, usually set to 1 by default
    muCoeff2 : float
        Factors used to calibrate mu, i.e. the SLED parameter, based on a "normal" galaxy. Following Karoumpis et al. 2024, usually set to 2.5 by default
    Transition : int
        The target CO transition for us (determines which SLED params we use)
    SLEDFilename : str
        The file used to hold the parameters for each SLED. This is a 2D array, with 3 columns. The first column is the transition, the second is "lower SLR",
        the third is "upper SLR". To explain the latter two, these are the factors you need to multiply by to get the other CO luminosities for calibration
        galaxies (the upper and lower ones). As we use log luminosities, we add by these instead
    MSFilename : str
        The file used to hold the main sequence parameters for the sample for each redshift 0.5 interval
    SLEDType : str, optional
        With the SLED, we can choose to follow it... or we can fix the location of each galaxy. "Upper" makes all galaxies follow max SLED,
        "Lower" makes all follow min, Average sets to evenly fall between both. The default is "Normal", i.e. the regular procedure.

    Returns
    -------
    COLumArray : numpy.ndarray
        The propagated CO luminositiy

    """
    #load up the Upper and Lower elements of the SLED
    
    
    
    SLEDArray=np.load(SLEDFilename)
    SLEDIndex=np.where(SLEDArray[:,0]==Transition)
    LowerSLR,UpperSLR=SLEDArray[SLEDIndex][1],SLEDArray[SLEDIndex][2]
            
            
    #we load the main sequence of COSMOS2020 for each z~0.5 interval, derived earlier.
    MSArray=np.load(MSFilename)
    Redshifts=VariableList["Redshift"]
    log_mass=VariableList["log_mass"]
    log_SFR=VariableList["log_SFR"]
    
    
    #dummy, placeholder
    COLumArray=np.zeros(len(CO10LumArray))
    log_SFR_MSGal=np.zeros(len(Redshifts))
    dMS=np.zeros(len(Redshifts)) 
    mu=np.zeros(len(Redshifts))
    
    #for each galaxy, we check where it is within redshift to determine main sequences
    #for this, we calculate the delta in main sequence, which is then used to calibrate mu
    #Upper/Lower SLR are calibration values from example galaxies
    #These are log values, so we add to logLCO instead of multiplying

    
    #need to do this separately for each Main Sequence, as a given cube may have overlap with different z~0.5 segments, and these have separate MS values
    for j in range(len(MSArray)):
        ValidIndex=(Redshifts>=MSArray[j][0])*(Redshifts<MSArray[j][1])
        #if Redshifts[i]>=MSArray[j][0] and Redshifts[i]<MSArray[j][1]:
        if SLEDType=="Normal":
            log_SFR_MSGal[ValidIndex]=galMainSeq(log_mass[ValidIndex],MSArray[j][3],MSArray[j][2])
            dMS[ValidIndex]=log_SFR[ValidIndex]-log_SFR_MSGal[ValidIndex]
            mu[ValidIndex]=(dMS[ValidIndex]+muCoeff1)/muCoeff2
        #we can force maximum or minimum CO models if needed
        elif SLEDType=="Upper":
            mu[ValidIndex]=1
        elif SLEDType=="Lower":
            mu[ValidIndex]=0
        elif SLEDType=="Average":
            mu[ValidIndex]=0.5
        #use linear combination based on main sequence
        COLumArray[ValidIndex]=mu[ValidIndex]*(UpperSLR+CO10LumArray[ValidIndex])+(1-mu)*(LowerSLR+CO10LumArray[ValidIndex])
    return COLumArray



def fluxUnitConversion(cosmo,Redshift,LumArray,SliceNumber,BeamSizeRad,StartingRedshift,EndingRedshift,RestFrequency,FreqSpreadParameter=1,MapSpreadParameter=1):
    """
    Once we have the luminosities,we must generate the fluxes. These are in Jy/sr, and are generated independant of any subgridding
    The core equation is taken from Karoumpis et al (2021) - L/4pi*lumDist², /bandsize and beamsize²
    
    Parameters
    ----------
    cosmo: astropy cosmology object (e.g. FlatLambdaCDM)
        The cosmology you choose to use for your simulation, and therefore your Ho, etc. How you determine luminosity distances
    Redshift: numpy.ndarray
        a
    LumArray: numpy.ndarray
        the log galaxy luminosities generated earlier
    SliceNumber: integer
        the number of divisions of the total frequency range covered by the cube. Does NOT include subgridding
    BeamSizeRad: float
        the size of the beam, i.e. the size of the un-subgridded pixel. This is the FWHM of the beam
    StartingRedshift, EndingRedshift: floats
        The redshift range covered by the ube
    RestFrequency: float
        Of the line we are making this cube for
    ModelCoefficients: list
        A list of 6 parameters which govern the 2D quadratic. 
        log L=MC[0]+MC[1]*Variable1+MC[2]*Variable1²+MC[3]*Variable2+MC[4]*Variable2²+MC[5]*Variable1*Variable2
    FreqSpreadParameter/MapSpreadParameter: int
        subgridding of the cube, as mentioned earlier. Need to include to determine appropriate normalisation
    Returns
    -------
    numpy array
        The flux array from the luminosity, in NON-logarithmic units
    """
    #calculate luminosity distance based on cosmology
    lumDist=cosmo.luminosity_distance(Redshift).value 
    #the frequency slice in GHz (total freq band covered div by number of slices, making sure appropriately normalised by FSP if subgridding)
    GHzBand=RestFrequency*(1/(1+StartingRedshift) - 1/(1+EndingRedshift))/(SliceNumber*FreqSpreadParameter)
    #equation from Karoumpis+21. We do it in units of Lsun/(Mpc²*GHz*rad²)
    FluxArray=np.power(10,LumArray)/(4*np.pi*(lumDist**2)*GHzBand*((BeamSizeRad/MapSpreadParameter)**2))*u.Lsun/((u.Mpc**2)*u.GHz*(u.rad**2))
    #we can use astropy to convert nicely
    FluxArray=FluxArray.to(u.Jy/u.sr).value 
    
    return FluxArray

#This is the inverse function.
def inverseFluxUnitConversion(cosmo,Redshift,FluxArray,SliceNumber,BeamSizeRad,StartingRedshift,EndingRedshift,RestFrequency,FreqSpreadParameter=1,MapSpreadParameter=1):
    """
    The inverse of the above - recovering luminosity from an input flux array. All else is the same
    """
    lumDist=cosmo.luminosity_distance(Redshift).value 
    GHzBand=RestFrequency*(1/(1+StartingRedshift) - 1/(1+EndingRedshift))/(SliceNumber*FreqSpreadParameter)
    FluxArray=FluxArray*u.Jy/u.sr
    FluxArray=FluxArray.to(u.Lsun/((u.Mpc**2)*u.GHz*(u.rad**2))).value 
    LumArray=np.log10(FluxArray*4*np.pi*(lumDist**2)*GHzBand*((BeamSizeRad/MapSpreadParameter)**2))
    
    return LumArray



def createSimple3DMap(VariableList,FluxArray,SliceNumber,MapLength,MapSpreadParameter=1,FreqSpreadParameter=1):
    """
    This stores all the flux from before in a 3D array. The flux from each galaxy is neatly stored in an individual voxel, without any beam convolution
    This also sets the stage for beam convolution
    Despite this, we make the grid accounting for the subgridding, in the x/y and z directions. This allows for convolution later
    the 3D array is z*x*y, i.e. FluxCube[0] is the first "slice"
    NOTE: the earlier indexes in the z direction are for low frequency, i.e. high redshift. As determined by the z pix from earlier

    Parameters
    ----------
    VariableList : dict of numpy.ndarrays
        Carries the x, y, z positions of the galaxies
    FluxArray : numpy.ndarray
        The flux of the corresponding galaxies
    SliceNumber : int
        The number of non-subgridded slices of the total tomography, in frequency space. Assumes they are evenly spread
    MapLength : int
        The number of non-subgridded pixels making up the maps. 1 pixel = 1 beam FWHM
    MapSpreadParameter : int, optional
        The subgridding factor for the map
    FreqSpreadParameter : int, optional
        The subgridding factor for the bands

    Returns
    -------
    numpy.ndarray
        The flux cube with voxels filled, before any convolving (if wanted)

    """
    xpix,ypix,zpix=VariableList["xpix"],VariableList["ypix"],VariableList["zpix"]
    #in the case where we don't subdivide, this is simply inserting flux into correct voxel
    #if we do subdivide, this is just stage 1, as we then convolve
    FluxCube=np.zeros((SliceNumber*FreqSpreadParameter,MapLength*MapSpreadParameter,MapLength*MapSpreadParameter),dtype=np.float32)
    for i in tqdm(range(len(FluxArray))):
        FluxCube[zpix[i]][ypix[i]][xpix[i]]=FluxCube[zpix[i]][ypix[i]][xpix[i]]+FluxArray[i]
    return FluxCube



def lorentzian_kernel_1d(fwhm,x_size):
    """
    Create a 1D Lorentzian kernel normalized so that the sum of its elements is 1. Used for beam convolution of the above
    
    Parameters
    ----------
    fwhm : float
        The full-width at half-maximum (FWHM) of the Lorentzian.
    x_size : int
        Length of the kernel array. 
    
    Returns
    -------
    numpy.ndarray
        Normalized 1D Lorentzian kernel.
    """

    # Create symmetric x positions centered at zero.
    gamma=fwhm/2 #the half-width (HWHM)
    half_size = x_size // 2
    x = np.arange(-half_size, half_size + 1)

    # Evaluate the Lorentzian function
    kernel = (1 / np.pi) * gamma / (x**2 + gamma**2)

    # Normalize the kernel so its sum equals 1.
    kernel /= kernel.sum()
    return kernel

def gaussian_kernel_2d(fwhm, grid_size):
    """
    Create a 2D gaussian kernel normalized so that the sum of its elements is 1. Used for beam convolution of the above
    
    Parameters
    ----------
    fwhm : float
        The full-width at half-maximum (FWHM) of the Lorentzian.
    grid_size : int
        Length of one side of the kernel array. Assume is a square
    
    Returns
    -------
    numpy.ndarray
        Normalized 2D gaussian kernel.
    """    

    sigma=fwhm/2.355
    # Create symmetric grid centered at zero. The half-size is chosen so that the actual grid size is always odd(as grid size being even can lead to problems)

    half_size = grid_size // 2
    xmesh, ymesh = np.meshgrid(np.linspace(-half_size,half_size+1,half_size*2+1), np.linspace(-half_size,half_size+1,half_size*2+1))

    # Evaluate gaussian. Note, need "-0.5" to ensure centered around the center of pixel
    kernel=(np.exp(-(((xmesh-0.5)**2 +(ymesh-0.5)**2)/( 2.0 * sigma**2 ) )))/(2*np.pi*sigma**2) 

    # Normalize the kernel so its sum equals 1.
    kernel /= kernel.sum()
    return kernel


def convolveKernel3DMap(FluxCube,SliceNumber,MapLength,ConvMap=False,ConvZ=False,MapSpreadParameter=1,FreqSpreadParameter=1,pixel_kern=15):
    """
    Unless we want the more simplified map, where all flux is stored in voxels of size=beamFWHM, we want to convolve the flux with the beam
    To do so, we subgrid the map (done in previous step as noted), then we convolve with the beam - a lorentzian in the freq direction, a gaussian in the map
    This uses the above two functions
    As the MSP, FSP are the subgridding factors, they also serve as the FWHM of the beam in each direction, very conveniantly for us
    This is by far the longest step of the process of making each map
    Parameters
    ----------
    FluxCube : np.ndarray
        The flux cube before convolving with the beam
    SliceNumber : integer
        The number of non-subgridded slices of the total tomography, in frequency space. Assumes they are evenly spread
    MapLength : integer
        The number of non-subgridded pixels making up the maps. 1 pixel = 1 beam FWHM
    ConvMap : boolean, optional
        Whether we are doing a map beam convolution. The default is False.
    ConvZ : boolean, optional
        Whether we are doing a frequency beam convolution. The default is False.
    MapSpreadParameter : integer, optional
        Subgridding parameter for the x/y direction, thereby also the beamFWHM in pixel units in that direction. The default is 1.
    FreqSpreadParameter : integer, optional
        Subgridding parameter for the z direction, thereby also the beamFWHM in pixel units in that direction. The default is 1.

    Returns
    -------
    numpy.ndarray
        The convolved flux cube (if not convolving in either direction, no changes are made)

    """
    if ConvMap:
        #note - the kernel size being too large (i.e. size of map) would make this far too slow
        #therefore, this should help reduce the kernel size. Or, use ML*MSP-2 (largest safe value), just in case that's smaller than 15,
        #as if kernelsize>mapsize, can run into problems. In 99% of cases, a size of 15 should be more than enough
        kernsize=int(min(MapLength*MapSpreadParameter-2,pixel_kern))
        gausskern=gaussian_kernel_2d(MapSpreadParameter, kernsize)
        #we convolve each 2D slice separately
        for j in tqdm(range(len(FluxCube))):
            FluxCube[j]=convolve(FluxCube[j], gausskern)

    if ConvZ:
        #this taken from Ankur Dev's data_cube_scripts
        kernsize=int(min(SliceNumber*FreqSpreadParameter-2,pixel_kern))
        lornkern=lorentzian_kernel_1d(FreqSpreadParameter,kernsize)
        #this applies to each pixel in the map plane
        FluxCube = np.apply_along_axis(lambda spectrum: convolve(spectrum, lornkern, boundary='extend'),
                                                axis=0, arr=FluxCube)
    return FluxCube


def runFMC(cosmo,
           SourceFiles,Mapname,
           StartingRedshift,EndingRedshift,RestFrequency,
           VariableLabel1,VariableLabel2,ModelCoefficients,
           SliceNumber,PixelMapLength,FreqSpreadParameter=1,MapSpreadParameter=1,ConvMap=False,ConvZ=False,
           NameParams="",PixelScalingFactor="",OriginalMapParams="",
           COModel=False,COTransition=1,muCoeff1=1,muCoeff2=2.5,SLEDType="Normal",SLEDFilename="",MSFilename=""):
    """
    The entire procedure! Using the source files, and the relevant indexes/columns, we load all necessary catalogue data
    We then use a bulk property model to calculate the log Luminosity of the line we are interested in
    (if a CO line, we typically propagate along a SLED)
    Flux is then calculated (units of Jy/sr), sorted into the flux cube, then any beam convolution is done
    We then save the cube in a .npz file, with valuable metadata for power spectra calcs
    This is currently compatible with COSMOS2020 and FYST/CCAT, but edits to fetchVariables should make this compatible with any catalogue set

    
    Parameters
    ----------
    cosmo : astropy cosmology object
        The cosmology we use for determining e.g. luminositiy distances
    SourceFiles : list of str
        The list of files with the catalogues within. We can have several, load individually
    Mapname : str
        The output name we save to
    StartingRedshift, EndingRedshift  : float
        The starting and ending redshifts for this line for this cube
    RestFrequency : float
        The rest frequency for the given line we are covering
    VariableLabel1,VariableLabel2 : str
        The model labels we use for calculating logLsun (see calcLsun)
    ModelCoefficients : list of floats
        The parameteres used for said model, assuming it is a 2D quadratic (see calcLsun)
    SliceNumber, PixelMapLength  : int
        The dimensions of the final flux cube, before subgridding. SliceNumber is divisions in frequency space (assume even), PixelMapLength is number of on-sky beams covering tomography, where 1 beam FWHM=1 pixel
    #####################all following parameters are optional. The first few determine if we want to "subgrid" for convolving with a beam
    FreqSpreadParameter,MapSpreadParameter : int, optional
        If we wish to convolve with a beam, we should subgrid the voxels. These are the corresponding parameters for each direction. The default is 1.
    ConvZ, ConvMap : boolean
        If we wish to convolve with a beam, this confirms it. Separate for each direction. The default is False.

    #####################these parameters are about loading data
    NameParams : dict of str, optional
        For above, we need to load the right columns. We use a "dict" object tying the properties we want to given values. The default is "".
    ############for COSMOS2020 specifically, we need to use these two
    PixelScalingFactor : int, optional
        COSMSOS2020 uses a co-ordinate system with 1 pixel=0.15 arcsec, which we need to convert to the beams of FYST. PixelScalingFactor*0.15'' gives the beam size, e.g. for 410GHz, PSF=248-> beam size 37.2 arcsec. The default is "".
    OriginalMapParams : list, optional
        This is a list of integers that crops out the edges of COSMOS2020 data, and provides coords to renormalise the coord system to start from 0,0. Normally use [45000,29000,29000,7000,36000,7600,36600]  for 1.2x1.2deg². The default is "".
    #####################THESE PARAMETERS USED IF MAKING CO CUBE
    COModel : boolean, optional
        IF we use a CO model, we need to take certain things into account. The default is False.
    COTransition : int, optional
        The exact CO transisiton we use (the upper J value used). The default is 1.
    muCoeff1,muCoeff2 : floats, optional
        The CO SLED calibration values (see conversionCOSLED). The default is 1 and 2.5.
    SLEDType : str, optional
        Any deviation of the SLED, instead of following the default SLED (conversionCOSLED) The default is "Normal".
    SLEDFilename, MSFilename : str, optional
        These filenames have the parameters of the SLEDs we use, as well as the main sequence for the samples we use. The default is "".
    ifLDASH : boolean, optional
        By default, we find LCO. However, if our model/SLED find L'CO, we need to convert back to LCO, which this checks for. The default is False.

    Returns
    -------
    None. (we save the cube and various metadata for power spectra in a file)

    """
    #we make an unconvolved flux cube for each sourcefile, before convolving, as that step is the longest
    FluxCube=np.zeros((SliceNumber*FreqSpreadParameter,PixelMapLength*MapSpreadParameter,PixelMapLength*MapSpreadParameter), dtype=np.float32)
    #to make a map, you typically need more that one file, hence doing this iteratively
    for SourceFile in SourceFiles:
        #for each file, run through the basic process
        Data=CE.loadSample(SourceFile)
        #arcsecs covered by a unit pixel in the COSMOS2020 catalogue. Required to calc beamsizerad from the PSF
        OriginalPixelLength=0.15 #this is fixed for COSMOS2020. If using a different catalogue with different arbitrary value, you will need to change
        BeamSizeRad=((PixelScalingFactor*OriginalPixelLength)*u.arcsecond).to(u.rad).value
        #depending on the catalogue we use, we need to load data in different ways
        VariableList,Failure=fetchVariablesCOSMOS(Data,StartingRedshift,EndingRedshift,OriginalMapParams,PixelScalingFactor,SliceNumber,RestFrequency,NameParams,MapSpreadParameter=MapSpreadParameter,FreqSpreadParameter=FreqSpreadParameter)

        
        #if we actually have some valid galaxies, we make an updated cube
        if not Failure:
            LumArray=calcLsun(VariableList,VariableLabel1,VariableLabel2,ModelCoefficients)
            if COModel: #need to propagate from CO(1-0) calculated above
                LumArray=conversionCOSLED(VariableList, LumArray,muCoeff1,muCoeff2,COTransition,SLEDFilename,MSFilename,SLEDType=SLEDType)

            FluxArray=fluxUnitConversion(cosmo,VariableList["Redshift"],LumArray,SliceNumber,BeamSizeRad,StartingRedshift,EndingRedshift,RestFrequency,FreqSpreadParameter,MapSpreadParameter)
            FluxCube+=createSimple3DMap(VariableList,FluxArray,SliceNumber,PixelMapLength,MapSpreadParameter=MapSpreadParameter,FreqSpreadParameter=FreqSpreadParameter)
    #only after we are done with each sample file to we convolve
    FluxCube=convolveKernel3DMap(FluxCube,SliceNumber,PixelMapLength,MapSpreadParameter=MapSpreadParameter,FreqSpreadParameter=FreqSpreadParameter,ConvMap=ConvMap,ConvZ=ConvZ)
    #if the cube was made (which we check for), we save the cube with metadata
    if np.sum(FluxCube)!=0:
        np.savez(Mapname,
                  SliceNumber=SliceNumber,StartingRedshift=StartingRedshift,EndingRedshift=EndingRedshift,BeamSizeRad=BeamSizeRad,
                  RestFrequency=RestFrequency,VariableLabel1=VariableLabel1,VariableLabel2=VariableLabel2,
                  OriginalMapParams=OriginalMapParams,ModelCoefficients=ModelCoefficients,FluxMap=FluxCube)
        print("Map Making Complete")
    else:
        print("Failure to make cube, returning nothing")



