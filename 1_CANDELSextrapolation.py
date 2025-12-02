"""
This determines completeness from CANDELS and COSMOS2020 files
Note, it is often far more efficient for memory management to trim COSMOS2020 into smaller chunks
As we treat each z~0.5 chunk separately in our methodology, we do so for our sample files as well, in a code we do not include here
This calibration is specifically done above the mass completeness limits from Weaver+2022. Fitting mass functions extrapolates below those limits
"""

#imports
from astropy.io import fits
import numpy as np



def loadSample(Samplename): 
    """
    Load the appropriate fits file with the relevant catalogue
    
    Parameters
    ----------
    Samplename: string
        The pathname of the file to be loaded
    
    Returns
    -------
    numpy.ndarray
        Data cube, in fits format. It may or may not have column data
    """    
    hdul=fits.open(Samplename,memmap=True)
    Data=hdul[1].data
    hdul.close()
    
    return Data


def calcCANDELSDensity(Data_mass,Data_multiband,z_lower, mass_cutoff,map_area=(23.8/60)*(8.6/60),mass_index=14,z_index=5,IRlum_index=43): 
    """
    This calculates the number density of galaxies above the mass completeness limit in a Dz~0.5 range. We take properties from the CANDELS catalogues to determine this
    Parameters
    ----------
    Data_mass : numpy array
        the 2D array containing data from the mass file of CANDELS
    
    Data_multiband : numpy array
        the 2D array containing data from the photoz file of CANDELS
    
    z_lower : float
        the lower bound of redshift we are interested in (the upper bound is fixed at 0.5 higher)
    
    mass_cutoff : float
        the lower limit, in log of stellar units, above which we examine galaxies in the sample. Determined by equations in Weaver+22
        
    map_area : float, optional
        Area of CANDELS in deg²
        
    mass_index : int, optional
        Index of mass we use in CANDELS data
        
    z_index : int, optional
        Index of redshift we use in CANDELS data
        
    IRlum_index_index : int, optional
        Index of IR luminosity we use in CANDELS data (used to determine the magnitude of the galaxy, another cutoff we abide by)
    
    Returns
    -------
    CANDELS_density : float
        number of valid galaxies above mass cutoff per degree squared

    """
    z_higher=z_lower+0.5 #0.5 redshift intervals
    
    
    
    #these conditions you will need to manually set depending on the situation
    #in this case, we make sure galaxies are above the mass completeness cutoff, as well as within redshift range, and within reliable magnitude limits for COSMOS2020 as well
    ValidGalaxies=(Data_mass[:,z_index]>z_lower)*(Data_mass[:,z_index]<z_higher)*(Data_mass[:,mass_index]>mass_cutoff)*(23.9-2.5*np.log10(np.absolute(Data_multiband[:,IRlum_index]))<26)
    num_galaxies=np.sum(ValidGalaxies) #number of galaxies in the range
    
    CANDELS_density=num_galaxies/map_area #number of galaxies per deg² for CANDELS in the range
    return CANDELS_density






def calcCOSMOSDensity(z_lower, mass_cutoff,map_area=1.2*1.2,mass_index=8,X_index=3,Y_index=4,OriginalMapParams=[45000,29000,29000,7000,36000,7600,36600],file_folder=""):
    """
    Repeating the above but for COSMOS2020    

    Parameters
    ---------- 
    z_lower : float
        the lower bound of redshift we are interested in (the upper bound is fixed at 0.5 higher)
    
    mass_cutoff : float
        the lower limit, in log of stellar units, above which we examine galaxies in the sample. Determined by equations in Weaver+22
        
    map_area : float, optional
        Area of COSMOS2020 in deg²
        
    mass_index : int, optional
        Index of mass we use in COSMOS2020 data
    X/Y_index : int, optional
        Index of x and y coordinates we use in COSMOS2020 data
    OriginalMapParams : list, optional
        a list of 7 values describing COSMOS. The first is the size of the cube in its arbitrary pixel units (of 0.15 arcsec). The second and third are the desired cube 
        dimensions, in those same units, for x and y. The fourth and fifth are the lower and upper limits, in the same pixel units, for x (sixth and seventh are for y) 
    file_folder : string, optional
        folder where the COSMOS2020 files are kept in

    Returns
    -------
    COSMOS_density : float
        number of valid galaxies above mass cutoff per degree squared

    """
    
    z_higher=z_lower+0.5 #similar to the above
    num_galaxies=0 
    File_base=file_folder+("SampleCONDENSED_COSMOS2020_z~Z1-Z2_BASE.fits").replace("Z1-Z2",str(z_lower).replace(".",",")+"-"+str(z_higher).replace(".",",")) #filename we use to load
    Data_base=loadSample(File_base)
    #finding all galaxies within the map range, and the mass cutoff
    ValidGalaxies=(Data_base[:,X_index]>=OriginalMapParams[3])*(Data_base[:,X_index]<=OriginalMapParams[4])*(Data_base[:,Y_index]>=OriginalMapParams[5])*(Data_base[:,Y_index]<=OriginalMapParams[6])*(Data_base[:,mass_index]>mass_cutoff)
    num_galaxies=np.sum(ValidGalaxies)
    COSMOS_density=num_galaxies/map_area #number of galaxies per deg² for COSMOS2020 in the range
    return COSMOS_density



def createCANDELSRatios(file_save):
    """
    Finding the ratio of the CANDELS and COSMOS2020 for all redshifts in z~0.5 range, finding the "CANDELS" ratios

    Parameters
    ----------
    file_save : string
        the file location where we store the file

    Returns
    -------
    None.

    """
    #sample files we use for CANDELS, which include physical parameters, various stellar mass calculations, multiband wavelength data, and redshift data. We need these to calibrate our extrapolation
    Samplename_mass="hlsp_candels_hst_wfc3_cos_multi_v1_mass-cat.fits"
    Samplename_multiband="lsp_candels_hst_wfc3_cos-tot-multiband_f160w_v1_cat.fits"
    Data_mass=loadSample(Samplename_mass)
    Data_multiband=loadSample(Samplename_multiband)


    ###we run our processes iteratively for each  z~0.5 chunk. We calibrate these based on the "total" completeness equation in Weaver et al. 2023

    z_bands_lower=np.arange(0,8.5,0.5) #redshift
    z_bands_upper=z_bands_lower+0.5
    z_bands_midpoints=(z_bands_upper+z_bands_lower)/2
    
    #the mass limits are from the mass completeness equation, taken from Weaver+2022 (COSMOS2020 paper)
    mass_cutoffs=0-3.23e7*(1 + z_bands_midpoints) + 7.83e7*(1 + z_bands_midpoints)**2 #in Msol units
    
    
    CANDELS_factors=np.zeros(len(z_bands_lower))
    #we run the procedure. We always take a maximum ratio of "1"
    for i in range(len(z_bands_lower)):
        z=z_bands_lower[i]
        cutoff=mass_cutoffs[i]
        CANDELS_factors[i]=np.min([1,calcCANDELSDensity(Data_mass,Data_multiband,z, cutoff)/calcCOSMOSDensity(Data_mass, Data_multiband, z, cutoff)])
    #we then save in a savez file
    np.savez(file_save,z_bands_lower=z_bands_lower,z_bands_higher=z_bands_upper,CANDELS_factors=np.asarray(CANDELS_factors))
    

    
    
    


