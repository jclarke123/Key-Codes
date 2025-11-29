#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This determines completeness from CANDELS and COSMOS2020 files
Note, it is often far more efficient for memory management to trim COSMOS2020 into smaller chunks
As we treat each z~0.5 chunk separately in our methodology, we do so for our sample files as well, in a code we do not include here
This calibration is specifically done above the 
"""

#imports
from astropy.io import fits
import numpy as np



def loadSample(Samplename): 
    """
    

    Parameters
    ----------
    Filename : TYPE
        DESCRIPTION.

    Returns
    -------
    Data : TYPE
        DESCRIPTION.

    """
    hdul=fits.open(Samplename,memmap=True)
    Data=hdul[1].data
    hdul.close()
    
    return Data


def calcCANDELSDensity(Data_mass,Data_multiband,z_lower, mass_cutoff,map_area=(23.8/60)*(8.6/60),mass_index=14,z_index=5,IRlum_index=43): #for a given redshift range, return the density
    """
    
    Parameters
    ----------
    Data_mass : TYPE
        DESCRIPTION.
    
    Data_multiband : TYPE
        DESCRIPTION.
    
    z_lower : TYPE
        DESCRIPTION.
    
    mass_cutoff : TYPE
        DESCRIPTION.
        
    map_area : TYPE, optional
        DESCRIPTION.
        
    mass_index : TYPE, optional
        DESCRIPTION.
        
    z_index : TYPE, optional
        DESCRIPTION.
        
    IRlum_index_index : TYPE, optional
        DESCRIPTION.
    
    Returns
    -------
    CANDELS_density : TYPE
        DESCRIPTION.

    """
    z_higher=z_lower+0.5 #0.5 redshift intervals
    num_galaxies=0 #number of galaxies in the range
    for i, Galaxy in enumerate(Data_mass):
        #these conditions you will need to manually set depending on the situation
        #in this case, we make sure galaxies
        if z_lower<Galaxy[z_index] and Galaxy[z_index]<z_higher and Galaxy[mass_index]>mass_cutoff and 23.9-2.5*np.log10(np.absolute(Data_multiband[i][IRlum_index]))<26:
            num_galaxies+=1
    CANDELS_density=num_galaxies/map_area #number of galaxies per deg² for CANDELS in the range
    return CANDELS_density



################################re-imagining slightly - take density for COSMOS, using same knee



def calcCOSMOSDensity(Data_mass,Data_multiband,z_lower, mass_cutoff,map_area=1.2*1.2,mass_index=8,X_index=3,Y_index=4,OriginalMapParams=[45000,29000,29000,7000,36000,7600,36600],file_folder=""):
    """
    

    Parameters
    ----------
    Data_mass : TYPE
        DESCRIPTION.
    Data_multiband : TYPE
        DESCRIPTION.
    z_lower : TYPE
        DESCRIPTION.
    mass_cutoff : TYPE
        DESCRIPTION.
    map_area : TYPE, optional
        DESCRIPTION. The default is (23.8/60)*(8.6/60).
    mass_index : TYPE, optional
        DESCRIPTION. The default is 14.
    z_index : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    COSMOS_density : TYPE
        DESCRIPTION.

    """
    
    z_higher=z_lower+0.5
    num_galaxies=0 
    File_base=file_folder+("SampleCONDENSED_COSMOS2020_z~Z1-Z2_BASE.fits").replace("Z1-Z2",str(z_lower)+"-"+str(z_higher))
    File_mask=file_folder+("SampleCONDENSED_COSMOS2020_z~Z1-Z2_STELLARMASK.fits").replace("Z1-Z2",str(z_lower)+"-"+str(z_higher))
    Data_base=loadSample(File_base)
    Data_mask=loadSample(File_mask)
    for Galaxy in Data_base:
        if Galaxy[X_index]>=OriginalMapParams[3] and Galaxy[X_index]<=OriginalMapParams[4] and Galaxy[Y_index]>=OriginalMapParams[5] and Galaxy[Y_index]<=OriginalMapParams[6]:
            if Galaxy[mass_index]>mass_cutoff:
                num_galaxies+=1
    for Galaxy in Data_mask:
        if Galaxy[X_index]>=OriginalMapParams[3] and Galaxy[X_index]<=OriginalMapParams[4] and Galaxy[Y_index]>=OriginalMapParams[5] and Galaxy[Y_index]<=OriginalMapParams[6]:
            if Galaxy[mass_index]>mass_cutoff:
                num_galaxies+=1
    COSMOS_density=num_galaxies/map_area #number of galaxies per deg² for CANDELS in the range
    return COSMOS_density







def createCANDELSRatios(file_save):
    """
    

    Parameters
    ----------
    file_save : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #sample files we use for CANDELS, which include physical parameters, various stellar mass calculations, multiband wavelength data, and redshift data. We need these to calibrateour extrapolation
    Samplename_mass="hlsp_candels_hst_wfc3_cos_multi_v1_mass-cat.fits"
    Samplename_multiband="lsp_candels_hst_wfc3_cos-tot-multiband_f160w_v1_cat.fits"
    Data_mass=loadSample(Samplename_mass)
    Data_multiband=loadSample(Samplename_multiband)


    ###we run our processes iteratively for each  z~0.5 chunk. We calibrate these based on the "total" completeness equation in Weaver et al. 2023

    z_bands_lower=np.arange(0,8.5,0.5) #redshift
    z_bands_upper=z_bands_lower+0.5
    z_bands_midpoints=(z_bands_upper+z_bands_lower)/2
    mass_cutoffs=0-3.23e7*(1 + z_bands_midpoints) + 7.83e7*(1 + z_bands_midpoints)**2 #Msol
    
    
    CANDELS_factors=[]
    
    for i,z in enumerate(z_bands_lower):
        cutoff=mass_cutoffs[i]
        CANDELS_factors.append(np.min([1,calcCANDELSDensity(Data_mass,Data_multiband,z, cutoff)/calcCOSMOSDensity(Data_mass, Data_multiband, z, cutoff)]))
    np.savez(file_save,z_bands_lower=z_bands_lower,z_bands_higher=z_bands_upper,CANDELS_factors=np.asarray(CANDELS_factors))
    

    


