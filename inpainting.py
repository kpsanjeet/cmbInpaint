import healpy as hp
import numpy as np

class hWavlates:
    """
    This class use to decompse the CMB maps or mask  into approximation and detail coefficients
    using the HEALPix wavelet(CMB map or mask must be in NESTED order).
    ref. https://inspirehep.net/literature/2623240
    
    Parameters =>
                nside_in -> NSIDE of the input map.
                nside_min -> Minimum NSIDE upto which we want downgrade(decompose) the maps.
                             Default value is 8.
                groupPixNum -> Numer of pixels in a group(Number of pixels which are combine to downgrade
                                the map). Default value is 4.
    """
    def __init__(self, nside_in, nside_min=8, groupPixNum=4):
        self.nside_in = nside_in
        self.nside_min = nside_min
        self.groupPixNum = groupPixNum
        self.j_init = int(np.log2(nside_in)) # nside_in = 2^(j_init)
        self.j_final = int(np.log2(nside_min)) # nside_min = 2^(j_final)
        
    def _decompose(self, map_in):
        """
        This function takes a map or mask in NESTED order and returns approximation and detail coefficients
        after decomposition. 
        """
        npix = map_in.size
        groups = map_in.reshape(npix//self.groupPixNum, -1)# Diveding maps into groups(4 pixels per group). Reshaping in (npix/4, 4)
        approx = np.mean(groups, axis=1) # Taking the average of each groups
        detail = groups - approx[:, None]
        # flatten() function used to merge the groups(Reshaping in (npix, ))
        return approx, detail.flatten()
        
    def decomposeCMB(self, map_in):
        """
        This function takes a map in NESTED order and returns approximations and details coefficients
        in the range j_init to j_final(nside_in to nside_min).
        Parameters =>
                map_in -> CMB map.
        Return => 
                approx -> approximations coefficients(1D arrar).
                details -> List of detail coefficients(detail maps). Nubers of detail maps depends on 
                           j_init(nside_in) and j_final(nside_min)
        """
        approx = np.copy(map_in)# Storing approximations coefficients(approximations maps)
        details = []          # Storing details coefficients(details maps)
        for i in range(self.j_init-self.j_final):
                approx, detail = self._decompose(approx)
                details.append(detail)
        return approx, details
    
    def _binaryMask(self, mask):
        """ 
        This function used to convert the downgraded mask(some values appears between 0 and 1) 
        into binary mask(0,1)
        """
        mask[mask<0.5] = 0
        mask[mask>=0.5] = 1
    
    def decomposeMask(self, mask):
        """
        This function takes a mask in NESTED order and returns approximations and details coefficients
        in the range j_init to j_final(nside_in to nside_min).
        Parameters =>
                map_in -> CMB mask(1d array).
        Return => 
                approx -> approximations coefficients(1D arrar) of mask.
                details -> List of detail coefficients(detail maps) of mask. Nubers of detail maps depends on 
                           j_init(nside_in) and j_final(nside_min)
        """
        approx = np.copy(mask) # Storing approximations coefficients(approximations maps)
        details = []          # Storing details coefficients(details maps)
        for i in range(self.j_init-self.j_final):
                approx, detail = self._decompose(approx)
                self._binaryMask(approx)# Converting in binary mask
                self._binaryMask(detail)# Converting in binary mask
                details.append(detail) # Storing details coefficients(details maps)
        return approx, details
    
    def recostruct(self, approxCMB, detailsCMB):
        """
        This function used to recostruct the CMB map from approximations and details coefficients.
        Parameters =>
                approxCMB -> approximations coefficients(approx map).(1d array)
                detailsCMB -> List of approximations coefficients(detail maps).
        Return => 
                cmbMap -> Reconstructed CMB map(1d array).
        """
        cmbMap = np.copy(approxCMB)
        for i in range(len(detailsCMB)):
            Detail = detailsCMB[~i]
            sum_ = Detail.reshape(Detail.size//self.groupPixNum, self.groupPixNum) + cmbMap[:, None]
            cmbMap = sum_.flatten()
        return cmbMap
    
    class inpainting(hWavlates):
    """
    This class use to inpaint CMB map(NESTED order).
    ref. https://inspirehep.net/literature/2623240
    
    Parameters =>
                nside_in -> NSIDE of the input map.
                nside_min -> Minimum NSIDE upto which we want downgrade(decompose) the maps.
                             Default value is 8.
                groupPixNum -> Numer of pixels in a group(Number of pixels which are combine to downgrade
                                the map). Default value is 4.
                iterNum -> Number of iterations used in diffuse inpainting. Default value is 2000.
    """
    def __init__(self, nside_in, nside_min=8, groupPixNum=4, iterNum=2000):
        super().__init__(nside_in, nside_min, groupPixNum)
        self.iterNum = iterNum
        
    def _deffuseInpainting(self, map_in, mask):
        """
        Diffuse inpainting of CMB map.
        Parameters =>
                map_in -> CMB map(1d array).
                mask -> Mask(1d array).
        Return => 
                inpMap -> Inpainted CMB map(1d array).
        """
        nside_in = hp.npix2nside(map_in.size)
        inpMap = np.copy(map_in)
        idx = np.where(mask==0.)
        #Neighbours corresponding to each pixels in masked region
        nb = hp.get_all_neighbours(nside_in, idx[0],  nest=True).T
        for k in range(self.iterNum):
            for i, j in zip(idx[0], nb):
                #Filling masked pixel with the mean of it's  neighbours
                inpMap[i] = np.mean(inpMap[j[j!=-1]])
        return inpMap #Inpainted map
    
    def _deffuseInpainting2(self, map_in, mask):
        """
        Diffuse inpainting of CMB map.
        Parameters =>
                map_in -> CMB map(1d array).
                mask -> Mask(1d array).
        Return => 
                inpMap -> Inpainted CMB map(1d array).
        """
        nside_in = hp.npix2nside(map_in.size)
        inpMap = np.copy(map_in)
        idx = np.where(mask==0.)
        #Neighbours corresponding to each pixels in masked region
        nb = hp.get_all_neighbours(nside_in, idx[0],  nest=True).T
        sum_ = 0
        for k in range(self.iterNum):
            tem_inp_pix = inpMap[idx]
            for i, j in zip(idx[0], nb):
                #Filling masked pixel with the mean of it's  neighbours
                inpMap[i] = np.mean(inpMap[j[j!=-1]])
            inp_pix = inpMap[idx]
            cond = np.mean((inp_pix-tem_inp_pix)/tem_inp_pix)
            if abs(cond)<1e-3:
                break
        return inpMap
    
    def _fsky(self, mask):
        """Fractional part of the sky that is left unmasked.
            Parameters =>
                mask -> Mask(1d array).
            Return => 
                 A floating Point number.
        """
        return len(mask[mask!=0])/len(mask)
    
    def _ringToNest(self, map_in):
        if len(map_in.shape) == 1:
            map_in[:] = hp.reorder(map_in, r2n=True)
        elif map_in.shape[0] == 3 and len(map_in.shape)==2:
            for i in range(len(map_in)):
                map_in[i] = hp.reorder(map_in[i], r2n=True)
        else:
            raise Exception("The shape of input map must be (npix, ) or (3, npix).")
            
    def _nestToRing(self, map_in):
        if len(map_in.shape) == 1:
            map_in[:] = hp.reorder(map_in, n2r=True)
        elif map_in.shape[0] == 3 and len(map_in.shape)==2:
            for i in range(len(map_in)):
                map_in[i] = hp.reorder(map_in[i], n2r=True)    
        else:
            raise Exception("The shape of input map must be (npix, ) or (3, npix).")
            
    def _corrDetailMap(self, maskedDetailCMB, filledDetailCMB, maskedDetail, fsky_):
        """
        Correction in filled detail map(details coefficients).
        Parameters =>
                maskedDetailCMB -> Masked CMB detail map(1d array).
                filledDetailCMB -> Filled CMB detail map(1d array).
                maskedDetail -> Detail map of mask(1d array).
                fsky_ -> Fractional part of the sky that is left unmasked.
        Return => 
                corrDetailMap_ -> Corrected filled CMB detail map.
        """
        nside = hp.npix2nside(maskedDetailCMB.size)
        lmax = 3*nside-1
        #Power spectrum of masked CMB detail map after reordering from NESTED to RING.
        cl_obs = hp.anafast(hp.reorder(maskedDetailCMB, n2r=True), lmax=lmax)
        #Power spectrum and Alm of filled CMB detail map after reordering from NESTED to RING.
        cl_fill, alm_fill = hp.anafast(hp.reorder(filledDetailCMB, n2r=True), lmax=lmax, alm=True)
        #cl_corr = np.sqrt(cl_obs/fsky_/cl_fill)
        cl_corr = np.nan_to_num(np.sqrt(cl_obs/fsky_/cl_fill), posinf=0., neginf=0.) # Correction part
        #alm_corr = np.zeros_like(alm_fill, dtype=np.complex128)
        for l in range(lmax+1):
            a = hp.Alm.getidx(lmax, l, np.arange(0, l+1))
            #alm_corr[a] = alm_fill[a]*cl_corr[l]
            alm_fill[a] = alm_fill[a]*cl_corr[l] #apply correction in Alm
        
        #Converting Alm int map and reordering back from RING to NESTED.
        corrDetailMap_ = hp.reorder(hp.alm2map(alm_fill, nside=nside), r2n=True)
        
        #Replacing unmasked region of corrected detail map with the original detail map
        corrDetailMap_[maskedDetail!=0.] = maskedDetailCMB[maskedDetail!=0.]
        return corrDetailMap_

    def _OCDInpainting(self, cmb, mask):
        """
        Inpainting of CMB maps by using OCD method.
        Parameters =>
                cmb -> CMB map(1d array).
                mask -> mask(1d array).
        Return => 
                Inpainted CMB map.
        """
        approxCMB, detailsCMB = self.decomposeCMB(cmb)
        approxMask, detailsMask = self.decomposeMask(mask)
        #inpApprox = self._deffuseInpainting(approxCMB, approxMask)#Diffuse inpainting of approximation map
        inpApprox = self._deffuseInpainting2(approxCMB, approxMask)#Diffuse inpainting of approximation map
        for i in range(len(detailsCMB)):
            detailMask = detailsMask[~i]
            Detail = detailsCMB[~i]
            idxMasked = np.where(detailMask==0.)
            npixMasked = len(idxMasked[0])
            maskedDetailCMB = Detail*detailMask
            sigma = np.std(maskedDetailCMB[detailMask!=0.])
            #mean = np.mean(maskedDetailCMB[detailMask!=0.])
            Detail[idxMasked] = np.random.normal(0, 1, npixMasked)*sigma #Filling masked region
            #Correction in filled detail map
            corrDetailMap_ = self._corrDetailMap(maskedDetailCMB, Detail, detailMask, self._fsky(detailMask))
            #corrDetailMap_ = Detail

            npix = corrDetailMap_.size
            #Adding approximation and corrected detail map
            sum_ = corrDetailMap_.reshape(npix//self.groupPixNum, self.groupPixNum) + inpApprox[:, None]
            inpApprox = sum_.flatten()
            
        #Replacing unmasked region of recontructed map with the CMB map
        inpApprox[mask!=0.] = cmb[mask!=0.]
        return inpApprox
    
    def OCDInpainting(self, cmb, mask):
        """
        Inpainting of CMB maps by using OCD method.
        Parameters =>
                cmb -> I map(1d array) or in the form of np.array([I, Q, U]).
                mask -> mask(1d array) or in the form of np.array([tmask, pmask, pmask]).
                        Here tmask means temperatur mask and pmask means polarization mask.
        Return => 
                Inpainted CMB map(I) or array I, Q and U map according to the input parameters.
        """
        if len(cmb.shape) ==  len(mask.shape) == 1:
            return self._OCDInpainting(cmb, mask)
        elif cmb.shape[0] == mask.shape[0] == 3:
            IQU = np.zeros_like(cmb)
            for i in range(len(cmb)):
                IQU[i]= self._OCDInpainting(cmb[i], mask[i])
            return IQU
        else:
            raise Exception("The shape of inputs(cmb or mask) must be same.")
    
    def _MCDInpainting(self, cmb, mask, randomRealization):
        """
        Inpainting of CMB maps by using MCD method.
        Parameters =>
                cmb -> CMB map(1d array).
                mask -> mask(1d array).
                randomRealization -> Random realization(1d array) of CMB map. Gerated by using 'synfast'.
        Return => 
                Inpainted CMB map.
        """
        approxRR, detailsRR = self.decomposeCMB(randomRealization)
        approxCMB, detailsCMB = self.decomposeCMB(cmb)
        approxMask, detailsMask = self.decomposeMask(mask)
        #inpApprox = self._deffuseInpainting(approxCMB, approxMask)#Diffuse inpainting of approximation map
        inpApprox = self._deffuseInpainting2(approxCMB, approxMask)#Diffuse inpainting of approximation map
        for i in range(len(detailsCMB)):
            detailMask = detailsMask[~i]
            DetailCMB = detailsCMB[~i]
            DetailRR = detailsRR[~i]
            idxMasked = np.where(detailMask==0.)
            #Filling(replacing) masked region of detail CMB map with the detail map of random realization
            DetailCMB[idxMasked] = DetailRR[idxMasked]
            npix = DetailCMB.size
            #Adding approximation and detail map
            sum_ = DetailCMB.reshape(npix//self.groupPixNum, self.groupPixNum) + inpApprox[:, None]
            inpApprox = sum_.flatten()
            
        #Replacing unmasked region of recontructed map with the CMB map    
        inpApprox[mask!=0.] = cmb[mask!=0.]
        return inpApprox
    
    def MCDInpainting(self, cmb, mask, cls):
        """
        Inpainting of CMB maps by using MCD method.
        Parameters =>
                cmb -> I map(1d array) or in the form of np.array([I, Q, U]).
                mask -> mask(1d array) or in the form of np.array([tmask, pmask, pmask]).
                        Here tmask means temperatur mask and pmask means polarization mask.
                cls -> Theoretical power cpectrum cl or a array of cl (either 4 or 6). Use the new ordering of cl’s, 
                        ie by diagonal (e.g. TT, EE, BB, TE, EB, TB or TT, EE, BB, TE if 4 cl as input).
        Return => 
                Inpainted CMB map(I) or array I, Q and U map according to the input parameters.
        """
        randomRealization = hp.synfast(cls, nside=self.nside_in, new=True, pol=True)
        self._ringToNest(randomRealization)
        
        if len(cmb.shape) ==  len(mask.shape) == len(cls.shape) == 1:
            return self._MCDInpainting(cmb, mask, randomRealization)
        elif cls.shape[0] == 4 or cls.shape[0] == 6:
            assert len(cmb) == 3, "Input cmb must be in the form of np.array([I, Q, U])"
            assert len(mask) == 3, "Input mask must be in the form of np.array([tmask, pmask, pmask])"
            IQU = np.zeros_like(cmb)
            for i in range(len(cmb)):
                IQU[i]= self._MCDInpainting(cmb[i], mask[i], randomRealization[i])
            return IQU
        else:
            raise Exception("the Problem is related to shape of inputs(cmb or mask or cls).")