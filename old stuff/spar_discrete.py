# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:27:49 2015

@author: yhuang1
"""
from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float,Array,Str,Int
import math
import numpy as np
from scipy.optimize import fmin
from sympy.solvers import solve
from sympy import Symbol
pi=math.pi
from spar_utils import filtered_stiffeners_table

class spar_discrete(Component):
    # design variables 
    wall_thickness = np.array(Array(iotype='in', units='m',desc = 'wall thickness of each section'))
    number_of_rings = np.array(Array(iotype='in',desc = 'number of stiffeners in each section'))
    stiffener_index = Int(iotype='in',desc='index of stiffener from filtered table')
    # inputs 
    gravity = Float(9.806,iotype='in', units='m/s**2', desc='gravity')
    air_density = Float(1.198,iotype='in', units='kg/m**3', desc='density of air')
    water_density = Float(1025,iotype='in',units='kg/m**3',desc='density of water')
    water_depth = Float(iotype='in', units='m', desc='water depth')
    load_condition =  Str(iotype='in',desc='Load condition - N for normal or E for extreme')
    significant_wave_height = Float(iotype='in', units='m', desc='significant wave height')
    significant_wave_period = Float(iotype='in', units='m', desc='significant wave period')
    keel_cg_mooring = Float(iotype='in', units='m', desc='center of gravity above keel of mooring line')
    keel_cg_operating_system = Float(iotype='in', units='m', desc='center of gravity above keel of operating system')
    reference_wind_speed = Float(iotype='in', units='m/s', desc='reference wind speed')
    reference_height = Float(iotype='in', units='m', desc='reference height')
    alpha = Float(iotype='in', desc='power law exponent')
    material_density = Float(iotype='in', units='kg/m**3', desc='density of spar material')
    E = Float(iotype='in', units='Pa', desc='young"s modulus of spar material')
    nu = Float(iotype='in', desc='poisson"s ratio of spar material')
    yield_stress = Float(iotype='in', units='Pa', desc='yield stress of spar material')
    rotor_mass = Float(iotype='in', units='kg', desc='rotor mass')
    tower_mass = Float(iotype='in', units='kg', desc='tower mass')
    free_board = Float(iotype='in', units='m', desc='free board length')
    draft = Float(iotype='in', units='m', desc='draft length')
    fixed_ballast_mass = Float(iotype='in', units='kg', desc='fixed ballast mass')
    hull_mass = Float(iotype='in', units='kg', desc='hull mass')
    permanent_ballast_mass = Float(iotype='in', units='kg', desc='permanent ballast mass')
    variable_ballast_mass = Float(iotype='in', units='kg', desc='variable ballast mass')
    number_of_sections = Int(iotype='in',desc='number of sections in the spar')
    outer_diameter = np.array(Array(iotype='in', units='m',desc = 'outer diameter of each section'))
    length = np.array(Array(iotype='in', units='m',desc = 'wlength of each section'))
    end_elevation = np.array(Array(iotype='in', units='m',desc = 'end elevation of each section'))
    start_elevation = np.array(Array(iotype='in', units='m',desc = 'start elevation of each section'))
    bulk_head = Array(iotype='in',desc = 'N for none, T for top, B for bottom')
    VD = Float(1.2931,iotype='in', units='m/s**2', desc='acceleration')
    # VD = Float(iotype='in', units='m/s**2', desc='acceleration of system')
    # outputs
    VAL = Array(iotype='out',desc = 'unity check for axial load - local buckling')
    VAG = Array(iotype='out',desc = 'unity check for axial load - genenral instability')
    VEL = Array(iotype='out',desc = 'unity check for external pressure - local buckling')
    VEG = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    # shell_mass = Float(iotype='out',desc = 'mass of shell')
    # for checking stuff 
    FTHETAS = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    FTHETAR = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    FXEL = Array(iotype='out',desc = 'unity check for external pressure - general instability') 
    FXCL = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    FREL = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    FRCL = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    FXEG =Array(iotype='out',desc = 'unity check for external pressure - general instability')
    FXCG = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    PEG = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    FREG = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    FRCG = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    FTHETACL = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    FPHICL = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    FTHETACG = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    FPHICG = Array(iotype='out',desc = 'unity check for external pressure - general instability')
    shell_and_ring_mass = Float(iotype='out',desc = 'unity check for external pressure - general instability')
    def __init__(self):
        super(spar_discrete,self).__init__()
    def execute(self):
        ''' 
        '''
        def plasticityRF(F):
            dum = FY/F
            if F > FY/2:
                return F*dum*(1+3.75*dum**2)**(-0.25)
            else: 
                return F*1
        def samesign(a, b):
            return a * b > 0
        def bisect(func, low, high):
            'Find root of continuous function where f(low) and f(high) have opposite signs'
            assert not samesign(func(low), func(high))
            for i in range(54):
                midpoint = (low + high) / 2.0
                if samesign(func(low), func(midpoint)):
                    low = midpoint
                else:
                    high = midpoint
            return midpoint
        def frustrumVol(D1,D2,H): # array inputs
            N = len(D1)
            fV = np.array([0.]*N)
            r1 = D1/2.
            r2 = D2/2.
            fV = math.pi * (H / 3) * (r1**2 + r1*r2 + r2**2)
            return fV
        def frustrumCG(D1,D2,H):  # array inputs
            # frustrum vertical CG
            N = len(D1)
            fCG = np.array([0.]*N)
            r1 = D1/2.
            r2 = D2/2.    
            dum1 = r1**2 + 2.*r1*r2 + 3.*r2**2
            dum2 = r1**2 + r1 * r2 + r2**2
            fCG = H / 4. * (dum1/dum2)
            return fCG
        def ID(D,WALL):  # array inputs
            D = np.asarray(D)
            is_scalar = False if D.ndim>0 else True
            D.shape = (1,)*(1-D.ndim) + D.shape   
            N = len(D)
            ID = np.array([0.]*N)
            ID = D - 2.*WALL
            return ID if not is_scalar else (ID)
        def waveHeight(Hs): 
            return 1.1*Hs
        def wavePeriod(H):
            return 11.1*(H/G)**0.5   
        def waveNumber(T,D):
            k0 = 2 * pi / ( T * (G * D) **0.5)
            ktol =1 
            while ktol > 0.001:
                k = ( 2* pi/T) **2 / (G * np.tanh(k0*D))
                ktol = abs(k-k0)
                k0 = k
            return k
        def waveU(H,T,k,z,D,theta):
            return (pi*H/T)*(np.cosh(k*(z+D))/np.sinh(k*D))*np.cos(theta)
        def waveUdot(H,T,k,z,D,theta):
            return (2*pi**2*H/T**2)* (np.cosh(k*(z+D))/np.sinh(k*D))*np.sin(theta)
        def CD(U,D,DEN):
            RE = np.log10(abs(U) * D / DEN)
            if RE <= 5.:
                return 1.2
            elif RE < 5.301:
                return 1.1
            elif RE < 5.477:
                return  0.5
            elif RE < 6.:
                return 0.2
            elif RE < 6.301:
                return 0.4
            elif RE < 6.477:
                return 0.45
            elif RE < 6.699:
                return 0.5
            elif RE < 7.:
                return 0.6
            else:
                return 0.8
        def inertialForce(D,CA,L,A,VDOT,DEN):
            IF = 0.25 * pi * DEN * D** 2 * L * (A + CA * (A - VDOT))
            if A < 0:
                IF = -IF
            return IF
        def windPowerLaw(uref,href,alpha,H) :
            return uref*(H/href)**alpha
        def pipeBuoyancy(D):
            return pi/4 * D**2 *WDEN 
        def dragForce(D,CD,L,V,DEN):
            DF = 0.5 * DEN * CD * D * L * V**2
            if V < 0 :
                DF = -DF 
            return DF
        def calcPsi(F):
            dum = FY/2
            if F <= dum:
                return 1.2
            elif F > dum and F < FY:
                return 1.4 - 0.4*F/FY
            else: return 1
        def currentSpeed(XNEW):
            CDEPTH = [0.000, 61.000, 91.000, 130.000]
            CSPEED = [0.570, 0.570, 0.100, 0.100]
            return np.interp(abs(XNEW),CDEPTH,CSPEED)
        def curWaveDrag(Hs,Tp,WD,ODT,ODB,ELS,SL,CG,VDOT): 
            
            if Hs != 0: 
                H = waveHeight(Hs)
                k = waveNumber(Tp,WD) 
            # calculate current and wave drag 
            DL = SL /10.     # step sizes
            dtheta = pi/30. 
            S = (ODB - ODT)/SL 
            b = ODT 
            L = 0
            m = 0 
            for i in range(1,11):
                L2 = L + DL/2
                D2 = ELS -L2 
                D = S * L2 +b 
                fmax = 0.
                if (JMAX[i-1] == 0.):
                    JS = 1
                    JE = 31
                else: 
                    JS = JMAX[i-1]
                    JE = JMAX[i-1]
                for j in range(JS,JE+1):
                    V = currentSpeed(L2)
                    A = 0. 
                    if Hs != 0: 
                        V = V + waveU(H,Tp,k,D2,WD,dtheta*(j-1))
                        A = waveUdot(H,Tp,k,D2,WD,dtheta*(j-1))
                    if V != 0: 
                        CDT = CD(V,D,WDEN) 
                        f = dragForce(D,CDT,DL,V,WDEN) 
                    else:
                        f = 0
                    if Hs != 0:
                        f = f + inertialForce(D,1,DL,A,VDOT,WDEN) 
                    if f > fmax :
                        fmax = f
                        JMAX[i-1] =j
                m = m + fmax*L2 
                L = DL*i
            return m/(SL-CG)
        def calculateWindCurrentForces (flag): 
            if flag == 1: 
                VD = self.VD
            else: 
                VD = 0
            ODT = OD # outer diameter - top
            ODB = np.append(OD[1:NSEC],OD[-1]) # outer diameter - bottom
            WOD = (ODT+ODB)/2 # average outer diameter 
            COD = WOD # center outer diameter
            IDT = ID(ODT,T)
            IDB = ID(ODB,T)
            OV = frustrumVol(ODT,ODB,LB)
            IV = frustrumVol(IDT,IDB,LB)
            MV = OV - IV # shell volume 
            SHM = MV * MDEN # shell mass 
            SCG = frustrumCG(ODB,ODT,LB) # shell center of gravity
            KCG = DRAFT + ELE + SCG # keel to shell center of gravity
            KCB = DRAFT + ELE + SCG
            SHB = OV*WDEN #  outer volume --> displaced water mass
            coneH = np.array([0.]*NSEC)
            for i in range(0,len(LB)):
                if ODB[i]==ODT[i]:
                    coneH[i]=LB[i]
                else:
                    coneH[i] = -LB[i] * (ODB[i]/ODT[i]) / (1 - ODB[i]/ODT[i]) # cone height 
            # initialize arrays 
            SCF = np.array([0.]*NSEC)
            KCS = np.array([0.]*NSEC)
            SCM = np.array([0.]*NSEC)
            SWF = np.array([0.]*NSEC)
            KWS = np.array([0.]*NSEC)
            SWM = np.array([0.]*NSEC)
            BHM = np.array([0.]*NSEC)
            RGM = np.array([0.]*NSEC)
            for i in range (0,NSEC): 
            #for i in range(0,NSEC) :  # iterate through the sections
                if i <(NSEC-1) : # everything but the last section 
                    if ELE[i] <0 : # if bottom end underwater
                        HWL = abs(ELE[i]) 
                        if HWL >= LB[i]: # COMPLETELY UNDERWATER 
                            SCF[i] = curWaveDrag(Hs,Tp,WD,OD[i],OD[i+1],ELS[i],LB[i],SCG[i],VD)
                            KCS[i] = KCB[i]
                            if SCF[i] == 0: 
                                KCS[i] = 0.
                            SCM[i] = SCF[i] * (KCS[i]-KCGO)
                            SWF[i] = 0.
                            KWS[i] = 0.
                            SWM[i] = 0.
                        else: # PARTIALLY UNDER WATER 
                            if ODT[i] == ODB[i]:  # cylinder
                                SHB[i] = pipeBuoyancy(ODT[i])*HWL # redefine
                                SCG[i] = HWL/2 # redefine
                                KCB[i] = DRAFT + ELE[i] + SCG[i] # redefine 
                                ODW = ODT[i] # assign single variable
                            else: # frustrum
                                ODW = ODB[i]*(coneH[i]-HWL)/coneH[i] # assign single variable 
                                WOD[i] = (ODT[i]+ODW[i])/2  # redefine 
                                COD[i] = (ODW[i]+ODB[i])/2 # redefine 
                                SHB[i] = frustrumVol(ODW[i],ODB[i],HWL) # redefine 
                                SCG[i] = frustrumCG(ODW,ODB[i],HWL) # redefine 
                                KCB[i] = DRAFT + ELE[i] + SCG[i] # redefine 
                            SCF[i] = curWaveDrag(Hs,Tp,WD,ODW,ODB[i],0.,HWL,SCG[i],VD)
                            KCS[i] = KCB[i]
                            if SCF[i] == 0 : 
                                KCS[i] = 0.
                            SCM[i] = SCF[i]*(KCS[i]-KCGO)
                            if WREFS != 0: # if there is wind 
                                WSPEED = windPowerLaw(WREFS,WREFH,ALPHA,(LB[i]-HWL)/2) # assign single variable 
                                CDW = CD(WSPEED,WOD[i],ADEN) # assign single variable 
                                SWF[i] = dragForce(WOD[i],CDW,LB[i]-HWL,WSPEED,ADEN)
                                KWS[i]= KCG[i]
                                SWM[i] = SWF[i]*(KWS[i]-KCGO) 
                            else: # no wind 
                                SWF[i] = 0.
                                KWS[i] = 0.
                                SWM[i] = 0.
                    else: # FULLY ABOVE WATER 
                        SHB[i] = 0. # redefines 
                        KCB[i] = 0.
                        SCF[i] = 0.
                        KCS[i] = 0.
                        SCM[i] = 0.
                        if WREFS != 0: # if there is wind 
                            WSPEED = windPowerLaw(WREFS,WREFH,ALPHA,ELE[i]+LB[i]/2) # assign single variable 
                            CDW = CD(WSPEED,WOD[i],ADEN) # assign single variable 
                            SWF[i] = dragForce(WOD[i],CDW,LB[i],WSPEED,ADEN)
                            KWS[i]= KCG[i]
                            SWM[i] = SWF[i]*(KWS[i]-KCGO) 
                        else: # no wind 
                            SWF[i] = 0.
                            KWS[i] = 0.
                            SWM[i] = 0.
                    RGM[i] = N[i]*(pi*ID(WOD[i],T[i])*AR[i])*MDEN # ring mass
                else: # last section 
                    # SHM unchanged 
                    KCG[i] = DRAFT + ELE[i] + LB[i]/2  #redefine
                    # SHB already calculated 
                    # KCB already calculated 
                    SCF[i] = curWaveDrag(Hs,Tp,WD,OD[i],OD[i],ELS[i],LB[i],KCG[i],VD)
                    KCS[i] = KCG [i] # redefines
                    if SCF[i] ==0: 
                        KCS[i] = 0.
                    SCM[i] = SCF[i]*(KCS[i]-KCGO)
                    SWF[i] = 0.
                    KWS[i] = 0.
                    SWM[i] = 0.
                    RGM[i] = N[i]*(pi*ID(OD[i],T[i])*AR[i])*MDEN # ring mass
                if BH[i] == 'T':
                    BHM[i] = pi / 4 * IDT[i]**2 * T[i] * MDEN 
                    KCG[i] = (KCG[i] * SHM[i] + (DRAFT + ELS[i] - 0.5 * T[i]) * BHM[i]) / (SHM[i] + BHM[i])
                elif BH[i] == 'B' :
                    BHM[i] = pi / 4 * IDB[i]**2 * T[i] * MDEN
                    KCG[i] = (KCG[i] * SHM[i] + (DRAFT + ELE[i] + 0.5 * T[i]) * BHM[i]) / (SHM[i] + BHM[i])
                else: 
                    KCG[i] = KCG[i]
            return sum(SHM)+sum(RGM)
        allStiffeners = filtered_stiffeners_table()
        stiffener = allStiffeners[self.stiffener_index]
        # assign all varibles so its easier to read later
        G = self.gravity
        ADEN = self.air_density 
        WDEN = self.water_density
        WD = self.water_depth
        LOADC = self.load_condition
        Hs = self.significant_wave_height
        Tp = self.significant_wave_period
        if Hs!= 0: 
            WAVEH = 1.86*Hs
            WAVEP = 0.71*Tp
            WAVEL = G*WAVEP**2/(2*math.pi)
            WAVEN = 2*math.pi/WAVEL
        KCG = self.keel_cg_mooring
        KCGO = self.keel_cg_operating_system
        WREFS = self.reference_wind_speed
        WREFH = self.reference_height
        ALPHA = self.alpha 
        MDEN = self.material_density
        E = self.E
        PR = self.nu
        FY = self.yield_stress
        RMASS = self.rotor_mass
        TMASS = self.tower_mass
        FB = self.free_board
        DRAFT = self.draft
        FBM = self.fixed_ballast_mass
        SMASS = self.hull_mass
        PBM = self.permanent_ballast_mass
        OD = np.array(self.outer_diameter)
        T = np.array(self.wall_thickness)
        LB = np.array(self.length)
        ELE = np.array(self.end_elevation)
        ELS = np.array(self.start_elevation)
        BH = self.bulk_head
        N = np.array(self.number_of_rings)
        stiffenerName = stiffener[0]
        NSEC = self.number_of_sections
        AR = np.array(NSEC*[stiffener[1]])*0.00064516
        D = np.array(NSEC*[stiffener[2]])*0.0254
        TW = np.array(NSEC*[stiffener[3]])*0.0254
        BF=  np.array(NSEC*[stiffener[4]])*0.0254
        TFM = np.array(NSEC*[stiffener[5]])*0.0254
        YNA = np.array(NSEC*[stiffener[6]])*0.0254
        IR = np.array(NSEC*[stiffener[7]])*(0.0254)**4
        HW = D - TFM
        WBM = self.variable_ballast_mass
        # shell data
        RO = OD/2.  # outer radius 
        R = RO-T/2. # radius to centerline of wall/mid fiber radius 
        # ring data 
        LR = LB/(N+1.) # number of ring spacing
        #shell and ring data
        RF = RO - HW  # radius to flange
        MX = LR/(R*T)**0.5  # geometry parameter
        RR = R-YNA-T/2.  # radius to centroid of ring stiffener
        # effective width of shell plate in longitudinal direction 
        LE=np.array([0.]*NSEC)
        for i in range(0,NSEC):
            if MX[i] <= 1.56: 
                LE[i]=LR[i]
            else: 
                LE = 1.1*(2*R*T)**0.5+TW 
        # ring properties with effective shell plate
        AER = AR+LE*T  # cross sectional area with effective shell 
        YENA = (LE*T*T/2 + HW*TW*(HW/2+T) + TFM*BF*(TFM/2+HW+T))/AER 
        IER = IR+AR*(YNA+T/2.)**2*LE*T/AER+LE*T**3/12. # moment of inertia
        RC = RO-YENA-T/2. # radius to centroid of ring stiffener 
        # set loads (0 mass loads for external pressure) 
        MTURBINE = RMASS 
        MTOWER = TMASS 
        MBALLAST = PBM + FBM + WBM # sum of all ballast masses
        MHULL = SMASS 
        W = (MTURBINE + MTOWER + MBALLAST + MHULL) * G
        P = WDEN * G* abs(ELE)  # hydrostatic pressure at depth of section bottom 
        if Hs != 0: # dynamic head 
            DH = WAVEH/2*(np.cosh(WAVEN*(WD-abs(ELE)))/np.cosh(WAVEN*WD)) 
        else: 
            DH = 0 
        P = P + WDEN*G*DH # hydrostatic pressure + dynamic head
        #-----PLATE AND RING STRESS (SECTION 11)-----#
        # shell hoop stress at ring midway 
        Dc = E*T**3/(12*(1-PR**2))  # parameter D 
        BETAc = (E*T/(4*RO**2*Dc))**0.25 # parameter beta 
        TWS = AR/HW
        dum1 = BETAc*LR
        KT = 8*BETAc**3 * Dc * (np.cosh(dum1) - np.cos(dum1))/ (np.sinh(dum1) + np.sin(dum1))
        KD = E * TWS * (RO**2 - RF**2)/(RO * ((1+PR) * RO**2 + (1-PR) * RF**2))
        dum = dum1/2. 
        PSIK = 2*(np.sin(dum) * np.cosh(dum) + np.cos(dum) * np.sinh(dum)) / (np.sinh(dum1) + np.sin(dum1))
        PSIK = PSIK.clip(min=0) # psik >= 0; set all negative values of psik to zero
        SIGMAXA = -W/(2*pi*R*T)
        PSIGMA = P + (PR*SIGMAXA*T)/RO
        PSIGMA = np.minimum(PSIGMA,P) # PSIGMA has to be <= P
        dum = KD/(KD+KT)
        KTHETAL = 1 - PSIK*PSIGMA/P*dum
        FTHETAS = KTHETAL*P*RO/T
        # shell hoop stress at ring 
        KTHETAG = 1 - (PSIGMA/P*dum)
        FTHETAR = KTHETAG*P*RO/T
        #-----LOCAL BUCKLING (SECTION 4)-----# 
        # axial compression and bending 
        ALPHAXL = 9/(300+(2*R)/T)**0.4
        CXL = (1+(150/((2*R)/T))*(ALPHAXL**2)*(MX**4))**0.5
        FXEL = CXL * (pi**2 * E / (12 * (1 - PR**2))) * (T/LR)**2 # elastic 
        FXCL=np.array(NSEC*[0.])
        for i in range(0,len(FXEL)):
            FXCL[i] = plasticityRF(FXEL[i]) # inelastic 
        # external pressure
        BETA = np.array([0.]*NSEC)
        ALPHATHETAL = np.array([0.]*NSEC)
        global ZM
        ZM = 12*(MX**2 * (1-PR**2)**.5)**2/pi**4
        for i in range(0,NSEC):
            def f(x): # here x is beta
                return x**2*(1+x**2)**4/(2+3*x**2)-ZM[i]
            BETA[i] = bisect(f, 0, 100) # solve for x, or beta in this case
            if MX[i] < 5:
                ALPHATHETAL[i] = 1
            elif MX[i] >= 5:
                ALPHATHETAL[i] = 0.8  
        n = np.round(BETA*pi*R/LR) # solve for smallest whole number n 
        BETA = LR/(pi*R/n)
        left = (1+BETA**2)**2/(0.5+BETA**2)
        right = 0.112*MX**4 / ((1+BETA**2)**2*(0.5+BETA**2))
        CTHETAL = (left + right)*ALPHATHETAL 
        FREL = CTHETAL * pi**2 * E * (T/LR)**2 / (12*(1-PR**2)) # elastic
        FRCL=np.array(NSEC*[0.])
        for i in range(0,len(FREL)):
            FRCL[i] = plasticityRF(FREL[i]) # inelastic 
        #-----GENERAL INSTABILITY (SECTION 4)-----# 
        # axial compression and bending 
        AC = AR/(LR*T) # Ar bar 
        ALPHAX = 0.85/(1+0.0025*(OD/T))
        ALPHAXG = np.array([0.]*NSEC)
        for i in range(0,NSEC):
            if AC[i] >= 0.2 :
                ALPHAXG[i] = 0.72
            elif AC[i] > 0.06 and AC[i] <0.2:
                ALPHAXG[i] = (3.6-0.5*ALPHAX[i])*AC[i]+ALPHAX[i]
            else: 
                ALPHAXG[i] = ALPHAX[i]
        FXEG = ALPHAXG * 0.605 * E * T / R * (1 + AC)**0.5 # elastic
        FXCG = np.array(NSEC*[0.])
        for i in range(0,len(FXEG)):
            FXCG[i] = plasticityRF(FXEG[i]) # inelastic  
        # external pressure 
        ALPHATHETAG = 0.8
        LAMBDAG = pi * R / LB 
        k = 0.5 
        PEG = np.array([0.]*NSEC)
        for i in range(0,NSEC):
            t = T[i]
            r = R[i]
            lambdag = LAMBDAG[i]
            ier = IER[i]
            rc = RC[i]
            ro = RO[i]
            lr = LR[i]
            def f(x,E,t,r,lambdag,k,ier,rc,ro,lr):
                return E*(t/r)*lambdag**4/((x**2+k*lambdag**2-1)*(x**2+lambdag**2)**2) + E*ier*(x**2-1)/(lr*rc**2*ro)   
            x0 = [2]
            m = float(fmin(f, x0, xtol=1e-3, args=(E,t,r,lambdag,k,ier,rc,ro,lr))) # solve for n that gives minimum P_eG
            PEG[i] = f(m,E,t,r,lambdag,k,ier,rc,ro,lr)
        ALPHATHETAG = 0.8 #adequate for ring stiffeners 
        FREG = ALPHATHETAG*PEG*RO*KTHETAG/T # elastic 
        FRCG = np.array(NSEC*[0.])
        for i in range(0,len(FREG)):
            FRCG[i] = plasticityRF(FREG[i]) # inelastic  
        # General Load Case
        NPHI = W/(2*pi*R)
        NTHETA = P * RO 
        K = NPHI/NTHETA 
        #-----Local Buckling (SECTION 6) - Axial Compression and bending-----# 
        C = (FXCL + FRCL)/FY -1
        KPHIL = 1
        CST = K * KPHIL /KTHETAL 
        FTHETACL = np.array([0.]*NSEC)
        for i in range(0,NSEC):
            cst = CST[i]
            fxcl = FXCL[i]
            frcl = FRCL[i]
            c = C[i]
            x = Symbol('x')
            ans = solve((cst*x/fxcl)**2 - c*(cst*x/fxcl)*(x/frcl) + (x/frcl)**2 - 1, x)
            FTHETACL[i] =  float(min([a for a in ans if a>0]))
        FPHICL = CST*FTHETACL
        #-----General Instability (SECTION 6) - Axial Compression and bending-----# 
        C = (FXCG + FRCG)/FY -1
        KPHIG = 1
        CST = K * KPHIG /KTHETAG 
        FTHETACG = np.array([0.]*NSEC)
        for i in range(0,NSEC):
            cst = CST[i]
            fxcg = FXCG[i]
            frcg = FRCG[i]
            c = C[i]
            x = Symbol('x')
            ans = solve((cst*x/fxcg)**2 - c*(cst*x/fxcg)*(x/frcg) + (x/frcg)**2 - 1, x)
            FTHETACG[i] =  float(min([a for a in ans if a>0]))
        FPHICG = CST*FTHETACG
        #-----Allowable Stresses-----# 
        # factor of safety
        FOS = 1.25
        if LOADC == 'N' or LOADC == 'n': 
            FOS = 1.65
        FAL = np.array([0.]*NSEC)
        FAG = np.array([0.]*NSEC)
        FEL = np.array([0.]*NSEC)
        FEG = np.array([0.]*NSEC)
        for i in range(0,NSEC):
            # axial load    
            FAL[i] = FPHICL[i]/(FOS*calcPsi(FPHICL[i]))
            FAG[i] = FPHICG[i]/(FOS*calcPsi(FPHICG[i]))
            # external pressure
            FEL[i] = FTHETACL[i]/(FOS*calcPsi(FTHETACL[i]))
            FEG[i] = FTHETACG[i]/(FOS*calcPsi(FTHETACG[i]))
        # unity check 
        self.VAL = abs(SIGMAXA / FAL)
        self.VAG = abs(SIGMAXA / FAG)
        self.VEL = abs(FTHETAS / FEL)
        self.VEG = abs(FTHETAS / FEG)
        global JMAX
        JMAX = np.array([0]*10)
        self.shell_and_ring_mass=calculateWindCurrentForces(0)
        self.shell_and_ring_mass=calculateWindCurrentForces(1)
        # for checking stuff 
        self.FTHETAS = FTHETAS 
        self.FTHETAR = FTHETAR
        self.FXEL = FXEL 
        self.FXCL = FXCL
        self.FTHETAS = FTHETAS
        self.FXEL = FXEL
        self.FXCL = FXCL
        self.FREL = FREL
        self.FRCL = FRCL
        self.FXEG = FXEG
        self.FXCG = FXCG
        self.PEG = PEG
        self.FREG = FREG
        self.FRCG = FRCG
        self.FTHETACL = FTHETACL
        self.FPHICL = FPHICL
        self.FTHETACG = FTHETACG
        self.FPHICG = FPHICG