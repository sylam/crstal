'''
All Stochastic Processes (SP's) are defined here. All SP's run on the Cuda device and are constructed using the
ConstructProcess function. You must pass in the RiskFactor (constructed using the ConstructFactor function) to these
classes.
'''

#import standard libraries
import sys
import numpy as np
import scipy.interpolate
import scipy.integrate

#specific modules
if sys.version[:3]>'2.6':
	from collections import OrderedDict
	#import cuda stuff
	import pycuda.driver as drv
else:
	from ordereddict import OrderedDict
	
#Internal modules
import Utils
from Instruments import getFXZeroRateFactor

class GBMAssetPriceModel(object):
    '''The Geometric Brownian Motion Stochastic Process - defines the python interface and the low level cuda code'''
    
    cudacodetemplate = '''
        __global__ void GBMAssetPriceModel ( const REAL* __restrict__ Samples,
                                             const REAL* __restrict__ dt,
                                             int sample_offset,
                                             int factor_offset,
                                             REAL* Output,
                                             REAL drift,
                                             REAL volatility )
        {
            int index  		= ( blockIdx.x*blockDim.x + threadIdx.x ) * gridDim.y + blockIdx.y ;

            REAL timeStep 	= dt[blockIdx.y];
            REAL sample     = Samples[DIMENSION*index + sample_offset];
            
            Output[ScenarioFactorSize*index + factor_offset] = exp((drift - 0.5*(volatility*volatility))*timeStep + volatility*sample*sqrt(timeStep));
        }

        __global__ void GBMAssetPriceModel_Spot(REAL* Factors, int factor_offset, REAL spot)
        {
            //Have to loop over all timesteps because of path dependance
            
            int  index 		= ScenarioTimeSteps * ( blockIdx.x * blockDim.x + threadIdx.x );
            REAL current    = spot;
            
            for (int i=0; i<ScenarioTimeSteps; i++)
            {
                int offset = ScenarioFactorSize*(index+i) + factor_offset;
                current = Factors[offset] * current;
                Factors[offset] = current;
            }
        }
    '''

    documentation = ['The model is specified as follows:',
                      '',
                      '$$ dS = \\mu S dt + \\sigma S dZ$$',
                      '',
                      'Its final form is:',
                      '',
                      '$$ S = exp \\{ (\\mu-\\frac{1}{2}\\sigma^2)t + \\sigma \\sqrt{t}Z  \\}$$',
                      '',
                      'Where:',
                      '',
                      '- $S$ is the spot price of the asset',
                      '- $dZ$ is the standard Brownian motion',
                      '- $\\mu$ is the drift of the asset',
                      '- $\\sigma$ is the volatility of the asset']
    
    def __init__(self, factor, param):
        self.factor = factor
        self.param  = param

    def NumFactors(self):
        return 1

    def PreCalc(self, ref_date, time_grid):
        self.dt  = np.diff(np.hstack(([0],time_grid.scen_time_grid/365.25)))

    def CalcReferences(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass
    
    def TheoreticalMeanStd(self, t):
        mu  = self.factor.CurrentVal() * np.exp ( self.param['Drift'] * t )
        var = mu*mu*(np.exp(t*self.param['Vol']**2) - 1.0)
        return mu, np.sqrt(var)
    
    def CorrelationName(self):
        return 'LognormalDiffusionProcess', [()]
    
    def Generate(self, module, CudaMem, precision, time_grid, scenario_batch_size, scenarios_per_batch, process_ofs, factor_ofs):
        grid        		= (scenario_batch_size, time_grid.size)
        block       		= (scenarios_per_batch,1,1)
        
        GBMAssetPriceModel 	= module.get_function ('GBMAssetPriceModel')
        GBMAssetPriceModel ( CudaMem.d_random_numbers, drv.In( self.dt.astype(precision) ),
                             np.int32(process_ofs), np.int32(factor_ofs), CudaMem.d_Scenario_Buffer,
                             precision(self.param['Drift']), precision(self.param['Vol']), block=block, grid=grid )

        grid        		= (scenario_batch_size, 1)
        GBMAssetPriceModel_Spot	= module.get_function ('GBMAssetPriceModel_Spot')
        GBMAssetPriceModel_Spot( CudaMem.d_Scenario_Buffer, np.int32(factor_ofs), precision( self.factor.CurrentVal()[0] ), block=block, grid=grid )

class GBMAssetPriceCalibration(object):
    def __init__ ( self, model, param ):
        self.model              = model
        self.param              = param
        self.num_factors        = 1
    
    def calibrate ( self, data_frame, num_business_days=252.0, vol_cuttoff=0.5, drift_cuttoff=0.1 ):
        stats, correlation, delta   = Utils.Calc_statistics ( data_frame, method='Log', num_business_days=num_business_days )
        mu                          = (stats['Drift']+ 0.5*(stats['Volatility']**2)).values[0]
        sigma                       = stats['Volatility'].values[0]
        
        return Utils.CalibrationInfo({'Vol':np.clip(sigma,0.01,vol_cuttoff), 'Drift':np.clip(mu,-drift_cuttoff,drift_cuttoff)}, [[1.0]], delta)

class GBMAssetPriceTSModelImplied(object):
    '''The Geometric Brownian Motion Stochastic Process with implied drift and vol - defines the python interface and the low level cuda code'''
    
    cudacodetemplate = '''
        __global__ void GBMAssetPriceTSModelImplied(    const REAL* __restrict__ Samples,
                                                        const REAL* __restrict__ scenario_time_grid,
                                                        const REAL* __restrict__ static_factors,
                                                        const REAL* __restrict__ Vt,
                                                        int sample_offset,
                                                        int factor_offset,
                                                        REAL* Output,
                                                        const int* __restrict__ r,
                                                        const int* __restrict__ q )
        {            
            REAL p_Vt       = blockIdx.y > 0 ? Vt[ blockIdx.y - 1 ] : 0.0;            
            REAL d_t        = scenario_time_grid[ blockIdx.y ] - ( blockIdx.y > 0 ? scenario_time_grid[ blockIdx.y - 1 ] : 0.0 );
            
            int index  		= ( blockIdx.x*blockDim.x + threadIdx.x ) * gridDim.y + blockIdx.y ;

            REAL vol2       = Vt[blockIdx.y]-p_Vt;
            REAL sample     = Samples[DIMENSION*index + sample_offset];
            
            REAL r_Accrual 	= calcDayCountAccrual(d_t, r[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
            REAL q_Accrual 	= calcDayCountAccrual(d_t, q[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
                                                    
            //reconstruct (and interpolate) the yield on the asset (q) and the interest rate on it (r)
            REAL r_t = r_Accrual * ScenarioCurve1D ( NULL, r_Accrual, Scenario1d, InterpolateIntRate, r, index, static_factors, Output );
            REAL q_t = q_Accrual * ScenarioCurve1D ( NULL, q_Accrual, Scenario1d, InterpolateIntRate, q, index, static_factors, Output );
                
            Output[ScenarioFactorSize*index + factor_offset] = exp ( ( r_t - q_t - 0.5*vol2 ) + sample*sqrt(vol2) );
        }
    '''

    documentation = ['The model is specified as follows:',
                      '',
                      '$$ \\frac{dS(t)}{S(t)} = (r(t)-q(t)) dt + \\sigma(t) dW(t)$$',
                      '',
                      'Its final form is:',
                      '',
                      '$$ S(t+\\delta) = F(t,t+\\delta)exp \\{ (-\\frac{1}{2}(V(t+\\delta)) - V(t)) + \\sqrt{V(t+\\delta) - V(t)}Z  \\}$$',
                      '',
                      'Where:',                     
                      '',
                      '- $\\sigma(t)$ is the volatility of the asset at time t'
                      '- $V(t) = \int_{0}^{t} \sigma(s)^2 ds$',
                      '- $r$ is the interest rate in the asset currency'
                      '- $q$ is the yield on the asset. If S is a foreign exchange rate, q is the foreign interest rate',
                      '- $F(t,t+\\delta)$ is the forward asset price at time t',
                      '- $S$ is the spot price of the asset',
                      '- $Z$ is a sample from the standard normal distribution',
                      '- $\\delta$ is the increment in timestep between samples',
                      '']
    
    def __init__(self, factor, param):
        self.factor = factor
        self.param  = param

    def NumFactors(self):
        return 1

    def PreCalc(self, ref_date, time_grid):
        time_grid_in_years  = time_grid.scen_time_grid/365.0
        vols                = self.param['Vol'].array
        self.V              = np.array( [ scipy.integrate.quad(lambda s : np.interp(s, *vols.T )**2, 0 , t)[0] for t in time_grid_in_years ] )

    def CalcReferences(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        #this is valid for FX factors only - can change this to equities etc. by changing the getXXXFactor function below
        self.r_t            = getFXZeroRateFactor( self.factor.GetDomesticCurrency(None), static_ofs, stoch_ofs, all_tenors, all_factors )
        self.q_t            = getFXZeroRateFactor( factor.name, static_ofs, stoch_ofs, all_tenors, all_factors )
                                                   
    def CorrelationName(self):
        return 'LognormalDiffusionProcess', [()]

    def Generate(self, module, CudaMem, precision, time_grid, scenario_batch_size, scenarios_per_batch, process_ofs, factor_ofs):
        grid        		= (scenario_batch_size, time_grid.size)
        block       		= (scenarios_per_batch,1,1)
        
        GBMAssetPriceTSModelImplied = module.get_function ('GBMAssetPriceTSModelImplied')
        GBMAssetPriceTSModelImplied ( CudaMem.d_random_numbers, drv.In(time_grid.astype(precision)), CudaMem.d_Static_Buffer,
                                      drv.In(self.V.astype(precision)), np.int32(process_ofs), np.int32(factor_ofs),
                                      CudaMem.d_Scenario_Buffer, drv.In ( self.r_t ), drv.In ( self.q_t ), 
                                      block=block, grid=grid )

        grid        		     = (scenario_batch_size, 1)
        GBMAssetPriceModel_Spot	 = module.get_function ('GBMAssetPriceModel_Spot')
        GBMAssetPriceModel_Spot( CudaMem.d_Scenario_Buffer, np.int32(factor_ofs), precision( self.factor.CurrentVal()[0] ), block=block, grid=grid )
        
class GBMPriceIndexModel(object):
    '''The Geometric Brownian Motion Stochastic Process for Price Indices - can contain adjustments for seasonality - defines the python interface and the low level cuda code'''
    
    cudacodetemplate = '''
        __global__ void GBMPriceIndexModel( const REAL* __restrict__ Samples,
                                            const REAL* __restrict__ dt,
                                            int sample_offset,
                                            int factor_offset,
                                            REAL* Output,
                                            REAL drift,
                                            REAL volatility )
        {
            int index  		= ( blockIdx.x*blockDim.x + threadIdx.x ) * gridDim.y + blockIdx.y ;

            REAL timeStep 	= dt[blockIdx.y];
            REAL sample     = Samples[DIMENSION*index + sample_offset];
            
            Output[ScenarioFactorSize*index + factor_offset] = exp((drift - 0.5*(volatility*volatility))*timeStep + volatility*sample*sqrt(timeStep));
        }

        __global__ void GBMPriceIndexModel_Spot(REAL* Factors, int factor_offset, REAL spot)
        {
            //Have to loop over all timesteps because of path dependance
            
            int  index 		= ScenarioTimeSteps * ( blockIdx.x * blockDim.x + threadIdx.x );
            REAL current    = spot;
            
            for (int i=0; i<ScenarioTimeSteps; i++)
            {
                int offset = ScenarioFactorSize*(index+i) + factor_offset;
                current = Factors[offset] * current;
                Factors[offset] = current;
            }
        }
    '''
    
    documentation = ['The model is specified as follows:',
                      '',
                      '$$ dS = \\mu S dt + \\sigma S dZ$$',
                      '',
                      'Its final form is:',
                      '',
                      '$$ S = exp \\{ (\\mu-\\frac{1}{2}\\sigma^2)t + \\sigma \\sqrt{t}Z  \\}$$',
                      '',
                      'Where:',
                      '',
                      '- $S$ is the spot price of the asset',
                      '- $dZ$ is the standard Brownian motion',
                      '- $\\mu$ is the drift of the asset',
                      '- $\\sigma$ is the volatility of the asset']
    
    def __init__(self, factor, param):
        self.factor = factor
        self.param  = param

    def NumFactors(self):
        return 1

    def PreCalc(self, ref_date, time_grid):
        self.scenario_time_grid     = np.array( [( x - self.factor.param['Last_Period_Start']).days for x in self.factor.GetLastPublicationDates ( ref_date, time_grid.scen_time_grid.tolist() ) ], dtype=np.float64 )
        self.dt                     = np.diff(np.hstack(([0],self.scenario_time_grid/365.25)))
        #self.mtm_publication_dates  = self.factor.GetLastPublicationDates ( ref_date, time_grid.time_grid[:,Utils.TIME_GRID_MTM] )

    def CalcReferences(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass
    
    def CorrelationName(self):
        return 'LognormalDiffusionProcess', [()]

    def Generate(self, module, CudaMem, precision, time_grid, scenario_batch_size, scenarios_per_batch, process_ofs, factor_ofs):
        grid        		= (scenario_batch_size, time_grid.size)
        block       		= (scenarios_per_batch,1,1)
        
        GBMPriceIndexModel 	= module.get_function ('GBMPriceIndexModel')
        GBMPriceIndexModel ( CudaMem.d_random_numbers, drv.In( self.dt.astype(precision) ),
                             np.int32(process_ofs), np.int32(factor_ofs), CudaMem.d_Scenario_Buffer,
                             precision(self.param['Drift']), precision(self.param['Vol']), block=block, grid=grid )

        grid        		= (scenario_batch_size, 1)
        GBMPriceIndexModel_Spot	= module.get_function ('GBMPriceIndexModel_Spot')
        GBMPriceIndexModel_Spot( CudaMem.d_Scenario_Buffer, np.int32(factor_ofs), precision( self.factor.CurrentVal()[0] ), block=block, grid=grid )

class GBMPriceIndexCalibration(object):
    def __init__(self, model, param):
        self.model              = model
        self.param              = param
        self.num_factors        = 1
   
    def calibrate(self, data_frame, num_business_days=252.0):
        stats, correlation, delta   = Utils.Calc_statistics ( data_frame, method='Log', num_business_days=num_business_days )
        mu                          = (stats['Drift']+ 0.5*(stats['Volatility']**2)).values[0]
        sigma                       = stats['Volatility'].values[0]
        
        return Utils.CalibrationInfo({'Vol':sigma, 'Drift':mu, 'Seasonal_Adjustment':None}, [[1.0]], delta)

class HullWhite1FactorInterestRateModel(object):
    '''Hull White 1 factor model
    
        The basic model is as follows:

            BtT = (1.0-exp(-alpha*(T-t)))/alpha
            AtT = ( ( BtT*exp(-alpha*t) )/alpha ) * ( 2.0 * It - ( exp(-alpha*t) + exp(-alpha*T) ) * Jt );
            DtT = D0t/D0T * exp ( -0.5 * AtT - BtT * exp(-alpha*t)* ( sqrt( Jt )*Z + lambda*Ht )

        I've omitted the quanto correction for simplicity - it can be added in reasonably easily.
        Note that It and Jt are integrals involing the vol of the short rate.
        
        some code to illustrate . . .
        
        AtT        = np.array( [ [ ( ( ((1.0-np.exp(-alpha*Tmt))/alpha)*np.exp(-alpha*t) )/alpha ) * ( 2.0 * It - ( np.exp(-alpha*t) + np.exp(-alpha*(t+Tmt)) ) * Jt ) for Tmt in self.factor.GetTenor() ] for (It,Jt,t) in zip(self.I, self.J, time_grid_in_years) ] )
        BtT        = np.array( [ [ ( (1.0-np.exp(-alpha*Tmt))/alpha)*np.exp(-alpha*t)*np.sqrt(Jt) for Tmt in self.factor.GetTenor() ] for (Jt,t) in zip(self.J, time_grid_in_years) ] )

        fwd_A      = np.vstack( (AtT[0], AtT[1:] - AtT[:-1]) )
        fwd_B      = np.vstack( (BtT[0], BtT[1:]**2 - BtT[:-1]**2) )
        sample     = np.random.randn(1000, fwd_A.shape[0])
        
        exp_part   = np.array([ [np.exp( -0.5*self.AtT[i] -self.BtT[i]*x ) for i,x in zip(range(time_grid.scen_time_grid.size), s)] for s in self.sample])

        now just multiply exp_part with self.disc_curves and you're done.
    '''

    cudacodetemplate = '''
        __global__ void HullWhite1FactorInterestRateModel(  const REAL* __restrict__ Samples,
                                                            const REAL* __restrict__ Buffer,
                                                            int sample_offset,
                                                            int factor_offset,
                                                            REAL* Output,
                                                            REAL lambda,
                                                            REAL alpha,
                                                            int tenor_size )
        {
            //assumes that the time (t), It, Jt & tenors for each timestep is stored in the constant Buffer
            REAL  t 		        = Buffer [blockIdx.y];
            
            const REAL* It         = Buffer + ScenarioTimeSteps;
            const REAL* Jt         = Buffer + 2*ScenarioTimeSteps;
            const REAL* Ht         = Buffer + 3*ScenarioTimeSteps;
            const REAL* tenors     = Buffer + 4*ScenarioTimeSteps;

            //calculate the index
            int index  		= ( blockIdx.x*blockDim.x + threadIdx.x ) * gridDim.y + blockIdx.y;
            REAL rnd      	= Samples[DIMENSION*index + sample_offset];

            //get the previous timestep value of our state process . . . 
            REAL p_It       = blockIdx.y > 0 ? It[ blockIdx.y - 1 ] : 0.0;
            REAL p_Jt       = blockIdx.y > 0 ? Jt[ blockIdx.y - 1 ] : 0.0;
            REAL p_Ht       = blockIdx.y > 0 ? Ht[ blockIdx.y - 1 ] : 0.0;
            REAL p_t        = blockIdx.y > 0 ? Buffer[ blockIdx.y - 1 ] : 0.0;
            
            // Simulate the path for rates at one tenor at a time
            for (int k=0; k<tenor_size; k++)
            {
                REAL BtT = ( 1.0-exp(-alpha*tenors[k]))/alpha;
                REAL delta_AtT =  BtT * ( ( ( exp(-alpha*t) )/alpha  ) * ( 2.0 * It[ blockIdx.y ] - ( exp(-alpha*t) + exp(-alpha*(t+tenors[k])) ) * Jt[ blockIdx.y ] ) -
                                          ( ( exp(-alpha*p_t) )/alpha ) * ( 2.0 * p_It - ( exp(-alpha*p_t) + exp(-alpha*(p_t+tenors[k])) ) * p_Jt ) );
                                        
                REAL delta_BtT = max ( exp(-alpha*2.0*t) * Jt[ blockIdx.y] - exp(-alpha*2.0*p_t) * p_Jt , 0.0);
                REAL delta_HtT = lambda * ( exp(-alpha*t) * Ht[ blockIdx.y ] - exp(-alpha*p_t) * p_Ht );

                //REAL AtT = BtT * ( ( exp(-alpha*t) )/alpha ) * ( 2.0*It[ blockIdx.y ] - ( exp(-alpha*t) + exp(-alpha*(t+tenors[k])) ) * Jt[ blockIdx.y ] );
                //Output[ ScenarioFactorSize*index + factor_offset + k ] = ( 0.5*AtT + BtT*exp(-alpha*t)*(sqrt(Jt[blockIdx.y]) * rnd + lambda*Ht[blockIdx.y]) );
                //Full [ ScenarioFactorSize*index + factor_offset + k ] = ( 0.5*AtT + BtT*exp(-alpha*t)*(sqrt(Jt[blockIdx.y]) * rnd + lambda*Ht[blockIdx.y]) );
                
                Output [ ScenarioFactorSize*index + factor_offset + k ] = ( 0.5*delta_AtT + BtT * ( sqrt(delta_BtT) * rnd + delta_HtT ) );
            }
        }
 
        __global__ void HullWhite1FactorInterestRateModel_Spot ( REAL* Factors,
                                                                 const REAL* __restrict__ Buffer,
                                                                 const REAL* __restrict__ forward_factors,
                                                                 int factor_offset )
        {
            const REAL* tenors  = Buffer + 4*ScenarioTimeSteps;
            int  index 		    = ScenarioTimeSteps * ( blockIdx.x * blockDim.x + threadIdx.x );
            REAL current        = 0.0;
            
            for (int i=0; i<ScenarioTimeSteps; i++)
            {
                int offset   = ScenarioFactorSize*(index+i) + factor_offset + blockIdx.y;                
                current     += Factors[offset];
                
                Factors[offset] = ( forward_factors[ gridDim.y*i+blockIdx.y ] + current ) / tenors[blockIdx.y];
            }
        }
    '''
    
    documentation = ['The instantaneous spot rate (or short rate) which governs the evolution of the yield curve is modeled as:',
                     '',
                     '$$ dr(t) = (\\theta (t)-\\alpha r(t) - v(t)\\sigma(t)\\rho)dt + \\sigma(t) dW^*(t)$$',
                     '',
                     'Where:',
                     '',
                     '- $\\sigma (t)$ is a deterministic volatility curve',
                     '- $\\alpha$ is a constant mean reversion speed',
                     '- $\\theta (t)$ is a deterministic curve derived from the vol, mean reversion and initial discount factors',
                     '- $v(t)$ is the quanto FX volatility and $\\rho$ is the quanto FX correlation',
                     '- $dW^*(t)$ is the risk neutral Wiener process related to the real-world Wiener Process $dW(t)$ by $dW^*(t)=dW(t)+\\lambda dt$ where $\\lambda$ is the market price of risk (assumed to be constant)',
                     '',
                     'Final form of the model is:',
                     '$$ D(t,T) = \\frac{D(0,T)}{D(0,t)}exp\\Big(-\\frac{1}{2}A(t,T)-B(T-t)e^{-\\alpha t}(Y(t) -\\rho K(t) + \\lambda H(t))\\Big)$$',
                     '',
                     'Where:',
                     '- $B(t) = \\frac{(1-e^{-\\alpha t})}{\\alpha}$, $Y(t)=\\int\\limits_0^t e^{\\alpha s}\\sigma (s) dW$',
                     '- $A(t,T)=\\frac{B(T-t)e^{-\\alpha T}}{\\alpha}(2I(t)-(e^{-\\alpha t}+e^{-\\alpha T})J(t))$',
                     '- $H(t)$, $I(t)$, $J(t)$ and $K(t)$ are all deterministic functions of $\\alpha$, $v(t)$ and $\\sigma(t)$']
    
    def __init__(self, factor, param):
        self.factor = factor
        self.param  = param
        self.H      = None
        self.I      = None
        self.J      = None

    def NumFactors(self):
        return 1

    def PreCalc(self, ref_date, time_grid):
        #ensures that tenors used are the same as the price factor
        factor_tenor        = self.factor.GetTenor()
        alpha               = self.param['Alpha']
        
        self.fwd_curves     = np.array ( [ self.factor.CurrentVal(self.factor.GetDayCountAccrual(ref_date,t)+self.factor.tenors)*(self.factor.GetDayCountAccrual(ref_date,t)+self.factor.tenors )-self.factor.CurrentVal(self.factor.GetDayCountAccrual(ref_date,t))*self.factor.GetDayCountAccrual(ref_date,t) for t in time_grid.scen_time_grid] )

        #calculate the times we need to evaluate the time grid
        time_grid_in_years  = time_grid.scen_time_grid/365.
        
        #Really should implement this . . . 
        quantofx        = self.param['Quanto_FX_Volatility'].array.T if self.param['Quanto_FX_Volatility'] else np.zeros(2)
        #grab the vols
        vols            = self.param['Sigma'].array
        
        #calculate known functions
        self.H          = np.array( [ scipy.integrate.quad(lambda s : np.exp(alpha*s)*np.interp(s, *vols.T ), 0 , t)[0] for t in time_grid_in_years ] )
        self.I          = np.array( [ scipy.integrate.quad(lambda s : np.exp(alpha*s)*(np.interp(s, *vols.T )**2), 0 , t)[0] for t in time_grid_in_years ] )
        self.J          = np.array( [ scipy.integrate.quad(lambda s : np.exp(2.0*alpha*s)*(np.interp(s, *vols.T )**2), 0 , t)[0] for t in time_grid_in_years ] )
        
        #store in one buffer
        self.Buffer     = np.hstack( ( time_grid.scen_time_grid/365., self.I, self.J, self.H, self.factor.GetTenor() ) )

    def CalcReferences(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass
    
    def CorrelationName(self):
        return 'HWInterestRate', [('F1',)]    
    
    def Generate(self, module, CudaMem, precision, time_grid, scenario_batch_size, scenarios_per_batch, process_ofs, factor_ofs):
        grid        		= (scenario_batch_size, time_grid.size)
        block       		= (scenarios_per_batch,1,1)

        HullWhite1FactorInterestRateModel = module.get_function ('HullWhite1FactorInterestRateModel')
        HullWhite1FactorInterestRateModel ( CudaMem.d_random_numbers, drv.In(self.Buffer.astype(precision)),
                                            np.int32(process_ofs), np.int32(factor_ofs), CudaMem.d_Scenario_Buffer,
                                            precision(self.param['Lambda']), precision(self.param['Alpha']), np.int32(self.factor.tenors.size), block=block, grid=grid )
        
        grid        		= (scenario_batch_size, self.factor.tenors.size)
        block       		= (scenarios_per_batch,1,1)
        
        HullWhite1FactorInterestRateModel_Spot	= module.get_function ('HullWhite1FactorInterestRateModel_Spot')
        HullWhite1FactorInterestRateModel_Spot( CudaMem.d_Scenario_Buffer, drv.In(self.Buffer.astype(precision)),
                                                drv.In(self.fwd_curves.astype(precision)), np.int32(factor_ofs), block=block, grid=grid )

class HWInterestRateCalibration(object):
    def __init__(self, model, param):
        self.model              = model
        self.param              = param
        self.num_factors        = 1
        
    def calibrate ( self, data_frame, num_business_days=252.0 ):
        tenor                       = np.array ( [ (x.split(',')[1]) for x in data_frame.columns ], dtype=np.float64 )
        stats, correlation, delta   = Utils.Calc_statistics ( data_frame, method='Diff', num_business_days=num_business_days, max_alpha = 4.0 )
        #alpha                       = np.percentile(stats['Mean Reversion Speed'], 50)#.mean()
        alpha                       = stats['Mean Reversion Speed'].mean()
        sigma                       = stats['Reversion Volatility'].mean()
        correlation_coef            = np.array([np.array([1.0/np.sqrt(correlation.values.sum())]*tenor.size)])
        
        return Utils.CalibrationInfo({ 'Lambda':0.0, 'Alpha':alpha, 'Sigma':Utils.Curve ( [], [(0.0, sigma)] ) , 'Quanto_FX_Correlation':0.0, 'Quanto_FX_Volatility':0.0 }, correlation_coef, delta)

class HWHazardRateModel(object):
    '''Hull White 1 factor hazard Rate model
    '''

    cudacodetemplate = '''
        __global__ void HWHazardRateModel ( const REAL* __restrict__ Samples,
                                            const REAL* __restrict__ Buffer,
                                            int sample_offset,
                                            int factor_offset,
                                            REAL* Output,
                                            REAL lambda,
                                            REAL sigma,
                                            REAL alpha,
                                            int tenor_size)
        {
            //assumes that the time (t) & tenors for each timestep is stored in the constant Buffer
            REAL  t 		    = Buffer [blockIdx.y];
            const REAL* tenors  = Buffer + ScenarioTimeSteps;

            //calculate the index
            int index  		= ( blockIdx.x*blockDim.x + threadIdx.x ) * gridDim.y + blockIdx.y;
            REAL rnd      	= Samples[DIMENSION*index + sample_offset];
            REAL Bt         = (1.0-exp(-alpha*t))/alpha;
            REAL B2t         = (1.0-exp(-alpha*2.0*t))/alpha;

            //get the previous timestep value of our state process . . . 
            REAL p_t        = blockIdx.y > 0 ? Buffer[ blockIdx.y - 1 ] : 0.0;
            REAL Bpt        = (1.0-exp(-alpha*p_t))/alpha;
            REAL B2pt       = (1.0-exp(-alpha*2.0*p_t))/alpha;
            
            // Simulate the path for rates at one tenor at a time
            for (int k=0; k<tenor_size; k++)
            {
                REAL BtT = ( 1.0-exp(-alpha*tenors[k]))/alpha;
                REAL delta_AtT =  sigma * sigma * BtT * ( ( BtT * B2t * 0.5 + Bt * Bt ) - ( BtT * B2pt * 0.5 + Bpt * Bpt ) );
                REAL delta_BtT = ( exp(-alpha*2.0*p_t) - exp(-alpha*2.0*t) ) / ( 2.0 * alpha );
                Output [ ScenarioFactorSize*index + factor_offset + k ] = 0.5*delta_AtT + sigma * BtT * ( sqrt(delta_BtT) * rnd + (Bt-Bpt) * lambda );
            }
        }
 
        __global__ void HWHazardRateModel_Spot(REAL* Factors, const REAL* __restrict__ forward_factors, int factor_offset)
        {
            int  index 		= ScenarioTimeSteps * ( blockIdx.x * blockDim.x + threadIdx.x );
            REAL current    = 0.0;
            
            for (int i=0; i<ScenarioTimeSteps; i++)
            {
                int offset   = ScenarioFactorSize*(index+i) + factor_offset + blockIdx.y;                
                current     += Factors[offset];
                
                Factors[offset] = forward_factors[ gridDim.y*i+blockIdx.y ] + current;
            }
        }
    '''

    documentation = [   'The instantaneous hazard rate process is modeled as:',
                        '',
                        '$$ dh(t) = (\\theta (t)-\\alpha h(t))dt + \\sigma dW^*(t)$$',
                        '',
                        'All symbols defined as per Hull White 1 factor for interest rates.'
                        '',
                        'Final form of the model is',
                        '',
                        '$$ S(t,T) = \\frac{S(0,T)}{S(0,t)}exp\\Big(-\\frac{1}{2}A(t,T)-\\sigma B(T-t)(Y(t) + B(t)\\lambda)\\Big)$$',
                        '',
                        'Where:',
                        '',
                        '- $B(t) = \\frac{1-e^{-\\alpha t}}{\\alpha}$, $Y(t) \\sim N(0, \\frac{1-e^{-2 \\alpha t}}{2 \\alpha})$',
                        '- $A(t,T)=\\sigma^2 B(T-t)\\Big(B(T-t)\\frac{B(2t)}{2}+B(t)^2\\Big)$']
    
    def __init__(self, factor, param):
        self.factor = factor
        self.param  = param

    def NumFactors(self):
        return 1

    def PreCalc(self, ref_date, time_grid):
        self.fwd_curves     = np.array ( [ self.factor.CurrentVal(self.factor.GetDayCountAccrual(ref_date,t)+self.factor.tenors) - self.factor.CurrentVal(self.factor.GetDayCountAccrual(ref_date,t)) for t in time_grid.scen_time_grid] )
        self.Buffer         = np.hstack( ( time_grid.scen_time_grid/365., self.factor.GetTenor() ) )

    def CalcReferences(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass
    
    def CorrelationName(self):
        return 'HullWhiteProcess', [()]
    
    def Generate(self, module, CudaMem, precision, time_grid, scenario_batch_size, scenarios_per_batch, process_ofs, factor_ofs):
        grid        		= (scenario_batch_size, time_grid.size)
        block       		= (scenarios_per_batch,1,1)

        HWHazardRateModel = module.get_function ('HWHazardRateModel')
        HWHazardRateModel ( CudaMem.d_random_numbers, drv.In(self.Buffer.astype(precision)),
                            np.int32(process_ofs), np.int32(factor_ofs), CudaMem.d_Scenario_Buffer,
                            precision(self.param['Lambda']), precision(self.param['Sigma']), precision(self.param['Alpha']), np.int32(self.factor.tenors.size), block=block, grid=grid )
        
        grid        		= (scenario_batch_size, self.factor.tenors.size)
        block       		= (scenarios_per_batch,1,1)
        
        HWHazardRateModel_Spot	= module.get_function ('HWHazardRateModel_Spot')
        HWHazardRateModel_Spot( CudaMem.d_Scenario_Buffer, drv.In(self.fwd_curves.astype(precision)), np.int32(factor_ofs), block=block, grid=grid )

class HWHazardRateCalibration(object):
    def __init__(self, model, param):
        self.model              = model
        self.param              = param
        self.num_factors        = 1
    
    def calibrate ( self, data_frame, num_business_days=252.0 ):
        tenor                       = np.array ( [ (x.split(',')[1]) for x in data_frame.columns ], dtype=np.float64 )
        stats, correlation, delta   = Utils.Calc_statistics ( data_frame, method='Diff', num_business_days=num_business_days, max_alpha = 4.0 )
        alpha                       = stats['Mean Reversion Speed'].mean()
        sigma                       = stats['Reversion Volatility'].values[0]/tenor[0]
        correlation_coef            = np.array([np.array([1.0/np.sqrt(correlation.values.sum())]*tenor.size)])
        
        return Utils.CalibrationInfo({'Alpha':alpha, 'Sigma':sigma, 'Lambda':0}, correlation_coef, delta)
        
class CSForwardPriceModel(object):
    '''Clewlow-Strickland Model'''

    cudacodetemplate = '''
        __global__ void CSForwardPriceModel (   const REAL* __restrict__ Samples,
                                                const REAL* __restrict__ Time,
                                                const REAL* __restrict__ all_tenors,
                                                int sample_offset,
                                                int factor_offset,
                                                REAL* Output,
                                                REAL mu,
                                                REAL sigma,
                                                REAL alpha,
                                                int tenor_size )
        {
            REAL  t 		 = Time [blockIdx.y];
            //get the previous timestep value of our state process . . . 
            REAL p_t         = blockIdx.y > 0 ? Time [ blockIdx.y - 1 ] : 0.0;
            //the timestep
            REAL dt          = t-p_t;
            
            const REAL* tenors     = all_tenors + blockIdx.y*tenor_size;

            //calculate the index
            int index  		= ( blockIdx.x*blockDim.x + threadIdx.x ) * gridDim.y + blockIdx.y;
            REAL rnd      	= Samples[DIMENSION*index + sample_offset];
            
            REAL vol_adjustment = ( exp(2.0*alpha*t) - exp(2.0*alpha*p_t) ) /( 2.0 * alpha );
            
            // Simulate the path for rates at one tenor at a time
            for (int k=0; k<tenor_size; k++)
            {
                REAL   vol = sigma * exp ( -alpha*tenors[k] ) * sqrt(vol_adjustment);
                Output [ ScenarioFactorSize*index + factor_offset + k ] = exp ( mu*dt - 0.5*vol*vol + vol*rnd );
            }
        }
 
        __global__ void CSForwardPriceModel_Spot(REAL* Factors, const REAL* __restrict__ initial_forward_curve, int factor_offset)
        {
            int  index 		= ScenarioTimeSteps * ( blockIdx.x * blockDim.x + threadIdx.x );
            REAL current    = 1.0;
            
            for (int i=0; i<ScenarioTimeSteps; i++)
            {
                int offset   = ScenarioFactorSize*(index+i) + factor_offset + blockIdx.y;
                current     *= Factors[offset];
                
                Factors [ offset ] = initial_forward_curve [ blockIdx.y ] * current;
            }
        }
    '''

    documentation = ['For commodity/Energy deals, the Forward price is modeled directly. For each settlement date T, the SDE for the forward price is:',
                     '',
                     '$$ dF(t,T) = \\mu F(t,T)dt + \\sigma e^{-\\alpha(T-t)}F(t,T)dW(t)$$',
                     '',
                     'Where:',
                     '',
                     '- $\\mu$ is the drift rate',
                     '- $\\sigma$ is the volatility',
                     '- $\\alpha$ is the mean reversion speed',
                     '- $W(t)$ is the standard Weiner Process',
                     '',
                     'Final form of the model is',
                     '',
                     '$$ F(t,T) = F(0,T)exp\\Big(\\mu t-\\frac{1}{2}\\sigma^2e^{-2\\alpha(T-t)}v(t)+\\sigma e^{-\\alpha(T-t)}Y(t)\\Big)$$',
                     '',
                     'Where Y is a standard Ornstein-Uhlenbeck Process with variance:',
                     '$$v(t) = \\frac{1-e^{-2\\alpha t}}{2\\alpha}$$']
    
    def __init__(self, factor, param):
        self.factor = factor
        self.param  = param

    def NumFactors(self):
        return 1
    
    def CalcReferences(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass

    def PreCalc(self, ref_date, time_grid):
        self.initial_curve      = self.factor.CurrentVal()
        excel_date_time_grid    = time_grid.scen_time_grid+(ref_date-self.factor.start_date).days
        self.tenors             = np.array([self.factor.GetTenor()-x for x in excel_date_time_grid])/365.0
        self.dt                 = time_grid.scen_time_grid/365.0

    def CorrelationName(self):
        return 'ClewlowStricklandProcess', [()]
    
    def Generate (self, module, CudaMem, precision, time_grid, scenario_batch_size, scenarios_per_batch, process_ofs, factor_ofs):
        grid        		= (scenario_batch_size, time_grid.size)
        block       		= (scenarios_per_batch,1,1)

        CSForwardPriceModel = module.get_function ('CSForwardPriceModel')
        CSForwardPriceModel ( CudaMem.d_random_numbers, drv.In(self.dt.astype(precision)), drv.In(self.tenors.astype(precision)),
                              np.int32(process_ofs), np.int32(factor_ofs), CudaMem.d_Scenario_Buffer,
                              precision(self.param['Drift']), precision(self.param['Sigma']), precision(self.param['Alpha']), np.int32(self.factor.tenors.size), block=block, grid=grid )
        
        grid        		= (scenario_batch_size, self.factor.tenors.size)
        block       		= (scenarios_per_batch,1,1)
        
        CSForwardPriceModel_Spot	= module.get_function ('CSForwardPriceModel_Spot')
        CSForwardPriceModel_Spot( CudaMem.d_Scenario_Buffer, drv.In(self.initial_curve.astype(precision)), np.int32(factor_ofs), block=block, grid=grid )

class CSForwardPriceCalibration(object):
    def __init__(self, model, param):
        self.model              = model
        self.param              = param
        self.num_factors        = 1
    
    def calibrate(self, data_frame, num_business_days=252.0):
        tenor                       = np.array ( [ (x.split(',')[1]) for x in data_frame.columns ], dtype=np.float64 )
        stats, correlation, delta   = Utils.Calc_statistics ( data_frame, method='Log', num_business_days=num_business_days, max_alpha = 5.0 )
        alpha                       = stats['Mean Reversion Speed'].values[0]
        sigma                       = stats['Reversion Volatility'].values[0]
        mu                          = stats['Drift'].values[0]+0.5*(stats['Volatility'].values[0])**2
        correlation_coef            = np.array([np.array([1.0/np.sqrt(correlation.values.sum())]*tenor.size)])
        
        return Utils.CalibrationInfo({'Sigma':sigma, 'Alpha':alpha, 'Drift':mu }, correlation_coef, delta)
        
class PCAInterestRateModel(object):
    '''The Principle Component Analysis model for interest rate curves Stochastic Process - defines the python interface and the low level cuda code'''
    
    cudacodetemplate = '''
        __global__ void PCAInterestRateModel (  const REAL* __restrict__ Samples,
                                                const REAL* __restrict__ Buffer,
                                                const REAL* __restrict__ drifts,
                                                int sample_offset,
                                                int factor_offset,
                                                REAL* Output,
                                                REAL reversionSpeed,
                                                int tenor_size )
        {
            //assumes that the time (t), volCurve and Principle Components for each timestep is stored in the constant Buffer
            //also (indirectly) specifies the length of each principle component (via the tenor_size parameter) 

            REAL  t 		        = Buffer [blockIdx.y];
            REAL  dt 		        = blockIdx.y > 0 ? Buffer [blockIdx.y] - Buffer [blockIdx.y-1] : 0.0;
            const REAL* PC          = Buffer + gridDim.y;
            const REAL* volCurve    = Buffer + gridDim.y + 3*tenor_size;			

            //calculate the index
            int index  		= ( blockIdx.x*blockDim.x + threadIdx.x ) * gridDim.y + blockIdx.y;

            REAL rnd1      	= Samples[DIMENSION*index + sample_offset];
            REAL rnd2      	= Samples[DIMENSION*index + sample_offset+1];
            REAL rnd3      	= Samples[DIMENSION*index + sample_offset+2];

            //adjust the vol to take into account the mean reversion and the change in time
            REAL vol_adjustment = (exp(-2*reversionSpeed*(t-dt)) - exp(-2*reversionSpeed*t))/(2*reversionSpeed);

            // Simulate the path for rates at one tenor at a time
            for (int k=0; k<tenor_size; k++)
            {
                // Calculate the PC portion of the equation (assume 3 PCs for now)
                REAL PCportion1 = PC[             k]*rnd1;
                REAL PCportion2 = PC[1*tenor_size+k]*rnd2;
                REAL PCportion3 = PC[2*tenor_size+k]*rnd3;
                REAL PCportion  = volCurve[k]*(PCportion1 + PCportion2 + PCportion3)*sqrt(vol_adjustment);

                // Calculate the first term of eq (27) p. 15 in Analytics doc
                REAL firstTerm = - 0.5*(volCurve[k]*volCurve[k])*vol_adjustment;
                Output[ ScenarioFactorSize*index + factor_offset + k ] = drifts[tenor_size*blockIdx.y + k] * exp(firstTerm + PCportion);
            }
        }
        
        __global__ void PCAInterestRateModel_Spot(REAL* Factors, const REAL* __restrict__ fwd_curve, int factor_offset)
        {
            int  index 		= ScenarioTimeSteps * ( blockIdx.x * blockDim.x + threadIdx.x );
            REAL current    = fwd_curve[blockIdx.y];
            
            for (int i=0; i<ScenarioTimeSteps; i++)
            {
                int offset = ScenarioFactorSize*(index+i) + factor_offset + blockIdx.y;
                current *= Factors[offset];
                Factors[offset] = current;
            }
        }
    '''
    
    documentation = [   'The stochastic process for the rate at each tenor on the interest rate curve is specified as:',
                        '',
                        '$$ dr_\\tau = r_\\tau ( u_\\tau  dt + \\sigma_\\tau dY )$$',
                        '$$ dY_t = -\\alpha Ydt + dZ$$',
                        '',
                        'with dY  a standard Ornstein-Uhlenbeck process and dZ a Brownian motion. It can be shown that:',
                        '$$ Y(t) \\sim N(0, \\frac{1-e^{-2 \\alpha t}}{2 \\alpha})$$ ',
                        '',
                        '- Rates at different points on the curve are correlated',
                        '- Only the first three principal components are used',
                        '',
                        'Final form of the model is',
                        '',
                        '$$ r_\\tau(t) = R_\\tau(t) exp \\{ -\\frac{1}{2} \\sigma_\\tau^2 (\\frac{1-e^{-2 \\alpha t}}{2 \\alpha}) + \\sigma_\\tau \\sum_{k=1}^{3} B_{k,\\tau} Y_k(t) \\}$$',
                        '',
                        'Where:',
                        '- $r_\\tau(t)$ is the zero rate with a tenor $\\tau$  at time $t$  where $t = 0$ denotes the current rate at tenor $\\tau$',
                        '- $R_\\tau(t)=[e^{-\\alpha t}r_\\tau (0) + (1-e^{-\\alpha t})\\theta_\\tau]$ a weighted average function of the current rate and a mean reversion level $\\theta_\\tau$',
                        '- $r_\\tau(t)$ is the zero rate with a tenor $\\tau$  at time $t$  where $t = 0$ denotes the current rate at tenor $\\tau$',
                        '- $\\theta_\\tau$ is the mean reversion level of zero rates with a tenor $\\tau$',
                        '- $B_{k,\\tau}$ are the weights derived from the principal component $k$ for the zero rates with tenor $\\tau$']
    
    def __init__(self, factor, param):
        self.factor = factor
        self.param  = param
        
        #need to precalculate these for a specific set of tenors
        self.evecs  = None
        self.vols   = None
        
    def NumFactors(self):
        return len(self.param['Eigenvectors'])

    def TheoreticalMeanStd(self, ref_date, time_in_days):
        t         = self.factor.GetDayCountAccrual(ref_date,time_in_days)
        #only works for drift to forward - todo - extend to other cases
        fwd_curve = (self.factor.CurrentVal(t+self.factor.tenors)*(t+self.factor.tenors)-self.factor.CurrentVal(t)*t)/self.factor.tenors
        sigma     = np.interp(self.factor.GetTenor(), *self.param['Yield_Volatility'].array.T)
        sigma2    = (1.0-np.exp(-2.0*self.param['Reversion_Speed']*t))/(2.0*self.param['Reversion_Speed']) * sigma*sigma
        std_dev   = np.sqrt(np.exp(sigma2)-1.0)*fwd_curve
        return fwd_curve, std_dev
    
    def PreCalc(self, ref_date, time_grid):
        #ensures that tenors used are the same as the price factor
        factor_tenor = self.factor.GetTenor()
        
        #rescale and precalculate the Eigenvectors
        evecs,evals = np.zeros((factor_tenor.size, self.NumFactors())), []
        for index, eigen_data in enumerate(self.param['Eigenvectors']):
            evecs[:,index] = np.interp(factor_tenor,*eigen_data['Eigenvector'].array.T)
            evals.append( eigen_data['Eigenvalue'] )
            
        evecs       = np.array( evecs )
        self.vols  	= np.interp(factor_tenor, *self.param['Yield_Volatility'].array.T)

        #note that I don't need to divide by the volatility because I normalize across tenors below . . .
        B  = evecs.dot ( np.diag ( np.sqrt ( evals ) ) ) 
        B  /= np.linalg.norm ( B, axis=1 ).reshape(-1,1)
        
        #normalize the eigenvectors, precalculate the vols and the historical_Yield
        self.evecs 			= B
        self.historic_mean  = scipy.interpolate.interp1d(*np.hstack(([[0.0],[self.param['Historical_Yield'].array.T[-1][0]]],self.param['Historical_Yield'].array.T)),
                                                         kind='linear', bounds_error=False,
                                                         fill_value=self.param['Historical_Yield'].array.T[-1][-1])
        
        #also need to pre-calculate the forward curves at time_grid given and pass that to the cuda kernel
        if self.param['Rate_Drift_Model']=='Drift_To_Blend':
            alpha 		= self.param['Reversion_Speed']
            curve_t0    = self.factor.CurrentVal(self.factor.tenors)
            omega       = self.historic_mean(self.factor.tenors)
            fwd_curves  = np.array ( [curve_t0*np.exp(-alpha*self.factor.GetDayCountAccrual(ref_date,t))+omega*(1.0-np.exp(-alpha*self.factor.GetDayCountAccrual(ref_date,t))) for t in time_grid.scen_time_grid] )
        else:
            fwd_curves  = np.array ( [ (self.factor.CurrentVal(self.factor.GetDayCountAccrual(ref_date,t)+self.factor.tenors)*(self.factor.GetDayCountAccrual(ref_date,t)+self.factor.tenors )-self.factor.CurrentVal(self.factor.GetDayCountAccrual(ref_date,t))*self.factor.GetDayCountAccrual(ref_date,t))/self.factor.tenors for t in time_grid.scen_time_grid] )
            
        #store the forward curves incrementally
        self.fwd_drift 		= np.vstack((np.ones_like(fwd_curves[0]),(fwd_curves[1:]/fwd_curves[:-1]).clip(0,np.inf)))
        self.Buffer         = np.hstack( (time_grid.scen_time_grid/365.25, self.evecs.T.flatten(), self.vols) )

    def CalcReferences(self, factor, static_ofs, stoch_ofs, all_tenors, all_factors):
        pass
        
    def CorrelationName(self):
        return 'InterestRateOUProcess', [('PC{}'.format(x),) for x in range(1, self.NumFactors()+1)]
    
    def Generate(self, module, CudaMem, precision, time_grid, scenario_batch_size, scenarios_per_batch, process_ofs, factor_ofs):
        grid        		= (scenario_batch_size, time_grid.size)
        block       		= (scenarios_per_batch,1,1)
        
        PCAInterestRateModel = module.get_function ('PCAInterestRateModel')
        PCAInterestRateModel ( CudaMem.d_random_numbers, drv.In(self.Buffer.astype(precision)), drv.In(self.fwd_drift.astype(precision)),
                               np.int32(process_ofs), np.int32(factor_ofs), CudaMem.d_Scenario_Buffer,
                               precision(self.param['Reversion_Speed']), np.int32(self.factor.tenors.size), block=block, grid=grid )

        grid        				= (scenario_batch_size, self.factor.tenors.size)
        PCAInterestRateModel_Spot	= module.get_function ('PCAInterestRateModel_Spot')
        PCAInterestRateModel_Spot( CudaMem.d_Scenario_Buffer, drv.In(self.factor.CurrentVal().astype(precision)), np.int32(factor_ofs), block=block, grid=grid )

class PCAInterestRateCalibration(object):
    def __init__(self, model, param):
        self.model              = model
        self.param              = param
        self.num_factors        = 3
    
    def calibrate(self, data_frame, num_business_days=252.0):
        tenor                       = np.array( [(x.split(',')[1]) for x in data_frame.columns], dtype=np.float64)
        stats, correlation, delta   = Utils.Calc_statistics ( data_frame, method='Log', num_business_days=num_business_days, max_alpha = 4.0 )

        standard_deviation          = stats['Reversion Volatility'].interpolate()
        covariance                  = np.dot ( standard_deviation.reshape(-1,1), standard_deviation.reshape(1,-1) ) * correlation 
        aki, evecs, evals	        = Utils.PCA ( covariance, self.num_factors )
        meanReversionSpeed          = stats['Mean Reversion Speed'].mean()
        volCurve 		            = standard_deviation
        reversionLevel              = stats['Long Run Mean'].interpolate()
        correlation_coef            = aki.T
        
        return Utils.CalibrationInfo(
                                        OrderedDict({
                                                    'Reversion_Speed'  : meanReversionSpeed,
                                                    'Historical_Yield' : Utils.Curve ( [], zip(tenor, reversionLevel) ),
                                                    'Yield_Volatility' : Utils.Curve ( [], zip(tenor, volCurve) ),
                                                    'Eigenvectors'     : [ OrderedDict({'Eigenvector': Utils.Curve ( [], zip(tenor, evec) ), 'Eigenvalue':eval}) for evec, eval in zip(evecs.real.T, evals.real) ],
                                                    'Rate_Drift_Model'  : self.param['Rate_Drift_Model'],
                                                    'Princ_Comp_Source' : self.param['Matrix_Type'],
                                                    'Distribution_Type' : self.param['Distribution_Type']
                                                    }),
                                        correlation_coef,
                                        delta
                                    )
    
def ConstructProcess(sp_type, factor, param):
    return globals().get(sp_type)(factor, param)

def ConstructCalibrationConfig(calibration_config, param):
    return globals().get(calibration_config['Method'])( calibration_config['PriceModel'], param )

def GetAllProcessCudaFunctions():
	return set([cls.cudacodetemplate for cls in globals().values() if isinstance(cls, type) and hasattr(cls, 'cudacodetemplate')])
