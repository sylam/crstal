'''
All risk factor are defined here. Use ConstructFactor to instantiate a the factor.
'''

#import standard libraries
import numpy as np
import pandas as pd

import Utils

class Factor0D(object):
	'''Represents an instantaneous Rate (0D) risk factor'''
	def __init__(self, param):
		self.param = param
		self.delta = 0.0

	def Bump(self, amount, relative=True):
		self.delta = self.param['Spot']*amount if relative else amount

	def GetDelta(self):
		return self.delta
	
	def CurrentVal(self):
		return np.array([self.param['Spot']+self.delta])

class Factor1D(object):
	'''Represents a risk factor with a term structure (1D)'''
	def __init__(self, param):
		self.param 			= param        
		self.tenors 		= self.GetTenor()
		self.delta 			= np.zeros_like(self.tenors)

	def GetTenor(self):
		'''Gets the tenor points stored in the Curve attribute'''
		if self.param['Curve'] is None:
			self.param['Curve'] = Utils.Curve([],[(0.0,0.0)])
		return self.param['Curve'].array[:,0]

	def Bump(self, amount, relative=True):
		self.delta = self.param['Curve'].array[:,1]*amount if relative else np.ones_like(self.tenors)*amount
		
	def GetDelta(self):
		return self.delta.mean()

	def CurrentVal(self, tenors=None):
		'''Returns the value of the rate at each tenor point (if set) else returns what's stored in the Curve parameter'''
		bumped_val = self.param['Curve'].array[:,1]+self.delta
		return np.interp(tenors, self.tenors, bumped_val)  if tenors is not None else bumped_val

class Factor2D(object):
	'''Represents a risk factor that's a surface (2D) - Currently this is only vol surfaces'''
	def __init__(self, param):
		self.param 			= param
		self.Update()

	def Update(self):
		self.moneyness 		= self.GetMoneyness()
		self.expiry 		= self.GetExpiry()
		self.vols           = self.GetVols()
		self.tenor_ofs      = np.array([0, self.moneyness.size])

	def GetMoneyness(self):
		'''Gets the moneyness points stored in the Surface attribute'''
		return np.unique ( self.param['Surface'].array[:,0] )
	
	def GetExpiry(self):
		'''Gets the expiry points stored in the Surface attribute'''
		return np.unique ( self.param['Surface'].array[:,1] )

	def GetVols(self):
		'''Uses flat extrapolation along moneyness and then linear interpolation along expiry'''
		surface = self.param['Surface'].array
		return np.array([np.interp ( self.moneyness, surface [ surface[:,1]==x ][:,0], surface[ surface[:,1]==x ][:,2] ) for x in self.expiry ] )
		
	def GetAllTenors(self):
		return np.hstack((self.moneyness, self.expiry))
		
	def CurrentVal(self):
		'''Returns the value of the vol surface'''
		return self.vols.ravel()

class Factor3D(object):
	'''Represents a risk factor that's a space (3D) - Things like swaption volatility spaces'''
	def __init__(self, param):
		self.param 			= param
		self.Update()

	def Update(self):
		self.moneyness 		= self.GetMoneyness()
		self.expiry 		= self.GetExpiry()
		self.tenor 		    = self.GetTenor()
		self.vols           = self.GetVols()
		self.tenor_ofs      = np.array([0, self.moneyness.size, self.moneyness.size+self.expiry.size])
		
	def GetMoneyness(self):
		'''Gets the moneyness points stored in the Surface attribute'''
		return np.unique ( self.param['Surface'].array[:,0] )
	
	def GetExpiry(self):
		'''Gets the expiry points stored in the Surface attribute'''
		return np.unique ( self.param['Surface'].array[:,1] )
	
	def GetTenor(self):
		'''Gets the tenor points stored in the Surface attribute'''
		return np.unique ( self.param['Surface'].array[:,2] )

	def GetVols(self):
		'''Uses flat extrapolation along moneyness and then linear interpolation along expiry'''
		vols = []
		for tenor in self.tenor:
			surface = self.param['Surface'].array [ self.param['Surface'].array[:,2]==tenor ]
			vols.append ( [ np.interp ( self.moneyness, surface [surface[:,1]==x][:,0], surface [surface[:,1]==x][:,3]) for x in self.expiry ] )
		return np.array(vols)
		
	def GetAllTenors(self):
		return np.hstack((self.moneyness, self.expiry, self.tenor))

	def CurrentVal(self, tenors=None):
		'''Returns the value of the Vol space'''
		return self.vols.ravel()
	
class FxRate(Factor0D):
	'''
	Represents the price of a currency relative to the base currency (snapped at end of day).
	'''
	def __init__(self, param):
		super(FxRate, self).__init__(param)
		
	def GetRepoCurveName(self, default):
		return Utils.CheckRateName(self.param['Interest_Rate'] if self.param['Interest_Rate'] else default)
	
	def GetDomesticCurrency(self, default):
		return Utils.CheckRateName(self.param['Domestic_Currency'] if self.param['Domestic_Currency'] else default)

class FuturesPrice(Factor0D):
	def __init__(self, param):
		super(FuturesPrice, self).__init__(param)
		
	def CurrentVal(self):
		return np.array([self.param['Price']])
		
class PriceIndex(Factor0D):
	'''
	Used to represent things like CPI/Stock Indices etc.
	'''
	def __init__(self, param):
		super(PriceIndex, self).__init__(param)
		#the start date for excel's date offset
		self.start_date   = pd.datetime(1899, 12, 30)
		#the offset to the latest index value
		self.last_period  = self.param['Last_Period_Start']-self.start_date
		self.latest_index = np.where(self.param['Index'].array[:,0]>=self.last_period.days)[0]

	def CurrentVal(self):
		return np.array ( [ self.param['Index'].array[ self.latest_index[0] ][1] ] if self.latest_index.any() else [ self.param['Index'].array[-1][1] ] )
	
	def GetReferenceVal(self, ref_date):
		query = float( (ref_date - self.start_date).days )
		return np.interp ( query, *self.param['Index'].array.T )

	def GetLastPublicationDates(self, base_date, time_grid):
		roll_period			= pd.DateOffset(months=3) if self.param['Publication_Period']=='Quarterly' else pd.DateOffset(months=1)
		last_date			= base_date+pd.DateOffset(days=time_grid[-1])
		publication 		= pd.date_range(self.param['Last_Period_Start'], last_date, freq=roll_period)
		next_publication 	= pd.date_range(self.param['Next_Publication_Date'], last_date+roll_period, freq=roll_period)
		eval_dates			= [(base_date+pd.DateOffset(days=t)).to_datetime64() for t in time_grid]
		return publication [ np.searchsorted ( next_publication.tolist(), eval_dates, side='right')]

class EquityPrice(Factor0D):
	'''
	This is just the equity price on a particular end of day
	'''
	def __init__(self, param):
		super(EquityPrice, self).__init__(param)
		
	def GetRepoCurveName(self):
		return Utils.CheckRateName ( self.param['Interest_Rate'] if self.param['Interest_Rate'] else self.param['Currency'] )
	
	def GetCurrency(self):
		return Utils.CheckRateName( self.param['Currency'] )

class ForwardPriceSample(Factor0D):
	'''
	This is just the sampling method for Forward Prices
	'''
	def __init__(self, param):
		super(ForwardPriceSample, self).__init__(param)

	def CurrentVal(self):
		return np.array([self.param['Offset']])
	
	def GetHolidayCalendar(self):
		return self.param.get('Holiday_Calendar')
		
	def GetSampling_Convention(self):
		return self.param.get('Sampling_Convention')
		
class DiscountRate(object):
	'''
	Frankly speaking, not really sure why we have a discount factor rate as opposed to just another interest rate factor.
	But who knows.
	'''
	def __init__(self, param):
		self.param = param
		
	def GetInterestRate(self):
		return Utils.CheckRateName ( self.param['Interest_Rate'] )
	
class ReferenceVol(object):
	def __init__(self, param):
		self.param = param

	def GetForwardPriceVol(self):
		return Utils.CheckRateName( self.param['ForwardPriceVol'] )
	
	def GetForwardPrice(self):
		return Utils.CheckRateName( self.param['ReferencePrice'] )

class Correlation(object):
	def __init__(self, param):
		self.param = param

	def CurrentVal(self):
		return self.param.get( 'Value', 0.0 )

class DividendRate(Factor1D):
	'''
	Represents the Dividend Yield risk factor
	'''
	def __init__(self, param):
		super(DividendRate, self).__init__(param)
		
	def GetCurrency(self):
		return Utils.CheckRateName( self.param['Currency'] )
		
	def GetDayCount(self):
		'''Adaptiv hardcodes the daycount for dividend rates to act/365'''
		return Utils.DAYCOUNT_ACT365

class SurvivalProb(Factor1D):
	'''
	Represents the Probability of Survival risk factor
	'''
	def __init__(self, param):
		super(SurvivalProb, self).__init__(param)
		
	def GetDayCount(self):
		'''Adaptiv hardcodes the daycount for Survival Probability rates to act/365'''
		return Utils.DAYCOUNT_ACT365
	
	def GetDayCountAccrual(self, ref_date, time_in_days):
		return Utils.GetDayCountAccrual(ref_date, time_in_days, self.GetDayCount() )
	
	def RecoveryRate(self):
		return self.param.get('Recovery_Rate')
	
class InterestRate(Factor1D):	
	'''
	Represents an Interest Rate risk factor - basically a time indexed array of rates
	Remember that the tenors are normally expressed as year fractions - not days.
	'''
	def __init__(self, param):
		super(InterestRate, self).__init__(param)
		
	def GetCurrency(self):
		return Utils.CheckRateName( self.param['Currency'] )

	def GetDayCount(self):
		return Utils.GetDayCount( self.param['Day_Count'] )
	
	def GetSubType(self):
		return 'InterestRate' + ( self.param['Sub_Type'] if self.param['Sub_Type'] else '' )
	
	def GetDayCountAccrual(self, ref_date, time_in_days):
		return Utils.GetDayCountAccrual(ref_date, time_in_days, self.GetDayCount() )

class InflationRate(Factor1D):	
	'''
	Represents an Interest Rate (1D) risk factor - basically a time indexed array of rates
	Remember that the tenors are normally expressed as year fractions - not days.
	'''
	def __init__(self, param):
		super(InflationRate, self).__init__(param)

	def GetReferenceName(self):
		return self.param['Reference_Name']

	def GetDayCount(self):
		return Utils.GetDayCount( self.param['Day_Count'] )

	def GetDayCountAccrual(self, ref_date, time_in_days):
		return Utils.GetDayCountAccrual(ref_date, time_in_days, self.GetDayCount() )

class ForwardPrice(Factor1D):
	'''
	Used to represent things like Futures prices on OIL/GOLD/Platinum etc.
	'''
	def __init__(self, param):
		super(ForwardPrice, self).__init__(param)
		#the start date for excel's date offset
		self.start_date   = pd.datetime(1899, 12, 30)
		
	def GetCurrency(self):        
		return Utils.CheckRateName ( self.param['Currency'] )

	def GetRelativeTenor(self, reference_date):
		reference_date_excel   = (reference_date-self.start_date).days
		return self.GetTenor()-reference_date_excel
	
	def GetDayCount(self):
		return Utils.DAYCOUNT_ACT365# .GetDayCount( self.param['Day_Count'] )

class ReferencePrice(Factor1D):
	'''
	Used to represent how lookups on the Forward/Futures curve are performed.
	'''
	def __init__(self, param):
		super(ReferencePrice, self).__init__(param)
		#the start date for excel's date offset
		self.start_date   = pd.datetime(1899, 12, 30)
		#the offset to the latest index value

	def GetForwardPrice(self):
		return Utils.CheckRateName ( self.param['ForwardPrice'] )
	
	def GetFixings(self):
		return self.param['Fixing_Curve']

	def GetTenor(self):
		'''Gets the tenor points stored in the Curve attribute'''
		return self.param['Fixing_Curve'].array[:,0]

	def CurrentVal(self, tenors=None):
		'''Returns the value of the rate at each tenor point (if set) else returns what's stored in the Curve parameter'''
		return np.interp(tenors,*self.param['Fixing_Curve'].array.T)  if tenors is not None else self.param['Fixing_Curve'].array[:,1]

class FXVol(Factor2D):
	def __init__(self, param):
		super(FXVol, self).__init__(param)

class EquityPriceVol(Factor2D):
	def __init__(self, param):
		super(EquityPriceVol, self).__init__(param)
		
class InterestYieldVol(Factor3D):
	def __init__(self, param):
		super(InterestYieldVol, self).__init__(param)

class InterestRateVol(Factor3D):
	def __init__(self, param):
		super(InterestRateVol, self).__init__(param)

class ForwardPriceVol(Factor3D):
	def __init__(self, param):
		super(ForwardPriceVol, self).__init__(param)

def ConstructFactor(sp_type, param):
	#add a rule here to let the factor know wheather a price factor is added to its base or multiplied - for now assume its added
	return globals().get(sp_type)(param)

if __name__=='__main__':
	import matplotlib
	import matplotlib.pylab as plt
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	from matplotlib.ticker import LinearLocator, FormatStrFormatter
	if 0:
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		X = self.moneyness
		Y = self.expiry
		X, Y = np.meshgrid(X, Y)
		surf = ax.plot_surface(X, Y, linear, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
		fig.colorbar(surf, shrink=0.5, aspect=5)
		plt.show()