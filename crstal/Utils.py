import sys
import datetime
import calendar

from collections import namedtuple

#useful types
if sys.version[:3]>'2.6':
	from collections import OrderedDict
else:
	from ordereddict import OrderedDict

#try to import numpy and pandas - if it fails, it because we're using pypy 
try:
	import pandas as pd
	import numpy  as np
	
	#For dealing with excel dates and dataframes
	excel_offset 				= pd.Timestamp('1899-12-30 00:00:00')
	array_type					= lambda x:np.array(x)
except:
	#just use a list as is
	array_type					= lambda x:x

#daycount codes - must match the cuda code
DAYCOUNT_ACT365				= 0
DAYCOUNT_ACT360				= 1
DAYCOUNT_ACT365IDSA			= 2
DAYCOUNT_ACT30_360			= 3
DAYCOUNT_ACT30_E360			= 4
DAYCOUNT_ACTACTICMA         = 5

#cashflow codes - must match the cuda code
CASHFLOW_INDEX_Start_Day	= 0
CASHFLOW_INDEX_End_Day		= 1
CASHFLOW_INDEX_Pay_Day		= 2

CASHFLOW_INDEX_Year_Frac	= 3
#can also use this index for equity swaplet multipliers
CASHFLOW_INDEX_Start_Mult	= 3

CASHFLOW_INDEX_Nominal		= 4
#can also use this index for equity swaplet multipliers
CASHFLOW_INDEX_End_Mult		= 4

CASHFLOW_INDEX_FixedAmt		= 5

#Cashflow code for Float payments
CASHFLOW_INDEX_FloatMargin		= 6
#Cashflow code for Fixed payments
CASHFLOW_INDEX_FixedRate    	= 6
#Cashflow code for caps/floor payments
CASHFLOW_INDEX_Strike			= 6
#Cashflow code for equity swaplet multipliers
CASHFLOW_INDEX_Dividend_Mult	= 6
#Cashflow code for possible FX resets
CASHFLOW_INDEX_FXResetDate		= 7
CASHFLOW_INDEX_FXResetValue		= 8

#Number of resets/fixings for this cashflow (0 for fixed cashflows)
CASHFLOW_INDEX_NumResets	= 9
#offset in the reset/fixings array for this cashflow
CASHFLOW_INDEX_ResetOffset	= 10
#Boolean (0 or 1) value that determines if this cashflow is settled (1) or accumulated (0)
CASHFLOW_INDEX_Settle		= 11

#Cashflow calculation methods - must match the cuda code
CASHFLOW_METHOD_IndexReference2M 				= 1
CASHFLOW_METHOD_IndexReference3M 				= 2
CASHFLOW_METHOD_IndexReferenceInterpolated3M 	= 3
CASHFLOW_METHOD_IndexReferenceInterpolated4M 	= 4

CASHFLOW_METHOD_Equity_Shares	 				= 0
CASHFLOW_METHOD_Equity_Principal 				= 1
CASHFLOW_METHOD_Average_Interest 				= 0
#CASHFLOW_METHOD_Average_Rate	 				= 1

CASHFLOW_METHOD_Compounding_Include_Margin 		= 2
CASHFLOW_METHOD_Compounding_Flat		 		= 3
CASHFLOW_METHOD_Compounding_Exclude_Margin 		= 4
CASHFLOW_METHOD_Compounding_None				= 5

CASHFLOW_METHOD_Fixed_Compounding_No			= 0
CASHFLOW_METHOD_Fixed_Compounding_Yes			= 1

CASHFLOW_IndexMethodLookup = {	'IndexReference2M': CASHFLOW_METHOD_IndexReference2M,
								'IndexReference3M': CASHFLOW_METHOD_IndexReference3M,
								'IndexReferenceInterpolated3M': CASHFLOW_METHOD_IndexReferenceInterpolated3M,
								'IndexReferenceInterpolated4M': CASHFLOW_METHOD_IndexReferenceInterpolated4M }

CASHFLOW_CompoundingMethodLookup = {	'None': 	CASHFLOW_METHOD_Compounding_None,
										'Flat':		CASHFLOW_METHOD_Compounding_Flat,
										'Include_Margin': CASHFLOW_METHOD_Compounding_Include_Margin }
					
#reset codes - must match the cuda code
RESET_INDEX_Time_Grid		= 0
RESET_INDEX_Reset_Day		= 1
RESET_INDEX_Start_Day		= 2
RESET_INDEX_End_Day		    = 3
RESET_INDEX_Scenario		= 4
RESET_INDEX_Weight		    = 5
RESET_INDEX_Value			= 6
#used to store the reset accrual period
RESET_INDEX_Accrual			= 7
#used to store any fx averaging (can't be used with accrual periods)
RESET_INDEX_FXValue			= 7

#modifiers for dealing with a sequence of cashflows
SCENARIO_CASHFLOWS_FloatLeg = 0
SCENARIO_CASHFLOWS_Cap      = 1
SCENARIO_CASHFLOWS_Floor    = 2
SCENARIO_CASHFLOWS_Energy   = 3
SCENARIO_CASHFLOWS_Index   	= 4
SCENARIO_CASHFLOWS_Equity  	= 5

#Constants for the time grid
TIME_GRID_PriorScenarioDelta	= 0
TIME_GRID_MTM					= 1
TIME_GRID_ScenarioPriorIndex    = 2

#Collateral Cash Valuation mode
CASH_SETTLEMENT_Received_Only 	= 0
CASH_SETTLEMENT_Paid_Only 		= 1
CASH_SETTLEMENT_All 			= 2

#Factor sizes
FACTOR_SIZE_CURVE				= 4
FACTOR_SIZE_RATE				= 2

#Named tuples to make life easier
Factor 			= namedtuple('Factor', 'type name')
RateInfo  		= namedtuple('RateInfo', 'model_name archive_name calibration')
CalibrationInfo = namedtuple('CalibrationInfo', 'param correlation delta')
DealDataType	= namedtuple('DealDataType','Instrument Factor_dep Time_dep Calc_res')
Partition       = namedtuple('Partition','DealMTMs Collateral_Cash Funding_Cost Cashflows')

#define 1, 2 and 3d risk factors - add more as validation proceeds
DimensionLessFactors 		= ['DiscountRate', 'ReferenceVol', 'Correlation']
OneDimensionalFactors 		= ['InterestRate','InflationRate','DividendRate','SurvivalProb','ForwardPrice']
TwoDimensionalFactors 		= ['FXVol','EquityPriceVol']
ThreeDimensionalFactors 	= ['InterestRateVol', 'InterestYieldVol','ForwardPriceVol']

#weekends and weekdays
WeekendMap = {'Friday and Saturday': 'Sun Mon Tue Wed Thu',
			  'Saturday and Sunday': 'Mon Tue Wed Thu Fri',
			  'Sunday': 'Mon Tue Wed Thu Fri Sat',
			  'Friday': 'Sat Sun Mon Tue Wed Thu'}
	
def Filter_DF(df, from_date, to_date, rate=None):
	index1 = (pd.Timestamp(from_date)-excel_offset).days
	index2 = (pd.Timestamp(to_date)-excel_offset).days
	return df.ix[index1:index2] if rate is None else df.ix[index1:index2][[col for col in df.columns if col.startswith(rate)]]

class Descriptor:
	def __init__(self, value):
		self.data = value
		self.descriptor_type = 'X'
		
	def __str__(self):
		return self.descriptor_type.join([str(x) for x in self.data])

#Adaptiv Analytics Defined types - data that's parsed in via an Adaptiv Analytic config/trade file
class Percent:
	def __init__(self, amount):
		self.amount = amount/100.0
		
	def __str__(self):
		return '%g%%' % (self.amount*100.0)

#to parse things like 12 bp - why didn't Adaptiv just use 0.0012? or even 0.12%? I don't know
class Basis:
	def __init__(self, amount):
		self.amount = amount/10000.0
		self.points = amount
		
	def __str__(self):
		return '%d bp' % self.points

class Curve:
	def __init__(self, meta, data):
		self.meta  = meta
		self.array = array_type(data)
		
	def __str__(self):
		
		def format1darray(data):
			return '(%s)' % ','.join(['%.12g' % y for y in data])
		
		array_rep = format1darray(self.array) if len(self.array.shape)==1 else ','.join( [format1darray(x) for x in self.array] )
		meta_rep  = ','.join([str(x) for x in self.meta])
		return '[%s,%s]' % (meta_rep, array_rep) if meta_rep else '[%s]' % array_rep
	
class Offsets:
	lookup = {'months':'m', 'days':'d', 'years':'y', 'weeks':'w'}
	def __init__(self, data, grid=False):
		self.grid = grid
		self.data = data
		
	def __str__(self):
		periods = [''.join(['%d%s' % (v,Offsets.lookup[k]) for k,v in value.kwds.items()]) for value in self.data]
		return '{0}'.format(' '.join ( periods ) ) if self.grid else '[{0}]'.format(','.join ( periods ) )

class DateList:
	def __init__(self, data):
		self.data = OrderedDict(data)
		self.dates = set()
		
	def Last(self):
		return self.data.values()[-1] if self.data.values() else 0.0
	
	def __str__(self):
		return '\\'.join( [ '%s=%.12g' % ('%02d%s%04d' % (x[0].day, calendar.month_abbr[x[0].month], x[0].year), x[1]) for x in self.data.items() ] )+'\\'

	def SumRange(self, run_date, cuttoff_date, index):
		return sum([val for date, val in self.data.items() if date<run_date and date>cuttoff_date],0.0)

	def PrepareDates(self):
		self.dates = set(self.data.keys())
		
	def Consume(self, cuttoff, date):
		datelist = set([x for x in self.dates if x>=cuttoff]) if cuttoff else self.dates
		if datelist:
			closest_date = min ( datelist, key=lambda x:np.abs((x-date).days) )
			if closest_date<=date:
				self.dates.remove( closest_date )
			return closest_date, self.data[closest_date]
		else:
			return None, 0.0

class CreditSupportList:
	def __init__(self, data):
		self.data = OrderedDict(data)

	def value(self):
		return self.data.values()[0]
	
	def __str__(self):
		return '\\'.join( [ '%d=%.12g' % (rating, amount) for rating, amount in self.data.items() ] )+'\\'

class DateEqualList:
	def __init__(self, data):
		self.data = OrderedDict([(x[0],x[1:]) for x in data])

	def value(self):
		return self.data.values()
	
	def SumRange(self, run_date, cuttoff_date, index):
		return sum([val[index] for date, val in self.data.items() if date<run_date and date>cuttoff_date],0.0)
	
	def __str__(self):
		return '['+','.join( [ '%s=%s' % ( '%02d%s%04d' % (date.day, calendar.month_abbr[date.month], date.year), '='.join([str(y) for y in value]) ) for date, value in self.data.items() ] )+']'

#Cuda specific classes that's used internally		
class CudaScedule(object):
	def __init__(self, schedule, offsets):
		self.schedule = np.array(schedule)
		self.offsets  = np.array(offsets)
		self.cache    = None
		self.dtype    = None
		
	def __getitem__(self, x):
		return self.schedule[x]
	
	def Count(self):
		return self.schedule.shape[0]
	
class CudaResets(CudaScedule):
	def __init__(self, schedule, offsets):
		super(CudaResets, self).__init__(schedule, offsets)

	def GetSchedule(self, precision):
		if self.cache is None or precision!=self.dtype:
			self.dtype       = precision
			offset           = np.int32 if precision==np.float32 else np.int64
			scenario_offsets = np.array(self.offsets, dtype=offset)
			self.cache       = self.schedule.astype(precision)
			self.cache[:,RESET_INDEX_Scenario] = scenario_offsets.view(precision)
		return self.cache
	
	def GetStartIndex(self, time_grid, offset=0):
		'''Read the start index (relative to the time_grid) of each reset'''
		return np.searchsorted ( self.schedule[:,RESET_INDEX_Reset_Day]-offset, time_grid.time_grid[:,TIME_GRID_MTM] ).astype(np.int32)
	
class CudaCashFlows(CudaScedule):
	def __init__(self, schedule, offsets):
		#check which cashflows are settlements (as opposed to accumulations)			
		for cashflow, next_cashflow, cash_ofs in zip( schedule[:-1], schedule[1:], offsets[:-1] ):
			if next_cashflow[ CASHFLOW_INDEX_Pay_Day ] != cashflow[ CASHFLOW_INDEX_Pay_Day ]:
				cash_ofs[2] = 1

		#last cashflow always settles
		offsets[-1][2] = 1
		
		#call superclass
		super(CudaCashFlows, self).__init__(schedule, offsets)
		self.Resets  = None

	def GetSchedule(self, precision):
		if self.cache is None or precision!=self.dtype:
			self.dtype          = precision
			offset              = np.int32 if precision==np.float32 else np.int64
			cashflow_reset_offsets = np.array(self.offsets, dtype=offset)
			self.cache          = np.hstack((self.schedule.astype(precision), cashflow_reset_offsets.view(precision)))
		return self.cache
	
	def GetRawSchedule(self, precision):
		if self.cache is None or precision!=self.dtype:
			self.dtype          	= precision
			cashflow_reset_offsets  = np.array(self.offsets, dtype=self.dtype)
			self.cache          	= np.hstack((self.schedule.astype(precision), cashflow_reset_offsets))
		return self.cache

	def InsertCashflow(self, cashflow):
		'''Inserts a cashflow at the beginning of the cashflow schedule - useful to model a fixed payment at the beginning of a schedule of cashflows'''
		self.schedule = np.vstack ( (cashflow, self.schedule) )
		self.offsets  = np.vstack ( ([0,0,1], self.offsets) )

	def AddMaturityAccrual(self, reference_date, daycount_code):
		'''Adjusts the last cashflow's daycount accrual fraction to include the maturity date'''
		last_cashflow = self.schedule[-1]
		last_cashflow[CASHFLOW_INDEX_Year_Frac] = GetDayCountAccrual( reference_date, last_cashflow[CASHFLOW_INDEX_End_Day]- last_cashflow[CASHFLOW_INDEX_Start_Day] +1, daycount_code )
		
	def SetResets(self, schedule, offsets):
		self.Resets = CudaResets(schedule, offsets)

	def OverwriteRate(self, attribute_index, value):
		'''
		Overwrites the strike/fixed_amount/float_rate defined in the cashflow schedule
		'''
		for cashflow in self.schedule:
			cashflow[attribute_index] = value
		self.cache = None

	def AddFixedPayments(self, base_date, Principal_Exchange, Effective_Date, Day_Count, principal):
		if (Principal_Exchange in ['Start_Maturity', 'Start']) and base_date<=Effective_Date:
			self.InsertCashflow ( MakeCashFlow ( base_date, Effective_Date, Effective_Date, Effective_Date, 0.0, GetDayCount( Day_Count ), -principal, 0.0 ) )
			
		if (Principal_Exchange  in ['Start_Maturity','Maturity']):
			self.schedule[-1][CASHFLOW_INDEX_FixedAmt] = principal
			
	def GetCashflowStartIndex(self, time_grid, last_payment=None):
		'''Read the start index (relative to the time_grid) of each cashflow'''
		t_grid = time_grid.time_grid[:,TIME_GRID_MTM]
		if last_payment:
			t_grid = time_grid.time_grid[:,TIME_GRID_MTM].copy()
			t_grid [t_grid > last_payment] = self.schedule[:,CASHFLOW_INDEX_Pay_Day].max()+1
		return np.searchsorted ( self.schedule[:,CASHFLOW_INDEX_Pay_Day], t_grid ).astype(np.int32)

#Math Type stuff
	
def PCA(matrix, nRedDim=0):
	# Compute eigenvalues and sort into descending order
	evals,evecs = np.linalg.eig(matrix)
	indices = np.argsort(evals)[::-1]
	evecs = evecs[:,indices]
	evals = evals[indices]

	if nRedDim>0:
		evecs = evecs[:,:nRedDim]

	var = np.diag(matrix)
	aki = evecs[:,:nRedDim] * np.sqrt ( var.reshape(-1,1).dot ( 1.0 / evals[:nRedDim].reshape ( 1, -1 ) ) )
	
	return aki, evecs, evals[:nRedDim]

def Calc_statistics ( data_frame, method='Log', num_business_days=252.0, frequency=1, max_alpha = 4.0 ):
	'Currently only frequency==1 is supported'
	
	def calc_alpha(x, y):
		return ( -num_business_days * np.log ( 1.0 + ((x-x.mean(axis=0))*(y-y.mean(axis=0))).mean(axis=0)/((y-y.mean(axis=0))**2.0).mean(axis=0) ) ).clip(-max_alpha, max_alpha)
	
	def calc_sigma2(x, y, alpha):
		return (x.var(axis=0) - ((1 - np.exp(-alpha/num_business_days))**2) * y.var(axis=0))*((2.0*alpha)/(1-np.exp(-2.0*alpha/num_business_days)))
	
	def calc_theta(x, y, alpha):
		return y.mean(axis=0) + x.mean(axis=0)/(1.0-np.exp(-alpha/num_business_days))

	def calc_Theta(theta, sigma2, alpha):
		return np.exp(theta+sigma2/(4.0*alpha))
		
	delta           = frequency/num_business_days
	transform       = {'Diff':lambda x:x, 'Log':lambda x:np.log(x.clip(0.0001,np.inf))}[method]
	transformed_df  = transform(data_frame)
	
	#can implement decay weights here if needed
	#transformed_df['_Decay'] = [weight**i for i in range(data_frame.shape[0])]
	
	data    = transformed_df.diff(frequency).shift(-frequency)
	y       = transformed_df#
	alpha   = calc_alpha(data, y)
	theta   = calc_theta(data, y, alpha)
	sigma2  = calc_sigma2(data, y, alpha)
	
	if method=='Log':
		theta   = calc_Theta(theta, sigma2, alpha)
	
	stats = pd.DataFrame({
							  'Volatility':data.std(axis=0)*np.sqrt(num_business_days),
							  'Drift':data.mean(axis=0)*num_business_days,
							  'Mean Reversion Speed':alpha,
							  'Long Run Mean':theta,
							  'Reversion Volatility':np.sqrt(sigma2)
						 })
	
	correlation = data.corr()
	
	return stats, correlation, data

#Graph operation type stuff
 
def Topolgical_Sort(graph_unsorted):
	"""
	Repeatedly go through all of the nodes in the graph, moving each of
	the nodes that has all its edges resolved, onto a sequence that
	forms our sorted graph. A node has all of its edges resolved and
	can be moved once all the nodes its edges point to, have been moved
	from the unsorted graph onto the sorted one.
	
	NB - this destroys the graph_unsorted dictionary that was passed in
	and just returns the keys of the sorted graph
	"""

	graph_sorted = []

	# Run until the unsorted graph is empty.
	while graph_unsorted:

		acyclic = False
		for node, edges in graph_unsorted.items():
			for edge in edges:
				if edge in graph_unsorted:
					break
			else:
				acyclic = True
				del graph_unsorted[node]
				graph_sorted.append(node)

		if not acyclic:
			raise RuntimeError("A cyclic dependency occurred")

	return graph_sorted

#Data transformation type stuff

def GetDayCount(code):
	if code=='ACT_365':
		return DAYCOUNT_ACT365
	elif code=='ACT_360':
		return DAYCOUNT_ACT360
	elif code=='_30_360':
		return DAYCOUNT_ACT30_360
	elif code=='_30E_360':
		return DAYCOUNT_ACT30_E360
	elif code=='ACT_365_ISDA':
		return DAYCOUNT_ACT365IDSA
	elif code=='ACT_ACT_ICMA':
		return DAYCOUNT_ACTACTICMA
	else:
		raise Exception('Daycount {} Not implemented'.format(code))
	
def GetDayCountAccrual(reference_date, time_in_days, code):
	'''Need to complete this implementation - and it needs to match the cuda device function of the same name - note that time_in_days is incremental'''
	if code == DAYCOUNT_ACT360:
		return time_in_days/360.0
	elif code == DAYCOUNT_ACT365:
		return time_in_days/365.0
	elif code in (DAYCOUNT_ACT365IDSA,DAYCOUNT_ACTACTICMA):
		in_leap=0
		#TODO
		#end_date = reference_date+pd.DateOffset(days=time_in_days)
		#for date in range(reference_date.year, end_date.year):
		#	if calendar.isleap(reference_date.year):
		#		in_leap+=366.0
		return time_in_days/365.0
	elif code == DAYCOUNT_ACT30_360:
		e1 					= min(reference_date.day,30)
		new_date = end_date = reference_date
		if isinstance(time_in_days,np.ndarray):
			ret = []
			for ed in time_in_days.astype(np.int32):
				end_date += pd.DateOffset(days=ed)
				e2 = 30 if end_date.day>=30 and new_date.day>=30 else end_date.day
				ret.append( ( (e2-e1)+30*(end_date.month-new_date.month)+360*(end_date.year-new_date.year) ) / 360.0 )
				new_date = end_date
			return ret
		else:
			end_date = reference_date+pd.DateOffset(days=time_in_days)
			e2 = 30 if end_date.day>=30 and reference_date.day>=30 else end_date.day
			return ( (e2-e1)+30*(end_date.month-reference_date.month)+360*(end_date.year-reference_date.year) ) / 360.0
	elif code == DAYCOUNT_ACT30_E360:
		e1 					= min(reference_date.day,30)
		new_date = end_date = reference_date
		if isinstance(time_in_days,np.ndarray):
			ret = []
			for ed in time_in_days.astype(np.int32):
				end_date += pd.DateOffset(days=ed)
				e2 = min(end_date.day,30)
				ret.append( ( (e2-e1)+30*(end_date.month-new_date.month)+360*(end_date.year-new_date.year) ) / 360.0 )
				new_date = end_date
			return ret
		else:
			end_date = reference_date+pd.DateOffset(days=time_in_days)
			e2 = min(end_date.day,30)
			return ( (e2-e1)+30*(end_date.month-reference_date.month)+360*(end_date.year-reference_date.year) ) / 360.0

def GetFieldName(field, obj):
	'''Needed to evaluate nested fields - e.g. collateral fields. TODO - extend to return more than the first value'''
	return ( [element.get(field[0]) for element in obj][0] if len(field)==1 else GetFieldName(field[1:], obj[field[0]] if obj.get(field[0]) else ({} if len(field)>2 else [{}]) ) ) if isinstance(field, tuple) else obj.get(field)
	
def CheckRateName(name):
	'''Needed to ensure that name is a tuple - Rate names need to be in upper case'''
	#return tuple([x.upper() for x in name]) if type(name)==tuple else (name.upper(),)
	return tuple([x.upper() for x in name]) if type(name)==tuple else tuple(name.split('.'))

def CheckTupleName(factor):
	'''Opposite of CheckRateName - used to make sure the name is a flat name'''
	#return (factor.type,)+factor.name if type(factor.name)==tuple else factor
	return '.'.join((factor.type,)+factor.name) if type(factor.name)==tuple else factor

def MakeCashFlow (reference_date, start_date, end_date, pay_date, nominal, daycount_code, fixed_amount, spread_or_rate):
	'''
	Constructs a single cashflow vector with the provided paramters - can be used to manually construct nominal
	or fixed payments.
	'''
	cashflow_days = [ (x-reference_date).days for x in [start_date, end_date, pay_date] ]
	return np.array( cashflow_days + [ GetDayCountAccrual(reference_date, cashflow_days[1]-cashflow_days[0],daycount_code), nominal, fixed_amount, spread_or_rate, 0, 0 ] )
					  
def GetCashflows ( reference_date, reset_dates, nominal, amort, daycount_code, spread_or_rate ):
	'''
	Generates a vector of Start_day, End_day, Pay_day, Year_Frac, Nominal, FixedAmount (=0)
	and rate/spread from the parameters provided. Note that the length of the nominal array must
	be 1 less than the reset_dates (Since there is no nominal on the first reset date i.e.
	Effective date).
	The nominal could also be just a single number representing a vanilla (constant) profile
	
	Returns a vector of days (and nominals) relative to the reference date
	'''
	
	amort_offsets       = np.array ( [ ( ( k-reference_date).days, v ) for k,v in amort.data.items() ] if amort else [] )
	day_offsets         = np.array ( [ (x-reference_date).days for x in reset_dates ] )
	
	nominal_amount, nominal_sign = [np.abs(nominal)], 1 if nominal>0 else -1
	amort_index = 0
	for offset in day_offsets[1:]:
		amort_to_add = 0.0
		while amort_index<amort_offsets.shape[0] and amort_offsets[amort_index][0]<=offset:
			amort_to_add+=amort_offsets[amort_index][1]
			amort_index+=1
		nominal_amount.append(nominal_amount[-1]-amort_to_add)
	nominal_amount = nominal_sign*np.array( nominal_amount )
	
	#we want the earliest negative number
	last_payment   = np.where(day_offsets>=0)[0]
	
	#calculate the index of the earliest cashflow
	previous_index = max(last_payment[0]-1 if last_payment.size else day_offsets.size,0)
	cashflows_left = day_offsets [ previous_index : ]
	rates          = spread_or_rate if isinstance(nominal,np.ndarray) else [spread_or_rate]*(reset_dates.size-1)
	ref_date	   = (reference_date+pd.DateOffset(days=cashflows_left[0].astype(np.int32))) if cashflows_left.any() else reference_date
	#order is start_day, end_day, pay_day, daycount_accrual, nominal, fixed amount, FxResetDate, FXResetValue - must match the cuda definitions
	return zip ( cashflows_left[:-1], cashflows_left[1:], cashflows_left[1:], GetDayCountAccrual (ref_date, np.diff(cashflows_left), daycount_code ), nominal_amount[previous_index:], np.zeros( cashflows_left.size - 1 ), rates[previous_index:], np.zeros( cashflows_left.size - 1 ), np.zeros( cashflows_left.size - 1 ) )

def GenerateFloatCashflows ( reference_date, time_grid, reset_dates, nominal, amort, known_rate_list, reset_tenor, reset_frequency, daycount_code, spread ):
	'''
	Generates a vector of Start_day, End_day, Pay_day, Year_Frac, Nominal, FixedAmount (=0)
	and spread from the parameters provided. Note that the length of the nominal array must
	be 1 less than the reset_dates (Since there is no nominal on the first reset date i.e.
	Effective date).
	The nominal could also be just a single number representing a vanilla (constant) profile
	
	Returns a vector of days (and nominals) relative to the reference date, as well as as
	the structure of resets
	'''
	
	cashflow_schedule       = GetCashflows ( reference_date, reset_dates, nominal, amort, daycount_code, spread )
	cashflow_reset_offsets  = []
	all_resets              = []
	reset_scenario_offsets  = []
	
	#prepare to consume reset dates
	known_rates = known_rate_list if known_rate_list is not None else DateList({})
	known_rates.PrepareDates()
	
	min_date = None
	for cashflow in cashflow_schedule:
		r           = []
		if reset_frequency.kwds.values()[0]==0.0:
			reset_days  = np.array( [ reference_date + pd.DateOffset(days=int(cashflow[CASHFLOW_INDEX_Start_Day])) ] )
			reset_tenor = pd.offsets.Day(cashflow[CASHFLOW_INDEX_End_Day]-cashflow[CASHFLOW_INDEX_Start_Day])
		else:
			reset_days  = pd.date_range ( reference_date + pd.DateOffset(days=int(cashflow[CASHFLOW_INDEX_Start_Day])), reference_date + pd.DateOffset(days=int(cashflow[CASHFLOW_INDEX_End_Day])), freq=reset_frequency, closed='left' )
			reset_tenor = reset_frequency if reset_tenor.kwds.values()[0]==0.0 else reset_tenor
			
		for reset_day in reset_days:
			Reset_Day		    = (reset_day-reference_date).days
			Start_Day		    = (reset_day-reference_date).days
			End_Day		        = (reset_day+reset_tenor-reference_date).days
			Accrual             = GetDayCountAccrual(reference_date, End_Day-Start_Day, daycount_code)
			Weight		        = 1.0/reset_days.size
			Time_Grid, Scenario = time_grid.GetScenarioOffset( Reset_Day )
			
			#match the closest reset
			closest_date, Value = known_rates.Consume(min_date, reset_day)
			if closest_date is not None:
				min_date = closest_date if min_date is None else max(min_date,closest_date)
				
			#only add a reset if its in the past
			r.append([Time_Grid,Reset_Day,Start_Day,End_Day,-1,Weight, Value/100.0 if reset_day<reference_date else 0.0, Accrual])
			reset_scenario_offsets.append(Scenario)
			
			if Start_Day==End_Day:
				raise Exception("Reset Start and End Days co-incide")
						
		#attach the reset_offsets to the cashflow - assume each cashflow is a settled one (not accumulated)
		cashflow_reset_offsets.append([len(r), len(all_resets), 1])
		#store resets    
		all_resets.extend (r)
		
	cashflows = CudaCashFlows(cashflow_schedule, cashflow_reset_offsets)
	cashflows.SetResets(all_resets, reset_scenario_offsets)
	
	return cashflows

def GenerateFixedCashflows ( reference_date, reset_dates, nominal, amort, daycount_code, fixed_rate ):
	'''
	Generates a vector of Start_day, End_day, Pay_day, Year_Frac, Nominal, FixedAmount (=0)
	and the fixed rate from the parameters provided. Note that the length of the nominal array must
	be 1 less than the reset_dates (Since there is no nominal on the first reset date i.e.
	Effective date).
	The nominal could also be just a single number representing a vanilla (constant) profile
	
	Returns a vector of days (and nominals) relative to the reference date
	'''
	cashflow_schedule   = GetCashflows ( reference_date, reset_dates, nominal, amort, daycount_code, fixed_rate )
	#Add the null resets to the end
	dummy_resets        = np.zeros((len(cashflow_schedule),3))
	
	return CudaCashFlows(cashflow_schedule, dummy_resets) 
	
def MakeFixedCashflows(reference_date, position, cashflows):
	'''
	Generates a vector of fixed cashflows from an adaptiv analytic data source taking nominal amounts into account.
	'''
	cash = []
	for cashflow in cashflows['Items']:		
		rate = cashflow['Rate'] if isinstance(cashflow['Rate'], float) else cashflow['Rate'].amount
		if cashflow['Payment_Date']>=reference_date:
			Accrual_Start_Date = cashflow['Accrual_Start_Date'] if cashflow['Accrual_Start_Date'] else cashflow['Payment_Date']
			Accrual_End_Date   = cashflow['Accrual_End_Date'] if cashflow['Accrual_End_Date'] else cashflow['Payment_Date']
			cash.append([ (Accrual_Start_Date - reference_date).days, (Accrual_End_Date - reference_date).days, (cashflow['Payment_Date'] - reference_date).days,
						  cashflow['Accrual_Year_Fraction'], position*cashflow['Notional'], position*cashflow.get('Fixed_Amount',0.0), rate, 0.0, 0.0 ] )
			
	#Add the null resets to the end
	dummy_resets        = np.zeros((len(cash),3))
	
	return CudaCashFlows( cash, dummy_resets )

def MakeSamplingData ( reference_date, time_grid, samples ):
	all_resets  			= []
	reset_scenario_offsets  = []
	D						= float(sum([x[2] for x in samples]))
	
	for sample in sorted(samples):
		Reset_Day			= (sample[0]-reference_date).days
		Start_Day		    = Reset_Day
		End_Day		        = Reset_Day
		Weight		        = sample[2]/D
		Time_Grid, Scenario = time_grid.GetScenarioOffset( Reset_Day )
		#only add a reset if its in the past    
		all_resets.append([Time_Grid,Reset_Day,Start_Day,End_Day,-1,Weight, sample[1] if sample[0]<reference_date else 0.0, 0.0])
		reset_scenario_offsets.append(Scenario)
		
	return CudaResets ( all_resets, reset_scenario_offsets )

def MakeSimpleFixedCashflows(reference_date, position, cashflows):
	'''
	Generates a vector of fixed cashflows from an adaptiv analytic data source only looking at the actual fixed value.
	'''
	cash = []
	for cashflow in cashflows['Items']:
		if cashflow['Payment_Date']>=reference_date:
			cash.append([ (cashflow['Payment_Date'] - reference_date).days, (cashflow['Payment_Date'] - reference_date).days, (cashflow['Payment_Date'] - reference_date).days,
						  1.0, 0.0, position*cashflow['Fixed_Amount'], 0.0, 0.0, 0.0 ] )
			
	#Add the null resets to the end
	dummy_resets        = np.zeros((len(cash),3))
	
	return CudaCashFlows( cash, dummy_resets)

def MakeEnergyFixedCashflows(reference_date, position, cashflows):
	'''
	Generates a vector of fixed cashflows from an adaptiv analytic data source only looking at the actual fixed value.
	'''
	cash = []
	for cashflow in cashflows['Items']:
		if cashflow['Payment_Date']>=reference_date:
			cash.append([ (cashflow['Payment_Date'] - reference_date).days, (cashflow['Payment_Date'] - reference_date).days, (cashflow['Payment_Date'] - reference_date).days,
						  1.0, 0.0, position*cashflow['Volume']*cashflow['Fixed_Price'], 0.0, 0.0, 0.0 ] )
			
	#Add the null resets to the end
	dummy_resets        = np.zeros((len(cash),3))
	
	return CudaCashFlows( cash, dummy_resets )

def MakeEquitySwapletCashflows ( base_date, time_grid, position, cashflows):
	'''
	Generates a vector of equity cashflows from an adaptiv analytic data source.
	'''
	cash 					= []
	all_resets  			= []
	cashflow_reset_offsets  = []
	reset_scenario_offsets  = []
	
	for cashflow in cashflows['Items']:
		if cashflow['Payment_Date']>=base_date:			
			cash.append([ (cashflow['Start_Date'] - base_date).days, (cashflow['End_Date'] - base_date).days, (cashflow['Payment_Date'] - base_date).days,
						  cashflow['Start_Multiplier'], cashflow['End_Multiplier'], position*cashflow['Amount'], cashflow['Dividend_Multiplier'], 0.0, 0.0 ] )
			
			r           = []
			for reset in ['Start','End']:
				Reset_Day		    = (cashflow[reset+'_Date'] - base_date).days
				Start_Day		    = Reset_Day
				#we map the weight of the reset with the prior dividends
				Weight		        = cashflow['Known_Dividend_Sum']
				
				#Need to use this reset to estimate future dividends
				Time_Grid, Scenario = time_grid.GetScenarioOffset( max(Reset_Day,0) )
				
				#only add a reset if its in the past    
				r.append([Time_Grid, Reset_Day, Start_Day, 0, -1, Weight,
						  cashflow['Known_'+reset+'_Price'] if Start_Day<=0 else 0.0,
						  cashflow['Known_'+reset+'FX_Rate'] if Start_Day<=0 else 0.0])
				reset_scenario_offsets.append(Scenario)
			
			#attach the reset_offsets to the cashflow
			cashflow_reset_offsets.append([len(r), len(all_resets), 0])
			#store resets    
			all_resets.extend (r)
		
	cashflows = CudaCashFlows(cash, cashflow_reset_offsets)
	cashflows.SetResets(all_resets, reset_scenario_offsets)
	
	return cashflows

def MakeIndexCashflows(base_date, time_grid, position, cashflows, price_index, index_rate, settlement_date, isBond=True):
	'''
	Generates a vector of index-linked cashflows from an adaptiv analytic data source given the price_index and index_rate price factors.
	'''
	
	def IndexReference2M ( pricing_date, lagged_date, resets, offsets ):
		Fixing_Day 		= (pricing_date - pd.DateOffset(months=2)).to_period('M').to_timestamp('D')
		Rel_Day 		= ( Fixing_Day - lagged_date ).days
		Value			= index_rate.GetReferenceVal(Fixing_Day) if Fixing_Day<=lagged_date else 0.0
		
		Time_Grid, Scenario = time_grid.GetScenarioOffset( Rel_Day ) if Rel_Day>=0.0 else ( 0, 0 )
		resets.append ( [ Time_Grid, Rel_Day, Rel_Day, Rel_Day, -1, 1.0, Value, 0.0 ] )
		offsets.append(Scenario)
		
	def IndexReference3M ( pricing_date, lagged_date, resets, offsets ):
		Fixing_Day 		= (pricing_date - pd.DateOffset(months=3)).to_period('M').to_timestamp('D')
		Rel_Day 		= ( Fixing_Day - lagged_date ).days
		Value			= index_rate.GetReferenceVal(Fixing_Day) if Fixing_Day<=lagged_date else 0.0
		
		Time_Grid, Scenario = time_grid.GetScenarioOffset( Rel_Day ) if Rel_Day>=0.0 else ( 0, 0 )
		resets.append ( [ Time_Grid, Rel_Day, Rel_Day, Rel_Day, -1, 1.0, Value, 0.0 ] )
		offsets.append(Scenario)
		
	def IndexReferenceInterpolated3M ( pricing_date, lagged_date, resets, offsets ):
		T1 				= pricing_date.to_period('M').to_timestamp('D')
		Sample_Day_1	= (pricing_date - pd.DateOffset(months=3)).to_period('M').to_timestamp('D')
		Sample_Day_2	= (pricing_date - pd.DateOffset(months=2)).to_period('M').to_timestamp('D')
		w 				= ( pricing_date - T1 ).days / float( ( ( T1+pd.DateOffset(months=1) ) - T1 ).days )
		Weights		    = [( Sample_Day_1, (1.0-w) ), (Sample_Day_2, w)]

		for Day, Weight in Weights:
			Rel_Day 			= ( Day - lagged_date ).days
			Value				= index_rate.GetReferenceVal(Day) if Day<=lagged_date else 0.0
			Time_Grid, Scenario = time_grid.GetScenarioOffset( Rel_Day ) if Rel_Day>=0.0 else ( 0, 0 )
			
			resets.append ( [ Time_Grid, Rel_Day, Rel_Day, Rel_Day, -1, Weight, Value, 0.0 ] )
			offsets.append(Scenario)
		
	def IndexReferenceInterpolated4M ( pricing_date, lagged_date, resets, offsets ):
		T1 				= pricing_date.to_period('M').to_timestamp('D')
		Sample_Day_1	= (pricing_date - pd.DateOffset(months=4)).to_period('M').to_timestamp('D')
		Sample_Day_2	= (pricing_date - pd.DateOffset(months=3)).to_period('M').to_timestamp('D')
		w 				= ( pricing_date - T1 ).days / float( ( ( T1+pd.DateOffset(months=1) ) - T1 ).days )
		Weights		    = [( Sample_Day_1, (1.0-w) ), (Sample_Day_2, w)]

		for Day, Weight in Weights:
			Rel_Day 			= ( Day - lagged_date).days
			Value				= index_rate.GetReferenceVal(Day) if Day<=lagged_date else 0.0
			Time_Grid, Scenario = time_grid.GetScenarioOffset( Rel_Day ) if Rel_Day>=0.0 else ( 0, 0 )
			
			resets.append ( [ Time_Grid, Rel_Day, Rel_Day, Rel_Day, -1, Weight, Value, 0.0 ] )
			offsets.append(Scenario)
		
	cash					= []
	cashflow_reset_offsets	= []
	#resets at different points in time
	time_resets				= []
	time_scenario_offsets 	= []
	#resets per cashflow
	base_resets           	= []
	base_scenario_offsets 	= []
	final_resets           	= []
	final_scenario_offsets 	= []
	
	for cashflow in cashflows['Items']:
		if cashflow['Payment_Date'] >= base_date and ((cashflow['Payment_Date'] >= settlement_date) if settlement_date else True):
			Pay_Date		 	= (cashflow['Payment_Date'] - base_date).days
			Accrual_Start_Date	= (cashflow['Accrual_Start_Date'] - base_date).days if cashflow['Accrual_Start_Date'] else Pay_Date
			Accrual_End_Date	= (cashflow['Accrual_End_Date'] - base_date).days if cashflow['Accrual_End_Date'] else Pay_Date	
			
			cash.append ( [ Accrual_Start_Date, Accrual_End_Date, Pay_Date, cashflow['Accrual_Year_Fraction'],
							position*cashflow['Notional'], cashflow['Rate_Multiplier'], cashflow['Yield'].amount, 0.0, 0.0 ] )
			
			#attach the base and final reference dates to the cashflow
			cashflow_reset_offsets.append([cashflow['Base_Reference_Value'] if cashflow['Base_Reference_Value'] else -(cashflow['Base_Reference_Date'] - base_date).days,
										   cashflow['Final_Reference_Value'] if cashflow['Final_Reference_Value'] else -(cashflow['Final_Reference_Date'] - base_date).days,
										   Pay_Date if settlement_date is None else -(settlement_date - base_date).days ])
			
			if isBond:
				locals()[ price_index.param['Reference_Name'] ] ( cashflow['Base_Reference_Date'] if cashflow['Base_Reference_Date'] else base_date, base_date, base_resets, base_scenario_offsets )
				locals()[ price_index.param['Reference_Name'] ] ( cashflow['Final_Reference_Date'] if cashflow['Final_Reference_Date'] else base_date, base_date, final_resets, final_scenario_offsets )

	#set the cashflows
	cashflows = CudaCashFlows ( sorted ( cash ), cashflow_reset_offsets )

	if isBond:
		mtm_grid = time_grid.time_grid[:,TIME_GRID_MTM]
		
		for last_published_date in index_rate.GetLastPublicationDates ( base_date, mtm_grid ):
			#calc the number of days since last published date to the base date
			Rel_Day 					= (last_published_date - base_date).days
			Value						= index_rate.GetReferenceVal(last_published_date) if last_published_date<=index_rate.param['Last_Period_Start'] else 0.0
			
			time_resets.append ( [ 0.0, Rel_Day, Rel_Day, Rel_Day, -1, 1.0, Value, 0.0 ] )
			time_scenario_offsets.append(0)

		cashflows.SetResets( time_resets, time_scenario_offsets )
		
		return cashflows, CudaResets ( base_resets, base_scenario_offsets ), CudaResets ( final_resets, final_scenario_offsets )

	else:
		for eval_time in time_grid.time_grid[:,TIME_GRID_MTM]:
			actual_time     = base_date+pd.DateOffset(days=eval_time)
			
			locals()[ price_index.param['Reference_Name'] ] ( actual_time, index_rate.param['Last_Period_Start'], time_resets, time_scenario_offsets )

		cashflows.SetResets( time_resets, time_scenario_offsets )

		return cashflows
	
def MakeFloatCashflows(reference_date, time_grid, position, cashflows):
	'''
	Generates a vector of floating cashflows from an adaptiv analytic data source.
	'''
	cash 					= []
	all_resets  			= []
	cashflow_reset_offsets  = []
	reset_scenario_offsets  = []
	
	for cashflow in cashflows['Items']:
		if cashflow['Payment_Date']>=reference_date:
			#potential FX resets
			fx_reset_date = ( cashflow.get('FX_Reset_Date') - reference_date).days if cashflow.get('FX_Reset_Date') else 0.0
			fx_reset_val  = cashflow.get('Known_FX_Rate', 0.0)
			
			cash.append([ (cashflow['Accrual_Start_Date'] - reference_date).days, (cashflow['Accrual_End_Date'] - reference_date).days, (cashflow['Payment_Date'] - reference_date).days,
						  cashflow['Accrual_Year_Fraction'], position*cashflow['Notional'], position*cashflow.get('Fixed_Amount',0.0), cashflow['Margin'].amount, fx_reset_date, fx_reset_val ] )
			
			r           = []
			for reset in cashflow['Resets']:
				#check if the reset end day is valid
				Actual_End_Day      = reset[1]+cashflow['Rate_Tenor'] if reset[2]==reset[1] else reset[2]
				
				#create the reset vector
				Reset_Day			= (reset[0]-reference_date).days
				Start_Day		    = (reset[1]-reference_date).days
				End_Day		        = (Actual_End_Day-reference_date).days
				Accrual             = reset[3]
				Weight		        = 1.0/len(cashflow['Resets'])
				Time_Grid, Scenario = time_grid.GetScenarioOffset( Reset_Day )
				#only add a reset if its in the past    
				r.append([Time_Grid,Reset_Day,Start_Day,End_Day,-1,Weight, reset[5].amount if reset[0]<reference_date else 0.0, Accrual])
				reset_scenario_offsets.append(Scenario)

			#attach the reset_offsets to the cashflow 
			cashflow_reset_offsets.append([len(r), len(all_resets), 0])
			#store resets    
			all_resets.extend (r)
			
	cashflows = CudaCashFlows(cash, cashflow_reset_offsets)
	cashflows.SetResets(all_resets, reset_scenario_offsets)
	
	return cashflows

def MakeEnergyCashflows(reference_date, time_grid, position, cashflows, reference, forwardsample, fxsample, calendars):
	'''
	Generates a vector of floating/fixed cashflows from an adaptiv analytic data source
	using the energy model. NOTE - Need to allow for fxSample different from the forwardsample - TODO!
	'''
	cash 					= []
	all_resets  			= []
	cashflow_reset_offsets  = []
	reset_scenario_offsets  = []
	forward_calendar_bday   = calendars.get(forwardsample.GetHolidayCalendar(),{'businessday':'B'})['businessday']
	
	for cashflow in cashflows['Items']:
		if cashflow['Payment_Date']>=reference_date:
			cash.append([ (cashflow['Period_Start'] - reference_date).days, (cashflow['Period_End'] - reference_date).days,
						  (cashflow['Payment_Date'] - reference_date).days, cashflow.get('Price_Multiplier', 1.0),
						  position*cashflow['Volume'], 0.0, cashflow.get('Fixed_Basis', 0.0), 0.0, 0.0 ] )
			
			r           	= []
			bunsiness_dates = pd.date_range ( cashflow['Period_Start'], cashflow['Period_End'], freq=forward_calendar_bday )
			
			if forwardsample.GetSampling_Convention()=='ForwardPriceSampleDaily':
				#create daily samples
				reset_dates = bunsiness_dates 
					
			elif forwardsample.GetSampling_Convention()=='ForwardPriceSampleBullet':
				#create one sample
				reset_dates = [ bunsiness_dates[-1] ]
				
			resets_in_excel_format = [(x-reference.start_date).days for x in reset_dates]
			reference_date_excel   = (reference_date-reference.start_date).days
			
			#retrieve the fixing dates from the reference curve and adding an offset
			fixing_dates = reference.GetFixings().array[ np.searchsorted(reference.GetTenor(),resets_in_excel_format) + int(forwardsample.param.get('Offset')) ][:,1]
			
			for reset_day, fixing_day in zip(resets_in_excel_format, fixing_dates):
				Reset_Day			= reset_day
				Start_Day		    = reset_day
				End_Day		        = fixing_day
				Weight		        = 1.0/len(reset_dates)
				Time_Grid, Scenario = time_grid.GetScenarioOffset( Start_Day - reference_date_excel )
				#only add a reset if its in the past    
				r.append([Time_Grid,Reset_Day,Start_Day,End_Day,-1,Weight, cashflow['Realized_Average'], cashflow['FX_Realized_Average'] ])
				reset_scenario_offsets.append(Scenario)

			#attach the reset_offsets to the cashflow
			cashflow_reset_offsets.append([len(r), len(all_resets), 0])
			#store resets    
			all_resets.extend (r)
				
	cashflows = CudaCashFlows(cash, cashflow_reset_offsets)
	cashflows.SetResets(all_resets, reset_scenario_offsets)
	
	return cashflows

if __name__=='__main__':
	import pandas as pd
	import cPickle
	
	#amort = cPickle.load(file('datelist.obj','rb'))
	#rates = cPickle.load(file('knownrates.obj','rb'))
	#GetCashflows ( pd.datetime(2014,6,25), pd.bdate_range(pd.datetime(2013,6,10), pd.datetime(2015,6,10), freq=pd.DateOffset(months=1) ), -139040206.4, [], [], DAYCOUNT_ACT365, 0.0628 )
	#reference_date, time_grid, reset_dates, nominal, amort, known_rates, reset_tenor, reset_frequency, daycount_code, spread = cPickle.load( file('testprime.obj','rb') )
	reference_date, time_grid, reset_dates, nominal, amort, known_rates, reset_tenor, reset_frequency, daycount_code, spread = cPickle.load( file('normalswap.obj','rb') )
	flows, resets = GenerateFloatCashflowsAndResets ( reference_date, time_grid, reset_dates, nominal, amort, known_rates, reset_tenor, reset_frequency, daycount_code, spread )