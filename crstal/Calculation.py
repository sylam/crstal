#import standard libraries
import json
import time
import string
import logging
import operator
import itertools
import pandas as pd
import numpy as np

#import the necessary pycuda libs
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import pycuda.curandom

from pycuda import gpuarray
from pycuda.compiler import SourceModule

#load up some useful datatypes
from collections import OrderedDict, namedtuple
#import the market and trade data
from Config import Parser
#import the riskfactors
from RiskFactors import ConstructFactor
#import the stochastic processes
from StochasticProcess import ConstructProcess, GetAllProcessCudaFunctions
#import the currency/curve lookup factors 
from Instruments import getFXRateFactor, getFXZeroRateFactor, GetAllInstrumentCudaFunctions
#import field definitions for useful groupings
from Fields import FieldMappings
#import the Utils
import Utils

#useful bit of code if you want to print data inside a cuda code block - must be in a global function
#Note if you see the word FUCK anywhere in the comments, its a technical phrase meaning "For Use Case Keynote"

##if (CashflowScenario==1 && threadIdx.x==0 && i==0 && blockIdx.x==0)
##{
##	for (int ci=float_starttime_index[mtm_time_index]; ci<num_float_cashflow; ci++)
##	{
##		const REAL* cashflow = float_cashflow+ci*CASHFLOW_INDEX_Size;				
##	
##		if ( cashflow[CASHFLOW_INDEX_Start_Day]>t[TIME_GRID_MTM] )
##		{
##			REAL forward_rate = CalcSimpleForwardCurveIndex ( t, cashflow[CASHFLOW_INDEX_Start_Day], cashflow[CASHFLOW_INDEX_End_Day], forward, scenario_prior_index, static_factors, stoch_factors );
##			REAL expiry 		 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Start_Day] - t[TIME_GRID_MTM], discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
##			REAL vol 		 = ScenarioSurface3D  ( t, forward_rate - cashflow[CASHFLOW_INDEX_Strike], expiry, cashflow[CASHFLOW_INDEX_Year_Frac], interest_rate_vol, scenario_prior_index, static_factors, stoch_factors );
##			printf("cashflow,%d,start_day,%g,forward,%.4f,moneyness,%.5f,expiry,%.4f,tenor,%.4f,vol,%.4f\\n",ci,cashflow[CASHFLOW_INDEX_Start_Day],forward_rate,forward_rate - cashflow[CASHFLOW_INDEX_Strike],expiry,cashflow[CASHFLOW_INDEX_Year_Frac],vol);
##		}
##	}
##}

class InstrumentExpired(Exception):
	def __init__(self, message):
		self.message = message

class Aggregation(object):
	def __init__(self, name):
		self.name = name
	
	def Aggregate(self, CudaMem, parent_partition, index, partition):
		parent_partition.DealMTMs 	[ index ] += partition.DealMTMs  [ index ].clip(0.0, np.inf)
		parent_partition.Cashflows  [ index ] += partition.Cashflows [ index ].clip(0.0, np.inf)
		
class DealStructure(object):
	def __init__(self, obj, settlement_currencies, num_scenarios, num_timepoints, precision, Deal_levelMTM=False):
		#parent object - note that all parent objects MUST have an Aggregate method that can aggregate partitons
		self.obj 				= Utils.DealDataType(Instrument=obj, Factor_dep=None, Time_dep=None, Calc_res=None)
		#gather a list of deal dependencies
		self.dependencies 		= []
		#maintain a list of container objects
		self.sub_structures		= []
		#place to store the results
		self.partition			= Utils.Partition( DealMTMs 		= np.zeros( ( num_scenarios, num_timepoints ), dtype=precision ),
												   Collateral_Cash 	= np.zeros( ( num_scenarios, num_timepoints ), dtype=precision ),
												   Funding_Cost		= np.zeros( ( num_scenarios, num_timepoints ), dtype=precision ),
												   Cashflows		= np.zeros( ( num_scenarios, sum([x.max()+1 for x in settlement_currencies.values()]) ), dtype=precision ) )
		#do we need to create sub_partitions?
		self.sub_partitions		= OrderedDict()
		#Do we want to store each deal level MTM explicitly?
		self.Deal_levelMTM		= Deal_levelMTM

	def ResetStructure(self):
		self.partition.DealMTMs.fill(0.0)
		self.partition.Collateral_Cash.fill(0.0)
		self.partition.Funding_Cost.fill(0.0)
		self.partition.Cashflows.fill(0.0)
		
		for sub_partition in self.sub_partitions.values():
			sub_partition.DealMTMs.fill(0.0)
			sub_partition.Collateral_Cash.fill(0.0)
			sub_partition.Funding_Cost.fill(0.0)
			sub_partition.Cashflows.fill(0.0)
				
		for sub_structure in self.sub_structures:
			sub_structure.ResetStructure()
		
	def CalcTimeDependency(self, base_date, deal, time_grid):
		#calculate the additional (dynamic) dates that this instrument needs to be evaluated at
		deal_time_dep = None
		try:
			reset_dates = deal.get_reval_dates()
			if len(time_grid.scenario_dates)==1:
				if len(reset_dates)>0 and max(reset_dates)<base_date:
					raise InstrumentExpired(deal.field.get('Reference','Unknown Instrument Reference'))
				deal_time_dep = time_grid.CalcDealGrid(set([base_date]))
			else:
				deal_time_dep = time_grid.CalcDealGrid(reset_dates)
		except InstrumentExpired, e:
			logging.warning('skipping expired deal {0}'.format(e.message))
			
		return deal_time_dep
	
	def AddDealToStructure(self, base_date, deal, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		'''
		The logic is as follows: a structure contains deals - a set of deals are netted off and then the rules that the
		structure itself contains is applied.
		'''
		deal_time_dep = self.CalcTimeDependency(base_date, deal, time_grid)
		if deal_time_dep is not None:
			#calculate dependencies based on field names
			try:
				self.dependencies.append ( Utils.DealDataType  ( Instrument = deal,
																 Factor_dep = deal.calc_dependancies(base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars),
																 Time_dep   = deal_time_dep,
																 Calc_res   = [] if self.Deal_levelMTM else None ) )
			except Exception, e:
				logging.error('{0}.{1} {2} - Skipped'.format(deal.field['Object'], deal.field['Reference'], e.message))
	
	def AddStructureToStructure ( self, struct, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars ):
		#get the dependencies
		struct_time_dep = self.CalcTimeDependency(base_date, struct.obj.Instrument, time_grid)
		if struct_time_dep is not None:
			struct.obj = Utils.DealDataType  (  Instrument = struct.obj.Instrument,
												Factor_dep = struct.obj.Instrument.calc_dependancies(base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars),
												Time_dep   = struct_time_dep,
												Calc_res   = [] if self.Deal_levelMTM else None )
		#Structure object representing a netted set of cashflows
		self.sub_structures.append( struct )

	def BuildPartitions(self):
		#sort the data by the tags
		self.dependencies.sort(key=lambda x:x.Instrument.field['Tags'])
		#get the distinct tags
		unique_tags=set([x.Instrument.field['Tags'] for x in self.dependencies])
		if len(unique_tags)>1:
			for tag in sorted(unique_tags):
				self.sub_partitions[tag] = Utils.Partition( DealMTMs 		= np.zeros_like( self.partition.DealMTMs ),
															Collateral_Cash = np.zeros_like( self.partition.Collateral_Cash ),
															Funding_Cost	= np.zeros_like( self.partition.Funding_Cost ),
															Cashflows		= np.zeros_like( self.partition.Cashflows ) )
	
	def ResolveStructure(self, module, CudaMem, precision, revals_per_batch, prev_block_num, block_num ):
		'''
		Resolves the Structure
		'''
		def UpdatePartition (CudaMem, sub_partitions, current_tag, previous_tag, mtm_offset):
			drv.memcpy_dtoh ( sub_partitions[current_tag].DealMTMs[mtm_offset], CudaMem.d_MTM_Accum_Buffer.ptr )
			if CudaMem.d_Cashflows:
				drv.memcpy_dtoh ( sub_partitions[current_tag].Cashflows[mtm_offset], CudaMem.d_Cashflows.ptr  )
			if previous_tag:
				#need to subtract it from the previous partition
				sub_partitions[current_tag].DealMTMs[mtm_offset]  -= sub_partitions[previous_tag].DealMTMs[mtm_offset]
				if CudaMem.d_Cashflows:
					sub_partitions[current_tag].Cashflows[mtm_offset] -= sub_partitions[previous_tag].Cashflows[mtm_offset]
						
		#work out the batch size 
		batch_size = block_num - prev_block_num
		#work out where to place the calcs in the deal array
		mtm_offset = slice(revals_per_batch*prev_block_num, revals_per_batch*block_num )
		#use a local variable for the accumulation buffer
		accum = CudaMem.d_MTM_Accum_Buffer
		
		if self.sub_structures:
			#clear out the accumulation and cashflow buffers
			accum.fill(0.0)
			if CudaMem.d_Cashflows:
				CudaMem.d_Cashflows.fill(0.0)
			#process sub structures	
			for structure in self.sub_structures:
				structure.ResolveStructure ( module, CudaMem, precision, revals_per_batch, prev_block_num, block_num )
				if hasattr(self.obj.Instrument, 'Aggregate'):
					self.obj.Instrument.Aggregate ( CudaMem, self.partition, mtm_offset, structure.partition )
					
		#currenly, only netting sets accumulate dependencies (which means it used the d_MTM_Accum_Buffer buffer)			
		if self.dependencies and self.obj.Instrument.accum_dependencies:
			#check if partitioning is required
			partition_names = self.sub_partitions.keys()
			current_tag 	= partition_names[0] if partition_names else None
			previous_tag    = None
			
			#accumulate the mtm's
			for deal_data in self.dependencies:								
				#update partitions ( if necessary )
				if current_tag and deal_data.Instrument.field['Tags']!=current_tag:
					UpdatePartition(CudaMem, self.sub_partitions, current_tag, previous_tag, mtm_offset)
					#update the tags	
					previous_tag = current_tag
					current_tag = deal_data.Instrument.field['Tags']

				#d_MTM_Buffer is used for pricing, then it's interpolated to the full scenario grid and then added to the accumulation buffer
				deal_data.Instrument.Calculate ( module, CudaMem, precision, batch_size, revals_per_batch, deal_data )
				
				#m=CudaMem.d_MTM_Buffer.get()
				#if (m[:revals_per_batch*batch_size,-1]!=0).any():
				#	print 'Fuck - nonzero mtm in last position'
				#if m[m!=m].any():
				#	print deal_data.Instrument.field['Reference'], 'NANS!!!!!!! - FUCK'
					
				#now add the MTM to the Accumulation buffer
				accum += CudaMem.d_MTM_Buffer
				
			#update the final partition (if necessary)	
			if current_tag:	
				UpdatePartition(CudaMem, self.sub_partitions, current_tag, previous_tag, mtm_offset)

		#postprocessing code for working out the mtm of all deals, collateralizaton etc..
		if hasattr(self.obj.Instrument, 'PostProcess'):
			#the actual answer for this netting set
			self.obj.Instrument.PostProcess ( module, CudaMem, precision, batch_size, revals_per_batch, self.obj, self.dependencies, self.partition, mtm_offset )
			#now work out the answer for each partition
			for parition_name, partition in self.sub_partitions.items():
				drv.memcpy_htod ( CudaMem.d_MTM_Accum_Buffer.ptr, partition.DealMTMs [mtm_offset] )
				if CudaMem.d_Cashflows:
					drv.memcpy_htod ( CudaMem.d_Cashflows.ptr, partition.Cashflows [mtm_offset] )
				#apply the postprocssing to this netting set
				self.obj.Instrument.PostProcess ( module, CudaMem, precision, batch_size, revals_per_batch, self.obj, self.dependencies, partition, mtm_offset )

class DealTimeDependencies(object):
	def __init__(self, deal_time_grid):
		self.deal_time_grid = deal_time_grid.astype(np.int32)
		to_interpolate		= np.where(np.diff(self.deal_time_grid)>1)[0]
		self.interp_index	= np.vstack((self.deal_time_grid[to_interpolate],self.deal_time_grid[to_interpolate+1]-self.deal_time_grid[to_interpolate])).T.astype(np.int32)
		
class TimeGrid(object):
	def __init__( self, scenario_dates, MTM_dates, base_MTM_dates ):
		self.scenario_dates = scenario_dates
		self.base_MTM_dates = base_MTM_dates
		self.CurrencyMap    = {}
		self.SetMtMDates(MTM_dates)

	def SetMtMDates(self, MTM_dates):
		self.mtm_dates	    = MTM_dates
		self.date_lookup    = dict( [(x,i) for i,x in enumerate(sorted(MTM_dates))] )
	
	def SetBaseDate(self, base_date, precision, indexoffset):
		#leave the grids in terms of the number of days
		self.scen_time_grid = np.array([(x-base_date).days for x in sorted(self.scenario_dates)])
		self.mtm_time_grid  = np.array([(x-base_date).days for x in sorted(self.mtm_dates)])
				
		#calculate the MTM/Scenario grid differential - used later for instruments to price off of
		time_grid,time_grid_index  = [], []
		for i,t in enumerate(self.mtm_time_grid):
			prev_scen_index 	= self.scen_time_grid[self.scen_time_grid<=t].size-1
			time_grid_index.append( prev_scen_index )
			#order here must match the cuda code - so, prior, then mtm_time then scenario_delta time
			scenario_grid_delta = precision ( ( self.scen_time_grid[prev_scen_index+1] - self.scen_time_grid[prev_scen_index] ) if self.scen_time_grid.size>1 else 0.0 )
			time_grid.append ( [ (t-self.scen_time_grid[prev_scen_index])/scenario_grid_delta, t, 0.0 ] )
			
		self.base_time_grid 	= set([self.date_lookup[x] for x in self.base_MTM_dates])
		self.time_grid	   		= np.array(time_grid, dtype=precision)
		self.time_grid_index	= np.array(time_grid_index, dtype=indexoffset)
		#now put the time_grid_index inside the timegrid
		self.time_grid[:,Utils.TIME_GRID_ScenarioPriorIndex] = self.time_grid_index.view(precision)

	def GetScenarioOffset(self, days_from_base):
		prev_scen_index 	= self.scen_time_grid[self.scen_time_grid<=days_from_base].size-1
		scenario_grid_delta = np.float64 ( (self.scen_time_grid[prev_scen_index+1] - self.scen_time_grid[prev_scen_index]) if (self.scen_time_grid.size>1 and self.scen_time_grid.size>prev_scen_index+1) else 0.0 )
		return (days_from_base-self.scen_time_grid[prev_scen_index])/scenario_grid_delta, prev_scen_index

	def SetCurrencySettlement(self, currencies):
		self.CurrencyMap = {}
		for currency, dates in currencies.items():
			settlement_dates = sorted ( [ self.date_lookup[x] for x in dates if x in self.date_lookup ] )
			if settlement_dates:
				currency_lookup  = np.zeros(self.mtm_time_grid.size, dtype=np.int32)-1
				currency_lookup [ settlement_dates ] = np.arange( len(settlement_dates) )
				self.CurrencyMap.setdefault(currency, currency_lookup )

	def GetCurrencySettlementMap(self, currency_map):
		currencies 	 = OrderedDict()
		currency_ofs = []
		currency_len = []
		currency_all = []
		for index, currency in sorted({v:k for k,v in currency_map.items()}.items()):
			currencies[currency] = self.CurrencyMap[currency]
			currency_ofs.append(index)
			currency_len.append(self.CurrencyMap[currency].max()+1)
			currency_all.append(self.CurrencyMap[currency])
			
		#update the order in the CurrencyMap 
		self.CurrencyMap = currencies
		
		return currency_ofs, currency_len, np.concatenate(currency_all)
		
	def CalcDealGrid(self, dates):
		try:
			dynamic_dates = self.base_time_grid.union([self.date_lookup[x] for x in dates])
		except KeyError, e:
			#if there is at least one reset date in the set of dates, then return it, else the deal has expired
			r = [ self.date_lookup[x] for x in dates if x in self.date_lookup ]
			if r:
				dynamic_dates = self.base_time_grid.union(r)
			else:
				if max(dates)<min(self.date_lookup.keys()):
					raise InstrumentExpired(e)
				
				#include this instrument but don't bother pricing it through time
				return DealTimeDependencies( np.array ( [0] ) )
			
		#add the last mtm date to the list so that the instrument prices right at the end (should force a zero right at the end)
		dynamic_dates.add(self.mtm_time_grid.size-1)
		#now construct the deal grid
		deal_time_grid = np.array ( list ( dynamic_dates ) )
		#sort it
		deal_time_grid.sort()
		#calculate the interpolation points etc.
		return DealTimeDependencies( deal_time_grid )
	
class Calculation(object):
	cudacodeheader = '''
	#include <stdio.h>

	typedef $PRECISION REAL;
	typedef $INDEXTYPE OFFSET;
	
	//Function pointers for caps/floors/floating legs
	typedef REAL (*FloatPV)( 	const REAL* __restrict__ t,
								int   cashflow_method,
								int   mtm_time_index,
								int   num_cashflows,
								REAL& settlement_cashflow,
								const int*  __restrict__ cashflow_starttime_index,
								const REAL* __restrict__ cashflows,
								const REAL* __restrict__ resets,
								const int* __restrict__ interest_rate_vol_offset,
								const int* __restrict__ forward_offset,
								const int* __restrict__ discount_offset,
								const OFFSET scenario_prior_index,
								const REAL* __restrict__ static_factors,
								const REAL* __restrict__ stoch_factors );

	//Function pointers for how to interpolate between scenarios
	typedef REAL (*ScenarioFn) ( const REAL* __restrict__ t,
								 const OFFSET scenario_prior_index,
								 const REAL* __restrict__ stoch_factors,
								 const int offset );
								 
	//Function pointers for how to interpolate curves (e.g. dividends vs interest rates)
	typedef REAL (*CurveFn) ( 	const REAL* curve_tenor,
								int prev_scenario_point,
								int next_scenario_point,
								REAL tenor_point );

	//Factor offset type
	const int FACTOR_TYPE_STATIC			= 1;
	const int FACTOR_TYPE_STOCHASTIC		= 2;
	
	//Daycount conventions - must match the python code
	const int DAYCOUNT_ACT365				= 0;
	const int DAYCOUNT_ACT360				= 1;
	const int DAYCOUNT_ACT365IDSA			= 2;
	const int DAYCOUNT_ACT30_360			= 3;
	const int DAYCOUNT_ACT30_E360			= 4;
	const int DAYCOUNT_ACTACTICMA			= 5;

	//Used to unpack the factor indexes in the scenario routines 
	const int NUM_FACTOR_INDEX				= 0;
	const int FACTOR_INDEX_START			= 1;
	
	//The size of data stored for each risk factor that is a curve (tenor+rate)
	const int CURVE_INDEX_Size				= 4;
	//The size of data stored for each risk factor that is a 2D Vol Surface (tenors+rate)
	const int VOL2D_INDEX_Size				= 4;
	//The size of data stored for each risk factor that is a 3D Vol Surface (tenors+underlying_tenor+rate)
	const int VOL3D_INDEX_Size				= 5;
	//The size of data stored for each risk factor that is a rate (just rate)
	const int RATE_INDEX_Size				= 2;
	
	//Index to determine if the factor is static or stochastic
	const int FACTOR_INDEX_Type				= 0;
	//Index into either the stochasic or static lookup tables
	const int FACTOR_INDEX_Offset			= 1;
	//Index in to the ScenarioTenor(Size/Offset) constant arrays to determine the tenor points - only applicable to CURVE_INDICES
	const int FACTOR_INDEX_Tenor_Index		= 2;
	//Index to determine the daycount convention - only applicale to CURVE_INDICES
	const int FACTOR_INDEX_Daycount			= 3;
	//Index to determine the calculation date in excel format (some curves are expressed in excel days)
	const int FACTOR_INDEX_ExcelCalcDate	= 3;
	//Index in to the ScenarioTenor(Size/Offset) constant arrays to determine the moneyness tenor points - only applicable to VOLSURFACE_INDICES
	const int FACTOR_INDEX_Moneyness_Index	= 2;
	//Index in to the ScenarioTenor(Size/Offset) constant arrays to determine the expiry tenor points - only applicable to VOLSURFACE_INDICES
	const int FACTOR_INDEX_Expiry_Index		= 3;
	//Index in to the ScenarioTenor(Size/Offset) constant arrays to determine the expiry tenor points - only applicable to VOLSURFACE_INDICES
	const int FACTOR_INDEX_VolTenor_Index	= 4;

	//The size of data stored for each point needed to perform MTM interpolation
	const int INTERP_GRID_Size 				= 2;
	//The size of data stored at each time grid point
	const int TIME_GRID_Size				= 3;
	
	//Index for time in years of the difference between the base_MTM and the closest prior scenario point	
	const int TIME_GRID_PriorScenarioDelta	= 0;
	//Index for time in days of the base_MTM 
	const int TIME_GRID_MTM					= 1;
	//Index for time in days of size of scenario step size
	const int TIME_GRID_ScenarioPriorIndex  = 2;

	//Cashflow indices - common to all cashflow types
	const int CASHFLOW_INDEX_Size			= 12;
	const int CASHFLOW_INDEX_Start_Day		= 0;
	const int CASHFLOW_INDEX_End_Day		= 1;
	const int CASHFLOW_INDEX_Pay_Day		= 2;
	
	const int CASHFLOW_INDEX_Year_Frac		= 3;
	//Cashflow indices specific to equity swaplet payments
	const int CASHFLOW_INDEX_Start_Mult		= 3;
	
	const int CASHFLOW_INDEX_Nominal		= 4;
	//Cashflow indices specific to equity swaplet payments
	const int CASHFLOW_INDEX_End_Mult		= 4;
	
	const int CASHFLOW_INDEX_FixedAmt		= 5;
	//Cashflow indices specific to floating payments
	const int CASHFLOW_INDEX_FloatMargin	= 6;
	//Cashflow indices specific to fixed payments
	const int CASHFLOW_INDEX_FixedRate		= 6;
	//Cashflow indices specific to caps/floors
	const int CASHFLOW_INDEX_Strike			= 6;
	//Cashflow indices specific to equity swaplet payments
	const int CASHFLOW_INDEX_Dividend_Mult	= 6;
	//Cashflow indices for possible FX resets
	const int CASHFLOW_INDEX_FXResetDate	= 7;
	const int CASHFLOW_INDEX_FXResetValue	= 8;
	
	//Cashflow index for number of resets - specific only to cashflows with attached resets
	const int CASHFLOW_INDEX_NumResets		= 9;
	//Cashflow index for number of resets - specific only to index-linked cashflows
	const int CASHFLOW_INDEX_BaseReference	= 9;
	//Cashflow index for number of resets - specific only to cashflows with attached resets
	const int CASHFLOW_INDEX_ResetOffset	= 10;
	//Cashflow index for number of resets - specific only to index-linked cashflows
	const int CASHFLOW_INDEX_FinalReference	= 10;
	//Cashflow index for settlement - i.e. does this cashflow settle (or is it just accumulated?)
	const int CASHFLOW_INDEX_Settle			= 11;

	//Cashflow calculation methods - useful for compounding/averaging/indexing etc.
	const int CASHFLOW_METHOD_IndexReferenceLastPublished	= 0;
	const int CASHFLOW_METHOD_IndexReference2M 				= 1;
	const int CASHFLOW_METHOD_IndexReference3M 				= 2;
	const int CASHFLOW_METHOD_IndexReferenceInterpolated3M 	= 3;
	const int CASHFLOW_METHOD_IndexReferenceInterpolated4M 	= 4;
	const int CASHFLOW_METHOD_Average_Interest 				= 0;
	
	const int CASHFLOW_METHOD_Compounding_Include_Margin 	= 2;
	const int CASHFLOW_METHOD_Compounding_Flat		 		= 3;
	const int CASHFLOW_METHOD_Compounding_Exclude_Margin 	= 4;
	const int CASHFLOW_METHOD_Compounding_None				= 5;
	const int CASHFLOW_METHOD_Fixed_Compounding_No			= 0;
	const int CASHFLOW_METHOD_Fixed_Compounding_Yes			= 1;

	const int CASHFLOW_METHOD_Fixed_Rate_Standard			= 0;
	const int CASHFLOW_METHOD_Ignore_Fixed_Rate 			= 1;

	//Reset indices 
	const int RESET_INDEX_Size			= 8;
	const int RESET_INDEX_Time_Grid		= 0;
	const int RESET_INDEX_Reset_Day		= 1;	
	const int RESET_INDEX_Start_Day		= 2;
	const int RESET_INDEX_End_Day		= 3;
	const int RESET_INDEX_Scenario		= 4;
	const int RESET_INDEX_Weight		= 5;
	const int RESET_INDEX_Value			= 6;
	//Either Accrual or FX Value is used (can't use both)
	const int RESET_INDEX_Accrual		= 7;
	const int RESET_INDEX_FXValue		= 7;

	//Cash settlement constants
	const int CASH_SETTLEMENT_Received_Only = 0;
	const int CASH_SETTLEMENT_Paid_Only 	= 1;
	const int CASH_SETTLEMENT_All 			= 2;

	//Payoff currency constants
	const int PAYOFF_CURRENCY_Currency	 	= 0;
	const int PAYOFF_CURRENCY_Underlying	= 1;

	//Constants for valuing options and barriers - Cuda has an issue with float constants - hence the #define
	#define BARRIER_UP						-1.0
	#define BARRIER_DOWN					1.0
	#define BARRIER_IN						-1.0
	#define BARRIER_OUT						1.0
	#define OPTION_PUT						-1.0
	#define OPTION_CALL						1.0
	const int PAYMENT_TOUCH					= 0;
	const int PAYMENT_EXPIRY				= 1;

	//Constants for system wide use
	const int MAX_CURRENCIES 			= 120;
	const int MAX_CURVE_FACTORS			= 500;
	
	/*
	Arbitrary constant buffer
		- used to store a cholesky matrix for correlation
		- used later to store instrument tenors, currency repo curves etc.
	Note that the memory for this pointer must be allocated separately.
	*/
	__device__ __constant__ REAL* Buffer;

	//Scenario specific constants for calculating the mtm's of instruments after the scenarios have been generated
	__device__ __constant__ int ScenarioTimeSteps;
	__device__ __constant__ int MTMTimeSteps;
	__device__ __constant__ int ScenarioFactorSize;
	__device__ __constant__ int NumSettlementCurrencies;
	__device__ __constant__ int NumCurveCurrencies;
	__device__ __constant__ int NumTenors;
	__device__ __constant__ int CurrencyCurveMap[MAX_CURRENCIES];
	__device__ __constant__ int CurrencyCurveOffset[MAX_CURRENCIES];
	__device__ __constant__ int CurrencySettlementMap[MAX_CURRENCIES];
	__device__ __constant__ int CurrencySettlementOffset[MAX_CURRENCIES];
	__device__ __constant__ int DIMENSION;
	__device__ __constant__ int ReportCurrency[FACTOR_INDEX_START+RATE_INDEX_Size];
	__device__ __constant__ int ScenarioTenorSize[MAX_CURVE_FACTORS];
	__device__ __constant__ int ScenarioTenorOffset[MAX_CURVE_FACTORS];

	//Perform a Binary Search - need to reduce the branching . . .
	//This part is out of the extern "C" because we need C++ and pycuda can only handle C linkage.
	template<typename T> 
	__device__ int binarySearch( T query, const T* __restrict__ xaxis, int size_axis )
	{
		int down = 0;
		int up   = size_axis-1;
		int mid;
		while (up-down>1)
		{
			mid = (up+down)/2;
			if (xaxis[mid] > query)
			{
				if ( xaxis[mid-1] < query ) return mid-1; else up=mid;
			}
			else
			{
				if ( xaxis[mid+1] > query ) return mid; else down=mid;
			}
		}
		return ( query < xaxis[size_axis-1] ) ? 0 : size_axis-1;
	}
	
	extern "C"
	{
	'''
	
	cudacodefooter = '''			
	}
	'''
	
	cudacodetemplate =	'''
		//need to flesh this out correctly - but hopefully most curves are just 360 or 365 (no funny 30/360 or 365ISDA etc.)
		__device__ REAL calcDayCountAccrual ( REAL time_in_days, int DaycountConvention )
		{
			return time_in_days / ( DaycountConvention == DAYCOUNT_ACT360 ? 360.0 : 365.0 );
		}		

		//The cumulative normal distribution function
		__device__ REAL CND( REAL x )
		{
			//Save the sign of x
			int sign = (x < 0) ? -1 : 1 ;
			x = fabsf(x)/sqrt(2.0);

			REAL t = 1.0/(1.0 + 0.3275911*x);
			REAL y = 1.0 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736 )*t + 0.254829592)*t*exp(-x*x);
			
			return 0.5*(1.0 + sign*y);
		}
		
		__device__ REAL blackEuropeanOption( REAL F, REAL X, REAL r, REAL vol, REAL tenor, REAL buyOrSell, REAL callOrPut)
		{
			REAL sigma = vol * sqrt(tenor);

			REAL d1 = ( log ( F / X) + 0.5 * sigma*sigma ) / sigma;
			REAL d2 = d1 - sigma;
			
			return  buyOrSell * ( ( (sigma > 0) && ( X>0 ) && (F>0) ) ? ( callOrPut * ( F * CND (callOrPut * d1) - X * CND (callOrPut * d2) ) ) : max ( callOrPut * ( F - X ), (REAL) 0.0 ) ) * exp ( -r * tenor ); 
		}
		
		__device__ int barrierPayoffType( REAL direction, REAL eta, REAL phi, REAL strike, REAL barrier )
		{
			if (direction==BARRIER_IN)
			{
				if ( ( phi==OPTION_CALL && eta==BARRIER_UP   && strike>barrier ) ||
					 ( phi==OPTION_PUT  && eta==BARRIER_DOWN && strike<=barrier ) )
				{
					return 0;
				}
				else if ( ( phi==OPTION_CALL && eta==BARRIER_UP   && strike<=barrier ) ||
						  ( phi==OPTION_PUT  && eta==BARRIER_DOWN && strike>barrier ) )
				{
					return 1;
				}
				else if ( ( phi==OPTION_PUT  && eta==BARRIER_UP   && strike>barrier ) ||
						  ( phi==OPTION_CALL && eta==BARRIER_DOWN && strike<=barrier ) )
				{
					return 2;
				}
				else if ( ( phi==OPTION_PUT  && eta==BARRIER_UP   && strike<=barrier ) ||
						  ( phi==OPTION_CALL && eta==BARRIER_DOWN && strike>barrier ) )
				{
					return 3;
				}
			}
			else
			{
				if ( ( phi==OPTION_CALL && eta==BARRIER_UP   && strike>barrier ) ||
					 ( phi==OPTION_PUT  && eta==BARRIER_DOWN && strike<=barrier ) )
				{
					return 4;
				}
				else if ( ( phi==OPTION_CALL && eta==BARRIER_UP   && strike<=barrier ) ||
						  ( phi==OPTION_PUT  && eta==BARRIER_DOWN && strike>barrier ) )
				{
					return 5;
				}
				else if ( ( phi==OPTION_PUT  && eta==BARRIER_UP   && strike>barrier ) ||
						  ( phi==OPTION_CALL && eta==BARRIER_DOWN && strike<=barrier ) )
				{
					return 6;
				}
				else if ( ( phi==OPTION_PUT  && eta==BARRIER_UP   && strike<=barrier ) ||
						  ( phi==OPTION_CALL && eta==BARRIER_DOWN && strike>barrier ) )
				{
					return 7;
				}
			}
			
			// Error - should never get here
			return -1;
		}
		
		__device__ REAL barrierOption ( int barrier_payoff_type, REAL eta, REAL phi, REAL sigma, REAL expiry, REAL cash_rebate, REAL b, REAL r, REAL spot, REAL strike, REAL barrier )
		{
			REAL sigma2		= sigma*sigma;
			REAL vol      	= sigma*sqrt(expiry);
			REAL mu 		= ( b - 0.5*sigma2 ) / sigma2;
			REAL log_bar    = log  ( barrier / spot );
			REAL x1			= log  ( spot/strike  ) /vol + (1.0 + mu)*vol;
			REAL x2			= log  ( spot/barrier ) /vol + (1.0 + mu)*vol;

			REAL y1			= log  ( ( barrier*barrier ) / ( spot * strike ) ) / vol + (1.0 + mu)*vol;
			REAL y2			= log_bar / vol + (1.0 + mu)*vol;
			REAL lambda 	= sqrt ( mu*mu + 2.0*r/sigma2 );
			REAL z 			= log_bar / vol + lambda*vol;
			
			REAL A 			= phi * ( spot * exp ( (b-r)*expiry ) * CND ( phi * x1 ) - strike * exp( -r*expiry ) * CND ( phi * ( x1 - vol ) ) );
			REAL B 			= phi * ( spot * exp ( (b-r)*expiry ) * CND ( phi * x2 ) - strike * exp( -r*expiry ) * CND ( phi * ( x2 - vol ) ) );
			REAL C 			= phi * ( spot * exp ( (b-r)*expiry + log_bar*2*(mu+1) ) * CND (eta*y1) - strike*exp ( -r*expiry + log_bar*2*mu ) * CND ( eta * ( y1 - vol ) ) );
			REAL D 			= phi * ( spot * exp ( (b-r)*expiry + log_bar*2*(mu+1) ) * CND (eta*y2) - strike*exp ( -r*expiry + log_bar*2*mu ) * CND ( eta * ( y2 - vol ) ) );
			REAL E 			= cash_rebate * exp ( -r*expiry ) * ( CND ( eta * ( x2 - vol ) ) - exp(log_bar*2*mu) * CND ( eta * ( y2 - vol ) ) );
			REAL F 			= cash_rebate * ( exp ( log_bar*(mu+lambda) ) * CND (eta*z) + exp(log_bar*(mu-lambda)) * CND ( eta * ( z - 2*lambda*vol) ) );
			
			switch (barrier_payoff_type) {
				case 0:
					return A + E;
				case 1:
					return B - C + D + E;
				case 2:
					return A - B + D + E;
				case 3:
					return C + E;
				case 4:
					return F;
				case 5:
					return A - B + C - D + F;
				case 6:
					return B - D + F;
				case 7:
					return A - C + F;
				default:
					return 0.0;
			}
		}
				
		//Utility function to lookup the offset of a FACTOR (not curve) in a sorted list of offsets (the index_list) - useful to map currencies to indexes in arrays for output
		__device__ int LookupRate(const int* __restrict__ offset, const int* __restrict__ index_list, int index_size)
		{
			return binarySearch ( (offset[FACTOR_INDEX_START+FACTOR_INDEX_Type]==FACTOR_TYPE_STATIC) ? -offset[FACTOR_INDEX_START+FACTOR_INDEX_Offset]-1 : offset[FACTOR_INDEX_START+FACTOR_INDEX_Offset], index_list, index_size );
		}
		
		//Utility function that returns a currency's repo Curve
		__device__ const int * CurrencyRepoCurve(const int* __restrict__ currency)
		{
			return ( reinterpret_cast <const int *> ( Buffer + NumTenors ) ) + CurrencyCurveOffset [ LookupRate( currency, CurrencyCurveMap, NumCurveCurrencies ) ];
		}
				
		//Interpolates a scenario risk factor across time
		__device__ REAL interpScenario1d ( const REAL* __restrict__ t, const OFFSET scenario_prior_index, const REAL* __restrict__ stoch_factors, const int offset )
		{
			return stoch_factors[ScenarioFactorSize*scenario_prior_index+offset] * (1.0-t[TIME_GRID_PriorScenarioDelta]) + stoch_factors[ScenarioFactorSize*(scenario_prior_index+1)+offset] * t[TIME_GRID_PriorScenarioDelta];
		}
		
		//Reads a scenario risk factor across time
		__device__ REAL Scenario1d ( const REAL* __restrict__ t, const OFFSET scenario_prior_index, const REAL* __restrict__ stoch_factors, const int offset )
		{
			return stoch_factors[ScenarioFactorSize*scenario_prior_index+offset];
		}
		
		__device__ REAL InterpolateDividend ( 	const REAL* curve_tenor,
												int prev_scenario_point,
												int next_scenario_point,
												REAL tenor_point )
		{
			REAL dt    = curve_tenor [ next_scenario_point ] - curve_tenor[ prev_scenario_point ];
			return max(
						min (
								1.0, ( ( curve_tenor[ prev_scenario_point ] <= 0.0 ) || curve_tenor[ prev_scenario_point ]==curve_tenor[ next_scenario_point ] ) ?
									 ( ( tenor_point - curve_tenor[prev_scenario_point] ) / ( dt > 0 ? dt : 1.0 ) ) :
									 ( ( 1.0/curve_tenor[prev_scenario_point] - 1.0/tenor_point ) / ( 1.0/curve_tenor[prev_scenario_point] - 1.0/curve_tenor[next_scenario_point] ) )
							) , 0.0
					) ;
		}
		
		__device__ REAL InterpolateIntRate ( 	const REAL* curve_tenor,
												int prev_scenario_point,
												int next_scenario_point,
												REAL tenor_point )
		{
			REAL dt    = curve_tenor [ next_scenario_point ] - curve_tenor[ prev_scenario_point ];
			return max ( min ( 1.0, ( tenor_point - curve_tenor[prev_scenario_point] ) / ( dt > 0 ? dt : 1.0 ) ) , 0.0 ) ;
		}
		
		//returns a single rate and interpolates if necessary - e.g. FxRates, Equities, Commodities etc.
		__device__ REAL ScenarioRate (	const REAL* __restrict__ t,
										const int* __restrict__ offsets,
										const OFFSET scenario_prior_index,
										const REAL* __restrict__ static_factors,
										const REAL* __restrict__ stoch_factors )
		{
			REAL output 		= 0.0;
			
			for (int j=0; j<offsets[NUM_FACTOR_INDEX]; j++)
			{
				const int* factor_offset = offsets+j*RATE_INDEX_Size+FACTOR_INDEX_START;
				
				if (factor_offset[FACTOR_INDEX_Type]==FACTOR_TYPE_STATIC)
				{
					output += static_factors[ factor_offset[FACTOR_INDEX_Offset] ];
				}
				else
				{
					output += interpScenario1d (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] );
				}
			}
			
			return output;
		}

		//Function assumes that each factor (stored in offsets) that references a curve, has its tenor points stored in the Buffer variable - needs to
		//be set prior to calling - and the tenors MUST be in ascending order. Also needs functions for scenario interpolation and tenor interpolation
		__device__ REAL ScenarioCurve1D ( 	const REAL* __restrict__ t,
											REAL  tenor_point,
											ScenarioFn calc_scenario,
											CurveFn    calc_weight,
											const int* __restrict__ offsets,
											const OFFSET scenario_prior_index,
											const REAL* __restrict__ static_factors,
											const REAL* __restrict__ stoch_factors )
		{
			int  prev_scenario_point;
			int  next_scenario_point;
			REAL current_rate 	= 0.0;
			REAL prior, post;
			
			for (int j=0; j<abs(offsets[NUM_FACTOR_INDEX]); j++)
			{
				const int* factor_offset = offsets + j*CURVE_INDEX_Size + FACTOR_INDEX_START;
				const REAL* curve_tenor	 = Buffer + ScenarioTenorOffset [ factor_offset[FACTOR_INDEX_Tenor_Index] ];
				int scen_tenor_size      = ScenarioTenorSize [ factor_offset[FACTOR_INDEX_Tenor_Index] ];
				
				prev_scenario_point 	= binarySearch ( tenor_point, curve_tenor, scen_tenor_size );
				next_scenario_point		= min( prev_scenario_point + 1, (int)(scen_tenor_size-1) );
			
				if (factor_offset[FACTOR_INDEX_Type]==FACTOR_TYPE_STATIC)
				{
					prior = static_factors[ factor_offset[FACTOR_INDEX_Offset] + prev_scenario_point ];
					post  = static_factors[ factor_offset[FACTOR_INDEX_Offset] + next_scenario_point ]; 
				}
				else
				{
					prior = calc_scenario (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] + prev_scenario_point );
					post  = calc_scenario (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] + next_scenario_point );
				}
				
				REAL w     = calc_weight (curve_tenor, prev_scenario_point, next_scenario_point, tenor_point);

				//using linear interpolation - might want to extend this to use cubic splines
				current_rate += prior*(1.0-w)+post*w;
			}
			return current_rate;
		}
		
		//Function assumes that each factor (stored in offsets) that references a curve, has its tenor points stored in the Buffer variable - needs to
		//be set prior to calling - and the tenors MUST be in ascending order
		__device__ REAL ScenarioCurve ( const REAL* __restrict__ t,
										REAL  tenor_point,
										const int* __restrict__ offsets,
										const OFFSET scenario_prior_index,
										const REAL* __restrict__ static_factors,
										const REAL* __restrict__ stoch_factors )
		{
			return ScenarioCurve1D ( t, tenor_point, interpScenario1d, InterpolateIntRate, offsets, scenario_prior_index, static_factors, stoch_factors );
		}
		
		//Function assumes that each factor (stored in offsets) that references a curve, has its tenor points stored in the Buffer variable - needs to
		//be set prior to calling - and the tenors MUST be in ascending order - This is exactly like the function above but uses a different interpolation scheme
		__device__ REAL ScenarioDividendYield ( const REAL* __restrict__ t,
												REAL  tenor_point,
												const int* __restrict__ offsets,
												const OFFSET scenario_prior_index,
												const REAL* __restrict__ static_factors,
												const REAL* __restrict__ stoch_factors )
		{
			return ScenarioCurve1D ( t, tenor_point, interpScenario1d, InterpolateDividend, offsets, scenario_prior_index, static_factors, stoch_factors );
		}
		
		__device__ REAL ScenarioSurface2D ( const REAL* __restrict__ t,
											REAL  moneyness,
											REAL  expiry,
											const int* __restrict__ offsets,
											const OFFSET scenario_prior_index,
											const REAL* __restrict__ static_factors,
											const REAL* __restrict__ stoch_factors )
		{
			int  prev_moneyness_point;
			int  next_moneyness_point;
			int  prev_expiry_point;
			int  next_expiry_point;
			REAL current_rate 	= 0.0;
			REAL prior_moneyness_prior_expiry, post_moneyness_prior_expiry, prior_moneyness_post_expiry, post_moneyness_post_expiry;
			
			for (int j=0; j<offsets[NUM_FACTOR_INDEX]; j++)
			{
				const int* factor_offset 	 = offsets + j*VOL2D_INDEX_Size + FACTOR_INDEX_START;
				const REAL* moneyness_tenor	 = Buffer + ScenarioTenorOffset [ factor_offset[FACTOR_INDEX_Moneyness_Index] ];
				const REAL* expiry_tenor	 = Buffer + ScenarioTenorOffset [ factor_offset[FACTOR_INDEX_Expiry_Index] ];
				
				int moneyness_tenor_size     = ScenarioTenorSize [ factor_offset[FACTOR_INDEX_Moneyness_Index] ];
				int expiry_tenor_size        = ScenarioTenorSize [ factor_offset[FACTOR_INDEX_Expiry_Index] ];
				
				prev_moneyness_point 	= binarySearch ( moneyness, moneyness_tenor, moneyness_tenor_size );
				next_moneyness_point	= min( prev_moneyness_point + 1, (int)(moneyness_tenor_size-1) );
				
				prev_expiry_point 		= binarySearch ( expiry, expiry_tenor, expiry_tenor_size );
				next_expiry_point		= min( prev_expiry_point + 1, (int)(expiry_tenor_size-1) );
			
				if (factor_offset[FACTOR_INDEX_Type]==FACTOR_TYPE_STATIC)
				{
					prior_moneyness_prior_expiry = static_factors[ factor_offset[FACTOR_INDEX_Offset] + prev_expiry_point*moneyness_tenor_size + prev_moneyness_point ];
					post_moneyness_prior_expiry  = static_factors[ factor_offset[FACTOR_INDEX_Offset] + prev_expiry_point*moneyness_tenor_size + next_moneyness_point ];
					prior_moneyness_post_expiry	 = static_factors[ factor_offset[FACTOR_INDEX_Offset] + next_expiry_point*moneyness_tenor_size + prev_moneyness_point ];
					post_moneyness_post_expiry   = static_factors[ factor_offset[FACTOR_INDEX_Offset] + next_expiry_point*moneyness_tenor_size + next_moneyness_point ];
				}
				else
				{
					prior_moneyness_prior_expiry = interpScenario1d (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] + prev_expiry_point*moneyness_tenor_size + prev_moneyness_point );
					post_moneyness_prior_expiry  = interpScenario1d (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] + prev_expiry_point*moneyness_tenor_size + next_moneyness_point );
					prior_moneyness_post_expiry  = interpScenario1d (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] + next_expiry_point*moneyness_tenor_size + prev_moneyness_point );
					post_moneyness_post_expiry	 = interpScenario1d (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] + next_expiry_point*moneyness_tenor_size + next_moneyness_point );
				}
				
				REAL dt1    = moneyness_tenor [ next_moneyness_point ] - moneyness_tenor[ prev_moneyness_point ];
				REAL dt2    = expiry_tenor [ next_expiry_point ] - expiry_tenor [ prev_expiry_point ];
				
				REAL w1     = max ( min ( 1.0, ( moneyness - moneyness_tenor[prev_moneyness_point] ) / ( dt1 > 0 ? dt1 : 1.0 ) ), 0.0 );
				REAL w2     = max ( min ( 1.0, ( expiry - expiry_tenor[prev_expiry_point] ) / ( dt2 > 0 ? dt2 : 1.0 ) ), 0.0 );

				//using linear interpolation - might want to extend this to use cubic splines
				current_rate += prior_moneyness_prior_expiry*(1.0-w1)*(1.0-w2) +
								post_moneyness_prior_expiry*w1*(1.0-w2) +
								prior_moneyness_post_expiry*(1.0-w1)*w2 +
								post_moneyness_post_expiry*w1*w2;
			}
			return current_rate;
		}
		
		__device__ REAL ScenarioSurface3D ( const REAL* __restrict__ t,
											REAL  moneyness,
											REAL  expiry,
											REAL  voltenor,
											const int* __restrict__ offsets,
											const OFFSET scenario_prior_index,
											const REAL* __restrict__ static_factors,
											const REAL* __restrict__ stoch_factors )
		{
			int  prev_moneyness_point;
			int  next_moneyness_point;
			int  prev_expiry_point;
			int  next_expiry_point;
			int  prev_tenor_point;
			int  next_tenor_point;
			
			REAL current_rate 	= 0.0;
			REAL prior_moneyness_prior_expiry_prior_tenor, post_moneyness_prior_expiry_prior_tenor, prior_moneyness_post_expiry_prior_tenor, post_moneyness_post_expiry_prior_tenor;
			REAL prior_moneyness_prior_expiry_post_tenor, post_moneyness_prior_expiry_post_tenor, prior_moneyness_post_expiry_post_tenor, post_moneyness_post_expiry_post_tenor;
			
			for (int j=0; j<offsets[NUM_FACTOR_INDEX]; j++)
			{
				const int* factor_offset 	 = offsets + j*VOL3D_INDEX_Size + FACTOR_INDEX_START;
				const REAL* moneyness_tenor	 = Buffer + ScenarioTenorOffset [ factor_offset[FACTOR_INDEX_Moneyness_Index] ];
				const REAL* expiry_tenor	 = Buffer + ScenarioTenorOffset [ factor_offset[FACTOR_INDEX_Expiry_Index] ];
				const REAL* voltenor_tenor	 = Buffer + ScenarioTenorOffset [ factor_offset[FACTOR_INDEX_VolTenor_Index] ];
				
				int moneyness_tenor_size     = ScenarioTenorSize [ factor_offset[FACTOR_INDEX_Moneyness_Index] ];
				int expiry_tenor_size        = ScenarioTenorSize [ factor_offset[FACTOR_INDEX_Expiry_Index] ];
				int voltenor_tenor_size      = ScenarioTenorSize [ factor_offset[FACTOR_INDEX_VolTenor_Index] ];
				
				prev_moneyness_point 	= binarySearch ( moneyness, moneyness_tenor, moneyness_tenor_size );
				next_moneyness_point	= min( prev_moneyness_point + 1, (int)(moneyness_tenor_size-1) );
				
				prev_expiry_point 		= binarySearch ( expiry, expiry_tenor, expiry_tenor_size );
				next_expiry_point		= min( prev_expiry_point + 1, (int)(expiry_tenor_size-1) );
				
				prev_tenor_point 		= binarySearch ( voltenor, voltenor_tenor, voltenor_tenor_size );
				next_tenor_point		= min( prev_tenor_point + 1, (int)(voltenor_tenor_size-1) );
			
				if (factor_offset[FACTOR_INDEX_Type]==FACTOR_TYPE_STATIC)
				{
					prior_moneyness_prior_expiry_prior_tenor = static_factors[ factor_offset[FACTOR_INDEX_Offset] + (prev_tenor_point*expiry_tenor_size + prev_expiry_point)*moneyness_tenor_size + prev_moneyness_point ];
					post_moneyness_prior_expiry_prior_tenor  = static_factors[ factor_offset[FACTOR_INDEX_Offset] + (prev_tenor_point*expiry_tenor_size + prev_expiry_point)*moneyness_tenor_size + next_moneyness_point ];
					prior_moneyness_post_expiry_prior_tenor	 = static_factors[ factor_offset[FACTOR_INDEX_Offset] + (prev_tenor_point*expiry_tenor_size + next_expiry_point)*moneyness_tenor_size + prev_moneyness_point ];
					post_moneyness_post_expiry_prior_tenor   = static_factors[ factor_offset[FACTOR_INDEX_Offset] + (prev_tenor_point*expiry_tenor_size + next_expiry_point)*moneyness_tenor_size + next_moneyness_point ];
					prior_moneyness_prior_expiry_post_tenor  = static_factors[ factor_offset[FACTOR_INDEX_Offset] + (next_tenor_point*expiry_tenor_size + prev_expiry_point)*moneyness_tenor_size + prev_moneyness_point ];
					post_moneyness_prior_expiry_post_tenor   = static_factors[ factor_offset[FACTOR_INDEX_Offset] + (next_tenor_point*expiry_tenor_size + prev_expiry_point)*moneyness_tenor_size + next_moneyness_point ];
					prior_moneyness_post_expiry_post_tenor	 = static_factors[ factor_offset[FACTOR_INDEX_Offset] + (next_tenor_point*expiry_tenor_size + next_expiry_point)*moneyness_tenor_size + prev_moneyness_point ];
					post_moneyness_post_expiry_post_tenor    = static_factors[ factor_offset[FACTOR_INDEX_Offset] + (next_tenor_point*expiry_tenor_size + next_expiry_point)*moneyness_tenor_size + next_moneyness_point ];
				}
				else
				{
					prior_moneyness_prior_expiry_prior_tenor = interpScenario1d (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] + (prev_tenor_point*expiry_tenor_size + prev_expiry_point)*moneyness_tenor_size + prev_moneyness_point );
					post_moneyness_prior_expiry_prior_tenor  = interpScenario1d (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] + (prev_tenor_point*expiry_tenor_size + prev_expiry_point)*moneyness_tenor_size + next_moneyness_point );
					prior_moneyness_post_expiry_prior_tenor  = interpScenario1d (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] + (prev_tenor_point*expiry_tenor_size + next_expiry_point)*moneyness_tenor_size + prev_moneyness_point );
					post_moneyness_post_expiry_prior_tenor	 = interpScenario1d (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] + (prev_tenor_point*expiry_tenor_size + next_expiry_point)*moneyness_tenor_size + next_moneyness_point );
					prior_moneyness_prior_expiry_post_tenor  = interpScenario1d (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] + (next_tenor_point*expiry_tenor_size + prev_expiry_point)*moneyness_tenor_size + prev_moneyness_point );
					post_moneyness_prior_expiry_post_tenor   = interpScenario1d (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] + (next_tenor_point*expiry_tenor_size + prev_expiry_point)*moneyness_tenor_size + next_moneyness_point );
					prior_moneyness_post_expiry_post_tenor   = interpScenario1d (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] + (next_tenor_point*expiry_tenor_size + next_expiry_point)*moneyness_tenor_size + prev_moneyness_point );
					post_moneyness_post_expiry_post_tenor	 = interpScenario1d (t, scenario_prior_index, stoch_factors, factor_offset[FACTOR_INDEX_Offset] + (next_tenor_point*expiry_tenor_size + next_expiry_point)*moneyness_tenor_size + next_moneyness_point );
				}
				
				REAL dt1    = moneyness_tenor [ next_moneyness_point ] 	- moneyness_tenor [ prev_moneyness_point ];
				REAL dt2    = expiry_tenor    [ next_expiry_point ] 	- expiry_tenor [ prev_expiry_point ];
				REAL dt3    = voltenor_tenor  [ next_tenor_point ] 		- voltenor_tenor [ prev_tenor_point ];
				
				REAL w1     = max ( min ( 1.0, ( moneyness - moneyness_tenor[prev_moneyness_point] ) / ( dt1 > 0 ? dt1 : 1.0 ) ), 0.0 );
				REAL w2     = max ( min ( 1.0, ( expiry - expiry_tenor[prev_expiry_point] ) / ( dt2 > 0 ? dt2 : 1.0 ) ), 0.0 );
				REAL w3     = max ( min ( 1.0, ( voltenor - voltenor_tenor[prev_tenor_point] ) / ( dt3 > 0 ? dt3 : 1.0 ) ), 0.0 );

				//using linear interpolation - might want to extend this to use cubic splines
				
				current_rate += prior_moneyness_prior_expiry_prior_tenor*(1.0-w1)*(1.0-w2)*(1-w3) +
								post_moneyness_prior_expiry_prior_tenor*w1*(1.0-w2)*(1-w3) +
								prior_moneyness_post_expiry_prior_tenor*(1.0-w1)*w2*(1-w3) +
								post_moneyness_post_expiry_prior_tenor*w1*w2*(1-w3) +
								prior_moneyness_prior_expiry_post_tenor*(1.0-w1)*(1.0-w2)*w3 +
								post_moneyness_prior_expiry_post_tenor*w1*(1.0-w2)*w3 +
								prior_moneyness_post_expiry_post_tenor*(1.0-w1)*w2*w3 +
								post_moneyness_post_expiry_post_tenor*w1*w2*w3;
								
			}
			return current_rate;
		}
		
		// Function used for Quanto deals (where FX forward prices are needed)
		__device__ REAL ScenarioFXForward ( const REAL* __restrict__ t,
										    REAL T,	
										    const int* __restrict__ local_curr,
										    const int* __restrict__ other_curr,
										    const int* __restrict__ local_curr_curve,
										    const int* __restrict__ other_curr_curve,
										    const OFFSET scenario_prior_index,
										    const REAL* __restrict__ static_factors,
										    const REAL* __restrict__ stoch_factors )
		{
			REAL LocalYearAccrual 	 = calcDayCountAccrual ( T, local_curr_curve[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
			REAL OtherYearAccrual 	 = calcDayCountAccrual ( T, other_curr_curve[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
			
			REAL localrate  = ScenarioCurve ( t, LocalYearAccrual, local_curr_curve, scenario_prior_index, static_factors, stoch_factors );
			REAL otherrate  = ScenarioCurve ( t, OtherYearAccrual, other_curr_curve, scenario_prior_index, static_factors, stoch_factors );
			
			//reconstruct the spot FX rate at this point in time - horribly hacky - need to find a nice way to do this
			REAL FX_Spot 	= ScenarioRate ( t, local_curr, scenario_prior_index, static_factors, stoch_factors ) / ScenarioRate ( t, other_curr, scenario_prior_index, static_factors, stoch_factors );

			//now calculate the FX forward price
			return FX_Spot * exp ( otherrate * OtherYearAccrual - localrate * LocalYearAccrual ) ;
		}		

		// Function used to PV a sequence of fixed cashflows - can be used for a fixed leg of a swap, a bond etc. 
		__device__ REAL ScenarioPVFixedLeg (	const REAL* __restrict__ t,
												int   cashflow_method,
												int   compounding,	
												int   from_cashflow_index,
												int   to_cashflow_index,
												REAL& settlement_cashflow,
												const REAL* __restrict__ cashflows,
												const int* __restrict__ discount_offset,
												const OFFSET scenario_prior_index,
												const REAL* __restrict__ static_factors,
												const REAL* __restrict__ stoch_factors )
		{
			REAL out = 0.0;
			REAL total_interest = 0.0;
			
			settlement_cashflow = 0.0;
			
			for (int i=from_cashflow_index; i<to_cashflow_index; i++)
			{
				const REAL* cashflow = cashflows+i*CASHFLOW_INDEX_Size;				
				REAL yT 			 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], discount_offset[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
				REAL discount_rate 	 = ScenarioCurve ( t, yT, discount_offset, scenario_prior_index, static_factors, stoch_factors );
				REAL Interest 		 = ( ( cashflow_method == CASHFLOW_METHOD_Fixed_Rate_Standard ) ? cashflow[CASHFLOW_INDEX_FixedRate] : 1.0 ) * cashflow[CASHFLOW_INDEX_Year_Frac];
				
				if (compounding==CASHFLOW_METHOD_Fixed_Compounding_Yes)
					total_interest 	 = Interest * cashflow [ CASHFLOW_INDEX_Nominal ] + (1.0+Interest) * total_interest ;
				else
					total_interest += Interest * cashflow[CASHFLOW_INDEX_Nominal];
				
				if ( cashflow[CASHFLOW_INDEX_Settle] )
				{
					REAL payment  =  total_interest + cashflow[CASHFLOW_INDEX_FixedAmt];
					
					//handle settlement
					if (cashflow[CASHFLOW_INDEX_Pay_Day]==t[TIME_GRID_MTM])
						settlement_cashflow = payment;
					
					out += payment * exp ( -discount_rate * yT );
					
					total_interest = 0.0;
				}
			}
			return out;
		}		

		__device__ REAL CalcDiscount (	const REAL* __restrict__ t,
										REAL  at_day,
										const int* __restrict__ curve,
										const OFFSET scenario_prior_index,
										const REAL* __restrict__ static_factors,
										const REAL* __restrict__ stoch_factors )
		{
			//Again, at_day needs to be greater than time t.
			REAL yt 		= calcDayCountAccrual ( at_day - t[TIME_GRID_MTM], curve[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
			REAL rt			= ScenarioCurve ( t, yt, curve, scenario_prior_index, static_factors, stoch_factors );
			
			return exp ( - rt * yt );
		}
																
		__device__ REAL CalcSimpleForwardCurveIndex (	const REAL* __restrict__ t,
														const REAL* __restrict__ reset,
														const int* __restrict__ forward_offset,
														const OFFSET scenario_prior_index,
														const REAL* __restrict__ static_factors,
														const REAL* __restrict__ stoch_factors )
		{
			
			//All this, of course, assumes that yt will be positive - otherwise, well . . fuck
			REAL yt 	= calcDayCountAccrual ( reset[RESET_INDEX_Start_Day] - t[TIME_GRID_MTM], forward_offset[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
			REAL rt		= ScenarioCurve ( t, yt, forward_offset, scenario_prior_index, static_factors, stoch_factors );
			REAL yT 	= calcDayCountAccrual ( reset[RESET_INDEX_End_Day] - t[TIME_GRID_MTM], forward_offset[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
			REAL rT		= ScenarioCurve ( t, yT, forward_offset, scenario_prior_index, static_factors, stoch_factors );

			return (exp(yT*rT - yt*rt)-1.0)/( reset[RESET_INDEX_Accrual] ? reset[RESET_INDEX_Accrual] : (yT-yt) );
		}		

		__device__ REAL Calc_Price( const REAL* __restrict__ t,
									const int* __restrict__ forward,
									const int* local_curr,
									const int* other_curr,
									const int* local_curr_curve,
									const int* other_curr_curve,
									const REAL* __restrict__ reset,
									const OFFSET scenario_index,
									const OFFSET scenario_prior_index,
									const REAL* __restrict__ static_factors,
									const REAL* __restrict__ stoch_factors )
		{
			REAL rate						= 0.0;
			REAL FX_contrib					= 1.0;
			REAL base_date_excel			= forward[FACTOR_INDEX_START+FACTOR_INDEX_Daycount];
			
			if ( reset[RESET_INDEX_Reset_Day] >= base_date_excel )
			{
				if ( t[TIME_GRID_MTM] > ( reset[RESET_INDEX_Reset_Day]-base_date_excel ) )
				{
					rate 	   = ScenarioCurve ( reset + RESET_INDEX_Time_Grid, reset[RESET_INDEX_End_Day], forward, scenario_index + reinterpret_cast<const OFFSET &> ( reset[RESET_INDEX_Scenario] ), static_factors, stoch_factors );
					FX_contrib = (forward[NUM_FACTOR_INDEX]<0) ? ScenarioFXForward (reset + RESET_INDEX_Time_Grid, reset[RESET_INDEX_Start_Day] - base_date_excel,
																					local_curr, other_curr, local_curr_curve, other_curr_curve,
																					scenario_index + reinterpret_cast<const OFFSET &>(reset[RESET_INDEX_Scenario]) , static_factors, stoch_factors )
																: 1.0;
				}
				else
				{
					rate 		= ScenarioCurve ( t, reset[RESET_INDEX_End_Day], forward, scenario_prior_index, static_factors, stoch_factors );
					FX_contrib 	= (forward[NUM_FACTOR_INDEX]<0) ? ScenarioFXForward (t, reset[RESET_INDEX_Start_Day] - base_date_excel,
																					 local_curr, other_curr, local_curr_curve, other_curr_curve,
																					 scenario_prior_index, static_factors, stoch_factors )
																: 1.0;
				}
			}
			else
			{
				//Don't need to look at the reset[RESET_INDEX_FXValue] value as fx_averaging is not used.
				rate 	   			= reset[RESET_INDEX_Value];
			}
			
			return rate * FX_contrib;
		}

		__device__ REAL AverageSample ( const int  num_samples,
										const int*  __restrict__ rate,
										const REAL* __restrict__ samples,
										const OFFSET scenario_index,
										const REAL* __restrict__ static_factors,
										const REAL* __restrict__ stoch_factors )
		{
			REAL sum = 0.0; 
			for ( int si=0; si<num_samples; si++ )
			{
				const REAL* sample_i    = samples+si*RESET_INDEX_Size;
				sum 					+= sample_i[RESET_INDEX_Weight] * ( ( sample_i [RESET_INDEX_Reset_Day] >= 0 ) ? ScenarioRate ( sample_i + RESET_INDEX_Time_Grid, rate, scenario_index + reinterpret_cast<const OFFSET &>(sample_i[RESET_INDEX_Scenario]) , static_factors, stoch_factors ) : sample_i[RESET_INDEX_Value] );
			}
			return sum;
		}
		
		__device__ REAL Calc_SingleCurrencyAssetVariance( 	const REAL* __restrict__ t,
															const REAL M1,
															const REAL b,
															const REAL sigma2,
															const REAL St,															
															const int  start_index,
															const int  num_samples,
															const REAL* __restrict__ samples,
															const OFFSET scenario_prior_index,
															const REAL* __restrict__ static_factors,
															const REAL* __restrict__ stoch_factors )
		{
			REAL  M2  = 0.0;

			for ( int si=start_index; si < num_samples; si++ )
			{
				const REAL* sample_i    		= samples+si*RESET_INDEX_Size;
				
				REAL ti = ( sample_i[RESET_INDEX_End_Day] - t[TIME_GRID_MTM] ) / 365.0;
				REAL Fi = St*exp(b*ti);
					
				for ( int sj=start_index; sj < num_samples; sj++ )
				{
					const REAL* sample_j    		= samples+sj*RESET_INDEX_Size;
					
					REAL tj 	= ( sample_j[RESET_INDEX_End_Day] - t[TIME_GRID_MTM] ) / 365.0;
					REAL Fj  	= St*exp(b*tj);
					REAL tau	= min(ti,tj);
					
					M2 += sample_i[RESET_INDEX_Weight] * sample_j[RESET_INDEX_Weight] * Fi * Fj * exp ( sigma2 * tau );
				}
			}

			return log ( M2/(M1*M1) );
		}
		
		__device__ REAL Calc_SinglePriceVariance( const REAL* __restrict__ t,
													const REAL M1,
													const REAL rho,
													const int* __restrict__ forward,
													const int* __restrict__ forward_vol,
													const int* __restrict__ fx_vol,
													const REAL moneyness,
													const int  start_index,
													const int  num_samples,
													const REAL* __restrict__ samples,
													const OFFSET scenario_prior_index,
													const REAL* __restrict__ static_factors,
													const REAL* __restrict__ stoch_factors )
		{
			REAL base_date_excel			= forward[FACTOR_INDEX_START+FACTOR_INDEX_Daycount];
			const OFFSET scenario_index 	= scenario_prior_index - reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			const int* local_curr			= NULL;
			const int* other_curr			= NULL;
			const int* local_curr_curve		= NULL;
			const int* other_curr_curve		= NULL;
			
			REAL  M2  		= 0.0;

			if (forward[NUM_FACTOR_INDEX]<0)
			{
				// quanto
				local_curr = forward + FACTOR_INDEX_START + CURVE_INDEX_Size*(-forward[NUM_FACTOR_INDEX]);
				other_curr = local_curr + FACTOR_INDEX_START + RATE_INDEX_Size;
				
				local_curr_curve = CurrencyRepoCurve ( local_curr );
				other_curr_curve = CurrencyRepoCurve ( other_curr );
			}
			
			for ( int si=start_index; si < num_samples; si++ )
			{
				const REAL* sample_i    		= samples+si*RESET_INDEX_Size;
				
				REAL Fi = Calc_Price( t, forward, local_curr, other_curr, local_curr_curve, other_curr_curve,
										sample_i, scenario_index, scenario_prior_index, static_factors, stoch_factors );
					
				for ( int sj=start_index; sj < num_samples; sj++ )
				{
					const REAL* sample_j    		= samples+sj*RESET_INDEX_Size;
					
					REAL Fj  = Calc_Price( t, forward, local_curr, other_curr, local_curr_curve, other_curr_curve,
											sample_j, scenario_index, scenario_prior_index, static_factors, stoch_factors );
											
					REAL tau = ( ( sample_i[RESET_INDEX_Start_Day] < sample_j[RESET_INDEX_Start_Day] ? sample_i[RESET_INDEX_Start_Day] : sample_j[RESET_INDEX_Start_Day] ) - base_date_excel - t[TIME_GRID_MTM] ) / 365.0 ;
					REAL T   = ( ( sample_i[RESET_INDEX_End_Day]   < sample_j[RESET_INDEX_End_Day] ? sample_i[RESET_INDEX_End_Day] : sample_j[RESET_INDEX_End_Day] ) - base_date_excel - t[TIME_GRID_MTM] ) / 365.0 ;
					
					//TODO - fix the call order here
					REAL vol_s 	= tau > 0 ? ScenarioSurface3D ( t, T, tau, moneyness, forward_vol, scenario_prior_index, static_factors, stoch_factors ) : 0.0;
					REAL vol_fx = ( fx_vol[NUM_FACTOR_INDEX] && tau > 0 ) ? ScenarioSurface2D ( t, 1.0, tau, fx_vol, scenario_prior_index, static_factors, stoch_factors ) : 0.0;
					//REAL vol2 	= ( vol_s*vol_s + vol_fx*vol_fx + 2.0*vol_fx*vol_s*rho );
					REAL vol2 	= ( vol_s*vol_s );
					M2 += sample_i[RESET_INDEX_Weight] * sample_j[RESET_INDEX_Weight] * Fi * Fj * exp ( vol2 * tau );
				}
			}

			return log ( M2/(M1*M1) );
		}
		
		__device__ REAL Calc_ForwardPrice ( const REAL* __restrict__ t,
											const int  from_sample,
											const int  to_sample,
											REAL& sum_weights,
											const int* __restrict__ forward,
											const REAL* __restrict__ resets,
											const OFFSET scenario_prior_index,
											const REAL* __restrict__ static_factors,
											const REAL* __restrict__ stoch_factors )
		{
			REAL base_date_excel		= 0.0;
			REAL total_reset 			= 0.0;
			REAL FX_contrib				= 1.0;
			const int* local_curr 		= NULL;
			const int* other_curr		= NULL;
			const int* local_curr_curve = NULL;
			const int* other_curr_curve = NULL;
			
			if (forward[NUM_FACTOR_INDEX]<0)
			{
				// quanto
				local_curr = forward + FACTOR_INDEX_START + CURVE_INDEX_Size*(-forward[NUM_FACTOR_INDEX]);
				other_curr = local_curr + FACTOR_INDEX_START + RATE_INDEX_Size;

				local_curr_curve = CurrencyRepoCurve ( local_curr );
				other_curr_curve = CurrencyRepoCurve ( other_curr );
									
				base_date_excel = forward [FACTOR_INDEX_START+FACTOR_INDEX_Daycount];
			}
			
			sum_weights = 0.0;
			
			for (int r=from_sample; r< to_sample; r++)
			{
				const REAL* reset   = resets+r*RESET_INDEX_Size;
				FX_contrib 			= (forward[NUM_FACTOR_INDEX]<0) ? ScenarioFXForward ( t, reset[RESET_INDEX_Start_Day] - base_date_excel,
																							local_curr, other_curr, local_curr_curve, other_curr_curve,
																							scenario_prior_index, static_factors, stoch_factors )
																			: 1.0;
																			
				total_reset += reset[RESET_INDEX_Weight] * ScenarioCurve ( t, reset[RESET_INDEX_End_Day], forward, scenario_prior_index, static_factors, stoch_factors ) * FX_contrib;
				sum_weights += reset[RESET_INDEX_Weight];
			}
			
			return total_reset;
		}
		
		__device__ REAL Calc_PastPrice( const REAL* __restrict__ t,
										const int  num_samples,
										const int* __restrict__ forward,
										const REAL* __restrict__ resets,
										const OFFSET scenario_prior_index,
										const REAL* __restrict__ static_factors,
										const REAL* __restrict__ stoch_factors )
		{
			const OFFSET scenario_index 	= scenario_prior_index - reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			REAL total_reset 				= 0.0;
			const int* local_curr 			= NULL;
			const int* other_curr			= NULL;
			const int* local_curr_curve 	= NULL;
			const int* other_curr_curve 	= NULL;

			if (forward[NUM_FACTOR_INDEX]<0)
			{
				// quanto
				local_curr = forward + FACTOR_INDEX_START + CURVE_INDEX_Size*(-forward[NUM_FACTOR_INDEX]);
				other_curr = local_curr + FACTOR_INDEX_START + RATE_INDEX_Size;
				
				local_curr_curve = CurrencyRepoCurve ( local_curr );
				other_curr_curve = CurrencyRepoCurve ( other_curr );
			}
			
			for (int r=0; r < num_samples; r++)
			{
				const REAL* reset 	= resets+r*RESET_INDEX_Size;
				total_reset 		+= reset[RESET_INDEX_Weight] * Calc_Price ( t, forward, local_curr, other_curr, local_curr_curve, other_curr_curve, reset, scenario_index, scenario_prior_index, static_factors, stoch_factors );
			}
			
			return total_reset;
		}
		
		//Function used to calculate index value at time t
		__device__ REAL Calc_Index (  	const REAL* __restrict__ t,
										int   cashflow_method,
										int   mtm_time_index,
										const REAL* __restrict__ resets,
										const int* __restrict__ ref_index,
										const OFFSET scenario_index,
										const REAL* __restrict__ static_factors,
										const REAL* __restrict__ stoch_factors )
		{
			REAL Index_val					= 0; 
			int	 num_resets;

			switch (cashflow_method) {
				case CASHFLOW_METHOD_IndexReference2M:
				case CASHFLOW_METHOD_IndexReference3M:
					num_resets = 1;
					break;
				case CASHFLOW_METHOD_IndexReferenceInterpolated3M:
				case CASHFLOW_METHOD_IndexReferenceInterpolated4M:
					num_resets = 2;
					break;
				default:
					num_resets = 1;
			}

			for (int r=0; r<num_resets; r++)
			{
				const REAL* reset  				= resets + ( num_resets * mtm_time_index + r ) * RESET_INDEX_Size;
				const OFFSET scenario_offset	= reinterpret_cast <const OFFSET &> ( reset[RESET_INDEX_Scenario] );
				
				Index_val += reset[RESET_INDEX_Weight] * ( ( reset[RESET_INDEX_Reset_Day] > 0 ) ?
															ScenarioRate ( reset + RESET_INDEX_Time_Grid, ref_index, scenario_index + scenario_offset, static_factors, stoch_factors ) :
															reset[RESET_INDEX_Value] );
			}

			return Index_val;
		}
		
		//Function used to calculate bond index value at time t
		__device__ REAL Calc_Bond_Index (  	const REAL* __restrict__ t,
											int   cashflow_method,
											int   cashflow_index,
											REAL  latest_index_value,
											const REAL* __restrict__ last_published,
											const REAL* __restrict__ resets,
											const int* __restrict__ ref_index,
											const int* __restrict__ growth_offset,
											const OFFSET scenario_prior_index,
											const REAL* __restrict__ static_factors,
											const REAL* __restrict__ stoch_factors )
		{
			REAL Index_val					= 0;			
			const OFFSET scenario_index 	= scenario_prior_index - reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			
			int	 num_resets;

			switch (cashflow_method) {
				case CASHFLOW_METHOD_IndexReference2M:
				case CASHFLOW_METHOD_IndexReference3M:
					num_resets = 1;
					break;
				case CASHFLOW_METHOD_IndexReferenceInterpolated3M:
				case CASHFLOW_METHOD_IndexReferenceInterpolated4M:
					num_resets = 2;
					break;
				default:
					num_resets = 1;
			}
			
			for (int r=0; r<num_resets; r++)
			{
				const REAL* reset  				= resets + ( num_resets * cashflow_index + r ) * RESET_INDEX_Size;
				const OFFSET scenario_offset	= reinterpret_cast <const OFFSET &> ( reset[RESET_INDEX_Scenario] );
				
				Index_val += reset[RESET_INDEX_Weight] * ( ( reset[RESET_INDEX_Reset_Day] >= last_published[RESET_INDEX_Reset_Day] ) ?
															 latest_index_value/CalcDiscount ( last_published + RESET_INDEX_Time_Grid, reset[RESET_INDEX_Reset_Day], growth_offset, scenario_prior_index, static_factors, stoch_factors ) :
															 ScenarioRate ( reset + RESET_INDEX_Time_Grid, ref_index, scenario_index+scenario_offset, static_factors, stoch_factors ) );
			}

			return Index_val;
		}

		__device__ REAL Calc_EquityForward (  	const REAL* __restrict__ t,
												REAL  T,
												const int* __restrict__ equity,
												const int* __restrict__ equity_repo,
												const int* __restrict__ dividend_yield,
												const OFFSET scenario_prior_index,
												const REAL* __restrict__ static_factors,
												const REAL* __restrict__ stoch_factors )
		{
				REAL expiry 	= calcDayCountAccrual ( T - t[TIME_GRID_MTM], equity_repo[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
				REAL r  		= ScenarioCurve ( t, expiry, equity_repo, scenario_prior_index, static_factors, stoch_factors );
				REAL q 			= ScenarioDividendYield ( t, expiry, dividend_yield, scenario_prior_index, static_factors, stoch_factors );
				REAL spot       = ScenarioRate  ( t, equity, scenario_prior_index, static_factors, stoch_factors );
				
				return spot * exp ( ( r - q ) * expiry ) ;
		}
		
		__device__ REAL Calc_EquityRealizedDividends (  const REAL* __restrict__ t,
														REAL  T,
														REAL  prior_dividends,
														const int* __restrict__ equity,
														const int* __restrict__ equity_repo,
														const int* __restrict__ dividend_yield,
														const OFFSET scenario_prior_index,
														const REAL* __restrict__ static_factors,
														const REAL* __restrict__ stoch_factors )
		{
				REAL expiry 	= calcDayCountAccrual ( T - t[TIME_GRID_MTM], equity_repo[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
				REAL r  		= ScenarioCurve ( t, expiry, equity_repo, scenario_prior_index, static_factors, stoch_factors );
				
				//REAL q 			= ScenarioDividendYield ( t, expiry, dividend_yield, scenario_prior_index, static_factors, stoch_factors );
				//REAL spot       = ScenarioRate  ( t, equity, scenario_prior_index, static_factors, stoch_factors );
				//REAL forward    = spot * exp ( ( r - q ) * expiry ) ;
				//below is useful if you have more than 1 month
				//return exp ( r * expiry ) * ( prior_dividends + spot ) - forward;
				
				return exp ( r * expiry ) * prior_dividends ;
		}

		__device__ REAL Calc_FutureReset (  const REAL* __restrict__ t,
											const int* __restrict__ forward,
											const REAL* __restrict__ cashflow,
											const REAL* __restrict__ resets,
											const OFFSET scenario_prior_index,
											const REAL* __restrict__ static_factors,
											const REAL* __restrict__ stoch_factors )
		{
			REAL total_reset 	= 0.0;
		
			for (int r=0; r< reinterpret_cast <const OFFSET &> (cashflow[CASHFLOW_INDEX_NumResets]); r++)
			{
				const REAL* reset    		= resets+r*RESET_INDEX_Size;			
				total_reset += reset[RESET_INDEX_Weight] * CalcSimpleForwardCurveIndex ( t, reset, forward, scenario_prior_index, static_factors, stoch_factors );
			}
			
			return total_reset;
		}
		
		__device__ REAL Calc_PastReset( const REAL* __restrict__ t,
										const int* __restrict__ forward,
										const REAL* __restrict__ cashflow,
										const REAL* __restrict__ resets,
										const OFFSET scenario_prior_index,
										const REAL* __restrict__ static_factors,
										const REAL* __restrict__ stoch_factors )
		{
			REAL total_reset 				= 0.0;
			const OFFSET scenario_index 	= scenario_prior_index - reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			
			for (int r=0; r < reinterpret_cast <const OFFSET &> (cashflow[CASHFLOW_INDEX_NumResets]); r++)
			{
				const REAL* reset 			 = resets+r*RESET_INDEX_Size;
				const OFFSET scenario_offset = reinterpret_cast<const OFFSET &>(reset[RESET_INDEX_Scenario]);
				
				REAL  rate = ( reset[RESET_INDEX_Reset_Day] >= 0 )
								? ( ( t[TIME_GRID_MTM] > reset[RESET_INDEX_Reset_Day] ) ? CalcSimpleForwardCurveIndex ( reset + RESET_INDEX_Time_Grid, reset, forward, scenario_index+scenario_offset, static_factors, stoch_factors )
																		: CalcSimpleForwardCurveIndex ( t, reset, forward, scenario_prior_index, static_factors, stoch_factors ) )
								: reset[RESET_INDEX_Value];

				total_reset += reset[RESET_INDEX_Weight] * rate;
			}
			
			return total_reset;
		}
		
		// Function used to PV a sequence of fixed index linked cashflows - assuming a growth curve - example CPI Bonds
		// Note here we grow the last published value and interpolate the index
		__device__ REAL ScenarioPVBondIndex (	const REAL* __restrict__ t,
												int   cashflow_method,
												int   mtm_time_index,
												int   num_cashflows,
												REAL& settlement_cashflow,
												const int*  __restrict__ cashflow_starttime_index,
												const REAL* __restrict__ cashflows,
												const REAL* __restrict__ time_resets,
												const REAL* __restrict__ base_resets,
												const REAL* __restrict__ final_resets,
												const int* __restrict__ ref_index,
												const int* __restrict__ growth_offset,
												const int* __restrict__ discount_offset,
												const OFFSET scenario_prior_index,
												const REAL* __restrict__ static_factors,
												const REAL* __restrict__ stoch_factors )
		{
			const REAL* last_published		= time_resets + mtm_time_index * RESET_INDEX_Size;			
			const REAL* cashflow	    	= NULL;
			REAL last_published_index		= ScenarioRate ( t, ref_index, scenario_prior_index, static_factors, stoch_factors );
			REAL out 						= 0.0;
			
			settlement_cashflow				= 0.0;

			for (int i=cashflow_starttime_index[mtm_time_index]; i<num_cashflows; i++)
			{
				cashflow 				= cashflows+i*CASHFLOW_INDEX_Size;

				REAL Index_val_Base		= cashflow[CASHFLOW_INDEX_BaseReference] > 0 ?
											cashflow[CASHFLOW_INDEX_BaseReference] :											
											Calc_Bond_Index ( t, cashflow_method, i, last_published_index, last_published, base_resets, ref_index, growth_offset, scenario_prior_index, static_factors, stoch_factors );
																
				REAL Index_val_Final 	= cashflow[CASHFLOW_INDEX_FinalReference] > 0 ?
											cashflow[CASHFLOW_INDEX_FinalReference] :
											Calc_Bond_Index ( t, cashflow_method, i, last_published_index, last_published, final_resets, ref_index, growth_offset, scenario_prior_index, static_factors, stoch_factors );
				
				REAL yT 			 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], discount_offset[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
				REAL discount_rate 	 = ScenarioCurve ( t, yT, discount_offset, scenario_prior_index, static_factors, stoch_factors );

				//equation 4.559 in the theory guide
				REAL payment		 = cashflow[CASHFLOW_INDEX_Nominal] * ( cashflow[CASHFLOW_INDEX_FixedAmt] * (Index_val_Final/Index_val_Base) + 0.0 ) * cashflow[CASHFLOW_INDEX_FixedRate] * cashflow[CASHFLOW_INDEX_Year_Frac];
				
				if (cashflow[CASHFLOW_INDEX_Settle]==t[TIME_GRID_MTM])
					settlement_cashflow = payment;
				
				out += payment * exp ( - discount_rate * yT );
			}
			
			//handle settlement - note we allow for physical settlement - the minus sign means this is a forward (as opposed to regular cash settled)
			if (-cashflow[CASHFLOW_INDEX_Settle]==t[TIME_GRID_MTM])
				settlement_cashflow = out;
					
			return out;
		}

		// Function used to PV a sequence of fixed cashflows - assuming a growth curve - example CPI
		// Note here we interpolate the index and then grow it
		__device__ REAL ScenarioPVIndex (	const REAL* __restrict__ t,
											int   cashflow_method,
											int   mtm_time_index,
											int   num_cashflows,
											REAL& settlement_cashflow,
											const int*  __restrict__ cashflow_starttime_index,
											const REAL* __restrict__ cashflows,
											const REAL* __restrict__ resets,
											const int* __restrict__ ref_index,
											const int* __restrict__ growth_offset,
											const int* __restrict__ discount_offset,
											const OFFSET scenario_prior_index,
											const REAL* __restrict__ static_factors,
											const REAL* __restrict__ stoch_factors )
		{
			REAL out 						= 0.0;
			const OFFSET scenario_index 	= scenario_prior_index - reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );			
			
			//Calculate the value of the index today (given that this is actually a lagged observation)
			REAL Index_t					= Calc_Index ( t, cashflow_method, mtm_time_index, resets, ref_index, scenario_index, static_factors, stoch_factors );
			const REAL* cashflow	    	= NULL;
			settlement_cashflow				= 0.0;
			
			for (int i=cashflow_starttime_index[mtm_time_index]; i<num_cashflows; i++)
			{
				cashflow 				= cashflows+i*CASHFLOW_INDEX_Size;
				
				REAL Index_val_Base		= cashflow[CASHFLOW_INDEX_BaseReference] > 0 ?
											cashflow[CASHFLOW_INDEX_BaseReference] :
											Index_t / CalcDiscount ( t, -cashflow[CASHFLOW_INDEX_BaseReference], growth_offset, scenario_prior_index, static_factors, stoch_factors );
																
				REAL Index_val_Final 	= cashflow[CASHFLOW_INDEX_FinalReference] > 0 ?
											cashflow[CASHFLOW_INDEX_FinalReference] :
											Index_t / CalcDiscount ( t, -cashflow[CASHFLOW_INDEX_FinalReference], growth_offset, scenario_prior_index, static_factors, stoch_factors );
				
				REAL yT 			 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], discount_offset[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
				REAL discount_rate 	 = ScenarioCurve ( t, yT, discount_offset, scenario_prior_index, static_factors, stoch_factors );

				//equation 4.559 in the theory guide
				REAL payment		 = cashflow[CASHFLOW_INDEX_Nominal] * ( cashflow[CASHFLOW_INDEX_FixedAmt] * (Index_val_Final/Index_val_Base) + 0.0 ) * cashflow[CASHFLOW_INDEX_FixedRate] * cashflow[CASHFLOW_INDEX_Year_Frac];
				
				if (cashflow[CASHFLOW_INDEX_Settle]==t[TIME_GRID_MTM])
					settlement_cashflow = payment;
				
				out += payment * exp ( - discount_rate * yT );
			}
			
			//handle settlement - note we allow for physical settlement - the minus sign means this is a forward (as opposed to regular cash settled)
			if (-cashflow[CASHFLOW_INDEX_Settle]==t[TIME_GRID_MTM])
				settlement_cashflow = out;
					
			return out;
		}

		// Function used to PV a sequence of floating MTM cashflows - need to allow different compounding conventions 
		
		__device__ REAL ScenarioPVMTMFloatLeg (	const REAL* __restrict__ t,
												int   mtm_time_index,
												int   num_cashflows,
												int   num_static_cashflows,
												REAL& settlement_cashflow,
												const int*  __restrict__ mtm_currency,
												const int*  __restrict__ mtm_cashflow_starttime_index,
												const REAL* __restrict__ mtm_cashflows,
												const REAL* __restrict__ mtm_resets,
												const int*  __restrict__ mtm_forward,
												const int*  __restrict__ mtm_discount,
												const int*  __restrict__ static_currency,
												const int*  __restrict__ static_cashflow_starttime_index,
												const REAL* __restrict__ static_cashflows,
												const OFFSET scenario_index,
												const OFFSET scenario_prior_index,
												const REAL* __restrict__ static_factors,
												const REAL* __restrict__ stoch_factors )
		{
			//need to store state information for the various scenario curve lookups
			REAL 	fx_rate, fx_rate_next;

			REAL 	out 				= 0.0;
			int     mtm_start_cash 		= mtm_cashflow_starttime_index ? mtm_cashflow_starttime_index[mtm_time_index] : mtm_time_index;
			int     static_start_cash 	= static_cashflow_starttime_index ? static_cashflow_starttime_index[mtm_time_index] : mtm_time_index;
			
			const int* static_curve 	= CurrencyRepoCurve ( static_currency );
			const int* mtm_curve 		= CurrencyRepoCurve ( mtm_currency );

			settlement_cashflow	 		= 0.0;
			
			for (int i=mtm_start_cash; i<num_cashflows; i++)
			{
				const REAL* 	static_cashflow 		= static_cashflows + ( static_start_cash + i - mtm_start_cash ) * CASHFLOW_INDEX_Size;
				const REAL* 	next_static_cashflow 	= ( ( static_start_cash + i + 1 - mtm_start_cash ) < num_static_cashflows ) ?
															static_cashflows + ( static_start_cash + i + 1 - mtm_start_cash ) * CASHFLOW_INDEX_Size :
															NULL;
															
				const REAL* 	mtm_cashflow 			= mtm_cashflows + i*CASHFLOW_INDEX_Size;
				const REAL* 	next_mtm_cashflow 		= ( ( i + 1 ) < num_cashflows ) ?
															mtm_cashflows + (i+1)*CASHFLOW_INDEX_Size:
															NULL;
															
				const OFFSET	reset_offset 			= ( reinterpret_cast <const OFFSET &> ( mtm_cashflow[CASHFLOW_INDEX_ResetOffset] ) ) * RESET_INDEX_Size;
				const REAL* 	resets 					= mtm_resets + reset_offset;
				
				REAL ydT 			= calcDayCountAccrual ( mtm_cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], mtm_discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
				REAL discount_rate 	= ScenarioCurve ( t, ydT, mtm_discount, scenario_prior_index, static_factors, stoch_factors );
				REAL forward_rate  	= Calc_PastReset ( t, mtm_forward, mtm_cashflow, resets, scenario_prior_index, static_factors, stoch_factors );
				
				if ( mtm_cashflow[CASHFLOW_INDEX_Start_Day] >= t[TIME_GRID_MTM] )
				{
					fx_rate			= mtm_cashflow [CASHFLOW_INDEX_FXResetValue] ?
										mtm_cashflow [CASHFLOW_INDEX_FXResetValue] :
										ScenarioFXForward ( t, mtm_cashflow[CASHFLOW_INDEX_FXResetDate] - t[TIME_GRID_MTM], static_currency, mtm_currency, static_curve, mtm_curve, scenario_prior_index, static_factors, stoch_factors );
				}
				else
				{
					fx_rate			=	mtm_cashflow [CASHFLOW_INDEX_FXResetValue] ?
										mtm_cashflow [CASHFLOW_INDEX_FXResetValue] :
										ScenarioFXForward ( resets + RESET_INDEX_Time_Grid, resets[RESET_INDEX_Start_Day],
															static_currency, mtm_currency, static_curve, mtm_curve, 
															scenario_index + reinterpret_cast<const OFFSET &>(resets[RESET_INDEX_Scenario]),
															static_factors, stoch_factors );
				}
				
				fx_rate_next = next_mtm_cashflow ?
					ScenarioFXForward ( t, next_mtm_cashflow[CASHFLOW_INDEX_FXResetDate] - t[TIME_GRID_MTM], static_currency, mtm_currency, static_curve, mtm_curve, scenario_prior_index, static_factors, stoch_factors ) :
					0.0;
				
				REAL Pi 		= -static_cashflow [ CASHFLOW_INDEX_Nominal ] * fx_rate;
				REAL Pi_1 		= next_static_cashflow ? -next_static_cashflow [ CASHFLOW_INDEX_Nominal ] * fx_rate_next : 0.0;
				REAL Interest	= ( forward_rate + mtm_cashflow[CASHFLOW_INDEX_FloatMargin] ) * mtm_cashflow[CASHFLOW_INDEX_Year_Frac];
				REAL payment	= Interest * Pi + ( Pi - Pi_1 );
				
				//handle settlement
				if (mtm_cashflow[CASHFLOW_INDEX_Pay_Day]==t[TIME_GRID_MTM])
					settlement_cashflow = payment;
				
				out += payment * exp ( -discount_rate * ydT );
			}
			
			return out;
		}
		
		// Function used to PV a sequence of floating cashflows - need to allow more different compounding conventions - TODO.
		__device__ REAL ScenarioPVFloatLeg (	const REAL* __restrict__ t,
												int   cashflow_method,
												int   mtm_time_index,
												int   num_cashflows,
												REAL& settlement_cashflow,
												const int*  __restrict__ cashflow_starttime_index,
												const REAL* __restrict__ cashflows,
												const REAL* __restrict__ resets,
												const int* __restrict__ interest_rate_vol_offset,
												const int* __restrict__ forward_offset,
												const int* __restrict__ discount_offset,
												const OFFSET scenario_prior_index,
												const REAL* __restrict__ static_factors,
												const REAL* __restrict__ stoch_factors )
		{
			//need to store state information for the various scenario curve lookups
			REAL 	forward_rate;
			REAL 	total_interest = 0.0;
			
			REAL 	out 		= 0.0;
			int     start_cash 	= cashflow_starttime_index ? cashflow_starttime_index[mtm_time_index] : mtm_time_index;

			settlement_cashflow	 = 0.0;
			
			for (int i=start_cash; i<num_cashflows; i++)
			{
				const REAL* 	cashflow 	 = cashflows+i*CASHFLOW_INDEX_Size;
				const OFFSET	reset_offset = ( reinterpret_cast <const OFFSET &> ( cashflow[CASHFLOW_INDEX_ResetOffset] ) ) * RESET_INDEX_Size;

				REAL ydT 			= calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], discount_offset[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
				REAL discount_rate 	= ScenarioCurve ( t, ydT, discount_offset, scenario_prior_index, static_factors, stoch_factors );
				
				//need to be able to add more curve indexes here
				forward_rate  = ( cashflow[CASHFLOW_INDEX_Start_Day]>=t[TIME_GRID_MTM] ) ?
								Calc_FutureReset ( t, forward_offset, cashflow, resets + reset_offset,
													 scenario_prior_index, static_factors, stoch_factors ) :
								Calc_PastReset ( t, forward_offset, cashflow, resets + reset_offset,
													 scenario_prior_index, static_factors, stoch_factors );
													 
				REAL Margin				= cashflow[CASHFLOW_INDEX_FloatMargin] * cashflow[CASHFLOW_INDEX_Year_Frac];
				REAL Interest 			= forward_rate * cashflow[CASHFLOW_INDEX_Year_Frac] + Margin;
				
				switch ( cashflow_method ) {
					case CASHFLOW_METHOD_Compounding_Include_Margin:
						total_interest 		= Interest * cashflow [ CASHFLOW_INDEX_Nominal ] + (1.0+Interest) * total_interest ;
						break;
					case CASHFLOW_METHOD_Compounding_Flat:
						total_interest 		= Interest * cashflow [ CASHFLOW_INDEX_Nominal ] + (1.0 + ( Interest - Margin ) ) * total_interest ;
						break;
					default:
						//None
						total_interest += Interest * cashflow[CASHFLOW_INDEX_Nominal];
				}
				
				if ( cashflow[CASHFLOW_INDEX_Settle] )
				{
					REAL payment  =  total_interest + cashflow[CASHFLOW_INDEX_FixedAmt];
					
					//handle settlement
					if (cashflow[CASHFLOW_INDEX_Pay_Day]==t[TIME_GRID_MTM])
						settlement_cashflow = payment;
					
					out += payment * exp ( -discount_rate * ydT );
					
					total_interest = 0.0;
				}
			}
			
			return out;
		}

		/* Function used to PV a sequence of equity swaplets - uses the same params as ScenarioPVFloatLeg (and assumptions) */
		__device__ REAL ScenarioPVEquitySwapLeg(	const REAL* __restrict__ t,
													int   cashflow_method,
													int   mtm_time_index,
													int   num_cashflows,
													REAL& settlement_cashflow,
													const int*  __restrict__ cashflow_starttime_index,
													const REAL* __restrict__ cashflows,
													const REAL* __restrict__ resets,
													const int* __restrict__ equity,
													const int* __restrict__ equity_repo,
													const int* __restrict__ discount,
													const OFFSET scenario_prior_index,
													const REAL* __restrict__ static_factors,
													const REAL* __restrict__ stoch_factors )
		{
			//need to store state information for the various scenario curve lookups
			//note that here forward_offset can optionally refer to (for quanto deals):
			// -	the cashflow currency
			
			REAL out 						= 0.0;
			const OFFSET scenario_index		= scenario_prior_index - reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			
			//Bit of a hack this - we pack the dividend yield immediately after the equity factor
			const int* dividend_yield		= equity + FACTOR_INDEX_START + RATE_INDEX_Size;

			settlement_cashflow				= 0.0;
			
			for (int i=cashflow_starttime_index[mtm_time_index]; i<num_cashflows; i++)
			{
				const REAL* cashflow 		= cashflows+i*CASHFLOW_INDEX_Size;
				const OFFSET reset_offset 	= ( reinterpret_cast <const OFFSET &> ( cashflow[CASHFLOW_INDEX_ResetOffset] ) ) * RESET_INDEX_Size;				
				
				REAL ydT 			 		= calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
				REAL discount_rate 	 		= ScenarioCurve ( t, ydT, discount, scenario_prior_index, static_factors, stoch_factors );
				
				const REAL* reset_t0 		= resets + reset_offset;
				const REAL* reset_t1 		= resets + reset_offset + RESET_INDEX_Size;
				
				REAL payoff;

				if (t[TIME_GRID_MTM]<cashflow[CASHFLOW_INDEX_Start_Day])
				{
					REAL Ft0 		= Calc_EquityForward (  t, cashflow[CASHFLOW_INDEX_Start_Day], equity, equity_repo, dividend_yield, scenario_prior_index, static_factors, stoch_factors );
					REAL Ft1 		= Calc_EquityForward (  t, cashflow[CASHFLOW_INDEX_End_Day], equity, equity_repo, dividend_yield, scenario_prior_index, static_factors, stoch_factors );

					REAL Drt0		= CalcDiscount ( t, cashflow[CASHFLOW_INDEX_Start_Day], equity_repo, scenario_prior_index, static_factors, stoch_factors ); 
					REAL Drt1		= CalcDiscount ( t, cashflow[CASHFLOW_INDEX_End_Day], equity_repo, scenario_prior_index, static_factors, stoch_factors );

					payoff		 = (cashflow[CASHFLOW_INDEX_End_Mult] - cashflow[CASHFLOW_INDEX_Dividend_Mult]) * (Ft1/( cashflow_method==1 ? Ft0 : 1.0 )) + ( cashflow[CASHFLOW_INDEX_Dividend_Mult] * (Drt0/Drt1) -  cashflow[CASHFLOW_INDEX_Start_Mult] ) * ( cashflow_method==1 ? 1.0 : Ft0 );
				}
				else if (t[TIME_GRID_MTM]<cashflow[CASHFLOW_INDEX_End_Day])
				{
					//Stock price at the start of the period
					REAL St0 		= ( reset_t0[RESET_INDEX_Start_Day] > 0 ) ? ScenarioRate ( reset_t0 + RESET_INDEX_Time_Grid, equity, scenario_index + reinterpret_cast<const OFFSET &>(reset_t0[RESET_INDEX_Scenario]) , static_factors, stoch_factors ) : reset_t0[RESET_INDEX_Value];
					REAL St 		= ScenarioRate ( t, equity, scenario_prior_index, static_factors, stoch_factors );
					REAL Ft1 		= Calc_EquityForward ( t, cashflow[CASHFLOW_INDEX_End_Day], equity, equity_repo, dividend_yield, scenario_prior_index, static_factors, stoch_factors );
					REAL Drt1		= CalcDiscount ( t, cashflow[CASHFLOW_INDEX_End_Day], equity_repo, scenario_prior_index, static_factors, stoch_factors );
					
					//Note we store past dividends in the RESET_INDEX_Weight field
					REAL Ht0_t		= Calc_EquityRealizedDividends ( reset_t0 + RESET_INDEX_Time_Grid, t[TIME_GRID_MTM], reset_t0[RESET_INDEX_Weight], equity, equity_repo, dividend_yield, scenario_index + reinterpret_cast<const OFFSET &>(reset_t0[RESET_INDEX_Scenario]), static_factors, stoch_factors );

					payoff		 	= ( (cashflow[CASHFLOW_INDEX_End_Mult] - cashflow[CASHFLOW_INDEX_Dividend_Mult]) * Ft1 + cashflow[CASHFLOW_INDEX_Dividend_Mult]*( St + Ht0_t ) / Drt1 - cashflow[CASHFLOW_INDEX_Start_Mult]*St0 ) / ( cashflow_method==1 ? St0 : 1.0 );
				}
				else
				{
					REAL  St0 		= ( reset_t0[RESET_INDEX_Start_Day] > 0 ) ? ScenarioRate ( reset_t0 + RESET_INDEX_Time_Grid, equity, scenario_index + reinterpret_cast<const OFFSET &>(reset_t0[RESET_INDEX_Scenario]) , static_factors, stoch_factors ) : reset_t0[RESET_INDEX_Value];
					REAL  St1 		= ( reset_t1[RESET_INDEX_Start_Day] > 0 ) ? ScenarioRate ( reset_t1 + RESET_INDEX_Time_Grid, equity, scenario_index + reinterpret_cast<const OFFSET &>(reset_t1[RESET_INDEX_Scenario]) , static_factors, stoch_factors ) : reset_t1[RESET_INDEX_Value];
					
					//Note we store past dividends in the RESET_INDEX_Weight field
					REAL Ht0_t1		= Calc_EquityRealizedDividends ( reset_t0 + RESET_INDEX_Time_Grid, cashflow[CASHFLOW_INDEX_End_Day], reset_t0[RESET_INDEX_Weight], equity, equity_repo, dividend_yield, scenario_index + reinterpret_cast<const OFFSET &>(reset_t0[RESET_INDEX_Scenario]), static_factors, stoch_factors );
					
					payoff        	= ( cashflow[CASHFLOW_INDEX_End_Mult] * St1 - cashflow[CASHFLOW_INDEX_Start_Mult] * St0 + cashflow[CASHFLOW_INDEX_Dividend_Mult]*Ht0_t1 ) / ( cashflow_method==1 ? St0 : 1.0 );
				}
				
				REAL payment  = cashflow[CASHFLOW_INDEX_FixedAmt] * payoff;
				
				//handle settlement - note - need to include dividend payment if Terminal
				if (cashflow[CASHFLOW_INDEX_Pay_Day]==t[TIME_GRID_MTM])
					settlement_cashflow += payment;
				
				out += payment * exp ( -discount_rate * ydT );
			}
			
			return out;
		}

		/* Function used to PV a sequence of energy forwards using reference forward prices - uses the same params as ScenarioPVFloatLeg (and assumptions) */
		__device__ REAL ScenarioPVEnergy(	const REAL* __restrict__ t,
											int   cashflow_method,
											int   mtm_time_index,
											int   num_cashflows,
											REAL& settlement_cashflow,
											const int*  __restrict__ cashflow_starttime_index,
											const REAL* __restrict__ cashflows,
											const REAL* __restrict__ resets,
											const int* __restrict__ interest_rate_vol_offset,
											const int* __restrict__ forward_offset,
											const int* __restrict__ discount_offset,
											const OFFSET scenario_prior_index,
											const REAL* __restrict__ static_factors,
											const REAL* __restrict__ stoch_factors )
		{
			//need to store state information for the various scenario curve lookups
			//note that here forward_offset can optionally refer to (for quanto deals):
			// -	the cashflow currency
			// -	forward price currency (should be different to the cashflow currency)
			
			REAL out 						= 0.0;
			REAL dummy_weights				= 0.0;			
			settlement_cashflow				= 0.0;
			
			for (int i=cashflow_starttime_index[mtm_time_index]; i<num_cashflows; i++)
			{
				const REAL* cashflow 		= cashflows+i*CASHFLOW_INDEX_Size;
				const OFFSET reset_offset 	= ( reinterpret_cast <const OFFSET &> ( cashflow[CASHFLOW_INDEX_ResetOffset] ) ) * RESET_INDEX_Size;
				
				REAL ydT 			 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], discount_offset[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
				REAL discount_rate 	 = ScenarioCurve ( t, ydT, discount_offset, scenario_prior_index, static_factors, stoch_factors );
				int  num_resets		 = reinterpret_cast <const OFFSET &> (cashflow[CASHFLOW_INDEX_NumResets]);
				
				REAL payoff;
				
				if ( cashflow[CASHFLOW_INDEX_Start_Day]>t[TIME_GRID_MTM] )
				{
					payoff        = Calc_ForwardPrice ( t, 0, num_resets, dummy_weights, forward_offset, resets + reset_offset, scenario_prior_index, static_factors, stoch_factors );
				}
				else
				{
					payoff		 = Calc_PastPrice ( t, num_resets, forward_offset, resets + reset_offset, scenario_prior_index, static_factors, stoch_factors );
				}				

				REAL payment  = cashflow[CASHFLOW_INDEX_Nominal] * (cashflow[CASHFLOW_INDEX_Start_Mult] * payoff + cashflow[CASHFLOW_INDEX_FloatMargin]);
				
				//handle settlement
				if (cashflow[CASHFLOW_INDEX_Pay_Day]==t[TIME_GRID_MTM])
					settlement_cashflow += payment;
				
				out += payment * exp ( -discount_rate * ydT );
			}
			
			return out;
		}
		
		/* Function used to PV a sequence of caplets but using a reference forward curve - uses the same params as ScenarioPVFloatLeg (and assumptions) */

		__device__ REAL ScenarioPVCap(	const REAL* __restrict__ t,
										int   cashflow_method,
										int   mtm_time_index,
										int   num_cashflows,
										REAL& settlement_cashflow,
										const int*  __restrict__ cashflow_starttime_index,
										const REAL* __restrict__ cashflows,
										const REAL* __restrict__ resets,
										const int* __restrict__ interest_rate_vol_offset,
										const int* __restrict__ forward_offset,
										const int* __restrict__ discount_offset,
										const OFFSET scenario_prior_index,
										const REAL* __restrict__ static_factors,
										const REAL* __restrict__ stoch_factors )
		{
			//need to store state information for the various scenario curve lookups
			REAL out = 0.0;
			REAL forward_rate;
			REAL vol, expiry;
			
			settlement_cashflow  = 0.0;
			
			for (int i=cashflow_starttime_index[mtm_time_index]; i<num_cashflows; i++)
			{
				const REAL* cashflow 		= cashflows+i*CASHFLOW_INDEX_Size;
				const OFFSET reset_offset 	= ( reinterpret_cast <const OFFSET &> ( cashflow[CASHFLOW_INDEX_ResetOffset] ) ) * RESET_INDEX_Size;
				
				REAL ydT 			 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], discount_offset[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
				REAL discount_rate 	 = ScenarioCurve ( t, ydT, discount_offset, scenario_prior_index, static_factors, stoch_factors );
				REAL payoff;
				
				if ( cashflow[CASHFLOW_INDEX_Start_Day]>t[TIME_GRID_MTM] )
				{
					forward_rate = Calc_FutureReset ( t, forward_offset, cashflow, resets + reset_offset, scenario_prior_index, static_factors, stoch_factors );
					expiry 		 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Start_Day] - t[TIME_GRID_MTM], discount_offset[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
					vol 		 = ScenarioSurface3D  ( t, forward_rate - cashflow[CASHFLOW_INDEX_Strike], expiry, cashflow[CASHFLOW_INDEX_Year_Frac], interest_rate_vol_offset, scenario_prior_index, static_factors, stoch_factors );
					payoff		 = blackEuropeanOption ( forward_rate, cashflow[CASHFLOW_INDEX_Strike], 0, vol, expiry, 1, 1 );
				}
				else
				{
					payoff		 = max ( Calc_PastReset ( t, forward_offset, cashflow, resets + reset_offset, scenario_prior_index, static_factors, stoch_factors ) - cashflow[CASHFLOW_INDEX_Strike], 0.0);
				}
				
				REAL payment  = cashflow[CASHFLOW_INDEX_Nominal] * payoff * cashflow[CASHFLOW_INDEX_Year_Frac];
				
				//handle settlement
				if (cashflow[CASHFLOW_INDEX_Pay_Day]==t[TIME_GRID_MTM])
					settlement_cashflow += payment;
				
				out += payment * exp ( -discount_rate * ydT );
			}
			
			return out;
		}
		
		/* Function used to PV a sequence of floorlets but using a reference forward curve - uses the same params as ScenarioPVFloatLeg (and assumptions) */
		
		__device__ REAL ScenarioPVFloor(	const REAL* __restrict__ t,
											int   cashflow_method,
											int   mtm_time_index,
											int   num_cashflows,
											REAL& settlement_cashflow,
											const int*  __restrict__ cashflow_starttime_index,
											const REAL* __restrict__ cashflows,
											const REAL* __restrict__ resets,
											const int* __restrict__ interest_rate_vol_offset,
											const int* __restrict__ forward_offset,
											const int* __restrict__ discount_offset,
											const OFFSET scenario_prior_index,
											const REAL* __restrict__ static_factors,
											const REAL* __restrict__ stoch_factors )
		{
			//need to store state information for the various scenario curve lookups
			REAL out = 0.0;
			REAL forward_rate;
			REAL vol, expiry;
			
			settlement_cashflow	= 0.0;
			
			for (int i=cashflow_starttime_index[mtm_time_index]; i<num_cashflows; i++)
			{
				const REAL* cashflow 		= cashflows+i*CASHFLOW_INDEX_Size;
				const OFFSET reset_offset 	= ( reinterpret_cast <const OFFSET &> ( cashflow[CASHFLOW_INDEX_ResetOffset] ) ) * RESET_INDEX_Size;
				
				REAL ydT 			 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], discount_offset[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
				REAL discount_rate 	 = ScenarioCurve ( t, ydT, discount_offset, scenario_prior_index, static_factors, stoch_factors );
				REAL payoff;
				
				if ( cashflow[CASHFLOW_INDEX_Start_Day]>t[TIME_GRID_MTM] )
				{
					forward_rate = Calc_FutureReset ( t, forward_offset, cashflow, resets + reset_offset, scenario_prior_index, static_factors, stoch_factors );
					expiry 		 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Start_Day] - t[TIME_GRID_MTM], discount_offset[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
					vol 		 = ScenarioSurface3D  ( t, cashflow[CASHFLOW_INDEX_Strike] - forward_rate, expiry, cashflow[CASHFLOW_INDEX_Year_Frac], interest_rate_vol_offset, scenario_prior_index, static_factors, stoch_factors );
					payoff		 = blackEuropeanOption ( forward_rate, cashflow[CASHFLOW_INDEX_Strike], 0, vol, expiry, 1, -1 );
				}
				else
				{
					payoff		 = max( cashflow[CASHFLOW_INDEX_Strike] - Calc_PastReset ( t, forward_offset, cashflow, resets + reset_offset, scenario_prior_index, static_factors, stoch_factors ), 0.0 );
				}
				
				REAL payment  = cashflow[CASHFLOW_INDEX_Nominal] * payoff * cashflow[CASHFLOW_INDEX_Year_Frac];
				
				//handle settlement
				if (cashflow[CASHFLOW_INDEX_Pay_Day]==t[TIME_GRID_MTM])
					settlement_cashflow += payment;
				
				out += payment * exp ( -discount_rate * ydT );
			}
			
			return out;
		}

		__device__ void ScenarioSettle ( REAL settlement_cashflow, const int* __restrict__ currency, int scenario_index, int mtm_time_index,
										 const int* __restrict__ cashflow_index, REAL* all_cashflows )
		{
			if (all_cashflows)
			{
				int currency_id		= LookupRate( currency, CurrencySettlementMap, NumSettlementCurrencies );
				int cashflow_offset = scenario_index * CurrencySettlementOffset[NumSettlementCurrencies] + CurrencySettlementOffset[currency_id] + cashflow_index[ MTMTimeSteps*currency_id + mtm_time_index ];
				all_cashflows[cashflow_offset] += settlement_cashflow;
			}
		}

		__device__ __constant__ FloatPV FloatScenario[6] = { ScenarioPVFloatLeg, ScenarioPVCap, ScenarioPVFloor, ScenarioPVEnergy, ScenarioPVIndex, ScenarioPVEquitySwapLeg };
			
		__global__ void AddFixedCashflow(	const REAL* __restrict__ stoch_factors,
											const REAL* __restrict__ static_factors,
											const int*  __restrict__ deal_time_index,
											const REAL* __restrict__ time_grid,
											const int*  __restrict__ cashflow_index,
											REAL* all_cashflows,
											REAL* Output,
											int  overwrite,
											int  compounding,
											int num_fixed_cashflow,
											const REAL* __restrict__ fixed_cashflow,
											const int*  __restrict__ fixed_starttime_index,
											const int* __restrict__ currency,
											const int* __restrict__ discount )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];			
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
			OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			
			//set the mtm of this deal
			REAL mtm 					= overwrite ? 0.0 : Output[ index ];
			//Is there a cashflow settlement here?
			REAL settlement_cashflow    = 0.0;
			
			//work out the remaining term
			int from_fixed_index = fixed_starttime_index[mtm_time_index];

			if ( from_fixed_index < num_fixed_cashflow )
			{
				//Get the reporting currency
				REAL FX_Base	= ScenarioRate ( t, currency, scenario_prior_index, static_factors, stoch_factors ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
				
				//PV the fixed leg
				REAL fixed_leg	= ScenarioPVFixedLeg ( t, CASHFLOW_METHOD_Fixed_Rate_Standard, compounding, fixed_starttime_index[mtm_time_index], num_fixed_cashflow,
														settlement_cashflow, fixed_cashflow,  discount, scenario_prior_index, static_factors, stoch_factors );
														
				//Settle any cashflow
				if (settlement_cashflow!=0)
					ScenarioSettle(settlement_cashflow, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows);

				//convert to the settlement currency
				mtm				+= FX_Base * fixed_leg;
			}			
			
			Output [ index ] = mtm ;
		}
		
		__global__ void AddIndexCashflow(	const REAL* __restrict__ stoch_factors,
											const REAL* __restrict__ static_factors,
											const int*  __restrict__ deal_time_index,
											const REAL* __restrict__ time_grid,
											const int*  __restrict__ cashflow_index,
											REAL* all_cashflows,
											REAL* Output,
											int   CashflowMethod,
											int   overwrite,
											int   num_index_cashflow,
											const REAL* __restrict__ index_cashflow,
											const int*  __restrict__ index_starttime_index,
											const REAL* __restrict__ time_resets,
											const REAL* __restrict__ base_resets,
											const REAL* __restrict__ final_resets,
											const int* __restrict__ currency,
											const int* __restrict__ interest_rate_vol,
											const int* __restrict__ forward,
											const int* __restrict__ discount )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
			OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			
			//set the mtm of this deal
			REAL mtm 					= overwrite ? 0.0 : Output[ index ];
			//Is there a cashflow settlement here?
			REAL settlement_cashflow    = 0.0;
			
			if ( index_starttime_index[mtm_time_index] < num_index_cashflow )
			{
				//Get the settlement currency
				REAL FX_Base	= ScenarioRate ( t, currency, scenario_prior_index, static_factors, stoch_factors ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
										
				//PV the index leg
				REAL index_leg	= ScenarioPVBondIndex ( t, CashflowMethod, mtm_time_index, num_index_cashflow, settlement_cashflow, index_starttime_index, index_cashflow,
														time_resets, base_resets, final_resets, interest_rate_vol, forward, discount, scenario_prior_index, static_factors, stoch_factors );
														
				/*
				if (blockIdx.x==0 && threadIdx.x==0 && blockIdx.y>=gridDim.y-10)
				{
					const REAL* last_published		= time_resets + mtm_time_index * RESET_INDEX_Size;			
					const REAL* cashflow	    	= NULL;
					REAL last_published_index		= ScenarioRate ( t, interest_rate_vol, scenario_prior_index, static_factors, stoch_factors );
					settlement_cashflow				= 0.0;
					REAL out 						= 0.0;
					
					for (int i=index_starttime_index[mtm_time_index]; i<num_index_cashflow; i++)
					{
						const REAL* 	cashflow 		= index_cashflow+i*CASHFLOW_INDEX_Size;
						
						REAL Index_val_Base		= cashflow[CASHFLOW_INDEX_BaseReference] > 0 ?
													cashflow[CASHFLOW_INDEX_BaseReference] :
													Calc_Bond_Index ( t, CashflowMethod, i, last_published_index, last_published, base_resets, interest_rate_vol, forward, scenario_prior_index, static_factors, stoch_factors );
																		
						REAL Index_val_Final 	= cashflow[CASHFLOW_INDEX_FinalReference] > 0 ?
													cashflow[CASHFLOW_INDEX_FinalReference] :
													Calc_Bond_Index ( t, CashflowMethod, i, last_published_index, last_published, final_resets, interest_rate_vol, forward, scenario_prior_index, static_factors, stoch_factors );
						
						REAL yT 			 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
						REAL discount_rate 	 = ScenarioCurve ( t, yT, discount, scenario_prior_index, static_factors, stoch_factors );

						//equation 4.559 in the theory guide
						REAL payment		 = cashflow[CASHFLOW_INDEX_Nominal] * ( cashflow[CASHFLOW_INDEX_FixedAmt] * (Index_val_Final/Index_val_Base) + 0.0 ) * cashflow[CASHFLOW_INDEX_FixedRate] * cashflow[CASHFLOW_INDEX_Year_Frac];
						
						out += payment * exp ( - discount_rate * yT );
						
						printf("cashflow,%d,start_day,%g,nominal,%.4f,fixed_amt,%.4f,fixed_rate,%.4f,Index,%.4f,Index_val_Base,%.5f,Index_val_Final,%.5f,tenor,%.4f,payment,%.4f\\n",i,cashflow[CASHFLOW_INDEX_Start_Day],cashflow[CASHFLOW_INDEX_Nominal],cashflow[CASHFLOW_INDEX_FixedAmt],cashflow[CASHFLOW_INDEX_FixedRate], last_published_index,Index_val_Base,Index_val_Final,cashflow[CASHFLOW_INDEX_Year_Frac],payment);
					}
				}
				*/
				
				//Settle any cashflow
				if (settlement_cashflow!=0)
					ScenarioSettle(settlement_cashflow, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows);

				//convert to the settlement currency
				mtm += FX_Base * index_leg;
			}
			
			Output [ index ] = mtm ;			
		}

		__global__ void AddFloatCashflow(	const REAL* __restrict__ stoch_factors,
											const REAL* __restrict__ static_factors,
											const int*  __restrict__ deal_time_index,
											const REAL* __restrict__ time_grid,
											const int*  __restrict__ cashflow_index,
											REAL* all_cashflows,
											REAL* Output,
											int   CashflowScenario,
											int   CashflowMethod,
											int   overwrite,
											int   num_float_cashflow,
											const REAL* __restrict__ float_cashflow,
											const int*  __restrict__ float_starttime_index,
											const REAL* __restrict__ resets,
											const int* __restrict__ currency,
											const int* __restrict__ interest_rate_vol,
											const int* __restrict__ forward,
											const int* __restrict__ discount )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
			OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			
			//set the mtm of this deal
			REAL mtm 					= overwrite ? 0.0 : Output[ index ];
			//Is there a cashflow settlement here?
			REAL settlement_cashflow    = 0.0;
			
			if ( float_starttime_index[mtm_time_index] < num_float_cashflow )
			{
				//Get the settlement currency
				REAL FX_Base	= ScenarioRate ( t, currency, scenario_prior_index, static_factors, stoch_factors ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
										
				//PV the float leg
				REAL float_leg	= FloatScenario[CashflowScenario] ( t, CashflowMethod, mtm_time_index, num_float_cashflow, settlement_cashflow, float_starttime_index,
																	float_cashflow, resets, interest_rate_vol, forward, discount, scenario_prior_index, static_factors, stoch_factors );
				/*													
				if ( CashflowScenario==1 && blockIdx.x==0 && threadIdx.x==1 )
				{
					REAL out = 0.0;
					REAL forward_rate;
					REAL vol, expiry;
					
					for (int i=float_starttime_index[mtm_time_index]; i<num_float_cashflow; i++)
					{
						const REAL* cashflow 		= float_cashflow+i*CASHFLOW_INDEX_Size;
						const OFFSET reset_offset 	= ( reinterpret_cast <const OFFSET &> ( cashflow[CASHFLOW_INDEX_ResetOffset] ) ) * RESET_INDEX_Size;
						
						REAL ydT 			 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
						REAL discount_rate 	 = ScenarioCurve ( t, ydT, discount, scenario_prior_index, static_factors, stoch_factors );
						REAL payoff;
						
						if ( cashflow[CASHFLOW_INDEX_Start_Day]>t[TIME_GRID_MTM] )
						{
							forward_rate = Calc_FutureReset ( t, forward, cashflow, resets + reset_offset, scenario_prior_index, static_factors, stoch_factors );
							expiry 		 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Start_Day] - t[TIME_GRID_MTM], discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
							vol 		 = ScenarioSurface3D  ( t, forward_rate - cashflow[CASHFLOW_INDEX_Strike], expiry, cashflow[CASHFLOW_INDEX_Year_Frac], interest_rate_vol, scenario_prior_index, static_factors, stoch_factors );
							payoff		 = blackEuropeanOption ( forward_rate, cashflow[CASHFLOW_INDEX_Strike], 0, vol, expiry, 1, 1 );
						}
						else
						{
							payoff		 = max ( Calc_PastReset ( t, forward, cashflow, resets + reset_offset, scenario_prior_index, static_factors, stoch_factors ) - cashflow[CASHFLOW_INDEX_Strike], 0.0);
						}
						
						REAL payment  = cashflow[CASHFLOW_INDEX_Nominal] * payoff * cashflow[CASHFLOW_INDEX_Year_Frac];
						
						printf ( "cashflow,%d,start_day,%g,nominal,%.4f,output,%.4f,forward_rate,%.4f,mtm_time_index,%d,vol,%.4f\\n", i, cashflow[CASHFLOW_INDEX_Start_Day], cashflow[CASHFLOW_INDEX_Nominal], out, forward_rate, mtm_time_index, vol );
						
						out += payment * exp ( -discount_rate * ydT );
					}
				}

				if (CashflowScenario==0 && blockIdx.x==0 && threadIdx.x==0)
				{
					REAL out 						= 0.0;
					const OFFSET scenario_index 	= scenario_prior_index - reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
					
					
					for (int i=float_starttime_index[mtm_time_index]; i<num_float_cashflow; i++)
					{
						const REAL* 	cashflow 		= float_cashflow+i*CASHFLOW_INDEX_Size;
						const OFFSET	reset_offset = ( reinterpret_cast <const OFFSET &> ( cashflow[CASHFLOW_INDEX_ResetOffset] ) ) * RESET_INDEX_Size;
						
						REAL yT 			 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
						REAL discount_rate 	 = ScenarioCurve ( t, yT, discount, scenario_prior_index, static_factors, stoch_factors );
						
						REAL forward_rate  = ( cashflow[CASHFLOW_INDEX_Start_Day]>=t[TIME_GRID_MTM] ) ?
												Calc_FutureReset ( t, forward, cashflow, resets + reset_offset,
																	 scenario_prior_index, static_factors, stoch_factors ) :
												Calc_PastReset ( t, forward, cashflow, resets + reset_offset,
																	 scenario_prior_index, static_factors, stoch_factors );

						REAL Margin				= cashflow[CASHFLOW_INDEX_FloatMargin] * cashflow[CASHFLOW_INDEX_Year_Frac];
						REAL Interest 			= forward_rate * cashflow[CASHFLOW_INDEX_Year_Frac] + Margin;
						
						//out += payment * exp ( - discount_rate * yT );
						
						printf("cashflow,%d,start_day,%g,nominal,%.4f,fixed_amt,%.4f,fixed_rate,%.4f,forward_rate,%.8f,Interest,%.5f,tenor,%.4f\\n",i,cashflow[CASHFLOW_INDEX_Start_Day],cashflow[CASHFLOW_INDEX_Nominal],cashflow[CASHFLOW_INDEX_FixedAmt],cashflow[CASHFLOW_INDEX_FixedRate], forward_rate,Interest,cashflow[CASHFLOW_INDEX_Year_Frac]);
					}
				}
				
				if (CashflowScenario==4 && blockIdx.x==0 && threadIdx.x==0)
				{
					REAL out 						= 0.0;
					const OFFSET scenario_index 	= scenario_prior_index - reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
					
					//Calculate the value of the index today (given that this is actually a lagged observation)
					REAL Index_t					= Calc_Index ( t, CashflowMethod, mtm_time_index, resets, interest_rate_vol, scenario_index, static_factors, stoch_factors );
					
					for (int i=float_starttime_index[mtm_time_index]; i<num_float_cashflow; i++)
					{
						const REAL* 	cashflow 		= float_cashflow+i*CASHFLOW_INDEX_Size;
						
						REAL Index_val_Base		= cashflow[CASHFLOW_INDEX_BaseReference] > 0 ?
													cashflow[CASHFLOW_INDEX_BaseReference] :
													Index_t/CalcDiscount ( t, -cashflow[CASHFLOW_INDEX_BaseReference], forward, scenario_prior_index, static_factors, stoch_factors );
																		
						REAL Index_val_Final 	= cashflow[CASHFLOW_INDEX_FinalReference] > 0 ?
													cashflow[CASHFLOW_INDEX_FinalReference] :
													Index_t/CalcDiscount ( t, -cashflow[CASHFLOW_INDEX_FinalReference], forward, scenario_prior_index, static_factors, stoch_factors );
						
						REAL yT 			 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
						REAL discount_rate 	 = ScenarioCurve ( t, yT, discount, scenario_prior_index, static_factors, stoch_factors );

						//equation 4.559 in the theory guide
						REAL payment		 = cashflow[CASHFLOW_INDEX_Nominal] * ( cashflow[CASHFLOW_INDEX_FixedAmt] * (Index_val_Final/Index_val_Base) + 0.0 ) * cashflow[CASHFLOW_INDEX_FixedRate] * cashflow[CASHFLOW_INDEX_Year_Frac];
						
						out += payment * exp ( - discount_rate * yT );
						
						printf("cashflow,%d,start_day,%g,nominal,%.4f,fixed_amt,%.4f,fixed_rate,%.4f,Index,%.4f,Index_val_Base,%.5f,Index_val_Final,%.5f,tenor,%.4f,payment,%.4f\\n",i,cashflow[CASHFLOW_INDEX_Start_Day],cashflow[CASHFLOW_INDEX_Nominal],cashflow[CASHFLOW_INDEX_FixedAmt],cashflow[CASHFLOW_INDEX_FixedRate], Index_t,Index_val_Base,Index_val_Final,cashflow[CASHFLOW_INDEX_Year_Frac],payment);
					}
				}
				*/
				
				//Settle any cashflow
				if (settlement_cashflow!=0)
					ScenarioSettle(settlement_cashflow, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows);

				//convert to the settlement currency
				mtm += FX_Base * float_leg;
			}
			
			Output [ index ] = mtm ;			
		}
		
		//Interpolate missing MTM numbers - Note that this must not be called if interp_time_index is null
		__global__ void MTM_Interpolate(const int*  __restrict__ deal_interp_index,
										const REAL* __restrict__ time_grid,
										REAL* Output)
		{
			const int* interp_time_index	= deal_interp_index + INTERP_GRID_Size*blockIdx.y;
			int   previous_pillar			= interp_time_index[0];
			int   next_pillar				= interp_time_index[0]+interp_time_index[1];
			
			//calculate the base Output index
			int	 base_index  				= ( blockIdx.x*blockDim.x + threadIdx.x ) * MTMTimeSteps;
			REAL mtm_prior					= Output[base_index+previous_pillar];
			REAL mtm_post					= Output[base_index+next_pillar];
			REAL prev_timepoint				= time_grid[TIME_GRID_Size*previous_pillar+TIME_GRID_MTM];
			REAL delta_time					= time_grid[TIME_GRID_Size*next_pillar+TIME_GRID_MTM]-prev_timepoint;
			
			for (int i=previous_pillar+1; i<next_pillar; i++)
			{
				REAL w = (time_grid[TIME_GRID_Size*i+TIME_GRID_MTM]-prev_timepoint)/delta_time;
				Output[base_index+i] = mtm_prior*(1.0-w)+mtm_post*w;
			}
		}
	'''

	def __init__(self, config, prec = np.float32):
		'''
		Construct a new calculation - all calculations must setup the GPU ( by setting the precision )
		and by specifying a configuration (market data + porfolio)
		'''
		
		self.config    				= config
		self.prec      				= prec
		
		#this is, of course assuming that there are only 2 choices - float32 and float64
		self.indexType      		= np.int32 if self.prec == np.float32 else np.int64
		self.time_grid				= None
		self.CudaMem				= None
		
		#the risk factor data
		self.stoch_factors			= OrderedDict()
		self.static_factors			= OrderedDict()
		self.static_ofs				= {}
		self.stoch_ofs				= {}
		self.all_factors			= {}
		self.all_tenors				= {}
		
		#the tenor and buffer data
		self.buffer					= None
		self.tenor_size				= None
		self.tenor_offset			= None
		self.currency_curve_map   	= None
		self.currency_curve_ofs   	= None
		self.max_buffer_size 		= 30000
		
		#the deal structure
		self.netting_sets			= None
		
		#generate slides from calculation?
		self.slides					= None
		#what's the name of this calculation?
		self.name					= 'Unnamed'

	def FiniteDifferences(self, factors, params, RelativeBumps=False, discount_curves=None):

		def	calc_approx_fva( exp_mtm, time_grid, spread, delta_t, collateral ):
			return np.sum([ spread[s] * mtm * np.exp( -( t/365.0 ) * collateral.CurrentVal ( t/365.0 ) ) for mtm,s,t in zip( exp_mtm, delta_t, time_grid ) ])
			
		#first establish the base (no changes) - Don't free the cuda memory
		base_results = self.Execute ( params, FreeMem=False)
		
		#get the raw (including negative) mtm
		mtms 		 = base_results['Netting'].sub_structures[0].partition.DealMTMs.astype(np.float64)
		base_emtm	 = mtms.mean(axis=0)
		results		 = {}
		bump_amount	 = 0.01
		calc_objs	 = []
		
		if discount_curves:
			time_grid      = base_results['Netting'].sub_structures[0].obj.Time_dep.deal_time_grid
			days 	  	   = self.time_grid.time_grid[:,1][time_grid]
			funding        = ConstructFactor('InterestRate', self.config.params['Price Factors'][discount_curves['funding']])
			collateral     = ConstructFactor('InterestRate', self.config.params['Price Factors'][discount_curves['collateral']])
			delta_t        = np.diff(days)
			spread         = {}
			for s in np.unique(delta_t):
				spread[s] = np.exp ( (s/365.0) * ( funding.CurrentVal(s/365.0) - collateral.CurrentVal(s/365.0) ) ) - 1.0
		
		#get the underlying factors 
		for index, factor in enumerate(factors):
			try:
				#first get the stochastic object
				stochastic_obj = self.stoch_factors[factor]
				#now get the factor
				factor_obj = stochastic_obj.factor
				#reset it
				factor_obj.Bump(0.0, relative=RelativeBumps)
				#store it
				calc_objs.append ( ( factor_obj , stochastic_obj ) )
			except:
				logging.warning('skipping finite differences for {0} - not part of calculation (or not "Bumpable")'.format(factor))
				
		approx_derivatives = {'f': calc_approx_fva ( base_emtm, days, spread, delta_t, collateral ) if discount_curves else base_emtm }
		
		if calc_objs:
			
			#1 basis point bumps
			for bumps in [x for x in itertools.product(*[[-bump_amount, 0.0, bump_amount]]*len(factors)) if np.any(x)]:
				deltas = []
				for index, ( factor_obj , stochastic_obj ) in enumerate(calc_objs):
					try:
						#bump it
						factor_obj.Bump(bumps[index], relative=RelativeBumps)
						#update the stochastic factor (if necessary)
						stochastic_obj.PreCalc(self.base_date, self.time_grid)
						#store the delta for this factor
						deltas.append(factor_obj.GetDelta())
					except:
						logging.warning('skipping finite differences for {0} - not part of calculation (or not "Bumpable")'.format(factor))
						
				#make sure the netting sets are clean before a raw_execution		
				self.netting_sets.ResetStructure()
				#execute with the existing cuda memory and netting set state
				self.RawExecute ( params )
				#get the results
				bumped_mtms 	= self.netting_sets.sub_structures[0].partition.DealMTMs.astype(np.float64).mean(axis=0)
				results[bumps]	= (deltas, calc_approx_fva ( bumped_mtms, days, spread, delta_t, collateral ) if discount_curves else bumped_mtms)

			if len(factors)==1:
				#single case, return the first and second derivatives
				h_times_2 = (results[(bump_amount,)][0][0]-results[(-bump_amount,)][0][0])
				approx_derivatives['f_x'] = (results[(bump_amount,)][1]-results[(-bump_amount,)][1])/h_times_2
				approx_derivatives['f_xx'] = (results[(bump_amount,)][1] - 2.0*approx_derivatives['f'] + results[(-bump_amount,)][1])/np.square(h_times_2/2.0)
			else:
				h_times_2 = (results[(bump_amount,0.0)][0][0]-results[(-bump_amount,0.0)][0][0])
				k_times_2 = (results[(0.0,bump_amount)][0][1]-results[(0.0,-bump_amount)][0][1])
				
				approx_derivatives['f_x']  = (results[(bump_amount,0.0)][1]-results[(-bump_amount,0.0)][1])/h_times_2
				approx_derivatives['f_xx'] = (results[(bump_amount,0.0)][1] - 2.0*approx_derivatives['f'] + results[(-bump_amount,0.0)][1])/np.square(h_times_2/2.0)
				
				approx_derivatives['f_y']  = (results[(0.0,bump_amount)][1]-results[(0.0,-bump_amount)][1])/k_times_2
				approx_derivatives['f_yy'] = (results[(0.0,bump_amount)][1] - 2.0*approx_derivatives['f'] + results[(0.0,-bump_amount)][1])/np.square(k_times_2/2.0)
				
				approx_derivatives['f_xy'] = (results[(bump_amount,bump_amount)][1] - results[(bump_amount,-bump_amount)][1] - results[(-bump_amount,bump_amount)][1] + results[(-bump_amount,-bump_amount)][1])/(k_times_2*h_times_2)
			
		#now explicitly free the memory
		self.FreeCudaMem()
		
		return approx_derivatives

	def Execute(self, params):
		pass

	def RawExecute(self, params):
		pass
					 
	def DistinctTenors(self, distinct_tenors, key, factor):
		if key.type in Utils.OneDimensionalFactors:
			distinct_tenors.setdefault( tuple ( factor.GetTenor() ), set() ).add(key+(0,))			
		elif key.type in Utils.TwoDimensionalFactors:
			distinct_tenors.setdefault( tuple ( factor.GetMoneyness() ), set() ).add(key+(0,))
			distinct_tenors.setdefault( tuple ( factor.GetExpiry() ), set() ).add(key+(1,))
		elif key.type in Utils.ThreeDimensionalFactors:
			distinct_tenors.setdefault( tuple ( factor.GetMoneyness() ), set() ).add(key+(0,))
			distinct_tenors.setdefault( tuple ( factor.GetExpiry() ), set() ).add(key+(1,))
			distinct_tenors.setdefault( tuple ( factor.GetTenor() ), set() ).add(key+(2,))

	def UpdateTenors(self, distinct_tenors):
		total_points, self.all_tenors = 0, {}
		for index, (k,v) in enumerate(distinct_tenors.items()):
			total_points+=len(k)
			if total_points>self.max_buffer_size-1000:
				raise Exception('total of all distinct tenor points cannot exceed {0} points (requested {1})'.format(self.max_buffer_size, total_points) )
			for factor_element in v:
				factor, dim_index = Utils.Factor(factor_element[0], factor_element[1]), factor_element[-1]
				if factor.type in Utils.OneDimensionalFactors:
					stoch_factor = self.stoch_factors.get(factor)
					self.all_tenors.setdefault(factor, [0])[0] = index
					self.all_tenors[factor] += [stoch_factor.factor.GetDayCount() if stoch_factor else self.static_factors[factor].GetDayCount()]
				#this is a surface of some kind
				elif factor.type in Utils.TwoDimensionalFactors:
					self.all_tenors.setdefault(factor,[0,0])[dim_index] = index
				elif factor.type in Utils.ThreeDimensionalFactors:
					self.all_tenors.setdefault(factor,[0,0,0])[dim_index] = index
				else:
					raise Exception('Unknown {0} while updating distinnct tenors'.format(factor))
	
	def SetBuffer(self, distinct_tenors):
		#store the distinct tenors per curve				
		tenors 				= np.array( reduce(operator.concat, distinct_tenors.keys(), ()), dtype=self.prec)
		self.tenor_size 	= np.array( [len(x) for x in distinct_tenors.keys()], dtype=np.int32 )
		self.tenor_offset 	= np.hstack(([0],self.tenor_size)).cumsum()[:-1].astype(np.int32)

		currency_curve_map = {}
		for factor, ofs in self.static_ofs.items():
			if factor.type == 'FxRate':
				currency_curve_map[factor.name[0]]=-ofs-1
		for factor, ofs in self.stoch_ofs.items():
			if factor.type == 'FxRate':
				currency_curve_map[factor.name[0]]=ofs
				
		#store the repo curve per currency	
		curve_code 				= np.array ( [], dtype = np.int32 )
		currency_curve_ofs 		= []
		currency_curve_lookup  	= []
		
		for index, currency in sorted({v:k for k,v in currency_curve_map.items()}.items()):
			
			try:
				curve_currency_code = getFXZeroRateFactor ( Utils.CheckRateName(currency), self.static_ofs, self.stoch_ofs, self.all_tenors, self.all_factors )
			except:
				logging.error('Currency Curve for {0} is not present in calculation'.format(currency))
				continue
			
			currency_curve_lookup.append ( index )
			currency_curve_ofs.append ( len ( curve_code ) )
			curve_code = np.hstack ( ( curve_code, curve_currency_code ) )

		#make sure it can be viewed at the desired precision
		try:
			curve_offsets = curve_code.view(self.prec)
		except:
			curve_offsets = np.hstack ( ( curve_code, [0] ) ).view(self.prec)
			
		self.currency_curve_ofs = np.array(currency_curve_ofs, dtype=np.int32)
		self.currency_curve_map = np.array(currency_curve_lookup, dtype=np.int32)
		self.buffer = np.hstack ( ( tenors, curve_offsets ) )
			
		if self.buffer.size >self.max_buffer_size:
			raise Exception('There is more tenor/currency curve data than has been allocated on the GPU - please increase the max_buffer_size (or use more granular curve data)')
			
	def FreeCudaMem(self):
		#explicitly free up any memory used in a previous calculation
		if self.CudaMem is not None:
			for field in self.CudaMemClass._fields:
				memblock = getattr(self.CudaMem,field)
				if memblock:
					memblock.gpudata.free()

	def SetDealStructures(self, deals, output, num_scenarios, tagging=None, DealLevelMTM=False):
		patitioning = False
		for node in deals:
			#get the instrument
			instrument = node['instrument']
			#should we skip it?
			if node.get('Ignore')=='True':
				continue
			#apply a tag if provided (used for partitioning)
			if tagging:
				instrument.field['Tags'] = tagging(instrument.field)
				patitioning = True
			if node.get('Children'):
				struct = DealStructure(instrument, self.time_grid.CurrencyMap, num_scenarios, self.time_grid.mtm_time_grid.size, self.prec, DealLevelMTM)
				logging.info ( 'Analysing Group {0}'.format( instrument.field.get('Reference','<undefined>') ) )
				self.SetDealStructures ( node['Children'], struct, num_scenarios, tagging, DealLevelMTM )				
				output.AddStructureToStructure( struct, self.base_date, self.static_ofs, self.stoch_ofs, self.all_factors, self.all_tenors, self.time_grid, self.config.holidays )
				continue
			output.AddDealToStructure ( self.base_date, instrument, self.static_ofs, self.stoch_ofs, self.all_factors, self.all_tenors, self.time_grid, self.config.holidays )
			
		#check if we need to partition this structure
		if patitioning:
			output.BuildPartitions()
										
class Credit_Monte_Carlo(Calculation):
	cudacodeheader = '''
		__global__ void CalcCorrelated( REAL* Samples )
		{
			int offset = ( DIMENSION * (DIMENSION + 1 ) ) / 2;
			int index  = DIMENSION * ( ( blockIdx.x*blockDim.x + threadIdx.x ) * gridDim.y + blockIdx.y );

			for (int k=DIMENSION-1; k>=0; k--)
			{
				REAL corr = 0.0;
				for (int l=k; l>=0; l--)
				{
					corr += Buffer [ --offset ] * Samples[index + l];
				}
				Samples[index + k] = corr;
			}
		}
		
		__global__ void AdjustNormals(REAL* Samples, int size)
		{
			//make sure that we use the right index
			int index  				= 	( blockIdx.x*blockDim.x + threadIdx.x ) * size;
			
			for (int timepoint = size-1; timepoint>0; timepoint--)
			{
				int offset 	= index + timepoint;
				
				//Sobolov numbers fill a static space - we need to make it fill an incremental space
				REAL var				= timepoint + 1.0;
				REAL current_weight 	= sqrt ( var );
				REAL previous_weight 	= sqrt ( var-1.0 );
				REAL normalize 			= 1+0*sqrt ( current_weight*current_weight + previous_weight*previous_weight );
				
				for (int k=0; k<DIMENSION; k++)
				{	
					Samples[DIMENSION*offset + k] = ( current_weight*Samples[DIMENSION*offset + k] - previous_weight*Samples[DIMENSION*(offset-1) + k] ) / normalize ;
				}
			}
		}
	'''
		
	def __init__(self, config, prec=np.float32, calcs_per_batch=64):
		super(Credit_Monte_Carlo, self).__init__(config, prec)		
		self.calcs_per_batch		= calcs_per_batch
		self.cholesky				= None
		self.dependend_factors 		= None
		self.stochastic_factors 	= None
		self.reset_dates			= None
		self.base_date 				= None
		self.settlement_currencies  = None		
		
		#Cuda related variables to store the state of the device between calculations
		self.CudaStateClass			= namedtuple('CudaState','BLOCK_DIM PRECISION INDEXTYPE')
		self.CudaState 				= None
		self.CudaMemClass			= namedtuple('CudaMem','d_Buffer d_random_numbers d_Scenario_Buffer d_Static_Buffer d_MTM_Accum_Buffer d_MTM_Buffer d_MTM_Net_Buffer d_Time_Grid d_Cashflows d_Cashflow_Index d_Cashflow_Pay d_Cashflow_Rec')
		self.CudaMem 				= None
		self.module 				= None
		
		#used to store the Scenarios (if requested)		
		self.Scenarios				= None 
		
		#map used to track settlement currencies
		self.currency_settle_map 	= {}

	def UpdateFactors(self, currency, base_date):		
		dependend_factors, stochastic_factors, reset_dates, settlement_currencies = self.config.CalculateDependencies(currency, base_date, self.Base_time_grid)

		#write an explantion of whats going on
		self.ExplainCalc( ('Dependent Risk Factors',),
						  ['The following are calculated/initialized:',
						   '- Base and Scenario Time Grids',
						   '- Potential cashflow settlement dates',
						   '- Stochastic Processes associated with each Risk Factor',
						   '- Correlation matrix for the Stochastic Processes'] )
		
		self.ExplainCalc( ('Dependent Risk Factors', 'Base and Scenario Time Grids'),
						  ['The Base time Grid of **{0}** is interpreted thus:'.format(self.Base_time_grid),
						   '- Allowed codes are **d,w,m** corresponding to days, weeks and months.',
						   '- Every symbol e.g. 4m is relative to the Run_Date.',
						   '- If there are parenthesis after the symbol e.g. 1m(1w) then:',
						   '	- A date is generated 1 month from the rundate and again at weekly intervals thereafter',
						   '	- The dates continue to be generated until another symbol or the expiry of the longest instrument is reached',
						   '',
						   'The Scenario time Grid of **{0}** is treated similarly'.format(self.Scenario_time_grid) ] )

		self.ExplainCalc( ('Dependent Risk Factors', 'Potential Settlement Dates'),
						  ['Settlement dates are calculated per instrument as all instruments have a maturity/expiry date with optional coupon dates.',
						   'For each instrument the following dates are determined:',
						   '- The potential cashflow date',
						   '- The day immediately after the cashflow date',
						   '',
						   'These extra dates are added to the Base time Grid yielding a complete time grid of **{0}** dates'.format(len(reset_dates)),
						   '',
						   'Note:',
						   '- Instruments that may knock out earlier (e.g. Barrier Options) are only evaluated on the Base time Grid',
						   '- Knock out in such cases is assumed to only be possible on the Base time grid'] )
		
		self.ExplainCalc( ('Dependent Risk Factors', 'Stochastic Processes'),
						  ['The associated Stochastic Process for each risk factor is obtained. ',
						   'These processes are used later on to generate the correlation matrix:',
						   ''], stochastic_factors=stochastic_factors )		
				
		#update the time grid
		if self.reset_dates!=reset_dates or self.base_date!=base_date or self.settlement_currencies!=settlement_currencies:
			logging.info('Updating timegrid - basedate, reset list or settlement_currencies different')
			self.UpdateTimeGrid(base_date, reset_dates, settlement_currencies)

		#update the factor dependencies
		if dependend_factors!=self.dependend_factors or self.stochastic_factors!=stochastic_factors or self.slides:
			logging.info('Updating Dependencies - Static or Stochastic factors changed')
			#now construct the stochastic factors and static factors for the simulation
			self.stoch_factors 	= OrderedDict( (price_factor, ConstructProcess(price_model.type, ConstructFactor ( price_factor.type, self.config.params['Price Factors'][ Utils.CheckTupleName(price_factor) ]), self.config.params['Price Models'][Utils.CheckTupleName(price_model)] )) for price_model, price_factor in stochastic_factors.items() )
			self.static_factors	= OrderedDict()
			for price_factor in dependend_factors.difference(stochastic_factors.values()):
				try:
					self.static_factors.setdefault( price_factor, ConstructFactor(price_factor.type, self.config.params['Price Factors'][Utils.CheckTupleName(price_factor)]) )
				except KeyError, e:
					logging.warning('Price Factor {0} missing in market data file - skipping'.format(e.message))

			self.all_factors    = OrderedDict( self.stoch_factors.items()+self.static_factors.items())
			self.num_factors   	= sum([v.NumFactors() for v in self.stoch_factors.values()])
			
			if self.num_factors>128:
				logging.warning('There are more than 128 factors in the simulation ({0} factors) - performance will be impeded depending on hardware'.format(self.num_factors))
			
			#prepare the risk factor output matrix . .		
			self.size_stoch = 0
			self.static_val = np.array([], dtype=self.prec)
			self.stoch_ofs	= OrderedDict()
			self.static_ofs	= OrderedDict()
			
			#work out distinct tenor sets
			distinct_tenors = OrderedDict()

			#now get the stochastic risk factors ready - these will be generated from the price models
			for key, value in self.stoch_factors.items():
				if key.type not in Utils.DimensionLessFactors:
					#record the offset of this risk factor
					self.stoch_ofs.setdefault(key, self.size_stoch)
					self.DistinctTenors(distinct_tenors, key, value.factor)
					#precalc any values for the stochastic process
					value.PreCalc(base_date, self.time_grid) 
					self.size_stoch += len( value.factor.CurrentVal() )
				
			#and then get the the static risk factors ready - these are just looked up
			for key, value in self.static_factors.items():
				if key.type not in Utils.DimensionLessFactors:
					#record the offset of this risk factor
					self.static_ofs.setdefault(key, len(self.static_val) )
					self.DistinctTenors(distinct_tenors, key, value)
					self.static_val = np.hstack ( ( self.static_val, value.CurrentVal() ) )

			#calculate a reverse lookup for the tenors and store the daycount code
			self.UpdateTenors(distinct_tenors)
					
			#store the full mtm timegrid and all the distinct tenors per curve
			self.SetBuffer(distinct_tenors)

			#now check if any of the stochastic processes depend on other processes
			for key, value in self.stoch_factors.items():
				if key.type not in Utils.DimensionLessFactors:
					#precalc any values for the stochastic process
					value.CalcReferences( key, self.static_ofs, self.stoch_ofs, self.all_tenors, self.all_factors )
			
			#store the state info
			self.dependend_factors, self.stochastic_factors = dependend_factors, stochastic_factors

			#update the correlations
			self.UpdateCorrelation()

			#update the number of settlement currencies
			self.currency_settle_map = {}
			for factor, ofs in self.static_ofs.items():
				if factor.type=='FxRate' and factor.name[0] in self.time_grid.CurrencyMap:
					self.currency_settle_map[factor.name[0]]=-ofs-1
			for factor, ofs in self.stoch_ofs.items():
				if factor.type=='FxRate' and factor.name[0] in self.time_grid.CurrencyMap:
					self.currency_settle_map[factor.name[0]]=ofs

	def UpdateTimeGrid(self, base_date, reset_dates, settlement_currencies):
		#work out the scenario and dynamic dates
		dynamic_dates  = set([x for x in reset_dates if x>base_date])
		scenario_dates = self.config.ParseGrid(base_date, max(dynamic_dates), self.Scenario_time_grid, past_max_date=True)
		base_MTM_dates = self.config.ParseGrid(base_date, max(dynamic_dates), self.Base_time_grid)
		MTM_dates 	   = base_MTM_dates.union(dynamic_dates)
		
		#setup the scenario and base time grids
		self.time_grid 	 = TimeGrid( scenario_dates, MTM_dates, base_MTM_dates )
		self.base_date   = base_date
		self.reset_dates = reset_dates
		self.time_grid.SetBaseDate(base_date, self.prec, self.indexType)

		#Set the settlement dates
		self.time_grid.SetCurrencySettlement(settlement_currencies)
		self.settlement_currencies = settlement_currencies		

		if self.time_grid.scen_time_grid.size>512:
			raise Exception('Scenario grid cannot contain more than 512 time_steps - are you crazy?')
		
	def UpdateCorrelation(self):
		#create the correlation matrix
		correlation_matrix = np.eye(self.num_factors, dtype=self.prec)
		
		#prepare the correlation matrix (and the offsets of each stochastic process)
		correlation_factors = []
		self.process_ofs    = OrderedDict()
		for key, value in self.stoch_factors.items():
			proc_corr_type, proc_corr_factors = value.CorrelationName()
			for sub_factors in proc_corr_factors:
				#record the offset of this factor model
				self.process_ofs.setdefault(key, len(correlation_factors))
				#record the name of needed correlation lookup
				correlation_factors.append( Utils.Factor(proc_corr_type, key.name+sub_factors) )

		for index1 in range(self.num_factors):
			for index2 in range(index1+1,self.num_factors):
				factor1,factor2 = Utils.CheckTupleName(correlation_factors[index1]), Utils.CheckTupleName(correlation_factors[index2])
				key = (factor1, factor2) if (factor1, factor2) in self.config.params['Correlations'] else (factor2, factor1)
				rho = self.config.params['Correlations'].get( key, 0.0 ) if factor1!=factor2 else 1.0
				correlation_matrix[index1,index2] = rho
				correlation_matrix[index2,index1] = rho

		self.ExplainCalc( ('Dependent Risk Factors', 'Correlation matrix'),
						  ['Each Stochastic Process requires at least 1 Source of Randomness.',
						   'Each source of randomness (its **Factor**) needs to be correlated to all other Factors',
						   '',
						   'Here is the resulting correlation matrix:',
						   ''],
						  correlation_factors=correlation_factors, correlation_matrix=correlation_matrix )

		#need to do cholesky
		try:
			self.cholesky    = np.linalg.cholesky ( correlation_matrix )
			#self.cholesky    = np.linalg.cholesky ( np.eye(self.num_factors, dtype=self.prec) )
		except:

##			n      = correlation_matrix.shape[0]
##			W      = np.identity(n) 
##			Yk     = correlation_matrix.astype(np.float64).copy()
##			deltaS = 0
##			
##			for k in range(10):
##				Rk  = Yk - deltaS
##				W05 = np.matrix(W**.5)
##				A   = W05 * Rk * W05
##				
##				eigval, eigvec = np.linalg.eig(A)
##				Q      = np.matrix(eigvec)
##				xdiag  = np.matrix(np.diag(np.maximum(eigval, np.spacing(1000.0))))
##				Xk     = W05.I * Q * xdiag * Q.T * W05.I 
##
##				deltaS = Xk - Rk
##				Aret = np.array(Xk.copy())
##				Aret[W > 0] = np.array(W)[W > 0]
##				Yk = np.matrix(Aret)
			
			logging.warning('Correlation matrix (size {0}) not positive definite - raising eigenvalues'.format(correlation_matrix.shape))
			#eigenvalue "healing" aka pretending 
			values, vector = np.linalg.eig( correlation_matrix.astype(np.float64) )
			values[values <= 0] = np.spacing(10.0)
			self.cholesky = np.linalg.cholesky ( vector.dot( np.diag ( values ) ).dot( vector.T ) ).astype(self.prec)
			
	def __RefreshCudaMem(self, seed, calc_cashflows, numscenario_batches, Scenario_nbytes, MemoryInMBperScenarioBatch = 128.0):
		#Talk a bit about what CUDA is		
		self.ExplainCalc( ('CUDA',),
						  ['- What is CUDA?',
						   '- {0} precision floating point capability'.format('Single' if self.prec == np.float32 else 'Double'),
						   '- How CUDA code works'] )
		
		self.ExplainCalc( ('CUDA','What is CUDA?'),
						  ['- Parallel computing platform and programming model invented by NVIDIA.',
						   '- Allows increases in computing performance by using a graphics processing unit (GPU)',
						   '- Needs specialized hardware (a CUDA compliant GPU)',
						   '- [Official NVIDIA page](http://www.nvidia.com/object/cuda_home_new.html "CUDA\'s Homepage")'] )
		
		self.ExplainCalc( ('CUDA','Floating point capability'),
						  ['CUDA can use single (32 bit) or double (64 bit) precision for its floating point calculations:',
						   '- For Monte Carlo calculations, single precision is usually good enough',
						   '- Double precision also requires double the memory',
						   '- CRSTAL can be recomplied to support Double precision if needed'] )
		
		self.ExplainCalc( ('CUDA','How Cuda code works'),
						  ['CRSTAL is built using Python and CUDA combining the flexibility of Python with the speed of CUDA.',
						   '',
						   'To achieve this:',
						   '- All instrument and scenario generation (stochastic process) code is fetched',
						   '- Python sets up the nvidia CUDA compiler at run-time',
						   '- If there is no change in the code, the last compiled code is loaded, otherwise it is fully re-compiled',
						   '',
						   'This means that it is possible to add new instruments/cuda code via the CRSTAL notebook and CUDA will automatically load and run it.'
						   ] )
		
		#set the RNG state to the right seed - this is annoying as it calls nvcc each time
		cuda_sobol_generator = pycuda.curandom.ScrambledSobol64RandomNumberGenerator(offset=seed)
		
		#work out a nice number of scenario processing batches and create memory on the device
		self.scenario_batch_size = int ( float(numscenario_batches) / np.ceil( Scenario_nbytes/(MemoryInMBperScenarioBatch*1024*1024.0) ) +.5 )
		
		#pc rng - to test random numbers
		#pc_rnd 		 = np.random.randn(self.numscenarios,self.time_grid.scen_time_grid.size*self.num_factors)
		
		#update the settlment currency indexes for the cuda code to lookup from
		CurrencySettlementMap, currencySettlementLen, CurrencySettlementOffset = self.time_grid.GetCurrencySettlementMap(self.currency_settle_map)

		drv.memcpy_htod ( self.module.get_global('NumSettlementCurrencies')[0],		np.array( [ len(self.currency_settle_map ) ], dtype=np.int32) )
		drv.memcpy_htod ( self.module.get_global('CurrencySettlementMap')[0],		np.array( CurrencySettlementMap, dtype=np.int32 ) )
		drv.memcpy_htod ( self.module.get_global('CurrencySettlementOffset')[0],	np.cumsum([0]+currencySettlementLen, dtype=np.int32) )
		drv.memcpy_htod ( self.module.get_global('ScenarioFactorSize')[0], 			np.array( [ self.size_stoch], dtype=np.int32 ) )
		drv.memcpy_htod ( self.module.get_global('ScenarioTimeSteps')[0],			np.array( [ self.time_grid.scen_time_grid.size ], dtype=np.int32 ) )
		drv.memcpy_htod ( self.module.get_global('MTMTimeSteps')[0],				np.array( [ self.time_grid.mtm_time_grid.size ], dtype=np.int32 ) )
		drv.memcpy_htod ( self.module.get_global('DIMENSION')[0],					np.array( [ self.num_factors ], dtype=np.int32 ) )
		drv.memcpy_htod ( self.module.get_global('NumTenors')[0],					np.array( [ self.tenor_size.sum() ], dtype=np.int32 ) )
		drv.memcpy_htod ( self.module.get_global('ScenarioTenorSize')[0],			self.tenor_size )
		drv.memcpy_htod ( self.module.get_global('ScenarioTenorOffset')[0], 		self.tenor_offset )
		drv.memcpy_htod ( self.module.get_global('NumCurveCurrencies')[0],			np.array( [ self.currency_curve_map.size ], dtype=np.int32 ) )
		drv.memcpy_htod ( self.module.get_global('CurrencyCurveMap')[0],			self.currency_curve_map )
		drv.memcpy_htod ( self.module.get_global('CurrencyCurveOffset')[0],			self.currency_curve_ofs )

		#talk about what's happening with random numbers - why can you bankers not just read code like normal people??
		self.ExplainCalc( ('Random Numbers',),
						  ['As discussed, Stochastic Processes (**SP\'s**) need random numbers for scenario generation.',
						   'The sum of all **Factors** across all SP\'s **({0})** determine the *size* of the calculation.'.format(self.num_factors),
						   '',
						   'CRSTAL uses the [Sobol](http://en.wikipedia.org/wiki/Sobol_sequence) quasi random generation algorithm as implemented by [nVidia](https://developer.nvidia.com/cuRAND).',
						   '',
						   'The nvidia CUDA library also allows the following:',
						   '- Pseudo Random Number Generators (RNG\'s)',
						   '	- MRG32k3a (Combined Multiple Recursive family of pseudorandom number generators)',
						   '	- MTGP Merseinne Twister',
						   '	- XORWOW',
						   '- Qausi Random Number Generators',
						   '	- Sobol scrambled',
						   '',
						   'The benefit of using a space-filling (i.e. quasi RNG) is faster convergence.',
						   '',
						   'CRSTAL can be configured to use any of the above generators with ease.'] )		

		#setup the GPU
		#convention used here is that all variables beginning with a "d_" are pointers to device memory
		#allocate memory on the device
		
		self.CudaMem = self.CudaMemClass (  d_Buffer			= gpuarray.GPUArray( (self.max_buffer_size,), dtype=self.prec ),
											d_random_numbers	= cuda_sobol_generator.gen_normal((self.numscenarios,self.time_grid.scen_time_grid.size*self.num_factors), dtype=self.prec),
											#d_random_numbers	= gpuarray.to_gpu( pc_rnd.astype(self.prec) ),
											d_Scenario_Buffer   = gpuarray.GPUArray( (self.scenario_batch_size*self.calcs_per_batch, self.size_stoch*self.time_grid.scen_time_grid.size), dtype=self.prec ),
											d_Static_Buffer     = gpuarray.to_gpu( self.static_val.astype(self.prec) ),
											d_MTM_Accum_Buffer  = gpuarray.GPUArray( (self.scenario_batch_size*self.calcs_per_batch, self.time_grid.mtm_time_grid.size), dtype=self.prec ),
											d_MTM_Buffer   		= gpuarray.GPUArray( (self.scenario_batch_size*self.calcs_per_batch, self.time_grid.mtm_time_grid.size), dtype=self.prec ),
											d_MTM_Net_Buffer	= gpuarray.GPUArray( (self.scenario_batch_size*self.calcs_per_batch, self.time_grid.mtm_time_grid.size), dtype=self.prec ),
											d_Time_Grid         = gpuarray.to_gpu ( self.time_grid.time_grid.ravel().astype(self.prec) ),
											d_Cashflows         = gpuarray.GPUArray( (self.scenario_batch_size*self.calcs_per_batch, sum([x.max()+1 for x in self.time_grid.CurrencyMap.values()]) ), dtype=self.prec ) if calc_cashflows else np.intp(0),
											d_Cashflow_Index    = gpuarray.to_gpu ( CurrencySettlementOffset.astype(np.int32) ) if calc_cashflows else np.intp(0),
											d_Cashflow_Pay      = gpuarray.GPUArray( (self.scenario_batch_size*self.calcs_per_batch, self.time_grid.mtm_time_grid.size), dtype=self.prec ) if calc_cashflows else np.intp(0),
											d_Cashflow_Rec      = gpuarray.GPUArray( (self.scenario_batch_size*self.calcs_per_batch, self.time_grid.mtm_time_grid.size), dtype=self.prec ) if calc_cashflows else np.intp(0) )
		
		#set the global buffer variable to the allocated buffer
		drv.memcpy_htod ( self.module.get_global('Buffer')[0],	np.array( [ self.CudaMem.d_Buffer.ptr ], dtype=np.intp ) )
		#talk about the random samples
		self.ExplainCalc( ('Random Numbers','Samples'), [], samples=self.CudaMem.d_random_numbers )
		
	def __ManageCudaState(self):
		'''
		Manage setting constants, compiling the pricing functions etc. - should not need to concern the user
		'''
		#Fill in the parameters for our cuda functions . .
		cudaBody		= Calculation.cudacodetemplate+reduce(operator.concat, GetAllProcessCudaFunctions(), '')+reduce(operator.concat, GetAllInstrumentCudaFunctions(), '')
		cudaTemplate    = string.Template ( Calculation.cudacodeheader + Credit_Monte_Carlo.cudacodeheader + cudaBody + Calculation.cudacodefooter )
		cudacode  		= cudaTemplate.substitute( self.CudaState._asdict() )
		#self.module    	= pycuda.compiler.SourceModule(cudacode, no_extern_c=True, options=['--ptxas-options=-v'])
		self.module    	= pycuda.compiler.SourceModule(cudacode, no_extern_c=True, options=['-O3'])
		
	def GetCashflows(self, n):
		import os
		for netting_set in n.sub_structures:
			offset 		= 0
			concat      = pd.DataFrame()
			for currency in self.time_grid.CurrencyMap.iterkeys():
				dates = sorted( [x for x in self.settlement_currencies[currency] if x>=self.base_date] )
				data  = netting_set.partition.Cashflows[:,offset:offset+len(dates)].T.astype(np.float64)
				dataframe  = pd.DataFrame ( index = dates, data = {'Average':data.mean(axis=1), 'Std':data.std(axis=1), 'Percentile (5%)':np.percentile(data, 5, axis=1), 'Percentile (95%)':np.percentile(data, 95, axis=1)} )
				dataframe['Currency'] = currency
				dataframe['Netting'] = netting_set.obj.Instrument.field.get('Reference', 'Unknown')
				concat = concat.append(dataframe)
				offset+=len(dates)
		return 	concat
				
	def FormatOutput(self, result, output):
		'''
		Function to present the output calc to the UI - not ideal. Should instead pass an object from the UI and have this method set
		it up without having to know about walking portfolios etc.
		'''
		
		def presentPFE(obj, partition, full_time_grid):
			if obj.Time_dep:
				time_grid		 = np.array(sorted(full_time_grid))[obj.Time_dep.deal_time_grid]
				mtm				 = partition.DealMTMs[:,obj.Time_dep.deal_time_grid].astype(np.float64)
				collateral		 = partition.Collateral_Cash[:,obj.Time_dep.deal_time_grid].astype(np.float64)
				funding			 = partition.Funding_Cost[:,obj.Time_dep.deal_time_grid].astype(np.float64)
			else:
				time_grid		 = np.array(sorted(full_time_grid))
				mtm				 = partition.DealMTMs.astype(np.float64)
				collateral		 = partition.Collateral_Cash.astype(np.float64)
				funding			 = partition.Funding_Cost.astype(np.float64)
				
			dates = [time.mktime(x.to_datetime().timetuple())*1000.0 for x in time_grid]
			pfb   = zip ( dates, np.percentile ( mtm, 5, axis=0).tolist() )
			pfe   = zip ( dates, np.percentile ( mtm, 95, axis=0).tolist() )
			ee    = zip ( dates, np.mean( mtm, axis=0).tolist() )
			
			coll  = []
			if collateral.any():
				coll_mean 		= zip ( dates, np.mean(collateral,axis=0).tolist() )
				coll_5percent 	= zip ( dates, np.percentile(collateral,5,axis=0).tolist() )
				coll			= [{"label":"Collateral (5%)", "data":coll_5percent},{"label":"Expected Collateral", "data":coll_mean}]

			fund  = []
			if funding.any():
				fva 			= zip ( dates, np.mean(funding,axis=0).tolist() )
				fund			= [{"label":"Expected Funding Cost", "data":fva}]
				
			return [{"label":"PFE (95%)", "data":pfe},{"label":"EE", "data":ee},{"label":"PFB (5%)", "data":pfb}] + coll + fund
		
		def presentCashflows(base_date, partition, CurrencyMap, settlement_currencies):
			offset 		= 0
			concat      = pd.DataFrame()
			for currency in CurrencyMap.iterkeys():
				dates = sorted ( [x for x in settlement_currencies[currency] if x>=base_date] )
				data  = partition.Cashflows[:,offset:offset+len(dates)].T.astype(np.float64)
				dataframe  = pd.DataFrame ( index = dates, data = {'Average':data.mean(axis=1), 'Std':data.std(axis=1), 'Percentile (5%)':np.percentile(data, 5, axis=1), 'Percentile (95%)':np.percentile(data, 95, axis=1)} )
				dataframe['Currency'] = currency
				concat = concat.append(dataframe)
				offset+=len(dates)

			output = []	
			for colname, coldata in sorted(concat.iteritems()):
				output.append({"label":colname, "data":zip(coldata.index.map(lambda x:x.strftime('%Y-%m-%d')),coldata)})
				
			return output
		
		def walkPortfolio ( deals, path, parent, profiles, full_time_grid ):
			
			profiles['PFE'].setdefault( json.dumps ( list(path), separators=(',', ':') ) , presentPFE(deals.obj, deals.partition, full_time_grid) )
			profiles['Cashflows'].setdefault( json.dumps ( list(path), separators=(',', ':') ) , presentCashflows(self.base_date, deals.partition,  self.time_grid.CurrencyMap, self.settlement_currencies) )

			for partition_name, partition in deals.sub_partitions.items():
				json_data = {  "text" : partition_name,
							   "type" : "default",
							   "data" : {},
							   "children" : [] }
				
				profiles['PFE'].setdefault( json.dumps ( list(path) + [partition_name], separators=(',', ':') ) , presentPFE(deals.obj, partition, full_time_grid) )
				profiles['Cashflows'].setdefault( json.dumps ( list(path) + [partition_name], separators=(',', ':') ) , presentCashflows(self.base_date, partition,  self.time_grid.CurrencyMap, self.settlement_currencies) )
				parent.append(json_data)
			
			for netting in deals.sub_structures:
				group_data  = {}
				name        = "{0}.{1}".format ( netting.obj.Instrument.field.get('Object'), netting.obj.Instrument.field.get('Reference') if netting.obj.Instrument.field.get('Reference') else len(profiles['PFE']) )
				
				group = {  "text" : name,
						   "type" : "group",
						   "data" : {},
						   "children" : [] }
				
				walkPortfolio ( netting, path+(name,), group['children'], profiles, full_time_grid )
				
				parent.append(group)

		deals_to_append = []
		profiles        = {'PFE':{}, 'Cashflows':{}}
		walkPortfolio ( result['Netting'], (), deals_to_append, profiles, self.time_grid.mtm_dates )
			
		tree_data = [{ "text" : "All",  
					   "type" : "root", 
					   "state" : { "opened" : True, "selected" : True },
					   "children" : deals_to_append} ]
			
		for field, field_data in output.items():
			if field_data.get('isvisible','True')=='True' and field in profiles:
				field_data['value'] = json.dumps ( tree_data )
				field_data['profiles'] = json.dumps ( profiles[field] )

	def PlotPFERiskFactor(self, scenarios, factor, closest_scenario_index, pfe_scenario, at_time):
		import matplotlib.pyplot as plt
		plt.style.use('ggplot')
		ofs 		= self.stoch_ofs[factor]
		fig 		= plt.figure()
		filename	= None
		
		if factor.type in Utils.OneDimensionalFactors:
			tenors 			= self.stoch_factors[factor].factor.GetTenor()
			simulated_vals 	= scenarios[pfe_scenario, closest_scenario_index*self.size_stoch+ofs:closest_scenario_index*self.size_stoch+ofs+tenors.size]
			
			if factor.type in ['InterestRate','InflationRate','DividendRate']:
				simulated_vals 	*= 100.0
				plt.plot ( tenors, simulated_vals, 'b-')
				plt.title ('{0}.{1} - {2}'.format(factor.type, '.'.join(factor.name), at_time ) )
				plt.ylabel('rate (%)')
				plt.xlabel('time (years)')
				
			elif factor.type in ['SurvivalProb','ForwardPrice']:
				plt.plot ( tenors, simulated_vals, 'b-')
				plt.title ('{0}.{1} - {2}'.format(factor.type, '.'.join(factor.name), at_time ) )
				plt.ylabel('Price' if factor.type=='ForwardPrice' else 'Log Probability')
				plt.xlabel('time (years)')
			filename = '{0}.PFE.{1}.{2}.{3}.png'.format(self.name, factor.type, '.'.join(factor.name), at_time)
			plt.savefig(filename)
			plt.close()
			
		return filename
				
	def PlotScenarioRiskFactor(self, scenarios, factor):
		import matplotlib.pyplot as plt
		plt.style.use('ggplot')
		ofs 	= self.stoch_ofs[factor]
		fig 	= plt.figure()

		if factor.type in Utils.OneDimensionalFactors:
			tenors 			= self.stoch_factors[factor].factor.GetTenor()
			image_tenors	= np.array([int(x*(tenors.size-1)/6.0) for x in range(1,7)]) if tenors.size>6 else np.arange(tenors.size)
			
			if factor.type in ['InterestRate','InflationRate','DividendRate']:
				for tenor, tenor_ofs in enumerate(image_tenors):
					simulated_vals 	= scenarios[:,ofs+tenor_ofs::self.size_stoch]*100.0

					plt.subplot(3, 2, tenor+1) if image_tenors.size==6 else plt.subplot(image_tenors.size, 1, tenor+1)
					plt.plot(self.time_grid.scen_time_grid/365.0, np.percentile(simulated_vals,97.5,axis=0), 'r-')
					plt.plot(self.time_grid.scen_time_grid/365.0, np.percentile(simulated_vals,50,axis=0), 'g-')
					plt.plot(self.time_grid.scen_time_grid/365.0, np.percentile(simulated_vals,2.5,axis=0), 'b-')
					
					plt.title ('{0}.{1} - {2}'.format(factor.type, '.'.join(factor.name), '{:.2f} yr'.format(tenors[tenor_ofs]) ) )
					plt.ylabel('rate (%)')
					plt.xlabel('time (years)')
			elif factor.type in ['SurvivalProb','ForwardPrice']:
				for tenor, tenor_ofs in enumerate(image_tenors):
					simulated_vals 	= scenarios[:,ofs+tenor_ofs::self.size_stoch]

					plt.subplot(3, 2, tenor+1) if image_tenors.size==6 else plt.subplot(image_tenors.size, 1, tenor+1)
					plt.plot(self.time_grid.scen_time_grid/365.0, np.percentile(simulated_vals,97.5,axis=0), 'r-')
					plt.plot(self.time_grid.scen_time_grid/365.0, np.percentile(simulated_vals,50,axis=0), 'g-')
					plt.plot(self.time_grid.scen_time_grid/365.0, np.percentile(simulated_vals,2.5,axis=0), 'b-')
					label = (self.stoch_factors[factor].factor.start_date + pd.DateOffset(days=tenors[tenor_ofs])).strftime('%d %b %Y') if factor.type=='ForwardPrice' else '{:.2f} yr'.format(tenors[tenor_ofs])
					plt.title ('{0}.{1} - {2}'.format(factor.type, '.'.join(factor.name), label ) )
					plt.ylabel('Price' if factor.type=='ForwardPrice' else 'Log Probability')
					plt.xlabel('time (years)')
					
			plt.tight_layout()
		else:
			simulated_vals 	= scenarios[:,ofs::self.size_stoch]
			plt.plot(self.time_grid.scen_time_grid/365.0, np.percentile(simulated_vals,97.5,axis=0), 'r-')
			plt.plot(self.time_grid.scen_time_grid/365.0, np.percentile(simulated_vals,50.0,axis=0), 'g-')
			plt.plot(self.time_grid.scen_time_grid/365.0, np.percentile(simulated_vals,2.5,axis=0), 'b-')
			plt.title('{0}.{1}'.format(factor.type, '.'.join(factor.name)) )
			plt.ylabel('rate')
			plt.xlabel('time (years)')
			
		filename = '{0}.{1}.{2}.png'.format(self.name, factor.type, '.'.join(factor.name))
		plt.savefig(filename)
		plt.close()
		return filename

	def ExplainCalc(self, calc_section, content, **kwargs):
		
		def presentCollateral(tkr, plt, calc_section, obj, currency, partition, label):
			#draw the PFE curve
			time_grid		= np.array(sorted(self.time_grid.mtm_dates))[obj.Time_dep.deal_time_grid]
			cash			= partition.Collateral_Cash[:,obj.Time_dep.deal_time_grid].astype(np.float64)
			dates 			= [x.to_datetime() for x in time_grid]
			
			fig, ax 		= plt.subplots()
			fifth, exp		= np.percentile ( cash, 5, axis=0 ), np.mean ( cash, axis=0 )
			min_coll_at 	= time_grid[fifth.argmin()]
			
			ax.plot ( dates, fifth, 'r-', dates, exp, 'g-')
			plt.title ( 'Stressed ( Red ) vs Expected ( Green ): {0}'.format ( label ) )
			plt.ylabel ( currency )
			plt.xlabel ('Date')
			fig.autofmt_xdate()
			ax.get_yaxis().set_major_formatter( tkr.FuncFormatter(lambda x, p: format(int(x), ',')) )
			ax.grid(True)
			filename  = '{0}.Collateral.{1}.png'.format(self.name, label )
			plt.savefig( filename )
			plt.close()
			
			cell 				= {}
			cell['metadata'] 	= slide_types['subslide']
			cell['text'] 		= '\n'.join(['## '+calc_section[0],'','### '+calc_section[1],'','#### Collateral (5th Percentile) {0:,} at {1}'.format(fifth.max(), min_coll_at.strftime('%Y-%m-%d')),'','<img src="{0}">'.format(filename)])
			#add the slide
			self.slides.append(cell)
		
		def presentPFE(tkr, plt, calc_section, obj, currency, partition, label, present_scenarios = True):
			#draw the PFE curve
			if obj.Time_dep:
				time_grid		 = np.array(sorted(self.time_grid.mtm_dates))[obj.Time_dep.deal_time_grid]
				mtm				 = partition.DealMTMs[:,obj.Time_dep.deal_time_grid].astype(np.float64).clip(0,np.inf)
			else:
				time_grid		 = np.array(sorted(self.time_grid.mtm_dates))
				mtm				 = partition.DealMTMs.astype(np.float64).clip(0,np.inf)
				
			dates 			= [x.to_datetime() for x in time_grid]
			fig, ax 		= plt.subplots()
			pfe, ee 		= np.percentile ( mtm, 95, axis=0 ), np.mean ( mtm, axis=0 )
			max_pfe_at 		= time_grid[pfe.argmax()]
			
			#just plot the net PFE/EE - show collateral in a separate section
			ax.plot ( dates, pfe, 'bo-', dates, ee, 'go-')
			plt.title ( '{0} PFE/EE: {1}'.format ( 'Net' if partition.Collateral_Cash.any() else 'Gross', label ) )
			plt.ylabel ( currency )			
			plt.xlabel ('Date')
			fig.autofmt_xdate()
			ax.get_yaxis().set_major_formatter( tkr.FuncFormatter(lambda x, p: format(int(x), ',')) )
			ax.grid(True)
			filename  = '{0}.PFE.{1}.png'.format( self.name, label )
			plt.savefig( filename )
			plt.close()
			
			cell 				= {}
			cell['metadata'] 	= slide_types['subslide']
			cell['text'] 		= '\n'.join(['## '+calc_section[0],'','### '+calc_section[1],'','#### PFE (95%) {0:,} at {1}'.format(pfe.max(), max_pfe_at.strftime('%Y-%m-%d')),'','<img src="{0}">'.format(filename)])
			#add the slide
			self.slides.append(cell)
			
			if present_scenarios:
				#draw the PFE curve
				relative_days 			= (max_pfe_at-min(self.time_grid.mtm_dates)).days
				closest_scenario_index 	= min([( (scen_date-relative_days)**2,index) for index,scen_date in enumerate(self.time_grid.scen_time_grid)])[1]
				pfe_mtm 				= mtm[:,pfe.argmax()]
				pfe_idx					= int((pfe_mtm.size-1)*.95)
				pfe_scenario			= np.argpartition(pfe_mtm, pfe_idx)[pfe_idx]
				pfe_time				= sorted(self.time_grid.scenario_dates)[ closest_scenario_index ].strftime('%Y-%m-%d')
				single_rates			= []
				
				for factor in self.stoch_factors.keys():
					if factor.type in Utils.OneDimensionalFactors:
						cell = {}
						cell['metadata'] = slide_types['subslide']
						filename = self.PlotPFERiskFactor(scenarios, factor, closest_scenario_index, pfe_scenario, pfe_time)
						cell['text'] = '\n'.join(['## '+calc_section[0],'','### '+calc_section[1],'','#### {0}.'.format(factor.type)+'.'.join(factor.name),'','<img src="{0}">'.format(filename)])
						self.slides.append(cell)
					else:
						ofs = self.stoch_ofs[factor]
						single_rates.append('- {0}.'.format(factor.type)+'.'.join(factor.name)+' at {0:.4}'.format(scenarios[pfe_scenario, closest_scenario_index*self.size_stoch+ofs]))
											
				cell = {}
				cell['metadata'] = slide_types['subslide']
				cell['text'] = '\n'.join(['## '+calc_section[0],'','### '+calc_section[1],'','#### Rates','','The following rates were realized at {0}'.format(pfe_time)]+single_rates)
				#add the slide
				self.slides.append(cell)
				
			#return the pfe and ee
			return pfe, ee, time_grid
				
		slide_types = {
						'slide': { 'internals': { 'slide_helper': 'subslide_end', 'slide_type': 'subslide' }, 'slide_helper': 'slide_end', 'slideshow': {'slide_type': 'slide'} },
						'subslide': { 'internals': { 'slide_helper': 'subslide_end', 'slide_type': 'subslide' }, 'slide_helper': 'slide_end', 'slideshow': {'slide_type': 'subslide'} }
						}
		
		if self.slides is not None:
			cell = {}
			if calc_section[0] == 'Intro':
				cell['metadata'] = slide_types['slide']
				cell['text'] = '\n'.join(['## '+calc_section[1],'','### '+calc_section[2]]+content)
			elif calc_section == ('Dependent Risk Factors', 'Stochastic Processes'):
				cell['metadata'] = slide_types['subslide']
				stoch_procs, procs = kwargs['stochastic_factors'], []
				for process_type in set([ x.type for x in sorted(stoch_procs) ]):
					procs += ['- {0}'.format(process_type)]
					procs += ['\t- {0}.{1}'.format(factor.type, '.'.join(factor.name)) for proc, factor in stoch_procs.items() if proc.type==process_type]
				cell['text'] = '\n'.join(['## '+calc_section[0],'','### '+calc_section[1]]+content+procs)
			elif calc_section == ('Dependent Risk Factors', 'Correlation matrix'):
				cell['metadata'] = slide_types['subslide']
				correlation_factors, correlation_matrix  = kwargs['correlation_factors'], kwargs['correlation_matrix']
				lines = [ '|Stochastic Process|'+'|'.join(['{0}'.format(Utils.CheckTupleName(correlation_factor)) for correlation_factor in correlation_factors])+'|' ]
				lines += ['|:'+':|:'.join(['----']*len(correlation_factors))+ ':|']
				for index,correlation_factor in enumerate(correlation_factors):
					lines += ['|{0}|{1}|'.format(Utils.CheckTupleName(correlation_factor), '|'.join( np.vectorize("%.3f".__mod__)(correlation_matrix[index]) ) ) ]
				cell['text'] = '\n'.join(['## '+calc_section[0],'','### '+calc_section[1]]+content+lines)
				
			elif calc_section == ('Scenario Generation', 'Stochastic Processes'):
				scenarios = kwargs['scenarios']
				process_documentation = {model.__class__.__name__:model.documentation for model in self.stoch_factors.values()}
				for process_type, process_doc in process_documentation.items():
					cell = {}
					cell['metadata'] = slide_types['subslide']
					cell['text'] = '\n'.join(['## '+calc_section[0],'','### '+calc_section[1],'','#### '+process_type,'']+ process_doc)
					self.slides.append(cell)
					for factor in [factor for factor, model in self.stoch_factors.items() if model.__class__.__name__==process_type]:
						cell = {}
						cell['metadata'] = slide_types['subslide']
						filename = self.PlotScenarioRiskFactor(scenarios, factor)
						cell['text'] = '\n'.join ( [ '## '+calc_section[0],'','### '+calc_section[1],'','#### {0}.'.format(factor.type)+'.'.join(factor.name),'','<img src="{0}">'.format(filename) ] )
						self.slides.append(cell)
				cell = {}
				
			elif calc_section == ('Random Numbers', 'Samples'):
				import matplotlib.pyplot as plt
				plt.style.use('ggplot')
				samples 	= kwargs['samples'].get()
				index		= 1
				
				for i, dim_i in enumerate ( np.linspace ( 1, samples.shape[1], 4) ):
					for j, dim_j in enumerate ( np.linspace ( 1, samples.shape[1], 4) ):
						if i!=j:
							plt.subplot (4, 3, index )
							plt.plot (samples[:,int(i)], samples[:,int(j)], 'b.' )
							plt.title ( '{0},{1}'.format(int(i), int(j)) )
							index	+=1

				filename  = '{0}.random_samples.png'.format(self.name)
				plt.savefig( filename )
				plt.close()
						
				cell 				= {}
				cell['metadata'] 	= slide_types['subslide']
				cell['text'] 		= '\n'.join(['## '+calc_section[0],'','### Scatter Plot','','#### Dimension {0}'.format(samples.shape[1]),'','<img src="{0}">'.format(filename)])
						
			elif calc_section == ('Instrument Valuation', 'PFE'):
				import matplotlib.pyplot as plt
				import matplotlib.ticker as tkr
				plt.style.use('ggplot')
				scenarios, netting, currency, external_result 	= kwargs['scenarios'], kwargs['netting'], kwargs['currency'], kwargs['compare_against']

				for sub_structure in netting.sub_structures:
					structure_label = sub_structure.obj.Instrument.field.get('Reference','Unnamed')
					#report the PFE and EE for each partition (if available)
					for partition_name, partition in sub_structure.sub_partitions.items():
						sub_pfe, sub_ee, sub_time_grid = presentPFE ( tkr, plt, ( 'Instrument Valuation', 'PFE: Partiton {0}'.format(partition_name) ), sub_structure.obj, currency, partition, '{0}.{1}'.format(structure_label,partition_name), present_scenarios = False )
						
					#report the PFE and EE
					pfe, ee, time_grid = presentPFE ( tkr, plt, calc_section, sub_structure.obj, currency,
													  sub_structure.partition, structure_label, present_scenarios = True )
					#we're done with the PFE graphs
					cell = {}
					
					#report the comparison against the reference PFE/EE if necessary
					if external_result is not None:
						#make sure the dates are aligned
						dates 				= [x.to_datetime() for x in time_grid]
						compare_against	    = external_result.ix[time_grid].interpolate(method='index')
						
						fig, ax = plt.subplots()
						ax.plot ( dates, pfe, 'bo-', dates, compare_against['PFE'], 'ro-')
						fig.autofmt_xdate()
						ax.get_yaxis().set_major_formatter( tkr.FuncFormatter(lambda x, p: format(int(x), ',')) )
						ax.grid(True)
						filename  = '{0}.PFE.Comparison.{1}.png'.format(self.name, sub_structure.obj.Instrument.field.get('Reference','Unnamed'))
						plt.savefig( filename )
						plt.close()
						
						cell 				= {}
						cell['metadata'] 	= slide_types['subslide']
						cell['text'] 		= '\n'.join(['## '+calc_section[0],'','### PFE Comparison','','#### Calculated PFE (Blue) {0:,} vs Reference PFE (Red) {1:,}'.format(pfe.max(), compare_against['PFE'].max()),'','<img src="{0}">'.format(filename)])
						self.slides.append(cell)
						
						fig, ax = plt.subplots()
						ax.plot ( dates, ee, 'bo-', dates, compare_against['EE'], 'ro-')
						fig.autofmt_xdate()
						ax.get_yaxis().set_major_formatter( tkr.FuncFormatter(lambda x, p: format(int(x), ',')) )
						ax.grid(True)
						filename  = '{0}.EE.Comparison.{1}.png'.format(self.name, sub_structure.obj.Instrument.field.get('Reference','Unnamed'))
						plt.savefig( filename )
						plt.close()
						
						cell 				= {}
						cell['metadata'] 	= slide_types['subslide']
						cell['text'] 		= '\n'.join(['## '+calc_section[0],'','### EE Comparison','','#### Calculated EE (Blue) {0:,} vs Reference EE (Red) {1:,}'.format(ee.max(), compare_against['EE'].max()),'','<img src="{0}">'.format(filename)])
					
			elif calc_section == ('Collateral', 'Plots'):
				import matplotlib.pyplot as plt
				import matplotlib.ticker as tkr
				plt.style.use('ggplot')
				netting 		= kwargs['netting'] 

				for sub_structure in netting.sub_structures:
					instrument			= sub_structure.obj.Instrument
					if instrument.field.get('Object')=='NettingCollateralSet' and instrument.field.get('Collateralized')=='True':
						currency 		= instrument.field.get('Balance_Currency')
						structure_label = sub_structure.obj.Instrument.field.get('Reference','Unnamed')
						
						#report the collateral for each partition (if available)
						for partition_name, partition in sub_structure.sub_partitions.items():
							presentCollateral ( tkr, plt, ( 'Collateral', 'Plot: Partiton {0}'.format(partition_name) ), sub_structure.obj, currency, partition, '{0}.{1}'.format(structure_label,partition_name) )
							
						#report the collateral
						presentCollateral ( tkr, plt, calc_section, sub_structure.obj, currency, sub_structure.partition, structure_label )
						
				#we're done with the Collateral graphs
				cell = {}
				
			else:
				if len(calc_section)==1:
					cell['metadata'] = slide_types['slide']
					cell['text'] = '\n'.join(['## Credit Monte Carlo Calculation','','### '+calc_section[0]]+content)
				else:
					cell['metadata'] = slide_types['subslide']
					cell['text'] = '\n'.join(['## '+calc_section[0],'','### '+calc_section[1]]+content)

			if cell:
				self.slides.append(cell)

	def RawExecute(self, params):
		
		prev_block_num 		= 0
		random_index   		= 0
		numscenario_batches = params['Number_Simulations'] / self.calcs_per_batch + 1		
			
		for block_num in range(self.scenario_batch_size,numscenario_batches,self.scenario_batch_size)+[numscenario_batches]:
			actual_batch_size = block_num-prev_block_num
			
			for key, value in self.stoch_factors.items():
				value.Generate ( self.module, self.CudaMem, self.prec, self.time_grid.scen_time_grid, actual_batch_size, self.calcs_per_batch, random_index+self.process_ofs[key], self.stoch_ofs[key] )
			
			#now ask the netting set to price each deal
			self.netting_sets.ResolveStructure ( self.module, self.CudaMem, self.prec, self.calcs_per_batch, prev_block_num, block_num )

			if params.get('Calc_Scenarios','No') == 'Yes':
				#store the results	
				self.Scenarios [ self.calcs_per_batch*prev_block_num : self.calcs_per_batch*block_num ] = self.CudaMem.d_Scenario_Buffer.get()[:actual_batch_size*self.calcs_per_batch]
				
			prev_block_num = block_num
			random_index  += actual_batch_size*self.calcs_per_batch*self.time_grid.scen_time_grid.size*self.num_factors
			
	def Execute(self, params, FreeMem=True):
		#get the rundate		
		base_date = pd.Timestamp(params['Run_Date'])
		#check if we need to return cashflows
		calc_cashflows = True if params['Generate_Cashflows'] == 'Yes' else False		
		#check if we need to produce a slideshow
		if params['Generate_Slideshow'] == 'Yes':
			self.slides = []
			#we need to generate scenarios to draw pictures . . .
			params['Calc_Scenarios'] = 'Yes'
			#check if we need to compare the results to an externally produced result
			compare_output = pd.read_csv ( params['PFE_Recon_File'], index_col = 0, parse_dates=True ) if params['PFE_Recon_File'] else None
			#set the name of this calculation
			self.name = params['calc_name'][0]
			#talk
			self.ExplainCalc(
				('Intro', 'Counterparty Risk Simulation & Trading Analytics Library', 'Credit Monte Carlo Calculation'),
				['The calculation proceeds as follows:',
				 '- Dependent Risk Factors are identified',
				 '- Cuda is initalized',
				 '- Correlated Random Numbers are generated',
				 '- Scenarios using Stochastic Processes are produced',
				 '- Instruments are valued',
				 '- Collateral (if any) applied'] )
		else:
			compare_output = ''
			self.slides = None

		#Define the base and scenario grids		
		self.Base_time_grid 		= params['Base_time_grid']
		self.Scenario_time_grid		= params['Scenario_time_grid']
					
		#first update the risk factors - and the correlation, time_grid etc
		self.UpdateFactors(params['Currency'], base_date)

		#determine the partition (if any)
		tagging_rule=None
		if params.get('Partition'):
			if params['Partition']=='Product_Type':
				headings   			= {'EN':'Energy', 'FX':'Forex', 'ED':'Equity', 'IR':'IntRate', 'CR':'Credit'}
				instrument_groups	= [[(instrument,headings[k]) for instrument in v[1]] for k,v in FieldMappings['Instrument']['groups'].items() if k<>'STR']
				groupings	 		= dict([item for sublist in instrument_groups for item in sublist])
				tagging_rule 		= lambda obj:groupings.get( obj.get('Object'), 'Not Classified' )
			elif params['Partition']=='Hedging':
				tagging_rule 		= lambda obj: 'Hedge' if obj.get( 'Reference', '' ).startswith('Hedge') else 'Existing'
			elif params['Partition']=='Custom':
				tagging_rule 		= lambda obj: params['Partition_String'] if reduce(operator.or_, [x in obj.get( 'Reference', '' ) for x in params['Partition_String'].split('|')]) else 'Everything else'

		#Work out the number of scenario batches	
		numscenario_batches		= params['Number_Simulations'] / self.calcs_per_batch + 1
		self.numscenarios  		= numscenario_batches * self.calcs_per_batch
		ScenarioBatchMemSize	= params.get( 'ScenarioBatchMemorySize', 1536.0 )
		
		cudaState	= self.CudaStateClass(  BLOCK_DIM = self.calcs_per_batch,
											PRECISION = 'float' if self.prec==np.float32 else 'double',
											INDEXTYPE = 'int' if self.prec==np.float32 else 'long' )

		#Store the Scenarios in main memory
		if params['Calc_Scenarios'] == 'Yes':
			self.Scenarios 			= np.empty ( (self.numscenarios, self.size_stoch*self.time_grid.scen_time_grid.size), dtype=self.prec )
			Scenario_bytes			= self.Scenarios.nbytes			
		else:
			Scenario_bytes			= self.numscenarios*self.size_stoch*self.time_grid.scen_time_grid.size*(4 if self.prec==np.float32 else 8)
		
		if cudaState!=self.CudaState:
			self.CudaState = cudaState
			self.__ManageCudaState()

		#setup the device and alloacte memory	
		self.__RefreshCudaMem ( params['Random_Seed'], calc_cashflows, numscenario_batches, Scenario_bytes, MemoryInMBperScenarioBatch = ScenarioBatchMemSize )
		
		#now that the memory is setup, all python objects have had a chance to sync with the device (in terms of indices, offsets etc.)
		self.netting_sets = DealStructure( Aggregation('root'), self.time_grid.CurrencyMap, self.numscenarios, self.time_grid.mtm_time_grid.size, self.prec )
		self.SetDealStructures(self.config.deals['Deals']['Children'], self.netting_sets, self.numscenarios, tagging=tagging_rule, DealLevelMTM=False)
			
		#define the grid
		grid        		= (numscenario_batches, self.time_grid.scen_time_grid.size)
		block       		= (self.calcs_per_batch,1,1)

		#Adjust the random numbers to take into account they came from a sobolov sequence ?? Might want to think about this more
		#AdjustNormals 		= self.module.get_function ('AdjustNormals')
		#AdjustNormals ( self.CudaMem.d_random_numbers, np.int32(self.time_grid.scen_time_grid.size) , block=block, grid= (numscenario_batches, 1) )
		
		#get the cuda functions for generating random numbers ready
		CalcCorrelated 		= self.module.get_function ('CalcCorrelated')

		#Set the reporting currency
		drv.memcpy_htod( self.module.get_global('ReportCurrency')[0], 	getFXRateFactor(Utils.CheckRateName(params['Currency']), self.static_ofs, self.stoch_ofs) )
		#Copy over the matrix and perform the correlation
		drv.memcpy_htod( self.CudaMem.d_Buffer.ptr, self.cholesky[np.tril_indices(self.num_factors)])
		#Now apply that cholesky to our random numbers
		CalcCorrelated ( self.CudaMem.d_random_numbers, block=block, grid=grid )
		
		#Talk about scenario generation
		self.ExplainCalc( ('Scenario Generation',),
						  ['All Risk Factors are classified as either Static (there is no SP attached) or Stochastic (a valid SP is attached).',
						   'Remember that there are {0} dates in the Scenario Time Grid'.format(self.time_grid.scen_time_grid.size),
						   '',
						   'For each Monte Carlo Simulation the following happens:',
						   '- {0} correlated random numbers are generated per Scenario time grid for a total of {1} ({0} x {2}) random numbers'.format(self.num_factors, self.num_factors*self.time_grid.scen_time_grid.size, self.time_grid.scen_time_grid.size),
						   '- At each scenario time point, the {0} correlated random numbers are transformed into realizations of its corresponding risk factor according to its SP'.format(self.num_factors),
						   '',
						   'Static Factors are simply read as-is',
						   '',
						   'In the figures below:',
						   '- The Red line represents the 97.5th Percentile',
						   '- The Green line represents the 50th Percentile',
						   '- The Blue line represents the 2.5th Percentile'
						   ] )
		
		#copy across the tenors to the constant memory buffer of the GPU	
		drv.memcpy_htod (self.CudaMem.d_Buffer.ptr, self.buffer )

		#Now use these numbers to calculate the risk factors as defined in each classes models . .
		self.RawExecute(params)
		
		#free the memory or the device runs out
		if FreeMem:
			self.FreeCudaMem()

		#talk some more . . . 
		self.ExplainCalc( ('Scenario Generation', 'Stochastic Processes'), [], scenarios=self.Scenarios )

		self.ExplainCalc( ('Instrument Valuation',),
						  ['Valuation for each instrument proceeds as follows:',
						   '- For each Monte Carlo simulation:'
						   '	- Determine the set of potential cashflow dates',
						   '	- Calculate (according to standard pricing funtions) the MTM on these dates AND on the day immediately after',
						   '		- Revaluation on the Base Time Grid is implicitly done',
						   '		- The effect of repricing instrument on cashflow dates highlights the effect of settlement on the exposure profile',
						   '	- Interpolate linearly the calculated MTM price between these dates to cover the full MTM time grid',
						   '		- The full MTM time grid consists of the Base Time Grid and the set of all instrument cashflow dates in the netting set',
						   '- This implicitly takes into consideration:',
						   '	- Past fixings (in the case of swaps/Fras)',
						   '	- Path dependency (in the case of barrier options)',
						   '',
						   'The output for each instrument valuation is a matrix of size {0} x {1} representing the number of simulations by the number of timesteps.'.format(self.numscenarios, self.time_grid.mtm_time_grid.size),
						   'Note that the number of simulations may not be exactly what was sent due to CUDA requiring it to be a multiple of {0}'.format(self.calcs_per_batch),
						   '',
						   'Each instrument matrix is then simply added together to obtain the valuation matrix of the netting set'] )
		
		self.ExplainCalc( ('Instrument Valuation', 'PFE'), [], netting = self.netting_sets, scenarios=self.Scenarios, currency=params['Currency'], compare_against=compare_output )
		
		self.ExplainCalc( ('Collateral',),
						  ['Currently, only cash collateral is supported in CRSTAL',
						   'The collateral is applied directly on the valuation matrix of the netting set and proceeds as follows:',
						   '- For each simulation:', 
						   '	- Determine cumulative net cashflows paid and received expressed in base currency',
						   '	- Determine actual cashflows paid/received during the settlment and liquidation period',
						   '	- Simulate collateral flows according to the Thresholds, Initial and Minimum Transfer amounts',
						   '	- If provided, calculate the discounted funding cost according to the collateral and funding curves',
						   '',
						   'Note that there is no optimization of collateral. This will be added at a later stage.'],
						  netting = self.netting_sets, currency=params['Currency'])
		
		self.ExplainCalc( ('Collateral','Plots'), [], netting = self.netting_sets )
		
		#return a dictionary of output
		output = {'Netting':self.netting_sets}
		if params.get('Calc_Scenarios','No') == 'Yes':
			output.update({'Scenarios':self.Scenarios})

		return output

class Base_Revaluation(Calculation):
	'''Simple deal revaluation - Use this to reconcile with the source system'''
	def __init__(self, config, prec=np.float64, calcs_per_batch=64):
		super(Base_Revaluation, self).__init__(config, prec)
		self.calcs_per_batch		= calcs_per_batch
		self.dependend_factors 		= None
		self.base_date 				= None
		
		#Cuda related variables to store the state of the device between calculations
		self.CudaStateClass			= namedtuple('CudaState','PRECISION INDEXTYPE')
		self.CudaState 				= None
		self.CudaMemClass			= namedtuple('CudaMem','d_Buffer d_Scenario_Buffer d_Static_Buffer d_MTM_Accum_Buffer d_MTM_Buffer d_Time_Grid d_Cashflows d_Cashflow_Index')
		self.CudaMem 				= None
		self.module 				= None
	
	def UpdateFactors(self, currency, base_date):
		dependend_factors, stochastic_factors, reset_dates, settlement_currencies = self.config.CalculateDependencies(currency, base_date, '0d', False)
		
		#update the time grid
		if self.base_date!=base_date:
			self.UpdateTimeGrid(base_date)

		#update the factor dependencies
		if dependend_factors!=self.dependend_factors:
			self.static_factors	= OrderedDict()
			for price_factor in dependend_factors:
				try:
					self.static_factors.setdefault( price_factor, ConstructFactor(price_factor.type, self.config.params['Price Factors'][ Utils.CheckTupleName(price_factor) ]) )
				except KeyError, e:
					logging.warning('Price Factor {0} missing in market data file - skipping'.format(e.message))

			self.all_factors    = self.static_factors

			self.static_val = np.array([], dtype=self.prec)
			self.static_ofs	= OrderedDict()
			
			#work out distinct tenor sets
			distinct_tenors = OrderedDict()

			for key, value in self.static_factors.items():
				if key.type not in Utils.DimensionLessFactors:
					#record the offset of this risk factor
					self.static_ofs.setdefault(key, len(self.static_val) )
					self.DistinctTenors(distinct_tenors, key, value)
					self.static_val = np.hstack ( ( self.static_val, value.CurrentVal() ) )

			#calculate a reverse lookup for the tenors and store the daycount code
			self.UpdateTenors(distinct_tenors)
			
			#Setup the buffer to (tenors + currency repo curves)
			self.SetBuffer(distinct_tenors)
			
			#store the state info
			self.dependend_factors = dependend_factors

	def UpdateTimeGrid(self, base_date):
		#setup the scenario and base time grids
		self.time_grid 	 = TimeGrid( set([base_date]), set([base_date]), set([base_date]) )
		self.base_date   = base_date
		self.reset_dates = None
		self.time_grid.SetBaseDate(base_date, self.prec, self.indexType)
		
	def __RefreshCudaMem(self, numscenario_batches):
		#work out a nice number of scenario processing batches and create memory on the device
		self.scenario_batch_size = int ( numscenario_batches )

		#copy over all the constants
		drv.memcpy_htod ( self.module.get_global('ScenarioTimeSteps')[0],		np.array( [ self.time_grid.scen_time_grid.size], dtype=np.int32) )
		drv.memcpy_htod ( self.module.get_global('MTMTimeSteps')[0],			np.array( [ self.time_grid.mtm_time_grid.size], dtype=np.int32) )
		drv.memcpy_htod ( self.module.get_global('NumTenors')[0],				np.array( [ self.tenor_size.sum() ], dtype=np.int32) )
		drv.memcpy_htod ( self.module.get_global('ScenarioTenorSize')[0],		self.tenor_size )
		drv.memcpy_htod ( self.module.get_global('ScenarioTenorOffset')[0], 	self.tenor_offset )
		drv.memcpy_htod ( self.module.get_global('NumCurveCurrencies')[0],		np.array( [ self.currency_curve_map.size ], dtype=np.int32) )
		drv.memcpy_htod ( self.module.get_global('CurrencyCurveMap')[0],		self.currency_curve_map )
		drv.memcpy_htod ( self.module.get_global('CurrencyCurveOffset')[0],		self.currency_curve_ofs )
		
		#allocate memory on the device
		self.CudaMem = self.CudaMemClass (  d_Buffer			= gpuarray.GPUArray( (self.max_buffer_size,), dtype=self.prec ),
											d_Scenario_Buffer   = np.intp(0),
											d_Static_Buffer     = gpuarray.to_gpu( self.static_val.astype(self.prec) ),
											d_MTM_Accum_Buffer  = gpuarray.GPUArray( (self.scenario_batch_size*self.calcs_per_batch, self.time_grid.mtm_time_grid.size), dtype=self.prec ),
											d_MTM_Buffer   		= gpuarray.GPUArray( (self.scenario_batch_size*self.calcs_per_batch, self.time_grid.mtm_time_grid.size), dtype=self.prec ),
											d_Time_Grid         = gpuarray.to_gpu ( self.time_grid.time_grid.ravel().astype(self.prec) ),
											d_Cashflows			= np.intp(0),
											d_Cashflow_Index	= np.intp(0)
										)
		
		#set the global buffer variable to the allocated buffer
		drv.memcpy_htod ( self.module.get_global('Buffer')[0],	np.array( [ self.CudaMem.d_Buffer.ptr ], dtype=np.intp ))
		
	def __ManageCudaState(self):
		'''
		Manage things setting constants, compiling the pricing functions etc. - should not need to concern the user
		'''
		#setup the GPU
		#convention used here is that all variables beginning with a "d_" are pointers to device memory
		#Fill in the parameters for our cuda functions . .
		#need to ask all the stoch parameters and pricing functions for their cuda functions, and dynamically compile this
		cudaBody		= Calculation.cudacodetemplate+reduce(operator.concat, GetAllInstrumentCudaFunctions(), '')
		cudaTemplate    = string.Template(Calculation.cudacodeheader+cudaBody+Calculation.cudacodefooter)		
		cudacode  		= cudaTemplate.substitute( self.CudaState._asdict() )		
		#self.module    	= pycuda.compiler.SourceModule(cudacode, no_extern_c=True, options=['--ptxas-options=-v'])
		self.module    	= pycuda.compiler.SourceModule(cudacode, no_extern_c=True, options=['-O3'])

	def FormatOutput(self, result, output):
		
		def GetMTM(path, structure):
			mtm   = []
			recon = []
			error = []
			objs  = []
			
			#add the object results first
			key 		= [ structure.obj.Instrument.field.get('Reference','.'.join(path) ) ]
			mtm.append   ( key + structure.partition.DealMTMs[0].tolist())
			recon.append ( key + [''] )
			error.append ( key + ['Unknown'] )
			objs.append  ( key + [structure.obj.Instrument.field.get('Object')] )
			
			for deal_data in structure.dependencies:
				key 		= [ str(deal_data.Instrument.field.get('Reference')) ]
				calc_mtm    = [ float(deal_data.Calc_res[0]) if deal_data.Calc_res else 0.0 ]
				try:
					recon_mtm 	= [ float(deal_data.Instrument.field.get('MtM')) ]
					error_mtm 	= [ '{0} %'.format( 100.0 * ( (calc_mtm[0]-recon_mtm[0])/(recon_mtm[0] if recon_mtm[0]!=0.0 else 1.0) ) ) ]
				except:
					recon_mtm 	= [ '' ]
					error_mtm   = [ 'Unknown' ]
					
				mtm.append   ( key + calc_mtm )
				recon.append ( key + recon_mtm )
				error.append ( key + error_mtm )
				objs.append  ( key + [deal_data.Instrument.field.get('Object')] )

			return [{"label":"Calc MTM", "data":mtm},{"label":"Ref MTM", "data":recon},{"label":"Relative Error", "data":error},{"label":"Object", "data":objs}]

		def presentStructure( structure, path, parent, out ):
			json_key = json.dumps ( list(path), separators=(',', ':') )
			
			if structure.dependencies:
				out.setdefault ( json_key, GetMTM(path, structure) )
			else:
				mtm   	= [structure.obj.Instrument.name] + structure.partition.DealMTMs[0].tolist()
				out.setdefault( json_key, [{"label":"Calc MTM", "data":[mtm]}] )
			
			for sub_structure in structure.sub_structures:
				group_data  = {}
				name        = "{0}.{1}".format( sub_structure.obj.Instrument.field.get('Object'), sub_structure.obj.Instrument.field.get('Reference') if sub_structure .obj.Instrument.field.get('Reference') else len(out) )
				
				group = {  "text" : name,
						   "type" : "group",
						   "data" : {},
						   "children" : [] }
				
				presentStructure ( sub_structure, path+(name,), group['children'], out )
				parent.append(group)

		out 			= {}
		deals_to_append = []
		
		presentStructure( result['Netting'], (), deals_to_append, out )
		
		tree_data = [{ "text" : "All",  
					   "type" : "root", 
					   "state" : { "opened" : True, "selected" : True }, 
					   "children" : deals_to_append} ]
			
		for field, field_data in output.items():
			if field=='MTM':
				field_data['value'] = json.dumps ( tree_data )
				field_data['profiles'] = json.dumps ( out )
					
	def Execute(self, params, FreeMem=True):
		#get the rundate		
		base_date = pd.Timestamp(params['Run_Date'])
					
		#first update the risk factors - and the correlation, time_grid etc
		self.UpdateFactors(params['Currency'], base_date)

		#reval just has 1 scenario - the identity scenario
		numscenario_batches	= 1		

		cudaState	= self.CudaStateClass(  PRECISION = 'float' if self.prec==np.float32 else 'double',
											INDEXTYPE = 'int' if self.prec==np.float32 else 'long' )

		if cudaState!=self.CudaState:
			self.CudaState = cudaState
			self.__ManageCudaState()

		#setup the device and alloacte memory	
		self.__RefreshCudaMem(numscenario_batches)
		
		#now that the cuda memory is setup, all python objects have had a change to sync with the device (in terms of indices, offsets etc.)
		self.netting_sets = DealStructure( Aggregation('root'), self.time_grid.CurrencyMap, self.calcs_per_batch, self.time_grid.mtm_time_grid.size, self.prec, True )
		self.SetDealStructures(self.config.deals['Deals']['Children'], self.netting_sets, self.calcs_per_batch, DealLevelMTM=True)

		#define the grid
		grid        		= (numscenario_batches, self.time_grid.scen_time_grid.size)
		block       		= (self.calcs_per_batch,1,1)
		#d_Buffer 			= self.module.get_global('Buffer')[0]

		#copy across the reporting currency
		drv.memcpy_htod ( self.module.get_global('ReportCurrency')[0], getFXRateFactor(Utils.CheckRateName(params['Currency']), self.static_ofs, {}) )
		
		#Now use these numbers to calculate the risk factors as defined in each classes models . .
		prev_block_num = 0
		random_index   = 0
		
		#copy across the tenors to the constant memory buffer of the GPU
		drv.memcpy_htod(self.CudaMem.d_Buffer.ptr, self.buffer)
		
		for block_num in range(self.scenario_batch_size,numscenario_batches,self.scenario_batch_size)+[numscenario_batches]:
			actual_batch_size = block_num-prev_block_num			
			
			#now ask the netting set to price each deal
			self.netting_sets.ResolveStructure( self.module, self.CudaMem, self.prec, self.calcs_per_batch, prev_block_num, block_num )

			prev_block_num = block_num
		
		#free the memory or the device runs out
		if FreeMem:
			self.FreeCudaMem()
		
		#return a dictionary of output
		output = {'Netting':self.netting_sets}
		logging.info('{0} Complete'.format(params['calc_name'][0]))
		return output
	
def ConstructCalculation(calc_type, config):
    return globals().get(calc_type)(config)

def blackEuropeanOption(F,X,r,vol,tenor,buyOrSell,callOrPut):
	import scipy.stats
	stddev = vol*np.sqrt(tenor)
	d1 = ( np.log (F / X) + 0.5 * stddev * stddev ) / stddev
	d2 = d1 - stddev
	return  buyOrSell * callOrPut * ( F * scipy.stats.norm.cdf(callOrPut * d1) - X * scipy.stats.norm.cdf (callOrPut * d2) ) * np.exp (-r * tenor)

def drawcf(date, offset, rate, notional, fixed):
	pay = date.strftime('%d%b%Y')
	start = (date-offset).strftime('%d%b%Y')
	end   = pay
	#return '[Payment_Date={},Notional={},Rate={},Accrual_Start_Date={},Accrual_End_Date={},Accrual_Day_Count=0,Accrual_Year_Fraction=,Fixed_Amount={},Discounted=0,FX_Reset_Date=,Known_FX_Rate=]'.format(pay, notional, rate, start, end, fixed)
	return '[Payment_Date={},Fixed_Price={},Volume={}]'.format(pay, notional, rate)

def check_prices(n, prefix=''):
	for sub_struct in n.sub_structures:
		ref = sub_struct.obj.Instrument.field.get('Reference','?')
		obj = sub_struct.obj.Instrument.field['Object']
		val = sub_struct.obj.Calc_res[0] if sub_struct.obj.Calc_res else 0.0
		print '{}{}\t{}\t{}'.format(prefix,ref,obj,val)
		check_prices(sub_struct, '\t'+prefix)
		
	for deal in n.dependencies:
		ref = deal.Instrument.field['Reference']
		obj = deal.Instrument.field['Object']
		val = deal.Calc_res[0] if deal.Calc_res else 0.0
		print '{}{}\t{}\t{}'.format(prefix,ref,obj,val)

def Report(cmc, mtm):
	if mtm.sub_structures[0].partition.Collateral_Cash.any():
		time_grid = mtm.sub_structures[0].obj.Time_dep.deal_time_grid
		dates 	  = np.array(sorted(cmc.time_grid.mtm_dates))[time_grid]
		deal_mtm  = mtm.partition.DealMTMs[:,time_grid].astype(np.float64)
		cash_col  = mtm.sub_structures[0].partition.Collateral_Cash[:,time_grid].astype(np.float64)
		fva		  = mtm.sub_structures[0].partition.Funding_Cost[:,time_grid].astype(np.float64)
		return pd.DataFrame ( index=dates, data={
												'PFE': np.percentile(deal_mtm, 95, axis=0),
												'EE': deal_mtm.mean(axis=0),
												'Collateral (5%)': np.percentile( cash_col, 5, axis=0),
												'Collateral (expected)': cash_col.mean(axis=0),
												'Funding Cost': fva.mean(axis=0)
											})
	else:
		return pd.DataFrame(index=sorted(cmc.time_grid.mtm_dates), data={'PFE':np.percentile(mtm.partition.DealMTMs.astype(np.float64), 95, axis=0), 'EE':mtm.partition.DealMTMs.astype(np.float64).mean(axis=0)})

def Whatif (calc, factors, derivatives, scenarios, relative=False):
	time_grid = calc.netting_sets.sub_structures[0].obj.Time_dep.deal_time_grid
	dates 	  = np.array(sorted(calc.time_grid.mtm_dates))[time_grid]
	factors   = [calc.stoch_factors[f].factor for f in factors]
	results   = OrderedDict()

	if len(factors)==1:
		for stress in scenarios:
			mult	= 100.0 if relative else 10000.0
			unit    = '%' if relative else 'bp'
			heading = 'Up {0} {1}'.format(mult*stress, unit) if stress>0 else 'Down {0} {1}'.format(mult*-stress, unit)
			
			#bump the factor 
			factors[0].Bump(stress, relative=relative)
			#get the effect
			h=factors[0].GetDelta()
			#calc the change
			results[heading] = derivatives['f']+h*derivatives['f_x']+0.5*h*h*derivatives['f_xx']
			
		return pd.DataFrame ( index=dates, data=results )
	else:
		#need to draw a picture
		pass
							
def ExamineRiskFactor(cmc, factor, Scenarios):
	ofs 	= cmc.stoch_ofs[factor]
	
	if factor.type in Utils.OneDimensionalFactors:
		result  		= OrderedDict()
		tenors 			= cmc.stoch_factors[factor].factor.GetTenor()
		for index, scenario_point in enumerate(sorted(cmc.time_grid.scenario_dates)):
			result[scenario_point] = Scenarios[:,cmc.size_stoch*index + ofs: cmc.size_stoch*index + ofs + tenors.size]
	else:
		result = Scenarios[:,ofs::cmc.size_stoch]
	return result

def ExamineScenarios(cmc, factor, scenario_date, rundate, scenarios):
	m,s = cmc.stoch_factors[factor].TheoreticalMeanStd(rundate, (scenario_date-rundate).days )
	r=ExamineRiskFactor(cmc, factor, scenarios)
	return pd.DataFrame({'sim_mean':r[scenario_date].mean(axis=0), 'sim_std':r[scenario_date].std(axis=0), 'theo_mean':m, 'theo_std':s, 'mean rel error':(m-r[scenario_date].mean(axis=0))/m, 'std rel error':(s-r[scenario_date].std(axis=0))/s})

def ir_matrix(aa):
	import Instruments
	
	Base_time_grid 			= '0d 2d 1w(1w) 3m(1m) 2y(3m)'
	Scenario_time_grid 		= '0d 2d 1w(1w) 3m(1m) 2y(3m)'
		
	pfe_params 				= { 'calc_name':('IR_Matrix',), 'Base_time_grid':Base_time_grid, 'Scenario_time_grid':Scenario_time_grid,
								'Run_Date':rundate, 'Currency':'ZAR', 'ScenarioBatchMemorySize': 2500.0, 'Number_Simulations':5000,
								'Random_Seed':4126, 'Calc_Scenarios':'No', 'Generate_Cashflows':'No', 'Partition':'None',
								'Generate_Slideshow':'No', 'PFE_Recon_File':''}

	years       = [1,2,5,10,15,20,25]
	up_matrix   = {}
	down_matrix = {}
	
	ins                   = OrderedDict()
	ins['Object']         = 'NettingCollateralSet'
	ins['Reference']      = 'Matrix'
	ins['Netted']         = True
	ins['Collateralized'] = False

	netting_set = {'instrument':Instruments.ConstructInstrument( ins ), 'Children':[]}
	
	eff_date    = pd.Timestamp(rundate)+pd.DateOffset(days=1)
	roll_period = pd.DateOffset(months=3)
	matrix      = {}
	#set the netting set
	aa.deals['Deals']['Children'] = [netting_set]
	
	for fx_rate, discount_rate, forecast_rate  in [ ('ZAR','ZAR-SWAP','ZAR-SWAP'), 
													('USD','USD-MASTER','USD-LIBOR-3M'),
													('EUR','EUR-MASTER', 'EUR-EURIBOR-3M'),
													('JPY', 'JPY-MASTER', 'JPY-LIBOR-6M.JPY-LIBOR-3M'),
													('AUD', 'AUD-MASTER', 'AUD-BBSW-6M.AUD-BBSW-3M'),
													('GBP', 'GBP-MASTER', 'GBP-LIBOR-6M.GBP-LIBOR-3M') ][:1]:
		for tenor in years:
			float_leg              = OrderedDict()
			float_leg['Object']    = 'CFFloatingInterestListDeal'
			float_leg['Reference'] = 'Float_{0}_yrs'.format(tenor)            
			float_leg['Currency']      = fx_rate
			float_leg['Discount_Rate'] = discount_rate
			float_leg['Forecast_Rate'] = forecast_rate
			float_leg['Buy_Sell']       = 'Sell'
			float_leg['Cashflows']      = {'Items':[], 'Compounding_Method':'None'}
			
			fixed_leg              = OrderedDict()
			fixed_leg['Object']    = 'CFFixedInterestListDeal'
			fixed_leg['Reference'] = 'Fixed_{0}_yrs'.format(tenor)
			fixed_leg['Currency']      = fx_rate
			fixed_leg['Discount_Rate'] = discount_rate
			fixed_leg['Buy_Sell']       = 'Buy'
			fixed_leg['Cashflows']      = {'Items':[], 'Compounding':'No'}
			
			for pay_date in Instruments.generatedatesBackward ( eff_date+pd.DateOffset(years=tenor), eff_date, roll_period ):
				start_date   = pay_date - roll_period
				daycount_acc = Utils.GetDayCountAccrual(eff_date, (pay_date-start_date).days, Utils.DAYCOUNT_ACT365)
				
				payment = { 'Payment_Date': pay_date,
							'Accrual_Start_Date': start_date,
							'Accrual_End_Date': pay_date,
							'Accrual_Year_Fraction':daycount_acc,
							'Rate':Utils.Percent(100.0),
							'Notional':1.0,
							'Margin':Utils.Basis(0.0),
							'Resets':[[start_date,start_date,pay_date,daycount_acc,'No',Utils.Percent(0.0)]]
							}
				
				float_leg['Cashflows']['Items'].append(payment)
				fixed_leg['Cashflows']['Items'].append(payment)
				
			netting_set['Children'] = [ {'instrument':Instruments.ConstructInstrument( float_leg )}, 
										{'instrument':Instruments.ConstructInstrument( fixed_leg )}]

			calc 	  = ConstructCalculation('Base_Revaluation', aa)
			res 	  = calc.Execute ( { 'calc_name':('test1',), 'Run_Date':rundate, 'Currency':'ZAR' } )
			
			swap      = res['Netting'].sub_structures[0].dependencies            
			swap_rate = -swap[0].Calc_res[0]/swap[1].Calc_res[0]
			
			for cf in fixed_leg['Cashflows']['Items']:
				cf['Rate'] = Utils.Percent(100.0*swap_rate)
				
			pfecalc 	  = ConstructCalculation('Credit_Monte_Carlo', aa)
			pfe 	      = pfecalc.Execute ( pfe_params )
			
			mtm           = pfe['Netting'].sub_structures[0]
			time_grid     = mtm.obj.Time_dep.deal_time_grid
			dates 	      = np.array(sorted(pfecalc.time_grid.mtm_dates))[time_grid]
			deal_mtm      = mtm.partition.DealMTMs[:,time_grid].astype(np.float64)
			
			matrix.setdefault('Swap Rate', []).append(swap_rate)
			matrix.setdefault('97.5 Perc (%)', []).append(100.0*np.percentile(deal_mtm, 97.5, axis=0))
			matrix.setdefault('2.5 Perc (%)', []).append(100.0*np.percentile(deal_mtm, 2.5, axis=0))
			
		pd.DataFrame (data = matrix,  index = years ).to_csv('IR_{0}_Matrix.csv'.format(fx_rate))

if __name__=='__main__':
	#pay_date = pd.Timestamp('2015-05-07').strftime('%d%b%Y')
	#exit(0)
	
	if 0 or 'aa' not in locals():
		aa 		= Parser(None)
		rundate = '2016-03-04'
		path    = 'G:\\Arena\\{0}\\{1}'
		#path	= 'C:\\utils\\Synapse\\binary_trading\\notebooks\\CCR\\CRSTAL\\{0}\\{1}'
		
		#load calendars - not used yet - TODO
		aa.ParseCalendarfile(path[:-4].format('calendars.cal'))
		
		#aa.ParseMarketfile(path.format(rundate, 'MarketData.dat'))
		aa.ParseJson(path.format(rundate, 'MarketData.json'))
		#aa.ParseJson(path.format(rundate, 'MarketData_ImpliedFX.json'))
		#aa.ParseMarketfile(path.format(rundate, 'MarketData2.dat'))

		#aa.ParseTradefile(path.format(rundate, 'trs.aap'))
		#aa.ParseTradefile(path.format(rundate, 'swap.aap'))
		#aa.ParseTradefile(path.format(rundate, 'CrB_tammy_ISDA.aap'))
		#aa.ParseTradefile(path.format(rundate, 'mtmswap.aap'))
		
		#aa.ParseJson(path.format(rundate, 'eskom.json'))
		#aa.ParseJson(path.format(rundate, 'CrB_Sanlam_Arcelormittal_SA_PF_ISDA.json'))
		#aa.ParseJson(path.format(rundate, 'CrB_Investec_Bank_Plc_ISDA.json'))
		#aa.ParseJson(path.format(rundate, 'CrB_Bakwena_PCC_ISDA.json'))
		#aa.ParseJson(path.format(rundate, 'CrB_JPMorgan_Chase_NYK_ISDA.json'))
		aa.ParseJson(path.format(rundate, 'CrB_Goldman_Sachs_Int_ISDA.json'))
		#aa.ParseJson(path.format(rundate, 'CrB_Standard_Bank_ISDA.json'))
		#aa.ParseJson(path.format(rundate, 'CrB_Shanta_Mining_ISDA.json'))
		#aa.ParseJson(path.format(rundate, 'CrB_Deutsche_Frankfurt_ISDA.json'))
				
		#aa.ParseJson(path.format(rundate, 'CrB_Growthpoint_Properties_Ltd_ISDA.json'))
		#aa.ParseJson(path.format(rundate, 'CrB_BNP_Paribas__Paris__ISDA.json'))		
		
		#aa.ParseTradefile(path.format(rundate, 'gold3.aap'))
		#aa.ProcessHedgeCounterparties( [ path.format(rundate, 'eskom.json'),  path.format(rundate, 'CrB_JPMorgan_Chase_NYK_ISDA.json')] )

		#exit(0)

	#aa.ParseTradefile(path.format(rundate, 'absa.aap'))

	#Base_time_grid 			= '0d 2d 1w(1w) 3m(1m) 2y(3m)'
	Base_time_grid 			= '0d 2d 1w(1w) 3m(1m) 2y(3m)'
	Scenario_time_grid 		= '0d 2d 1w(1w) 3m(1m) 2y(3m)'
	
	params 					= { 'calc_name':('test1',), 'Base_time_grid':Base_time_grid, 'Scenario_time_grid':Scenario_time_grid,
								'Run_Date':rundate, 'Currency':'ZAR', 'ScenarioBatchMemorySize': 2000.0, 'Number_Simulations':5000,
								'Random_Seed':4126, 'Calc_Scenarios':'No', 'Generate_Cashflows':'Yes', 'Partition':'None',
								'Generate_Slideshow':'No', 'PFE_Recon_File':''}
	#ir_matrix(aa)
	
	#turn off Collateral
	aa.deals['Deals']['Children'][0]['instrument'].field['Collateralized']='False'	
	aa.deals['Deals']['Children'][0]['instrument'].field['Collateralized']='False'	

	cmc 	 = ConstructCalculation('Credit_Monte_Carlo', aa)
	#cmc 	 = ConstructCalculation('Base_Revaluation', aa)

	#todo - work out the priority
	for scensize in np.arange(1500.0,3000.0,500.0):
		params['ScenarioBatchMemorySize']=scensize
		t0 = time.time()
		n = cmc.Execute ( params )
		t1 = time.time()
		total = t1-t0
		print 'total',scensize,total 

	#factors_to_bump = [ Utils.Factor('FxRate', ('ZAR',)), Utils.Factor('InterestRate', (u'USD-MASTER',)) ]
	#factors_to_bump = [ Utils.Factor('InterestRate', (u'USD-MASTER',)) ]
	
	#diffs = cmc.FiniteDifferences ( factors_to_bump, params, RelativeBumps=True, discount_curves = {'collateral':'InterestRate.USD-OIS', 'funding':'InterestRate.USD-FUNDING'} )
	
	#df = Whatif ( cmc, factors_to_bump, diffs, [-0.5, -0.1, -0.01, 0.01, 0.1, 0.5], relative=True )
	#df = Whatif ( cmc, factors_to_bump, diffs, [-50*0.01*0.01, -25*0.01*0.01, -0.01*0.01, 0.0, 0.01*0.01, 25*0.01*0.01, 50*0.01*0.01], relative=False )

	#c=cmc.GetCashflows(n['Netting'])
	#results
	#df = Report(cmc, n['Netting'])
	#print df
	#check_prices(n['Netting'])
	#factor = Utils.Factor('FxRate', ('ZAR',))
	#r=ExamineRiskFactor(cmc, factor, n['Scenarios'])
	#fx=pd.DataFrame(np.array([np.interp(cmc.time_grid.mtm_time_grid, cmc.time_grid.scen_time_grid, fx) for fx in ExamineRiskFactor(cmc, factor, n['Scenarios'])]) , columns=sorted(cmc.time_grid.mtm_dates)).to_clipboard()
	#expiry=n['Netting'].sub_structures[0].DealMTMs[:,16]
	#test=n['Netting'].sub_structures[0].DealMTMs[:,32].copy()
	#test[expiry>=0]=0
	#print i, np.percentile(test, 95)
	
##	s= ExamineScenarios(cmc, factor = Utils.Factor('InterestRate', ('AUD-BBSW-6M',)),
##							 scenario_date = pd.Timestamp('2023-10-23'),
##							 rundate = pd.Timestamp(rundate),
##							 scenarios = n['Scenarios'])
	
	#c=cmc.GetCashflows(n['Netting'])
	#now delete a netting set
		
	#n2 = cmc.Execute ( { 'Base_time_grid':Base_time_grid, 'Scenario_time_grid':Scenario_time_grid, 'Run_Date':'2014-10-05', 'Currency':'ZAR', 'Number_Simulations':5000, 'Random_Seed':10, 'Calc_Scenarios':'No', 'Generate_Cashflows':'No' } )
	#res = pd.DataFrame(data=n['Netting'].DealMTMs, columns = [pd.Timestamp(rundate)+pd.offsets.Day(int(x)) for x in cmc.time_grid.time_grid[:,1]])

	#cf=cmc.GetCashflows(n['Netting'])
	#cmc.PlotRiskFactorCurve(s, ('InterestRate', 'ZAR'))