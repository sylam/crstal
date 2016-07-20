#import standard libraries
import sys
import logging

if sys.version[:3]>'2.6':
    from collections import OrderedDict
else:
    from ordereddict import OrderedDict

#utility functions and constants
import Utils

#specific modules
try:
	import numpy as np
	import pandas as pd
	import pycuda.driver as drv
except:
    logging.warning('Import failed for numpy/pandas/pycuda')

def generatedatesBackward(end_date, start_date, offset, bus_day=None):
	dates = [end_date]
	while end_date>start_date:
		end_date-=offset
		new_date = max(end_date,start_date)
		dates.append( bus_day.rollforward(new_date) if bus_day else new_date )
	dates.reverse()	
	return pd.DatetimeIndex( dates )

##def generatedatesBackward(end_date, start_date, period, bus_day=None, clip=True):
##	i, new_date = 1, end_date
##	dates       = [new_date]
##	while new_date>start_date:
##		new_date = max ( end_date-period(i), start_date ) if clip else end_date-period(i)
##		adj_date = bus_day.rollforward(new_date) if bus_day else new_date
##		dates.append( adj_date if adj_date.month==new_date.month else bus_day.rollback(new_date) )
##		i+=1
##	dates.reverse()	
##	return pd.DatetimeIndex( dates )

def packoffsets(offsets):
	return np.hstack(([offsets.shape[0]],offsets.ravel())).astype(np.int32)
		
def calc_factor_index(field, static_offsets, stochastic_offsets, all_tenors={}):
	'''Utility function to determine if a factor is static or stochastic and returns its offset in the scenario block'''
	if static_offsets.get ( field ) != None:
		return [1, static_offsets.get ( field ) ] + all_tenors.get(field,[])
	else:
		return [2, stochastic_offsets.get ( field ) ] + all_tenors.get(field,[])

def calc_sub_factor(field, sub_field, static_offsets, stochastic_offsets, all_factors):
	if static_offsets.get ( field ) != None:
		return static_offsets.get ( field ).param.get ( sub_field )
	else:
		return stochastic_offsets.get ( field ).factor.param.get ( sub_field )

def getRecoveryRate( name, all_factors):
	'''Read the Recovery Rate on a Survival Probability Price Factor'''
	survival_prob = all_factors.get( Utils.Factor ('SurvivalProb',name ) )
	return survival_prob.factor.RecoveryRate() if hasattr(survival_prob, 'factor') else survival_prob.RecoveryRate()

def getInterestRateCurrency( name, all_factors ):
	'''Read the Recovery Rate on a Survival Probability Price Factor'''
	ir_curve = all_factors.get ( Utils.Factor('InterestRate',name) )
	return ir_curve.factor.GetCurrency() if hasattr(ir_curve, 'factor') else ir_curve.GetCurrency()
	
def getInflationIndexName ( fieldname, all_factors ):
	'''Read the Name of the Price Index price factor linked to this inflation index'''
	inflation = all_factors.get( Utils.Factor('InflationRate',fieldname) )
	return inflation.factor.param.get('Price_Index') if hasattr(inflation, 'factor') else inflation.param.get('Price_Index')

def getForwardPriceVol ( fieldname, all_factors ):
	'''Read the Forward Price volatility factor linked to this Reference Vol'''
	pricevol = all_factors.get(Utils.Factor('ReferenceVol',fieldname))
	return pricevol.GetForwardPriceVol()

def getInflationIndexObjects ( inflation_name, index_name, all_factors ):
	'''Read the Name of the Price Index price factor linked to this inflation index'''
	inflation 			= all_factors.get(Utils.Factor('InflationRate',inflation_name))
	inflation_factor 	= inflation.factor if hasattr(inflation, 'factor') else inflation
	index   			= all_factors.get(Utils.Factor('PriceIndex',index_name))
	index_factor     	= index.factor if hasattr(index, 'factor') else index
	return inflation_factor, index_factor

def getFXRateFactor(fieldname, static_offsets, stochastic_offsets):
	'''Read the index of the FX rate price factor'''
	return packoffsets ( np.array ( [ calc_factor_index( Utils.Factor('FxRate', fieldname ), static_offsets, stochastic_offsets) ] ) )

def getForwardPriceSampling(fieldname, all_factors):
	'''Read the sampling offset for the Sampling_type price factor'''
	return all_factors.get( Utils.Factor('ForwardPriceSample',fieldname ) )

def getForwardPriceFactor(fieldname, cashflow_currency, static_offsets, stochastic_offsets, all_tenors, reference_factor, forward_factor, base_date):
	'''Read the Forward price factor of a reference Factor - adjusts the all_tenors lookup to include the excel_date version of the base_date'''
	
	forward_price 	  								= reference_factor.GetForwardPrice()
	all_tenors[Utils.Factor('ForwardPrice',forward_price)][-1] = (base_date-reference_factor.start_date).days	
	forward_offset  								= packoffsets ( np.array ( [ calc_factor_index( Utils.Factor('ForwardPrice',forward_price), static_offsets, stochastic_offsets, all_tenors ) ] ) )
	
	if cashflow_currency != forward_factor.GetCurrency():
		#compo deal
		forward_price_ofs	= getFXRateFactor ( forward_factor.GetCurrency() , static_offsets, stochastic_offsets )
		cashflow_ofs		= getFXRateFactor ( cashflow_currency, static_offsets, stochastic_offsets )
		forward_offset[0]   = -forward_offset[0]
		#my word - this is bad - need to find a better way to do this
		return np.hstack( ( forward_offset, forward_price_ofs, cashflow_ofs ) ).astype(np.int32)
	else:
		return forward_offset

def getReferenceFactorObjects(fieldname, all_factors):
	'''Read the Reference and Forward price factors'''
	reference 			= all_factors.get( Utils.Factor('ReferencePrice',fieldname ) )
	reference_factor 	= (reference.factor if hasattr(reference, 'factor') else reference)
	forward_price_name 	= reference_factor.GetForwardPrice()
	forward   			= all_factors.get( Utils.Factor('ForwardPrice',forward_price_name ) )
	forward_factor 		= (forward.factor if hasattr(forward, 'factor') else forward)
	return reference_factor, forward_factor

def getImpliedCorrelation( rate1, rate2, all_factors ):
	correlation_name = rate1[:-1] + ('{0}/{1}'.format(rate1[-1],rate2[0]),) + rate2[1:]+(rate1[1],)
	implied_correlation = all_factors.get( Utils.Factor ( 'Correlation', correlation_name ) )
	return implied_correlation.CurrentVal() if implied_correlation else 0.0

def getEquityRateFactor(fieldname, static_offsets, stochastic_offsets):
	'''Read the index of the Equity rate price factor'''
	return packoffsets ( np.array ( [ calc_factor_index( Utils.Factor('EquityPrice',fieldname), static_offsets, stochastic_offsets) ] ) )

def getEquityCurrencyFactor(fieldname, static_offsets, stochastic_offsets, all_factors):
	'''Read the index of the Equity's Currency price factor'''
	equity_factor = all_factors.get( Utils.Factor('EquityPrice',fieldname) )
	fxrate 	      = (equity_factor.factor if hasattr(equity_factor, 'factor') else equity_factor).GetCurrency()
	return packoffsets ( np.array ( [ calc_factor_index( Utils.Factor('FxRate',fxrate), static_offsets, stochastic_offsets) ] ) )

def getDividendRateFactor(fieldname, static_offsets, stochastic_offsets, all_tenors):
	'''Read the index of the dividend rate price factor'''
	return packoffsets ( np.array ( [ calc_factor_index(Utils.Factor('DividendRate',fieldname), static_offsets, stochastic_offsets, all_tenors) ] ) )

def getEquityDividentFactors(fieldname, static_offsets, stochastic_offsets, all_tenors):
	'''Read the index of the equity and the dividend rate price factor and return it as 1 array'''
	equity_ofs 		= getEquityRateFactor ( fieldname, static_offsets, stochastic_offsets )
	dividend_ofs	= getDividendRateFactor ( fieldname, static_offsets, stochastic_offsets, all_tenors )
	#this is bad - need to find a better way to do this
	return np.hstack( ( equity_ofs, dividend_ofs ) ).astype(np.int32)

def getEquityZeroRateFactor (fieldname, static_offsets, stochastic_offsets, all_tenors, all_factors):
	'''Read the equity's interest rate price factor'''
	equity_factor = all_factors.get( Utils.Factor('EquityPrice',fieldname) )
	ir_curve 	  = (equity_factor.factor if hasattr(equity_factor, 'factor') else equity_factor).GetRepoCurveName()
	return packoffsets ( np.array ( [ calc_factor_index( Utils.Factor('InterestRate',ir_curve), static_offsets, stochastic_offsets, all_tenors ) ] ) )	

def getDiscountFactor(fieldname, static_offsets, stochastic_offsets, all_tenors, all_factors):
	'''Get the interest rate curve linked to this discount rate price factor'''
	discout_curve = all_factors.get( Utils.Factor('DiscountRate',fieldname) ).GetInterestRate()
	return packoffsets ( np.array ( [ calc_factor_index( Utils.Factor('InterestRate', discout_curve[:x]), static_offsets, stochastic_offsets, all_tenors ) for x in range(1,len(discout_curve)+1) ] ) )

def getInterestFactor(fieldname, static_offsets, stochastic_offsets, all_tenors):
	'''Read the index of the interest rate price factor'''
	return packoffsets ( np.array ( [ calc_factor_index(Utils.Factor('InterestRate',fieldname[:x]), static_offsets, stochastic_offsets, all_tenors) for x in range(1,len(fieldname)+1) ] ) )

def getFXZeroRateFactor (fieldname, static_offsets, stochastic_offsets, all_tenors, all_factors):
	'''Read the Currency's interest rate price factor'''
	fx_factor 	= all_factors.get ( Utils.Factor('FxRate',fieldname) )
	ir_curve 	= (fx_factor.factor if hasattr(fx_factor, 'factor') else fx_factor).GetRepoCurveName(fieldname)
	return getInterestFactor(ir_curve, static_offsets, stochastic_offsets, all_tenors)

def getPriceIndexFactor(fieldname, static_offsets, stochastic_offsets):
	'''Read the index of the Price Index price factor'''
	return packoffsets ( np.array ( [ calc_factor_index( Utils.Factor('PriceIndex',fieldname), static_offsets, stochastic_offsets) ] ) )

def getSurvivalFactor(fieldname, static_offsets, stochastic_offsets, all_tenors):
	'''Read the index of the inflation rate price factor'''
	return packoffsets ( np.array ( [ calc_factor_index( Utils.Factor('SurvivalProb',fieldname), static_offsets, stochastic_offsets, all_tenors) ] ) )

def getInflationFactor(fieldname, static_offsets, stochastic_offsets, all_tenors):
	'''Read the index of the inflation rate price factor'''
	return packoffsets ( np.array ( [ calc_factor_index(Utils.Factor('InflationRate',fieldname[:x]), static_offsets, stochastic_offsets, all_tenors) for x in range(1,len(fieldname)+1) ] ) )

def getFXVolFactor(fieldname, static_offsets, stochastic_offsets, all_tenors):
	'''Read the index of the fx vol price factor'''
	return packoffsets ( np.array ( [ calc_factor_index( Utils.Factor('FXVol',fieldname), static_offsets, stochastic_offsets, all_tenors) ] ) )

def getEquityPriceVolFactor(fieldname, static_offsets, stochastic_offsets, all_tenors):
	'''Read the index of the Equity Price vol price factor'''
	return packoffsets ( np.array ( [ calc_factor_index( Utils.Factor('EquityPriceVol',fieldname), static_offsets, stochastic_offsets, all_tenors) ] ) )

def getInterestVolFactor (fieldname, tenor, static_offsets, stochastic_offsets, all_tenors ):
	'''Read the index of the interest vol price factor'''
	pricefactor = 'InterestRateVol' if pd.Timestamp('1900-01-01')+tenor <= pd.Timestamp('1900-01-01') + pd.DateOffset(years=1) else 'InterestYieldVol'
	return packoffsets ( np.array ( [ calc_factor_index( Utils.Factor(pricefactor,fieldname), static_offsets, stochastic_offsets, all_tenors) ] ) )

def getForwardPriceVolFactor ( fieldname, static_offsets, stochastic_offsets, all_tenors ):
	'''Read the index of the forward vol price factor'''
	return packoffsets ( np.array ( [ calc_factor_index( Utils.Factor('ForwardPriceVol',fieldname), static_offsets, stochastic_offsets, all_tenors) ] ) )

class Deal(object):
	'''
	Base class for representing a trade/deal. Needs to be able to aggregate sub-deals (e.g. a netting set can be a "deal") and calculate dynamic dates for resets. 
	'''
	def __init__(self, params):
		self.field 				= params
		#is this instrument path dependent
		self.path_dependent 	= False
		#should this deal use the accumulator (evaluate child dependencies?)
		self.accum_dependencies = False
		
	def reset(self):
		self.reval_dates  = set()
		self.settlement_currencies = {}

	def add_grid_dates(self, parser, base_date, grid):
		pass

	def add_reval_date_offset(self, offset, relative_to_settlement=True):
		if relative_to_settlement:
			for fixings in self.settlement_currencies.values():
				self.reval_dates.update([x+pd.DateOffset(days=offset) for x in fixings])
		else:
			self.reval_dates.update([x+pd.DateOffset(days=offset) for x in self.reval_dates])
	
	def add_reval_dates(self, dates, currency=None):
		self.reval_dates.update(dates)
		if currency:
			self.settlement_currencies.setdefault(currency, set()).update(dates)
		
	def get_reval_dates(self):
		return self.reval_dates
	
	def finalize_dates ( self, parser, base_date, grid, node_children, node_resets, node_settlements ):
		if self.path_dependent:
			self.add_grid_dates ( parser, base_date, grid )

		node_resets.update( self.get_reval_dates() )
		for settlement_currency, settlement_dates in self.get_settlement_currencies().items():
			node_settlements.setdefault(settlement_currency, set()).update( settlement_dates )

	def get_settlement_currencies (self):
		return self.settlement_currencies

	def Calculate ( self, module, CudaMem, precision, batch_size, revals_per_batch, deal_data ):
		#generate the theo price
		self.Generate ( module, CudaMem, precision, batch_size, revals_per_batch, deal_data.Factor_dep, deal_data.Time_dep )
		#interpolate it
		self.InterpolateBlock ( module, CudaMem, deal_data, batch_size, revals_per_batch )

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, factor_dep, time_dep):
		pass
	
	def InterpolateBlock ( self, module, CudaMem, deal_data, batch_size, revals_per_batch, net = False ):
		#determine which buffer needs interpolation
		block_buffer = CudaMem.d_MTM_Net_Buffer if net else CudaMem.d_MTM_Buffer
		
		if deal_data.Time_dep.interp_index.size:
			MTM_Interpolate 	= module.get_function ('MTM_Interpolate')
			grid        		= (batch_size, deal_data.Time_dep.interp_index.shape[0])
			block       		= (revals_per_batch,1,1)
			
			MTM_Interpolate ( drv.In (deal_data.Time_dep.interp_index.ravel().astype(np.int32) ), CudaMem.d_Time_Grid, block_buffer, block=block, grid=grid )
			
		if deal_data.Calc_res is not None:
			MTM = block_buffer.get()
			deal_data.Calc_res.append ( MTM[0,0] )
			#check for NAN's - can be removed once everything is working properly
			if MTM  [ MTM != MTM ].size:
				logging.error('NAN is present in MTM buffer - investigate deal {0}'.format(deal_data.Instrument.field['Reference']))

class NettingCollateralSet(Deal):
	#dependent price factors for this instrument
	factor_fields 		= {'Agreement_Currency':										['FxRate'],
						   'Funding_Rate':												['DiscountRate'],
						   'Balance_Currency':											['FxRate'],
						   ('Collateral_Assets','Cash_Collateral','Currency'):			['FxRate'],
						   ('Collateral_Assets','Cash_Collateral','Funding_Rate'):		['InterestRate'],
						   ('Collateral_Assets','Cash_Collateral','Collateral_Rate'):	['InterestRate'],
						   ('Collateral_Assets','Equity_Collateral','Equity'):			['EquityPrice']
						   }
	#UI help documentation
	required_fields     = {'Netted':						'Whether contained trades net when calculating exposure ()',
						   'Collateralized':				'Whether the exposure of contained trades is collateralized. If Yes, then the remaining properties define the terms on which exposure is collateralized. If No, then the remaining properties are ignored',
						   'Agreement_Currency':			'ID of the FX rate price factor in which terms of collateralization such as thresholds are specified. Requires this FX rate price factor',
						   'Independent_Amount':			'The amount of collateral which is posted in addition to collateral to cover current exposure',
						   'Minimum_Received':				'Minimum transfer amount for received collateral',
						   'Minimum_Posted':				'Minimum transfer amount for posted collateral. A positive number',
						   'Opening_Balance':				'Opening collateral balance. Enter a value or leave as <undefined>',
						   'Received_Threshold':			'Received threshold value. If the portfolio value is above the received threshold then the agreed collateral amount is increased by the difference. Enter a value or leave as <undefined>',
						   'Posted_Threshold':				'Posted threshold value. If the portfolio value is below the posted threshold then the agreed collateral amount is reduced by the difference. Usually a negative number. Enter a value or leave as <undefined>',
						   'Settlement_Period':				'The number of days required to determine whether default has occurred, during which there is settlement risk',
						   'Liquidation_Period':			'The number of days required to liquidate the netting set after default has been established',
						   'Collateral_Call_Frequency':		'Frequency of collateral calls, with respect to the given Base_Collateral_Call_Date and Calendars',
						   'Cash_Collateral':				'List of cash collateral assets. Specific properties: 1) Currency - ID of the FX rate price factor for this cash. Requires this FX rate price factor. 2) Amount - Number of units of this cash in one unit of the collateral portfolio',
						   }

	#cuda code required to price this instrument
	cudacodetemplate = '''
		__global__ void RebaseCashflow(	const REAL* __restrict__ stoch_factors,
										const REAL* __restrict__ static_factors,
										const REAL* __restrict__ time_grid,
										const int*  __restrict__ cashflow_index,
										const REAL* __restrict__ all_cashflows,
										REAL* Cashflow_Pay,
										REAL* Cashflow_Rec,
										int   exclude_paid_today,
										int   num_currencies,
										const int* __restrict__ cashflow_currencies )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int scenario_index  		= scenario_number * ScenarioTimeSteps;
			REAL SumPayt				= 0.0;
			REAL SumRect				= 0.0;			
				
			for (int mtm_time_index=0; mtm_time_index<MTMTimeSteps; mtm_time_index++)
			{
				const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
				int scenario_prior_index  	= scenario_index + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
				int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
				REAL Payt					= 0.0;
				REAL Rect					= 0.0;
				
				for (int currency_id=0; currency_id<num_currencies; currency_id++)
				{									
					int cashflow_subindex = cashflow_index[ MTMTimeSteps*currency_id + mtm_time_index ];
					if ( cashflow_subindex >= 0 )
					{
						int  cashflow_offset 	= scenario_number * CurrencySettlementOffset[NumSettlementCurrencies] + CurrencySettlementOffset[currency_id] + cashflow_subindex;
						REAL Xt                 = ScenarioRate ( t, cashflow_currencies + currency_id*(RATE_INDEX_Size+FACTOR_INDEX_START), scenario_prior_index, static_factors, stoch_factors );
						
						Rect += Xt * max ((REAL)0.0, all_cashflows[cashflow_offset]);
						Payt += Xt * max ((REAL)0.0, -all_cashflows[cashflow_offset]);
					}
				}
				
				if (exclude_paid_today)
				{
					SumRect += Rect;
					SumPayt += Payt;
					
					Cashflow_Pay [index] = SumPayt;
					Cashflow_Rec [index] = SumRect;
				}
				else
				{
					Cashflow_Pay [index] = SumPayt;
					Cashflow_Rec [index] = SumRect;
					
					SumRect += Rect;
					SumPayt += Payt;
				}
			}
		}

		__device__ REAL St( REAL cash_haircut,
							REAL equity_haircut,
							const REAL* __restrict__ t,
							const int* __restrict__ collateral_currency,
							const int* __restrict__ collateral_equity,
							const int* __restrict__ collateral_equity_currency,
							int scenario_prior_index,
							const REAL* __restrict__ stoch_factors,
							const REAL* __restrict__ static_factors)
		{
			//Very basic implementation - assuming 1 unit each.
			//TODO - Generalize
			REAL St_i = 0.0;
			if (collateral_currency)
				St_i += ( 1.0-cash_haircut ) * ScenarioRate ( t, collateral_currency, scenario_prior_index, static_factors, stoch_factors );
			if (collateral_equity)
			{
				St_i += ( 1.0-equity_haircut ) * ScenarioRate ( t, collateral_equity, scenario_prior_index, static_factors, stoch_factors ) *
												ScenarioRate ( t, collateral_equity_currency, scenario_prior_index, static_factors, stoch_factors );
			}
			return St_i;
		}
		
		__global__ void CollateralCalcBt(	const REAL* __restrict__ stoch_factors,
											const REAL* __restrict__ static_factors,
											const REAL* __restrict__ time_grid,
											const REAL* __restrict__ Cashflow_Pay,
											const REAL* __restrict__ Cashflow_Rec,
											const REAL* __restrict__ Output,
											REAL* Collateral,
											int   exclude_paid_today,
											REAL receive_threshold,
											REAL posted_threshold,
											REAL independant_amount,
											REAL min_received,
											REAL min_posted,
											REAL cash_haircut,
											REAL equity_haircut,
											REAL opening_balance,
											const int* __restrict__ agreement_currency,
											const int* __restrict__ balance_currency,
											const int* __restrict__ collateral_currency,
											const int* __restrict__ collateral_equity,
											const int* __restrict__ collateral_equity_currency )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int scenario_index  		= scenario_number * ScenarioTimeSteps;

			REAL FX_balance				= ScenarioRate ( time_grid, balance_currency, scenario_index, static_factors, stoch_factors );
			
			//The value of collateral that should be held in base currency ignoring minimum transfer amounts - initialized to the opening balance
			REAL A_t 					= opening_balance * FX_balance;

			//The value in base currency of one unit of the collateral portfolio
			REAL S_t					= St(cash_haircut, equity_haircut, time_grid, collateral_currency, collateral_equity,
												collateral_equity_currency, scenario_index, stoch_factors, static_factors);
			
			//The number of units of collateral that ought to be held
			REAL B_t					= A_t/S_t;

			for ( int mtm_time_index=0; mtm_time_index<MTMTimeSteps; mtm_time_index++ )
			{
				const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
				int scenario_prior_index  	= scenario_index + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
				int index  					= scenario_number * MTMTimeSteps ;
				int total_index				= index+mtm_time_index;
				
				//will need to convert back to the base currency
				REAL FX_base				= ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
				
				//need the fx rate of the agreement currency
				REAL FX_Agreement			= ScenarioRate ( t, agreement_currency, scenario_prior_index, static_factors, stoch_factors );
				
				//Amount needed to adjust the mtm today
				REAL mtm_today_adj			= 0.0;
				if (exclude_paid_today)
				{
					mtm_today_adj			= 	( Cashflow_Rec[ total_index ] - ( mtm_time_index > 0 ? Cashflow_Rec[ total_index - 1 ] : 0.0 ) ) -
												( Cashflow_Pay[ total_index ] - ( mtm_time_index > 0 ? Cashflow_Pay[ total_index - 1 ] : 0.0 ) );
				}
				
				//The posted and recieved thresholds converted to base currency
				REAL H	= receive_threshold*FX_Agreement;
				REAL G	= posted_threshold*FX_Agreement;
				REAL Vt	= Output [ total_index ] * FX_base - mtm_today_adj;
				
				S_t = St(cash_haircut, equity_haircut, t, collateral_currency, collateral_equity, collateral_equity_currency, scenario_prior_index, stoch_factors, static_factors);
				A_t = independant_amount*FX_Agreement + ( Vt - H ) * ( Vt > H ) + ( Vt - G ) * ( Vt < G );
				B_t = ( mtm_time_index>0 && ( (A_t == 0) || (A_t - S_t*B_t >= min_received*FX_Agreement) || (S_t*B_t - A_t >= min_posted*FX_Agreement) ) ) ? A_t/S_t : B_t;
				//B_t = ( mtm_time_index>0 && ( (A_t - S_t*B_t >= min_received*FX_Agreement) || (S_t*B_t - A_t >= min_posted*FX_Agreement) ) ) ? A_t/S_t : B_t;

				//B_t is the number of units of the collateral currency that was placed/received
				Collateral [ total_index ] = B_t;
			}
		}
		
		__global__ void CollateralCalc(	const REAL* __restrict__ stoch_factors,
										const REAL* __restrict__ static_factors,
										const REAL* __restrict__ time_grid,
										const int*  __restrict__ deal_time_index,
										const REAL* __restrict__ Cashflow_Pay,
										const REAL* __restrict__ Cashflow_Rec,
										const REAL* __restrict__ Collateral,
										const REAL* __restrict__ Gross_MTM,
										REAL* Output,
										int   exclude_paid_today,
										const int* __restrict__ Ts,
										const int* __restrict__ Te,
										const int* __restrict__ Tl,
										int  settlement_style,
										REAL cash_haircut,
										REAL equity_haircut,
										const int* __restrict__ collateral_currency,
										const int* __restrict__ collateral_equity,
										const int* __restrict__ collateral_equity_currency )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps;
			OFFSET scenario_index		= scenario_number * ScenarioTimeSteps;
			OFFSET scenario_prior_index = scenario_index + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );

			//will need to convert back to the base currency
			REAL FX_base				= ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
			
			//The timepoints relevant to liquidation/settlement risk
			int  ts 					=  index + Ts[blockIdx.y];
			int  te 					=  index + Te[blockIdx.y];
			int  tl 					=  index + Tl[blockIdx.y];

			//Amount needed to adjust the mtm today
			REAL mtm_today_adj			= 0.0;
			if (exclude_paid_today)
			{
				mtm_today_adj			= 	( Cashflow_Rec[ te ] - ( mtm_time_index > 0 ? Cashflow_Rec[ te - 1 ] : 0.0 ) ) -
											( Cashflow_Pay[ te ] - ( mtm_time_index > 0 ? Cashflow_Pay[ te - 1 ] : 0.0 ) );
			}
				
			//get the uncolateralized value at time te
			REAL Vte 					= Gross_MTM [ te ] * FX_base - mtm_today_adj;
			
			//The cash retained during the settlement process
			REAL C_ts_te;
			
			switch (settlement_style) {
				case CASH_SETTLEMENT_Received_Only:
					C_ts_te = ( Cashflow_Rec[ te ] - Cashflow_Rec[ ts ] ) - ( Cashflow_Pay[ te ] - Cashflow_Pay[ ts ] );
					break;
				case CASH_SETTLEMENT_Paid_Only:
					C_ts_te = ( Cashflow_Rec[ te ] - Cashflow_Rec[ tl ] ) - ( Cashflow_Pay[ te ] - Cashflow_Pay[ tl ] );
					break;
				default:
					C_ts_te = ( Cashflow_Rec[ te ] - Cashflow_Rec[ ts ] ) - ( Cashflow_Pay[ te ] - Cashflow_Pay[ tl ] );
			}
			
			REAL S_t = St ( cash_haircut, equity_haircut, t, collateral_currency, collateral_equity, collateral_equity_currency, scenario_prior_index, stoch_factors, static_factors );
			
			//note - I'm cheating here - S_t is supposed to be S_te
			REAL min_B_t = Collateral [ tl ];

			for (int B_t=ts; B_t<tl; B_t++)
				min_B_t = min ( min_B_t, Collateral [ B_t ] );
			
			//Output [ index + mtm_time_index ] = ( Vte + C_ts_te - min_B_t*S_t ) / FX_base;
			Output [ index + mtm_time_index ] = ( Vte + C_ts_te - min_B_t * S_t ) / FX_base;
		}
		
		__global__ void FVACalc  (	    const REAL* __restrict__ stoch_factors,
										const REAL* __restrict__ static_factors,
										const REAL* __restrict__ time_grid,
										const int*  __restrict__ deal_time_index,
										const REAL* __restrict__ Collateral,
										REAL* Output,
										const int* __restrict__ collateral_currency,
										const int* __restrict__ collateral_rate,
										const int* __restrict__ funding_rate )
		{
			//Note that this routine must be called with the grid size the y dimension 1 less than the full time grid.
			
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			const REAL* t_next          = time_grid + TIME_GRID_Size*deal_time_index[blockIdx.y+1];
			int index  					= scenario_number * MTMTimeSteps;
			OFFSET scenario_index		= scenario_number * ScenarioTimeSteps;
			OFFSET scenario_prior_index = scenario_index + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			
			//will need to convert back to the base(reporting) currency			
			REAL S_t                    = collateral_currency ? ScenarioRate ( t, collateral_currency, scenario_prior_index, static_factors, stoch_factors ) : 1.0;
			REAL discount_funding  		= CalcDiscount ( t, t_next[TIME_GRID_MTM], funding_rate, scenario_prior_index, static_factors, stoch_factors );
			REAL discount_collateral	= CalcDiscount ( t, t_next[TIME_GRID_MTM], collateral_rate, scenario_prior_index, static_factors, stoch_factors );
			REAL discount_collateral_t0	= CalcDiscount ( time_grid, t[TIME_GRID_MTM], collateral_rate, scenario_index, static_factors, stoch_factors );
			REAL FX_base_t0				= ScenarioRate ( time_grid, ReportCurrency, scenario_index, static_factors, stoch_factors );
			
			Output [ index + mtm_time_index ] = ( Collateral [ index + mtm_time_index ] * S_t * ( discount_collateral/discount_funding - 1.0 ) * discount_collateral_t0 ) / FX_base_t0;
		}
	'''
	
	def __init__(self, params):
		super(NettingCollateralSet, self).__init__(params)
		self.path_dependent = True
		#need to make this read in from config file
		self.options 		= {'Cash_Settlement_Risk':Utils.CASH_SETTLEMENT_Received_Only,
							   'Forward_Looking_Closeout':False,
							   'Use_Optimal_Collateral':False,
							   'Exclude_Paid_Today':False}
		
		#make sure collateral default parameters are defined
		self.field.setdefault('Settlement_Period', 0)
		self.field.setdefault('Liquidation_Period', 0)
		self.field.setdefault('Opening_Balance', 0.0)

	def reset(self, calendars):
		super(NettingCollateralSet, self).reset()
		#allow the accumulator
		self.accum_dependencies = True
		#if we allow collateral, this instrument is now path dependent
		if self.field['Collateralized']=='True':
			self.path_dependent = True

	def finalize_dates ( self, parser, base_date, grid, node_children, node_resets, node_settlements ):
		
		def calc_liquidation_settlement_dates(date) :
			if self.options['Forward_Looking_Closeout']:
				tl = date+pd.DateOffset(days=self.field['Settlement_Period'])
				ts = date
			else:
				tl = max(date-pd.DateOffset(days=self.field['Liquidation_Period']), base_date)
				ts = max(date-pd.DateOffset(days=self.field['Settlement_Period']+self.field['Liquidation_Period']), base_date)
			return ts, tl

		#have to reset the original instrument and let the children nodes determine the outcome
		if self.field['Collateralized']=='True':
			#update each child element with extra reval dates
			for child in node_children:
				child.add_reval_date_offset(1)
				child.add_reval_date_offset(self.field['Settlement_Period'], relative_to_settlement=False )
				child.add_reval_date_offset(self.field['Settlement_Period']+self.field['Liquidation_Period'], relative_to_settlement=False )
				node_resets.update(child.get_reval_dates())
				
			#Load the time grid
			grid_dates = parser ( base_date, max(node_resets), grid )
			
			#load up all dynamic dates
			for dates in node_settlements.values():
				valid_fixings = np.clip( list(dates), base_date, max(dates) )
				grid_dates.update(valid_fixings)
				
			#determine the new dates to add
			node_additions 	= set()			
			#add a date just after the rundate
			node_additions.add ( base_date + pd.DateOffset(days=1) )
			
			#the current complete grid
			full_grid		= grid_dates.union ( np.clip(list(node_resets), base_date, max(node_resets)) )
			
			for date in sorted(grid_dates):
				ts, tl   = calc_liquidation_settlement_dates(date)
				if (tl not in full_grid) or (ts not in full_grid):
					node_additions.add(ts)
					node_additions.add(tl)
					
			#now set the reval dates
			fresh_grid 		 = full_grid.union(node_additions)
			self.reval_dates = set()

			for date in sorted(fresh_grid):
				ts, tl   = calc_liquidation_settlement_dates(date)
				if ts in fresh_grid and tl in fresh_grid:
					self.reval_dates.add(date)			
					
			#add more valuation nodes
			node_resets.update(node_additions)
		else:
			for child in node_children:
				child.add_reval_date_offset(1)
				node_resets.update(child.get_reval_dates())
				
			self.reval_dates = node_resets
			
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field 						= {}
		field_index				 	= {}
		
		#only set this up if this is a collateralized deal
		if self.field['Collateralized']=='True':
			field['Agreement_Currency'] = Utils.CheckRateName(self.field['Agreement_Currency'])
			field['Funding_Rate']		= Utils.CheckRateName(self.field['Funding_Rate']) if self.field.get('Funding_Rate') else None
			
			#apparently this should default to the base currency not the agreement currency, but I think this is a better default..
			field['Balance_Currency']	= Utils.CheckRateName(self.field['Balance_Currency']) if self.field['Balance_Currency'] else field['Agreement_Currency']
			
			field_index['Agreement_Currency'] 	= getFXRateFactor   ( field['Agreement_Currency'], static_offsets, stochastic_offsets )
			field_index['Balance_Currency'] 	= getFXRateFactor   ( field['Balance_Currency'], static_offsets, stochastic_offsets )

			#get the settlment currencies loaded
			field_index['Settlement_Currencies'] = OrderedDict()
			for currency in time_grid.CurrencyMap.keys():
				field_index['Settlement_Currencies'].setdefault ( currency, getFXRateFactor ( Utils.CheckRateName(currency), static_offsets, stochastic_offsets ) )

			#handle equity collateral
			Collateral_defined=False	
			if self.field.get('Collateral_Assets',{}).get('Equity_Collateral'):
				collateral_equity 						 = self.field['Collateral_Assets']['Equity_Collateral'][0]
				field['Equity_Collateral'] 				 = Utils.CheckRateName(collateral_equity['Equity'])
				field_index['Equity_Haircut'] 			 = collateral_equity['Haircut_Posted'] if isinstance(collateral_equity['Haircut_Posted'], float) else  collateral_equity['Haircut_Posted'].amount
				field_index['Equity_Currency']			 = getEquityCurrencyFactor ( field['Equity_Collateral'], static_offsets, stochastic_offsets, all_factors )
				field_index['Collateral_Equity']		 = getEquityRateFactor 	( field['Equity_Collateral'], static_offsets, stochastic_offsets )
				Collateral_defined						 = True
				#don't need this just yet
				#field_index['Collateral_Rate']		    = getInterestFactor ( Utils.CheckRateName ( self.field['Collateral_Assets']['Equity_Collateral'][0]['Collateral_Rate'] ),  static_offsets, stochastic_offsets, all_tenors )
				#field_index['Funding_Rate']			= getInterestFactor ( Utils.CheckRateName ( self.field['Collateral_Assets']['Equity_Collateral'][0]['Funding_Rate'] ),  static_offsets, stochastic_offsets, all_tenors )
			else:
				field_index['Equity_Haircut']			= 0.0
				field_index['Equity_Currency']			= np.zeros(1, dtype=np.int32)
				field_index['Collateral_Equity']	    = np.zeros(1, dtype=np.int32)
			
			#get the collateral currency loaded - not that I only simulate single currency collateral at the moment
			if self.field.get('Collateral_Assets',{}).get('Cash_Collateral'):
				collateral_cash 					= self.field['Collateral_Assets']['Cash_Collateral'][0]
				field_index['Cash_Haircut'] 		= collateral_cash['Haircut_Posted'] if isinstance(collateral_cash['Haircut_Posted'], float) else collateral_cash['Haircut_Posted'].amount
				field_index['Collateral_Currency'] 	= getFXRateFactor ( Utils.CheckRateName(collateral_cash['Currency']), static_offsets, stochastic_offsets )
				field_index['Collateral_Rate']		= getInterestFactor ( Utils.CheckRateName ( collateral_cash['Collateral_Rate'] ),  static_offsets, stochastic_offsets, all_tenors ) if collateral_cash.get('Collateral_Rate') else np.zeros(1, dtype=np.int32)
				field_index['Funding_Rate']			= getInterestFactor ( Utils.CheckRateName ( collateral_cash['Funding_Rate'] ),  static_offsets, stochastic_offsets, all_tenors ) if collateral_cash.get('Funding_Rate') else np.zeros(1, dtype=np.int32)
				Collateral_defined					= True
			else:
				field_index['Cash_Haircut'] 		= 0.0
				field_index['Collateral_Currency'] 	= np.zeros(1, dtype=np.int32)
				field_index['Collateral_Rate']		= np.zeros(1, dtype=np.int32)
				field_index['Funding_Rate']	 		= np.zeros(1, dtype=np.int32)

			if not Collateral_defined:
				#default the collateral to the be balance currency
				field_index['Collateral_Currency'] 	= getFXRateFactor ( field['Balance_Currency'], static_offsets, stochastic_offsets )
				
			#check if the independent amount has been mapped
			if self.field['Credit_Support_Amounts'].get('Independent_Amount'):
				field_index['Independent_Amount'] = self.field['Credit_Support_Amounts']['Independent_Amount'].value()
			else:
				field_index['Independent_Amount'] = 0.0
				
			#now get the closeout mechanics right
			t 			= time_grid.time_grid[:,Utils.TIME_GRID_MTM]
			#collateral call dates (interpolated - assuming daily)
			call_dates	= np.array([(x-base_date).days for x in sorted(self.get_reval_dates())])
			
			if self.options['Forward_Looking_Closeout']:
				field_index['Ts'] = ( np.searchsorted(t, call_dates, side='right').astype(np.int32)-1 ).clip(0,time_grid.mtm_time_grid.size-1)
				field_index['Tl'] = ( np.searchsorted(t, call_dates + self.field['Settlement_Period'], side='right').astype(np.int32)-1 ).clip(0,time_grid.mtm_time_grid.size-1)
				field_index['Te'] = ( np.searchsorted(t, call_dates + self.field['Settlement_Period'] + self.field['Liquidation_Period'], side='right').astype(np.int32)-1 ).clip(0,time_grid.mtm_time_grid.size-1)
			else:
				field_index['Ts'] = ( np.searchsorted(t, call_dates - self.field['Settlement_Period'] - self.field['Liquidation_Period'], side='right').astype(np.int32)-1 ).clip(0,time_grid.mtm_time_grid.size-1)
				field_index['Tl'] = ( np.searchsorted(t, call_dates - self.field['Liquidation_Period'], side='right').astype(np.int32)-1 ).clip(0,time_grid.mtm_time_grid.size-1)
				field_index['Te'] = ( np.searchsorted(t, call_dates, side='right').astype(np.int32)-1 ).clip(0,time_grid.mtm_time_grid.size-1)

		return field_index

	def Aggregate(self, CudaMem, parent_partition, index, partition):
		#pretty vanilla this - might need to generalize
		parent_partition.DealMTMs[ index ] += partition.DealMTMs [ index ]
		drv.memcpy_htod ( CudaMem.d_MTM_Accum_Buffer.ptr, parent_partition.DealMTMs [index] )
		
		#reset the cashflow buffer
		if CudaMem.d_Cashflows:
			parent_partition.Cashflows [ index ] += partition.Cashflows [ index ]
			drv.memcpy_htod ( CudaMem.d_Cashflows.ptr, parent_partition.Cashflows [ index ] )

		#in theory, we could also have ringfenced collateral balances that would need to be brought over too . . TODO

	def PostProcess( self, module, CudaMem, precision, batch_size, revals_per_batch, deal_data, child_dependencies, partition, mtm_offset ):
		#calc v^t = v(te) + C(ts,te) - min(B(u); ts<=u<=tl}S(te)
		dependencies		= deal_data.Factor_dep
		time_grid 			= deal_data.Time_dep
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, 1)

		if self.field['Collateralized']=='True' and CudaMem.d_Cashflows:
			RebaseCashflow 	= module.get_function ('RebaseCashflow')
			RebaseCashflow(	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, CudaMem.d_Time_Grid,
							CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_Cashflow_Pay, CudaMem.d_Cashflow_Rec,
							np.int32  ( self.options['Exclude_Paid_Today'] ),
							np.int32  ( len(dependencies['Settlement_Currencies']) ),
							drv.In    ( np.concatenate(dependencies['Settlement_Currencies'].values()) ),
							block=block, grid=grid )
			
			CollateralCalcBt = module.get_function ('CollateralCalcBt')
			CollateralCalcBt ( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, CudaMem.d_Time_Grid, 
								CudaMem.d_Cashflow_Pay, CudaMem.d_Cashflow_Rec, CudaMem.d_MTM_Accum_Buffer, CudaMem.d_MTM_Buffer,
								np.int32  ( self.options['Exclude_Paid_Today'] ),
								precision ( self.field['Credit_Support_Amounts']['Received_Threshold'].value() ),
								precision ( self.field['Credit_Support_Amounts']['Posted_Threshold'].value() ),
								precision ( dependencies['Independent_Amount'] ),
								precision ( self.field['Credit_Support_Amounts']['Minimum_Received'].value() ),
								precision ( self.field['Credit_Support_Amounts']['Minimum_Posted'].value() ),
								precision ( dependencies['Cash_Haircut'] ),
								precision ( dependencies['Equity_Haircut'] ),
								precision ( self.field['Opening_Balance'] ),
								drv.In    ( dependencies['Agreement_Currency'] ),
								drv.In    ( dependencies['Balance_Currency'] ),
								drv.In    ( dependencies['Collateral_Currency'] ),
								drv.In    ( dependencies['Collateral_Equity'] ),
								drv.In    ( dependencies['Equity_Currency'] ),
								block=block, grid=grid )

			CollateralCalc 	= module.get_function ('CollateralCalc')
			CollateralCalc( CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, CudaMem.d_Time_Grid, drv.In(time_grid.deal_time_grid), 
							CudaMem.d_Cashflow_Pay, CudaMem.d_Cashflow_Rec, CudaMem.d_MTM_Buffer, CudaMem.d_MTM_Accum_Buffer, CudaMem.d_MTM_Net_Buffer,
							np.int32  ( self.options['Exclude_Paid_Today'] ),
							drv.In    ( dependencies['Ts'] ),
							drv.In    ( dependencies['Te'] ),
							drv.In    ( dependencies['Tl'] ),
							np.int32  ( self.options['Cash_Settlement_Risk'] ),
							precision ( dependencies['Cash_Haircut'] ),
							precision ( dependencies['Equity_Haircut'] ),
							drv.In    ( dependencies['Collateral_Currency'] ),
							drv.In    ( dependencies['Collateral_Equity'] ),
							drv.In    ( dependencies['Equity_Currency'] ),
							block=block, grid = (batch_size, time_grid.deal_time_grid.size) )
			
			if dependencies['Funding_Rate'].any():
				FVACalc 	= module.get_function ('FVACalc')
				FVACalc  (	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, CudaMem.d_Time_Grid, drv.In(time_grid.deal_time_grid),
							CudaMem.d_MTM_Buffer, CudaMem.d_MTM_Accum_Buffer, 
							drv.In    ( dependencies['Collateral_Currency'] ),
							drv.In    ( dependencies['Collateral_Rate'] ),
							drv.In    ( dependencies['Funding_Rate'] ),
							block=block, grid = ( batch_size, time_grid.deal_time_grid.size - 1 ) )

				#copy across the Collateral cash
				partition.Funding_Cost [ mtm_offset ] = CudaMem.d_MTM_Accum_Buffer.get()[:batch_size*revals_per_batch]
			
			#copy across the Collateral cash
			partition.Collateral_Cash [ mtm_offset ] = CudaMem.d_MTM_Buffer.get()[:batch_size*revals_per_batch]

			#interpolate the Theo price
			self.InterpolateBlock ( module, CudaMem, deal_data, batch_size, revals_per_batch, net = True )
			
			#copy the mtm
			partition.DealMTMs  [ mtm_offset ] = CudaMem.d_MTM_Net_Buffer.get()[:batch_size*revals_per_batch]
			
		else:
			#copy the mtm
			partition.DealMTMs  [ mtm_offset ] = CudaMem.d_MTM_Accum_Buffer.get()[:batch_size*revals_per_batch]

		#copy the cashflows
		if CudaMem.d_Cashflows:
			partition.Cashflows [ mtm_offset ] = CudaMem.d_Cashflows.get()[:batch_size*revals_per_batch]

class MtMCrossCurrencySwapDeal(Deal):
	#dependent price factors for this instrument
	factor_fields = {'Pay_Interest_Rate_Volatility':		['InterestRateVol','InterestYieldVol'],
					 'Pay_Discount_Rate_Volatility':		['InterestRateVol','InterestYieldVol'],
					 'Receive_Interest_Rate_Volatility':	['InterestRateVol','InterestYieldVol'],
					 'Receive_Discount_Rate_Volatility':	['InterestRateVol','InterestYieldVol'],
					 'Pay_Currency':						['FxRate'],
					 'Pay_Discount_Rate':					['DiscountRate'],
					 'Pay_Interest_Rate':					['InterestRate'],
					 'Receive_Currency':					['FxRate'],
					 'Receive_Discount_Rate':				['DiscountRate'],
					 'Receive_Interest_Rate':				['InterestRate']}
	
	required_fields = {	'Effective_Date':					'The contract start date',
						'Maturity_Date':					'The contract end date',
						'Pay_Frequency':					'The payment frequency, which is the period between payments of the pay leg',
						'Pay_Margin':						'Margin rate added to the pay leg generalized cashflows. This property is required if the Pay_Rate_Type is set to "Floating" and ignored otherwise. Rates are entered in basis points. For example, enter 50 for 50 basis points',
						'Pay_Known_Rates':					'List of reset dates and corresponding known interest rates specified in the form of (date, value) pairs. This property is required if the Pay_Rate_Type is set to "Floating" and ignored otherwise',
						'Pay_Rate_Type':					'Interest rate type. Set to "Fixed" for a fixed pay leg or "Floating" for a floating pay leg',
						'Receive_Frequency':				'The payment frequency, which is the period between payments of the receive leg',
						'Receive_Margin':					'Margin rate added to the receive leg generalized cashflows. This property is required if the Receive_Rate_Type is set to "Floating" and ignored otherwise',
						'Receive_Known_Rates':				'List of reset dates and corresponding known interest rates specified in the form of (date, value) pairs. This property is required if the Receive_Rate_Type is set to "Floating" and ignored otherwise',
						'Pay_Currency':						'ID of the FX rate price factor used to define the pay leg settlement currency. For example, USD.',
						'Pay_Discount_Rate':				'ID of the discount rate price factor used to discount the pay leg cashflows. For example, USD.AAA',
						'Pay_Interest_Rate':				'ID of the interest rate price factor used to calculate forward interest rates. For example, USD.AAA',
						'Pay_Principal':					'Principal amount in units of settlement currency',
						'Pay_Amortisation':					'Permits an amortisation schedule for the principal to be specified as a list of (Amount, Date) pairs, where Amount represents the amount by which the principal is reduced by the Amortisation and Date represents the date of this Amortisation. A positive value for Amount represents a fall in the principal amount while a negative value for Amount represents a rise in the principal amount',
						'Pay_Fixed_Rate':					'Fixed interest rate. This property is required if the Pay_Rate_Type is set to "Fixed" and ignored otherwise',
						'Receive_Rate_Type':				'Interest rate type. Set to "Fixed" for a fixed receive leg or "Floating" for a floating receive leg',
						'Receive_Currency':					'ID of the FX rate price factor used to define the receive leg settlement currency. For example, USD',
						'Receive_Discount_Rate':			'ID of the discount rate price factor used to discount the receive leg cashflows. For example, USD.AAA',
						'Receive_Interest_Rate':			'ID of the interest rate price factor used to calculate forward interest rates. For example, USD.AAA',
						'Receive_Principal':				'Principal amount in units of settlement currency',
						'Receive_Amortisation':				'Permits an amortisation schedule for the principal to be specified as a list of (Amount, Date) pairs, where Amount represents the amount by which the principal is reduced by the Amortisation and Date represents the date of this Amortisation. A positive value for Amount represents a fall in the principal amount while a negative value for Amount represents a rise in the principal amount',
						'Receive_Fixed_Rate':				'Fixed interest rate. This property is required if the Receive_Rate_Type is set to "Fixed" and ignored otherwise',
						'Principal_Exchange':				'Type of principal exchange cashflows: no principal cashflows (None), principal cashflows at the effective date (Start), principal cashflows at the maturity date (Maturity), or principal cashflows at both the effective date and the maturity date (Start_Maturity)'
						}
	
	#cuda code required to price this instrument
	cudacodetemplate = '''
		__global__ void MtMCrossCurrencySwapDeal	(	const REAL* __restrict__ stoch_factors,
														const REAL* __restrict__ static_factors,
														const int*  __restrict__ deal_time_index,
														const REAL* __restrict__ time_grid,
														const int*  __restrict__ cashflow_index,
														REAL* all_cashflows,
														REAL* Output,
														int num_static_cashflow,
														const REAL* __restrict__ static_cashflows,
														const int*  __restrict__ static_starttime_index,
														const REAL* __restrict__ static_resets,
														const int*  __restrict__ static_currency,
														const int* __restrict__  static_forward,
														const int* __restrict__  static_discount,
														int   num_mtm_cashflow,
														const REAL* __restrict__ mtm_cashflows,
														const int*  __restrict__ mtm_starttime_index,
														const REAL* __restrict__ mtm_resets,
														const int* __restrict__  mtm_currency,
														const int* __restrict__  mtm_forward,
														const int* __restrict__  mtm_discount )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];			
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
			OFFSET scenario_index		= scenario_number * ScenarioTimeSteps;
			OFFSET scenario_prior_index = scenario_index + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			
			//set the mtm of this deal
			REAL mtm 					= 0.0;
			
			//FX rates for this deal
			REAL FX_Report				= ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
			REAL FX_Static_Base			= ScenarioRate ( t, static_currency, scenario_prior_index, static_factors, stoch_factors ) / FX_Report;
			REAL FX_MTM_Base			= ScenarioRate ( t, mtm_currency, scenario_prior_index, static_factors, stoch_factors ) / FX_Report;

			//static and mtm legs
			REAL static_leg, mtm_leg;
			
			//dummy variabe needed to call the cashflow pv functions
			REAL settlement_cashflow    = 0.0;
			
			if (static_starttime_index[mtm_time_index]<num_static_cashflow)
			{
				//if static_forward is Null, then the static leg is a fixed leg, otherwise it's floating
				if (static_forward)
				{
					static_leg	= ScenarioPVFloatLeg ( t, CASHFLOW_METHOD_Average_Interest, mtm_time_index, num_static_cashflow, settlement_cashflow, static_starttime_index,
														static_cashflows, static_resets, NULL, static_forward, static_discount, scenario_prior_index, static_factors, stoch_factors );
				}
				else
				{
					static_leg	= ScenarioPVFixedLeg ( t, CASHFLOW_METHOD_Fixed_Rate_Standard, CASHFLOW_METHOD_Fixed_Compounding_No, static_starttime_index[mtm_time_index], num_static_cashflow,
														settlement_cashflow, static_cashflows, static_discount, scenario_prior_index, static_factors, stoch_factors );
				}

				//Settle any cashflow									
				if (settlement_cashflow!=0)
					ScenarioSettle ( settlement_cashflow, static_currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows );

				mtm_leg		= ScenarioPVMTMFloatLeg (	t, mtm_time_index, num_mtm_cashflow, num_static_cashflow, settlement_cashflow,
														mtm_currency, mtm_starttime_index, mtm_cashflows, mtm_resets, mtm_forward, mtm_discount,
														static_currency, static_starttime_index, static_cashflows,
														scenario_index, scenario_prior_index, static_factors, stoch_factors );

				//Settle any cashflow
				if (settlement_cashflow!=0)
					ScenarioSettle ( settlement_cashflow, mtm_currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows );

				//convert to the settlement currency
				mtm 			= static_leg * FX_Static_Base + mtm_leg * FX_MTM_Base;
				/*
				if (threadIdx.x==2 && blockIdx.x==0 && blockIdx.y==2)
				{
					REAL 	forward_rate, fx_rate, fx_rate_next;

					int     mtm_start_cash 		= mtm_starttime_index ? mtm_starttime_index[mtm_time_index] : mtm_time_index;
					int     static_start_cash 	= static_starttime_index ? static_starttime_index[mtm_time_index] : mtm_time_index;
					
					const int* static_curve 	= CurrencyRepoCurve ( static_currency );
					const int* mtm_curve 		= CurrencyRepoCurve ( mtm_currency );

					settlement_cashflow	 		= 0.0;
					
					for (int i=mtm_start_cash; i<num_mtm_cashflow; i++)
					{
						const REAL* 	static_cashflow 		= static_cashflows + ( static_start_cash + i - mtm_start_cash ) * CASHFLOW_INDEX_Size;
						const REAL* 	next_static_cashflow 	= ( ( static_start_cash + i + 1 - mtm_start_cash ) < num_static_cashflow ) ?
																	static_cashflows + ( static_start_cash + i + 1 - mtm_start_cash ) * CASHFLOW_INDEX_Size :
																	NULL;
																	
						const REAL* 	mtm_cashflow 			= mtm_cashflows + i*CASHFLOW_INDEX_Size;
						const REAL* 	next_mtm_cashflow 		= ( ( i + 1 ) < num_mtm_cashflow ) ?
																	mtm_cashflows + (i+1)*CASHFLOW_INDEX_Size:
																	NULL;
																	
						const OFFSET	reset_offset 			= ( reinterpret_cast <const OFFSET &> ( mtm_cashflow[CASHFLOW_INDEX_ResetOffset] ) ) * RESET_INDEX_Size;
						const REAL* 	resets 					= mtm_resets + reset_offset;
						
						REAL ydT 			= calcDayCountAccrual ( mtm_cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], mtm_discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
						REAL discount_rate 	= ScenarioCurve ( t, ydT, mtm_discount, scenario_prior_index, static_factors, stoch_factors );
						
						if ( mtm_cashflow[CASHFLOW_INDEX_Start_Day] >= t[TIME_GRID_MTM] )
						{
							forward_rate  	= Calc_FutureReset ( t, mtm_forward, mtm_cashflow, resets, scenario_prior_index, static_factors, stoch_factors );
							fx_rate			= ScenarioFXForward ( t, mtm_cashflow[CASHFLOW_INDEX_FXResetDate] - t[TIME_GRID_MTM], static_currency, mtm_currency, static_curve, mtm_curve, scenario_prior_index, static_factors, stoch_factors );
						}
						else
						{
							forward_rate  	= Calc_PastReset ( t, mtm_forward, mtm_cashflow, resets, scenario_prior_index, static_factors, stoch_factors );
							fx_rate			=   mtm_cashflow [CASHFLOW_INDEX_FXResetValue] ?
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
						REAL payment	=  Interest * Pi + ( Pi - Pi_1 );
						
						printf("mtmswap,mtm_time_index,%d,t[TIME_GRID_MTM],%.2f,mtm_cashflow[CASHFLOW_INDEX_Start_Day],%.2f,mtm_cashflow[CASHFLOW_INDEX_FXResetDate],%.2f,mtm_cashflow [CASHFLOW_INDEX_FXResetValue],%.6f,cashflow,%d,forward_rate,%.6f,discount_rate,%.6f,fx_rate,%.6f,Pi,%.6f,payment,%.6f,mtm_leg,%.6f,static_leg,%.6f\\n",mtm_time_index,t[TIME_GRID_MTM],mtm_cashflow[CASHFLOW_INDEX_Start_Day],mtm_cashflow[CASHFLOW_INDEX_FXResetDate],mtm_cashflow [CASHFLOW_INDEX_FXResetValue],i,forward_rate,discount_rate,fx_rate,Pi,payment,mtm_leg,static_leg);
					}
				}
				*/
			}
			
			Output [ index ] = mtm ;
		}
		'''
	
	def __init__(self, params):
		super(MtMCrossCurrencySwapDeal, self).__init__(params)
		
	def reset(self, calendars):
		super(MtMCrossCurrencySwapDeal, self).reset()
		self.paydates	= generatedatesBackward ( self.field['Maturity_Date'], self.field['Effective_Date'], self.field.get('Pay_Frequency', pd.DateOffset(months=6) ) )
		self.recdates	= generatedatesBackward ( self.field['Maturity_Date'], self.field['Effective_Date'], self.field.get('Receive_Frequency', pd.DateOffset(months=6) ) )
		self.add_reval_dates ( self.paydates, self.field['Pay_Currency'] )
		self.add_reval_dates ( self.recdates, self.field['Receive_Currency'] )

	def finalize_dates ( self, parser, base_date, grid, node_children, node_resets, node_settlements ):
		#have to reset the original instrument and let the child node decide
		super(MtMCrossCurrencySwapDeal, self).reset()
		for currency, dates in node_settlements.items():
			self.add_reval_dates ( dates, currency )
		return super ( MtMCrossCurrencySwapDeal, self ).finalize_dates ( parser, base_date, grid, node_children, node_resets, node_settlements )
	
	def PostProcess( self, module, CudaMem, precision, batch_size, revals_per_batch, deal_data, child_dependencies, partition, mtm_offset ):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, deal_data.Time_dep.deal_time_grid.size)
		dependencies		= deal_data.Factor_dep
	
		child_map			= {}

		for index, child in enumerate(child_dependencies):
			if (child.Factor_dep['Currency']==dependencies[dependencies['MTM']]['Currency']).all():
				child_map.setdefault( 'MTM', child.Factor_dep)
			else:
				capital = child.Factor_dep['Cashflows'].schedule[-1][Utils.CASHFLOW_INDEX_Nominal]
				child.Factor_dep['Cashflows'].AddFixedPayments ( dependencies['base_date'], self.field['Principal_Exchange'], self.field['Effective_Date'], 'ACT_365', capital )
				child_map.setdefault( 'Static', child.Factor_dep)

		MtMCrossCurrencySwapDeal	= module.get_function ('MtMCrossCurrencySwapDeal')
		MtMCrossCurrencySwapDeal     (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(deal_data.Time_dep.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
										np.int32  ( child_map['Static']['Cashflows'].Count() ),
										drv.In    ( child_map['Static']['Cashflows'].GetSchedule(precision).ravel() ),
										drv.In    ( child_map['Static']['StartIndex'] ),
										drv.In 	  ( child_map['Static']['Cashflows'].Resets.GetSchedule(precision).ravel() ),
										drv.In 	  ( child_map['Static']['Currency'] ),
										drv.In 	  ( child_map['Static'].get('Forward',np.zeros(1) ) ),
										drv.In 	  ( child_map['Static']['Discount'] ),
										np.int32  ( child_map['MTM']['Cashflows'].Count() ),
										drv.In    ( child_map['MTM']['Cashflows'].GetSchedule(precision).ravel() ),
										drv.In    ( child_map['MTM']['StartIndex'] ),
										drv.In 	  ( child_map['MTM']['Cashflows'].Resets.GetSchedule(precision).ravel() ),
										drv.In 	  ( child_map['MTM']['Currency'] ),
										drv.In 	  ( child_map['MTM']['Forward'] ),
										drv.In 	  ( child_map['MTM']['Discount'] ),
										block=block, grid=grid )
		
		#interpolate the Theo price
		self.InterpolateBlock ( module, CudaMem, deal_data, batch_size, revals_per_batch )
		
		#copy the cashflows
		if CudaMem.d_Cashflows:
			partition.Cashflows [ mtm_offset ] = CudaMem.d_Cashflows.get()[:batch_size*revals_per_batch]
			
		#copy the mtm
		partition.DealMTMs  [ mtm_offset ] = CudaMem.d_MTM_Buffer.get()[:batch_size*revals_per_batch]
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Pay_Currency'] 		= Utils.CheckRateName(self.field['Pay_Currency'])
		field['Pay_Discount_Rate']	= Utils.CheckRateName(self.field['Pay_Discount_Rate']) if self.field['Pay_Discount_Rate'] else field['Pay_Currency']
		field['Pay_Interest_Rate']	= Utils.CheckRateName(self.field['Pay_Interest_Rate']) if self.field['Pay_Interest_Rate'] else field['Pay_Discount_Rate']
		
		field['Receive_Currency'] 		= Utils.CheckRateName(self.field['Receive_Currency'])
		field['Receive_Discount_Rate']	= Utils.CheckRateName(self.field['Receive_Discount_Rate']) if self.field['Receive_Discount_Rate'] else field['Receive_Currency']
		field['Receive_Interest_Rate']	= Utils.CheckRateName(self.field['Receive_Interest_Rate']) if self.field['Receive_Interest_Rate'] else field['Receive_Discount_Rate']

		field_index		= {'Pay':{}, 'Receive':{}}
		self.isQuanto 	= getInterestRateCurrency (field['Receive_Interest_Rate'],  all_factors) != field['Receive_Currency']
		if self.isQuanto:
			#TODO - Deal with Quanto Interest Rate swaps
			pass
		else:
			field_index['Pay']['Currency'] = getFXRateFactor   ( field['Pay_Currency'], static_offsets, stochastic_offsets )
			field_index['Pay']['Forward']  = getInterestFactor ( field['Pay_Interest_Rate'], static_offsets, stochastic_offsets, all_tenors )
			field_index['Pay']['Discount'] = getDiscountFactor ( field['Pay_Discount_Rate'],  static_offsets, stochastic_offsets, all_tenors, all_factors )
			
			field_index['Receive']['Currency'] = getFXRateFactor   ( field['Receive_Currency'], static_offsets, stochastic_offsets )
			field_index['Receive']['Forward']  = getInterestFactor ( field['Receive_Interest_Rate'], static_offsets, stochastic_offsets, all_tenors )
			field_index['Receive']['Discount'] = getDiscountFactor ( field['Receive_Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
			
		#TODO - complete cashflow definitions..

		#Which side is the mtm leg?
		field_index['MTM'] 					= self.field['MtM_Side']
		field_index['base_date'] 			= base_date
		
		return field_index
	
	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		#Set the block and grid dimensions - Not implemented - TODO - easy enough - just modify the PostProcess routine
		pass


class SwapCurrencyDeal(Deal):
	#dependent price factors for this instrument
	factor_fields = {'Pay_Interest_Rate_Volatility':		['InterestRateVol','InterestYieldVol'],
					 'Pay_Discount_Rate_Volatility':		['InterestRateVol','InterestYieldVol'],
					 'Receive_Interest_Rate_Volatility':	['InterestRateVol','InterestYieldVol'],
					 'Receive_Discount_Rate_Volatility':	['InterestRateVol','InterestYieldVol'],
					 'Pay_Currency':						['FxRate'],
					 'Pay_Discount_Rate':					['DiscountRate'],
					 'Pay_Interest_Rate':					['InterestRate'],
					 'Receive_Currency':					['FxRate'],
					 'Receive_Discount_Rate':				['DiscountRate'],
					 'Receive_Interest_Rate':				['InterestRate']}
	
	required_fields = {	'Effective_Date':					'The contract start date',
						'Maturity_Date':					'The contract end date',
						'Pay_Frequency':					'The payment frequency, which is the period between payments of the pay leg',
						'Pay_Margin':						'Margin rate added to the pay leg generalized cashflows. This property is required if the Pay_Rate_Type is set to "Floating" and ignored otherwise. Rates are entered in basis points. For example, enter 50 for 50 basis points',
						'Pay_Known_Rates':					'List of reset dates and corresponding known interest rates specified in the form of (date, value) pairs. This property is required if the Pay_Rate_Type is set to "Floating" and ignored otherwise',
						'Pay_Rate_Type':					'Interest rate type. Set to "Fixed" for a fixed pay leg or "Floating" for a floating pay leg',
						'Receive_Frequency':				'The payment frequency, which is the period between payments of the receive leg',
						'Receive_Margin':					'Margin rate added to the receive leg generalized cashflows. This property is required if the Receive_Rate_Type is set to "Floating" and ignored otherwise',
						'Receive_Known_Rates':				'List of reset dates and corresponding known interest rates specified in the form of (date, value) pairs. This property is required if the Receive_Rate_Type is set to "Floating" and ignored otherwise',
						'Pay_Currency':						'ID of the FX rate price factor used to define the pay leg settlement currency. For example, USD.',
						'Pay_Discount_Rate':				'ID of the discount rate price factor used to discount the pay leg cashflows. For example, USD.AAA',
						'Pay_Interest_Rate':				'ID of the interest rate price factor used to calculate forward interest rates. For example, USD.AAA',
						'Pay_Principal':					'Principal amount in units of settlement currency',
						'Pay_Amortisation':					'Permits an amortisation schedule for the principal to be specified as a list of (Amount, Date) pairs, where Amount represents the amount by which the principal is reduced by the Amortisation and Date represents the date of this Amortisation. A positive value for Amount represents a fall in the principal amount while a negative value for Amount represents a rise in the principal amount',
						'Pay_Fixed_Rate':					'Fixed interest rate. This property is required if the Pay_Rate_Type is set to "Fixed" and ignored otherwise',
						'Receive_Rate_Type':				'Interest rate type. Set to "Fixed" for a fixed receive leg or "Floating" for a floating receive leg',
						'Receive_Currency':					'ID of the FX rate price factor used to define the receive leg settlement currency. For example, USD',
						'Receive_Discount_Rate':			'ID of the discount rate price factor used to discount the receive leg cashflows. For example, USD.AAA',
						'Receive_Interest_Rate':			'ID of the interest rate price factor used to calculate forward interest rates. For example, USD.AAA',
						'Receive_Principal':				'Principal amount in units of settlement currency',
						'Receive_Amortisation':				'Permits an amortisation schedule for the principal to be specified as a list of (Amount, Date) pairs, where Amount represents the amount by which the principal is reduced by the Amortisation and Date represents the date of this Amortisation. A positive value for Amount represents a fall in the principal amount while a negative value for Amount represents a rise in the principal amount',
						'Receive_Fixed_Rate':				'Fixed interest rate. This property is required if the Receive_Rate_Type is set to "Fixed" and ignored otherwise',
						'Principal_Exchange':				'Type of principal exchange cashflows: no principal cashflows (None), principal cashflows at the effective date (Start), principal cashflows at the maturity date (Maturity), or principal cashflows at both the effective date and the maturity date (Start_Maturity)'
						}
	
	#cuda code required to price this instrument - none - as it's broken up into legs
	cudacodetemplate = ''
	
	def __init__(self, params):
		super(SwapCurrencyDeal, self).__init__(params)
		
	def reset(self, calendars):
		super(SwapCurrencyDeal, self).reset()
		self.paydates	= generatedatesBackward ( self.field['Maturity_Date'], self.field['Effective_Date'], self.field['Pay_Frequency'] )
		self.recdates	= generatedatesBackward ( self.field['Maturity_Date'], self.field['Effective_Date'], self.field['Receive_Frequency'] )
		self.add_reval_dates ( self.paydates, self.field['Pay_Currency'] )
		self.add_reval_dates ( self.recdates, self.field['Receive_Currency'] )
		
	def PostProcess( self, module, CudaMem, precision, batch_size, revals_per_batch, deal_data, child_dependencies, partition, mtm_offset ):
		logging.warning('SwapCurrencyDeal {0} - TODO'.format(self.field['Reference']))
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Pay_Currency'] 		= Utils.CheckRateName(self.field['Pay_Currency'])
		field['Pay_Discount_Rate']	= Utils.CheckRateName(self.field['Pay_Discount_Rate']) if self.field['Pay_Discount_Rate'] else field['Pay_Currency']
		field['Pay_Interest_Rate']	= Utils.CheckRateName(self.field['Pay_Interest_Rate']) if self.field['Pay_Interest_Rate'] else field['Pay_Discount_Rate']
		
		field['Receive_Currency'] 		= Utils.CheckRateName(self.field['Receive_Currency'])
		field['Receive_Discount_Rate']	= Utils.CheckRateName(self.field['Receive_Discount_Rate']) if self.field['Receive_Discount_Rate'] else field['Receive_Currency']
		field['Receive_Interest_Rate']	= Utils.CheckRateName(self.field['Receive_Interest_Rate']) if self.field['Receive_Interest_Rate'] else field['Receive_Discount_Rate']

		field_index		= {'Pay':{}, 'Receive':{}}
		self.isQuanto 	= getInterestRateCurrency (field['Receive_Interest_Rate'],  all_factors) != field['Receive_Currency']
		if self.isQuanto:
			#TODO - Deal with Quanto Interest Rate swaps
			pass
		else:
			field_index['Pay']['Currency'] = getFXRateFactor   ( field['Pay_Currency'], static_offsets, stochastic_offsets )
			field_index['Pay']['Forward']  = getInterestFactor ( field['Pay_Interest_Rate'], static_offsets, stochastic_offsets, all_tenors )
			field_index['Pay']['Discount'] = getDiscountFactor ( field['Pay_Discount_Rate'],  static_offsets, stochastic_offsets, all_tenors, all_factors )
			field_index['Pay']['InterestYieldVol'] = np.zeros(1, dtype=np.int32)
			
			field_index['Receive']['Currency'] = getFXRateFactor   ( field['Receive_Currency'], static_offsets, stochastic_offsets )
			field_index['Receive']['Forward']  = getInterestFactor ( field['Receive_Interest_Rate'], static_offsets, stochastic_offsets, all_tenors )
			field_index['Receive']['Discount'] = getDiscountFactor ( field['Receive_Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
			field_index['Receive']['InterestYieldVol'] = np.zeros(1, dtype=np.int32)
		
		#cashflows
		if self.field['Pay_Rate_Type']=='Floating':
			field_index['Pay']['Cashflows'] = Utils.GenerateFloatCashflows ( base_date, time_grid, self.paydates, -self.field['Pay_Principal'], self.field['Pay_Amortisation'], self.field['Pay_Known_Rates'], self.field['Pay_Index_Tenor'], self.field['Pay_Interest_Frequency'], Utils.GetDayCount( self.field['Pay_Day_Count'] ), self.field['Pay_Margin']/10000.0 )
		else:
			field_index['Pay']['Cashflows'] = Utils.GenerateFixedCashflows ( base_date, self.paydates, -self.field['Pay_Principal'], self.field['Pay_Amortisation'], Utils.GetDayCount( self.field['Pay_Day_Count'] ), self.field['Pay_Fixed_Rate']/100.0 )

		if self.field['Receive_Rate_Type']=='Floating':
			field_index['Receive']['Cashflows']  = Utils.GenerateFloatCashflows ( base_date, time_grid, self.recdates, self.field['Receive_Principal'], self.field['Receive_Amortisation'], self.field['Receive_Known_Rates'], self.field['Receive_Index_Tenor'], self.field['Receive_Interest_Frequency'], Utils.GetDayCount( self.field['Receive_Day_Count'] ), self.field['Receive_Margin']/10000.0 )
		else:
			field_index['Receive']['Cashflows'] = Utils.GenerateFixedCashflows ( base_date, self.recdates, self.field['Receive_Principal'], self.field['Receive_Amortisation'], Utils.GetDayCount( self.field['Receive_Day_Count'] ), self.field['Receive_Fixed_Rate']/100.0 )

		#add fixed payments
		field_index['Pay']['Cashflows'].AddFixedPayments(base_date, self.field['Principal_Exchange'], self.field['Effective_Date'], self.field['Pay_Day_Count'], -self.field['Pay_Principal'])
		field_index['Receive']['Cashflows'].AddFixedPayments(base_date, self.field['Principal_Exchange'], self.field['Effective_Date'], self.field['Receive_Day_Count'], self.field['Receive_Principal'])
		
		#cashflow start indexes
		field_index['Pay']['StartIndex'] 	 = field_index['Pay']['Cashflows'].GetCashflowStartIndex(time_grid)
		field_index['Receive']['StartIndex'] = field_index['Receive']['Cashflows'].GetCashflowStartIndex(time_grid)
		
		return field_index
	
	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		#Set the block and grid dimensions
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)
		
		overwrite			= 1
		for direction, leg in field_index.items():
			if self.field[direction+'_Rate_Type']=='Floating':		
				AddFloatCashflow 	= module.get_function ('AddFloatCashflow')
				AddFloatCashflow( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
									np.int32 (  Utils.SCENARIO_CASHFLOWS_FloatLeg ),
									np.int32 (  Utils.CASHFLOW_METHOD_Compounding_None ),
									np.int32( overwrite ),
									np.int32  ( leg['Cashflows'].Count() ),
									drv.In    ( leg['Cashflows'].GetSchedule(precision).ravel() ),
									drv.In    ( leg['StartIndex'] ),
									drv.In	  ( leg['Cashflows'].Resets.GetSchedule(precision).ravel() ),
									drv.In    ( leg['Currency'] ),
									drv.In    ( leg['InterestYieldVol'] ),
									drv.In    ( leg['Forward'] ),
									drv.In    ( leg['Discount'] ),
									block=block, grid=grid )
				overwrite			= 0
			else:
				AddFixedCashflow	= module.get_function ('AddFixedCashflow')
				AddFixedCashflow (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
									np.int32( overwrite ),
									np.int32( Utils.CASHFLOW_METHOD_Fixed_Compounding_No ),
									np.int32  ( leg['Cashflows'].Count() ),
									drv.In    ( leg['Cashflows'].GetSchedule(precision).ravel() ),
									drv.In    ( leg['StartIndex'] ),
									drv.In    ( leg['Currency'] ),
									drv.In    ( leg['Discount'] ),
									block=block, grid=grid )
				overwrite			= 0
				
class FXNonDeliverableForward(Deal):
	factor_fields = {'Buy_Currency':		['FxRate'],
					 'Discount_Rate':		['DiscountRate'],
					 'Sell_Currency':		['FxRate'],
					 'Settlement_Currency':	['FxRate']}
	
	required_fields = {	'Buy_Currency':			'ID of the FX rate price factor used to define foreign currency name, for example, USD. The underlying exchange rate asset used by the deal is the amount of Sell_Currency per unit of Buy_Currency. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Sell_Currency':		'ID of the FX rate price factor used to define domestic currency name, for example, GBP. The underlying exchange rate asset used by the deal is the amount of Sell_Currency per unit of Buy_Currency. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, GBP.AAA. The currency of this price factor (eg GBP in the GBP.AAA example) must be the Settlement_Currency. If left blank, the ID of the Settlement_Currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Buy_Amount':			'Amount of the currency bought',
						'Sell_Amount':			'Amount of the currency sold',
						'Settlement_Date':		'Settlement date'}
	
	#cuda code required to price this instrument
	cudacodetemplate = '''
		//all pricing functions will have the same 5 initial parameters
		__global__ void FXNonDeliverableForward(	const REAL* __restrict__ stoch_factors,
													const REAL* __restrict__ static_factors,
													const int*  __restrict__ deal_time_index,
													const REAL* __restrict__ time_grid,
													const int*  __restrict__ cashflow_index,
													REAL* all_cashflows,
													REAL* Output,
													REAL maturity_in_days,
													REAL buy_nominal,
													REAL sell_nominal,
													const int* __restrict__ buycurrency,
													const int* __restrict__ sellcurrency,
													const int* __restrict__ settlecurrency,
													const int* __restrict__ discount )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];			
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
			OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			
			//Set the MTM to 0
			REAL mtm 					= 0.0;

			//work out the remaining term
			REAL remainingMaturityinDays = maturity_in_days - t[TIME_GRID_MTM];

			if (remainingMaturityinDays>= 0.0)
			{
				const int* buy_curve 		= CurrencyRepoCurve ( buycurrency );
				const int* sell_curve 		= CurrencyRepoCurve ( sellcurrency );
				const int* settle_curve 	= CurrencyRepoCurve ( settlecurrency );
				
				REAL Accrual 				= calcDayCountAccrual ( remainingMaturityinDays, discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
				REAL discount_rate  		= ScenarioCurve ( t,  Accrual, discount, scenario_prior_index, static_factors, stoch_factors );
				REAL buy_forward  			= ScenarioFXForward ( t, remainingMaturityinDays, buycurrency, settlecurrency, buy_curve, settle_curve, scenario_prior_index, static_factors, stoch_factors );
				REAL sell_forward 			= ScenarioFXForward ( t, remainingMaturityinDays, sellcurrency, settlecurrency, sell_curve, settle_curve, scenario_prior_index, static_factors, stoch_factors );
				REAL FX_base				= ScenarioRate ( t, settlecurrency, scenario_prior_index, static_factors, stoch_factors ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
							
				//now calculate the MTM
				mtm 			= ( buy_forward * buy_nominal - sell_forward * sell_nominal ) * exp ( -discount_rate * Accrual ) * FX_base;
				
				//settle the currency_cashflows
				if (remainingMaturityinDays==0)
				{
					ScenarioSettle (  mtm/FX_base, settlecurrency,  scenario_number, mtm_time_index, cashflow_index, all_cashflows );
				}
			}
			
			Output [ index ] = mtm ;
		}		
	'''
	
	def __init__(self, params):
		super(FXNonDeliverableForward, self).__init__(params)

	def reset(self, calendars):
		super(FXNonDeliverableForward, self).reset()
		self.add_reval_dates ( set([self.field['Settlement_Date']]), self.field['Settlement_Currency'] )
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Buy_Currency'] 			= Utils.CheckRateName(self.field['Buy_Currency'])
		field['Sell_Currency']			= Utils.CheckRateName(self.field['Sell_Currency'])
		field['Settlement_Currency']	= Utils.CheckRateName(self.field['Settlement_Currency'])
		field['Discount_Rate']			= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Settlement_Currency']
		
		field_index				 	= {}		
		field_index['BuyFX']  		= getFXRateFactor 	( field['Buy_Currency'], static_offsets, stochastic_offsets )
		field_index['Discount']  	= getDiscountFactor ( field['Discount_Rate'],  static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['SellFX'] 		= getFXRateFactor 	( field['Sell_Currency'], static_offsets, stochastic_offsets )
		field_index['SettleFX']		= getFXRateFactor 	( field['Settlement_Currency'], static_offsets, stochastic_offsets )
		
		field_index['Maturity'] 	= (self.field['Settlement_Date'] - base_date).days
			
		return field_index
			
	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		grid        		= (batch_size, time_grid.deal_time_grid.size)
		block       		= (revals_per_batch,1,1)
		
		FXNonDeliverableForward	= module.get_function ('FXNonDeliverableForward')
		FXNonDeliverableForward ( CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
								precision ( field_index['Maturity'] ),
								precision ( self.field['Buy_Amount'] ),
								precision ( self.field['Sell_Amount'] ),
								drv.In    ( field_index['BuyFX'] ),
								drv.In    ( field_index['SellFX'] ),
								drv.In    ( field_index['SettleFX'] ),
								drv.In    ( field_index['Discount'] ),
								block=block, grid=grid )
	
class FXForwardDeal(Deal):
	#dependent price factors for this instrument
	factor_fields = {'Buy_Currency':		['FxRate'],
					 'Buy_Discount_Rate':	['DiscountRate'],
					 'Sell_Currency':		['FxRate'],
					 'Sell_Discount_Rate':	['DiscountRate']}
	
	required_fields = {	'Buy_Currency':			'ID of the FX rate price factor used to define foreign currency name, for example, USD. The underlying exchange rate asset used by the deal is the amount of Sell_Currency per unit of Buy_Currency. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Buy_Discount_Rate':	'ID of the discount rate price factor used to discount the foreign currency cashflows. For example, USD.AAA. The currency of this price factor (eg USD in the USD.AAA example) must be the Buy_Currency. If left blank, the ID of the Buy_Currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Sell_Currency':		'ID of the FX rate price factor used to define domestic currency name, for example, GBP. The underlying exchange rate asset used by the deal is the amount of Sell_Currency per unit of Buy_Currency. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Sell_Discount_Rate':	'ID of the discount rate price factor used to discount the domestic currency cashflows. For example, GBP.AAA. The currency of this price factor (eg GBP in the GBP.AAA example) must be the Sell_Currency. If left blank, the ID of the Sell_Currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Buy_Amount':			'Amount of the currency bought',
						'Sell_Amount':			'Amount of the currency sold',
						'Settlement_Date':		'Settlement date'}
	
	#cuda code required to price this instrument
	cudacodetemplate = '''
		//all pricing functions will have the same 5 initial parameters
		__global__ void FXForwardDeal(	const REAL* __restrict__ stoch_factors,
										const REAL* __restrict__ static_factors,
										const int*  __restrict__ deal_time_index,
										const REAL* __restrict__ time_grid,
										const int*  __restrict__ cashflow_index,
										REAL* all_cashflows,
										REAL* Output,
										REAL maturity_in_days,
										REAL buy_nominal,
										REAL sell_nominal,
										const int* __restrict__ buycurrency,
										const int* __restrict__ sellcurrency,
										const int* __restrict__ buy_discount,
										const int* __restrict__ sell_discount )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];			
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
			OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			
			//Set the MTM to 0
			REAL mtm 					= 0.0;

			//work out the remaining term
			REAL remainingMaturityinDays = maturity_in_days - t[TIME_GRID_MTM];

			if (remainingMaturityinDays>= 0.0)
			{
				//assume the daycount is stored in the first factor 
				REAL buyYearAccrual 	= calcDayCountAccrual(remainingMaturityinDays, buy_discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
				REAL sellYearAccrual 	= calcDayCountAccrual(remainingMaturityinDays, sell_discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
													
				//reconstruct (and interpolate) the discount curves
				REAL buyrate  = ScenarioCurve ( t,  buyYearAccrual, buy_discount, scenario_prior_index, static_factors, stoch_factors );
				REAL sellrate = ScenarioCurve ( t,  sellYearAccrual, sell_discount, scenario_prior_index, static_factors, stoch_factors );
				
				//reconstruct (and interpolate) the spot FX rates
				REAL FX_buy 	= ScenarioRate ( t, buycurrency, scenario_prior_index, static_factors, stoch_factors );
				REAL FX_sell 	= ScenarioRate ( t, sellcurrency, scenario_prior_index, static_factors, stoch_factors );
				REAL FX_base	= ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );

				//now calculate the MTM
				mtm 			= ( FX_buy * buy_nominal * exp ( -buyrate * buyYearAccrual ) - FX_sell * sell_nominal * exp ( -sellrate * sellYearAccrual ) ) / FX_base;

				//settle the currency_cashflows
				if (remainingMaturityinDays==0)
				{
					ScenarioSettle (  buy_nominal, buycurrency,  scenario_number, mtm_time_index, cashflow_index, all_cashflows);
					ScenarioSettle (-sell_nominal, sellcurrency, scenario_number, mtm_time_index, cashflow_index, all_cashflows);
				}
			}
			
			Output [ index ] = mtm ;
		}		
	'''
	
	def __init__(self, params):
		super(FXForwardDeal, self).__init__(params)

	def reset(self, calendars):
		super(FXForwardDeal, self).reset()
		self.add_reval_dates ( set([self.field['Settlement_Date']]), self.field['Buy_Currency'] )
		self.add_reval_dates ( set([self.field['Settlement_Date']]), self.field['Sell_Currency'] )
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Buy_Currency'] 		= Utils.CheckRateName(self.field['Buy_Currency'])
		field['Buy_Discount_Rate']	= Utils.CheckRateName(self.field['Buy_Discount_Rate']) if self.field['Buy_Discount_Rate'] else field['Buy_Currency']
		field['Sell_Currency']		= Utils.CheckRateName(self.field['Sell_Currency'])
		field['Sell_Discount_Rate']	= Utils.CheckRateName(self.field['Sell_Discount_Rate']) if self.field['Sell_Discount_Rate'] else field['Buy_Currency']
		
		field_index				 	= {}		
		field_index['BuyFX']  		= getFXRateFactor 	( field['Buy_Currency'], static_offsets, stochastic_offsets )
		field_index['BuyDiscount']  = getDiscountFactor ( field['Buy_Discount_Rate'],  static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['SellFX'] 		= getFXRateFactor 	( field['Sell_Currency'], static_offsets, stochastic_offsets )
		field_index['SellDiscount'] = getDiscountFactor ( field['Sell_Discount_Rate'],  static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Maturity'] 	= (self.field['Settlement_Date'] - base_date).days
			
		return field_index
			
	def Generate ( self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid ):
		#the tenors and the scenario grid should already be loaded up in the Buffer space in constant memory
		grid        		= (batch_size, time_grid.deal_time_grid.size)
		block       		= (revals_per_batch,1,1)
		
		FXForwardDeal	= module.get_function ('FXForwardDeal')
		FXForwardDeal ( CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
						precision ( field_index['Maturity'] ),
						precision ( self.field['Buy_Amount'] ),
						precision ( self.field['Sell_Amount'] ),
						drv.In    ( field_index['BuyFX'] ),
						drv.In    ( field_index['SellFX'] ),
						drv.In    ( field_index['BuyDiscount'] ),
						drv.In    ( field_index['SellDiscount'] ),
						block=block, grid=grid )
		
class SwapInterestDeal(Deal):
	#dependent price factors for this instrument
	factor_fields = {'Currency':				['FxRate'],
					 'Discount_Rate':			['DiscountRate'],
					 'Interest_Rate':			['InterestRate'],
					 'Interest_Rate_Volatility':['InterestRateVol','InterestYieldVol'],
					 'Discount_Rate_Volatility':['InterestRateVol','InterestYieldVol']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Amortisation':			'Permits an amortisation schedule for the principal to be specified as a list of (Date, Amount) pairs, where Date represents the date of this Amortisation and Amount represents the amount by which the principal is reduced by the Amortisation. A positive value for Amount represents a fall in the principal amount while a negative value for Amount represents a rise in the principal amount',
						'Known_Rates':			'List of reset dates and corresponding known interest rates specified in the form of (date, value) pairs. Rates are entered in percentage. For example, enter 5 for 5%',
						'Interest_Rate':		"ID of the interest rate price factor used to calculate forward interest rates. For example, USD.AAA",
						'Effective_Date':		'The contract start date',
						'Maturity_Date':		'The contract end date',
						'Principal':			'Principal amount in units of settlement currency',
						'Pay_Frequency':		'The payment frequency, which is the period between payments of the pay leg',
						'Receive_Frequency':	'The payment frequency, which is the period between payments of the receive leg',
						'Floating_Margin':		'Margin rate added to the floating interest rate. Rates are entered in basis points. For example, enter 50 for 50 basis points',
						'Swap_Rate':			'Fixed interest rate. Rates are entered in percentage. For example, enter 5 for 5%',
						'Rate_Multiplier':		'Rate multiplier of the floating interest rate',
						'Rate_Constant':		'Rate constant of the floating interest rate. Values are entered as a decimal or as a percentage. For example, enter 5% or 0.05 for 5%'
						}
	
	#cuda code required to price this instrument - none - as it's broken up into an addfixedleg and an addfloatleg
	cudacodetemplate = ''
	
	def __init__(self, params):
		super(SwapInterestDeal, self).__init__(params)

	def reset(self, calendars):
		super(SwapInterestDeal, self).reset()
		self.paydates	= generatedatesBackward ( self.field['Maturity_Date'], self.field['Effective_Date'], self.field['Pay_Frequency'] )
		self.recdates	= generatedatesBackward ( self.field['Maturity_Date'], self.field['Effective_Date'], self.field['Receive_Frequency'] )
		self.add_reval_dates ( self.paydates, self.field['Currency'] )
		self.add_reval_dates ( self.recdates, self.field['Currency'] )
		#this swap could be quantoed - TODO
		self.isQuanto 		= None
		
	def PostProcess( self, module, CudaMem, precision, batch_size, revals_per_batch, deal_data, child_dependencies, partition, mtm_offset ):
		logging.warning('SwapInterestDeal {0} - TODO'.format(self.field['Reference']))		
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 		= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']	= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Interest_Rate']	= Utils.CheckRateName(self.field['Interest_Rate']) if self.field['Interest_Rate'] else field['Discount_Rate']

		field_index		= {}		
		self.isQuanto 	= getInterestRateCurrency (field['Interest_Rate'],  all_factors) != field['Currency']
		if self.isQuanto:
			#TODO - Deal with Quanto Interest Rate swaps
			pass
		else:
			field_index['Forward']  = getInterestFactor ( field['Interest_Rate'], static_offsets, stochastic_offsets, all_tenors )
			field_index['Discount'] = getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
			field_index['Currency']	= getFXRateFactor ( field['Currency'], static_offsets, stochastic_offsets )
			
		if self.field['Pay_Rate_Type']=='Fixed':
			field_index['FixedCashflows'] = Utils.GenerateFixedCashflows ( base_date, self.paydates, -self.field['Principal'], self.field['Amortisation'], Utils.GetDayCount( self.field['Pay_Day_Count'] ), self.field['Swap_Rate']/100.0 )
			field_index['FloatCashflows'] = Utils.GenerateFloatCashflows ( base_date, time_grid, self.recdates, self.field['Principal'], self.field['Amortisation'], self.field['Known_Rates'], self.field['Receive_Interest_Frequency'], self.field['Index_Tenor'], Utils.GetDayCount( self.field['Receive_Day_Count'] ), self.field['Floating_Margin']/10000.0 )
		else:
			field_index['FixedCashflows'] = Utils.GenerateFixedCashflows ( base_date, self.recdates, self.field['Principal'], self.field['Amortisation'], Utils.GetDayCount ( self.field['Receive_Day_Count'] ), self.field['Swap_Rate']/100.0 )
			field_index['FloatCashflows'] = Utils.GenerateFloatCashflows ( base_date, time_grid, self.paydates, -self.field['Principal'], self.field['Amortisation'], self.field['Known_Rates'], self.field['Pay_Interest_Frequency'], self.field['Index_Tenor'], Utils.GetDayCount ( self.field['Pay_Day_Count'] ), self.field['Floating_Margin']/10000.0 )

		field_index['CompoundingMethod'] 	= Utils.CASHFLOW_CompoundingMethodLookup [ self.field['Compounding_Method'] ]
		field_index['Fixed_Compounding'] 	= Utils.CASHFLOW_METHOD_Fixed_Compounding_Yes if self.field['Fixed_Compounding']=='Yes' else Utils.CASHFLOW_METHOD_Fixed_Compounding_No
		field_index['InterestYieldVol'] 	= np.zeros(1, dtype=np.int32)
		field_index['FixedStartIndex']  	= field_index['FixedCashflows'].GetCashflowStartIndex(time_grid)
		field_index['FloatStartIndex']  	= field_index['FloatCashflows'].GetCashflowStartIndex(time_grid)
		
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid ):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)
		
		AddFloatCashflow 	= module.get_function ('AddFloatCashflow')
		AddFloatCashflow( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32 (  Utils.SCENARIO_CASHFLOWS_FloatLeg ),
							np.int32 (  field_index['CompoundingMethod'] ),
							np.int32( 1 ),
							np.int32  ( field_index['FloatCashflows'].Count() ),
							drv.In    ( field_index['FloatCashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['FloatStartIndex'] ),
							drv.In    ( field_index['FloatCashflows'].Resets.GetSchedule(precision).ravel() ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['InterestYieldVol'] ),
							drv.In    ( field_index['Forward'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )
				
		AddFixedCashflow	= module.get_function ('AddFixedCashflow')
		AddFixedCashflow (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32  ( 0 ),
							np.int32  ( field_index['Fixed_Compounding'] ),
							np.int32  ( field_index['FixedCashflows'].Count() ),
							drv.In    ( field_index['FixedCashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['FixedStartIndex'] ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )

class CFFixedInterestListDeal(Deal):
	factor_fields = {'Currency':				['FxRate'],
					 'Discount_Rate':			['DiscountRate']}

	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Cashflows':			'List of fixed interest cashflows'}
						
	#cuda code required to price this instrument - none - use addfixedleg
	cudacodetemplate = ''
	def __init__(self, params):
		super(CFFixedInterestListDeal, self).__init__(params)		

	def reset(self, calendars):
		super(CFFixedInterestListDeal, self).reset()		
		self.paydates	= set([x['Payment_Date'] for x in self.field['Cashflows']['Items']])
		self.add_reval_dates ( self.paydates, self.field['Currency'])
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 		= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']	= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']

		field_index	= {}		
		field_index['Currency']		= getFXRateFactor ( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 	= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Cashflows']  	= Utils.MakeFixedCashflows( base_date, 1 if self.field['Buy_Sell']=='Buy' else -1, self.field['Cashflows'] )
		field_index['Compounding'] 	= Utils.CASHFLOW_METHOD_Fixed_Compounding_Yes if self.field['Cashflows']['Compounding']=='Yes' else Utils.CASHFLOW_METHOD_Fixed_Compounding_No
		field_index['StartIndex'] 	= field_index['Cashflows'].GetCashflowStartIndex(time_grid)
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)

		AddFixedCashflow	= module.get_function ('AddFixedCashflow')
		AddFixedCashflow (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32  ( 1 ),
							np.int32  ( field_index['Compounding'] ),
							np.int32  ( field_index['Cashflows'].Count() ),
							drv.In    ( field_index['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['StartIndex'] ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )
		
class DepositDeal(Deal):
	factor_fields = {'Currency':				['FxRate'],
					 'Discount_Rate':			['DiscountRate']}

	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor'}
						
	#cuda code required to price this instrument - none - use addfixedleg
	cudacodetemplate = ''
	def __init__(self, params):
		super(DepositDeal, self).__init__(params)		

	def reset(self, calendars):
		super(DepositDeal, self).reset()
		self.paydates	= generatedatesBackward ( self.field['Maturity_Date'], self.field['Effective_Date'], self.field['Payment_Frequency'] )
		self.add_reval_dates ( self.paydates, self.field['Currency'])
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 		= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']	= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']

		field_index	= {}		
		field_index['Currency']		 = getFXRateFactor ( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 	 = getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Cashflows']     = Utils.GenerateFixedCashflows ( base_date, self.paydates, -self.field['Amount'], self.field['Amortisation'], Utils.GetDayCount( self.field['Accrual_Day_Count'] ), self.field['Interest_Rate']/100.0 )
		field_index['Cashflows'].AddFixedPayments ( base_date, 'Start_Maturity', self.field['Effective_Date'], self.field['Accrual_Day_Count'], -self.field['Amount'] )
		
		field_index['Compounding'] 	 = Utils.CASHFLOW_METHOD_Fixed_Compounding_Yes if self.field['Compounding']=='Yes' else Utils.CASHFLOW_METHOD_Fixed_Compounding_No
		field_index['StartIndex'] 	 = field_index['Cashflows'].GetCashflowStartIndex(time_grid)
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)

		AddFixedCashflow	= module.get_function ('AddFixedCashflow')
		AddFixedCashflow (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32  ( 1 ),
							np.int32  ( field_index['Compounding'] ),
							np.int32  ( field_index['Cashflows'].Count() ),
							drv.In    ( field_index['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['StartIndex'] ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )

class CFFixedListDeal(Deal):
	factor_fields = {'Currency':				['FxRate'],
					 'Discount_Rate':			['DiscountRate']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Cashflows':			'List of fixed cashflows'}
	
	#cuda code required to price this instrument - none - use addfixedleg
	cudacodetemplate = ''
	def __init__(self, params):
		super(CFFixedListDeal, self).__init__(params)		

	def reset(self, calendars):
		super(CFFixedListDeal, self).reset()		
		self.paydates	= set([x['Payment_Date'] for x in self.field['Cashflows']['Items']])
		self.add_reval_dates ( self.paydates, self.field['Currency'])
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 		= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']	= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']

		field_index	= {}		
		field_index['Currency']		= getFXRateFactor ( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 	= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Cashflows']  	= Utils.MakeSimpleFixedCashflows( base_date, 1 if self.field['Buy_Sell']=='Buy' else -1, self.field['Cashflows'] )
		field_index['StartIndex'] 	= field_index['Cashflows'].GetCashflowStartIndex(time_grid)
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)

		AddFixedCashflow	= module.get_function ('AddFixedCashflow')
		AddFixedCashflow (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32  ( 1 ),
							np.int32  ( Utils.CASHFLOW_METHOD_Fixed_Compounding_No ),
							np.int32  ( field_index['Cashflows'].Count() ),
							drv.In    ( field_index['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['StartIndex'] ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )

class FixedCashflowDeal(Deal):
	factor_fields = {'Currency':				['FxRate'],
					 'Discount_Rate':			['DiscountRate']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor'}
	
	#cuda code required to price this instrument 
	cudacodetemplate = '''
		//all pricing functions will have the same 5 initial parameters
		__global__ void FixedCashflowDeal(	const REAL* __restrict__ stoch_factors,
											const REAL* __restrict__ static_factors,
											const int*  __restrict__ deal_time_index,
											const REAL* __restrict__ time_grid,
											const int*  __restrict__ cashflow_index,
											REAL* all_cashflows,
											REAL* Output,
											REAL payment_date_in_days,
											REAL amount,											
											const int* __restrict__ currency,
											const int* __restrict__ discount)
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];			
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
			OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );

			//set the mtm of this deal to 0
			REAL mtm 					= 0.0;

			if (payment_date_in_days >= t[TIME_GRID_MTM])
			{
				REAL baseFX				    = ScenarioRate ( t, currency, scenario_prior_index, static_factors, stoch_factors ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
				mtm 						= amount * CalcDiscount ( t, payment_date_in_days, discount, scenario_prior_index, static_factors, stoch_factors) * baseFX;
				
				//settle the currency_cashflows
				if (payment_date_in_days==t[TIME_GRID_MTM])
					ScenarioSettle(mtm/baseFX, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows);
			}
			
			Output [ index ] = mtm ;
		}
	'''
	
	def __init__(self, params):
		super(FixedCashflowDeal, self).__init__(params)		

	def reset(self, calendars):
		super(FixedCashflowDeal, self).reset()
		self.add_reval_dates ( set([self.field['Payment_Date']]), self.field['Currency'] )
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field 					= {}
		field['Currency'] 		= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']	= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']

		field_index	= {}		
		field_index['Currency']		  	= getFXRateFactor ( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 		= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Payment_Date'] 	= (self.field['Payment_Date'] - base_date).days
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):		
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)

		FixedCashflowDeal	= module.get_function ('FixedCashflowDeal')
		FixedCashflowDeal (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							precision ( field_index['Payment_Date'] ),
							precision ( self.field ['Amount'] ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )

class CFFloatingInterestListDeal(Deal):
	#dependent price factors for this instrument 
	factor_fields = {'Currency':							['FxRate'],
					 'Discount_Rate':						['DiscountRate'],
					 'Discount_Rate_Cap_Volatility':		['InterestRateVol'],
					 'Discount_Rate_Swaption_Volatility':	['InterestYieldVol'],
					 'Forecast_Rate':						['InterestRate'],
					 'Forecast_Rate_Cap_Volatility':		['InterestRateVol'],
					 'Forecast_Rate_Swaption_Volatility':	['InterestYieldVol']}

	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Forecast_Rate':		'ID of the interest rate price factor used to calculate forward interest rates, which is used to obtain the underlying floating interest rate that determines the cashflow amounts. For example, USD.AAA',
						'Cashflows':			'List of floating interest cashflows'}

	#cuda code required to price this instrument - none - as it's just addfloatleg
	cudacodetemplate = ''
	
	def __init__(self, params):
		super(CFFloatingInterestListDeal, self).__init__(params)
		
	def reset(self, calendars):
		super(CFFloatingInterestListDeal, self).reset()
		reset_dates		= set([x['Payment_Date'] for x in self.field['Cashflows']['Items']])
		self.add_reval_dates ( reset_dates, self.field['Currency'] )
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 		= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']	= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Forecast_Rate']	= Utils.CheckRateName(self.field['Forecast_Rate']) if self.field['Forecast_Rate'] else field['Discount_Rate']

		field_index		= {}		
		field_index['Forward']  	= getInterestFactor ( field['Forecast_Rate'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Discount'] 	= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Currency']		= getFXRateFactor ( field['Currency'], static_offsets, stochastic_offsets )
		field_index['VolSurface'] 	= np.zeros(1, dtype=np.int32)
		field_index['Currency']		= getFXRateFactor ( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Cashflows']  	= Utils.MakeFloatCashflows ( base_date, time_grid, 1 if self.field['Buy_Sell']=='Buy' else -1, self.field['Cashflows'] )
		
		if self.field['Cashflows'].get('Properties'):
			first_prop = self.field['Cashflows']['Properties'][0]
			if first_prop['Cap_Multiplier'] or first_prop['Floor_Multiplier']:
				field_index['Model']  			= Utils.SCENARIO_CASHFLOWS_Cap if first_prop['Cap_Multiplier'] else Utils.SCENARIO_CASHFLOWS_Floor
				field_index['VolSurface']		= getInterestVolFactor ( Utils.CheckRateName(self.field['Forecast_Rate_Cap_Volatility']), pd.DateOffset(months=3), static_offsets, stochastic_offsets, all_tenors )
				
				field_index['Cashflows'].OverwriteRate( Utils.CASHFLOW_INDEX_Strike, first_prop['Cap_Strike'].amount if first_prop['Cap_Multiplier'] else first_prop['Floor_Strike'].amount )
		else:	
			field_index['Model']  			= Utils.SCENARIO_CASHFLOWS_FloatLeg
				
			#need to check if this is a generalized cap/floor structure
		field_index['CompoundingMethod'] 	= Utils.CASHFLOW_CompoundingMethodLookup [ self.field['Cashflows']['Compounding_Method'] ]
		field_index['StartIndex'] 			= field_index['Cashflows'].GetCashflowStartIndex(time_grid)
		
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)
		
		AddFloatCashflow 	= module.get_function ( 'AddFloatCashflow' )
		AddFloatCashflow( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32 (  field_index['Model'] ),
							np.int32 (  field_index['CompoundingMethod'] ),
							np.int32 ( 1 ),
							np.int32  ( field_index['Cashflows'].Count() ),
							drv.In    ( field_index['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['StartIndex'] ),
							drv.In    ( field_index['Cashflows'].Resets.GetSchedule(precision).ravel() ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['VolSurface'] ),
							drv.In    ( field_index['Forward'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )

class YieldInflationCashflowListDeal(Deal):
	#dependent price factors for this instrument 
	factor_fields = {'Currency':							['FxRate'],
					 'Discount_Rate':						['DiscountRate'],
					 'Index':								['InflationRate']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Index':				'ID of the inflation rate price factor. For example, RPI-GBP. This requires the Index inflation rate price factor, the price index price factor specified by the Price_Index property of that inflation rate price factor and optionally the seasonal price index price factor specified by the Seasonal_Adjustment property of that inflation rate price factor',
						'Cashflows':			'List of real yield cashflows'}

	#cuda code required to price this instrument - none - as it's just addfloatleg
	cudacodetemplate = ''
	def __init__(self, params):
		super(YieldInflationCashflowListDeal, self).__init__(params)
		
	def reset(self, calendars):
		super(YieldInflationCashflowListDeal, self).reset()

		if self.field['Is_Forward_Deal']=='Yes':
			paydates	= set ( [self.field['Settlement_Date']] )
		else:
			paydates	= set ( [x['Payment_Date'] for x in self.field['Cashflows']['Items']] )
			
		self.add_reval_dates ( paydates, self.field['Currency'] )
		
	def calc_dependancies ( self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars ):
		field = {}
		field['Currency'] 		= Utils.CheckRateName ( self.field['Currency'] )
		field['Discount_Rate']	= Utils.CheckRateName ( self.field['Discount_Rate'] ) if self.field['Discount_Rate'] else field['Currency']
		field['Index']			= Utils.CheckRateName ( self.field['Index'] )
		field['PriceIndex']		= Utils.CheckRateName ( getInflationIndexName ( field['Index'], all_factors ) )

		field_index						= {}		
		field_index['ForecastIndex']	= getInflationFactor ( field['Index'], static_offsets, stochastic_offsets, all_tenors )
		field_index['PriceIndex'] 		= getPriceIndexFactor ( field['PriceIndex'], static_offsets, stochastic_offsets ) 
		field_index['Discount'] 		= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Currency']			= getFXRateFactor ( field['Currency'], static_offsets, stochastic_offsets )
		
		inflation_factor, index_factor 	= getInflationIndexObjects ( field['Index'], field['PriceIndex'], all_factors )
		field_index['IndexMethod']  	= Utils.CASHFLOW_IndexMethodLookup [ inflation_factor.GetReferenceName() ]
		
		field_index['Cashflows'], field_index['Base_Resets'], field_index['Final_Resets']  = Utils.MakeIndexCashflows ( base_date, time_grid, 1 if self.field['Buy_Sell']=='Buy' else -1, self.field['Cashflows'], inflation_factor, index_factor, self.field.get('Settlement_Date') )
		field_index['StartIndex']  		= field_index['Cashflows'].GetCashflowStartIndex ( time_grid, ( self.field['Settlement_Date'] - base_date ).days if self.field['Is_Forward_Deal']=='Yes' else None )
		
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)
		
		AddIndexCashflow 	= module.get_function ('AddIndexCashflow')
		AddIndexCashflow( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32 (  field_index['IndexMethod'] ),
							np.int32 ( 1 ),
							np.int32  ( field_index['Cashflows'].Count() ),
							drv.In    ( field_index['Cashflows'].GetRawSchedule(precision).ravel() ),
							drv.In    ( field_index['StartIndex'] ),
							drv.In 	  ( field_index['Cashflows'].Resets.GetSchedule(precision).ravel() ),
							drv.In 	  ( field_index['Base_Resets'].GetSchedule(precision).ravel() ),
							drv.In 	  ( field_index['Final_Resets'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['PriceIndex'] ),
							drv.In    ( field_index['ForecastIndex'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )

class FloatingSwapLegDeal(Deal):
	#dependent price factors for this instrument 
	factor_fields = {'Currency':							['FxRate'],
					 'Discount_Rate':						['DiscountRate'],
					 'Forecast_Rate':						['InterestRate'],
					 'Forecast_Rate_Volatility':			['InterestRateVol','InterestYieldVol'],
					 'Discount_Rate_Volatility':			['InterestRateVol','InterestYieldVol']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Amortisation':			'Permits an amortisation schedule for the principal to be specified as a list of (Date, Amount) pairs, where Date represents the date of this Amortisation and Amount represents the amount by which the principal is reduced by the Amortisation. A positive value for Amount represents a fall in the principal amount while a negative value for Amount represents a rise in the principal amount',
						'Known_Rates':			'List of reset dates and corresponding known interest rates specified in the form of (date, value) pairs. Rates are entered in percentage. For example, enter 5 for 5%',
						'Forecast_Rate':		"ID of the interest rate price factor used to calculate forward interest rates. For example, USD.AAA. A quanto variant of the deal can be obtained by ensuring the currency of Forecast_Rate (USD in the case of the USD.AAA example), also called the interest rate currency, is different from the settlement currency given by Currency. If left blank, the deal Discount_Rate interest rate price factor will be used. An entry in this property requires the corresponding Interest_Rate price factor. Quanto deals also require: the FX rate price factor of the FX rate given by the interest rate currency and base currency; the FX volatility price factor of the FX rate given by the interest rate currency and the settlement currency; and the implied correlations price factor between the FX rate given by the interest rate currency and the settlement currency, the interest rate currency's interest rate and the settlement currency's interest rate",
						'Effective_Date':		'The contract start date',
						'Maturity_Date':		'The contract end date',
						'Payment_Frequency':	'The payment frequency, which is the period between payments. If this property is set to 0M, then the payment periods will be determined by the trade effective, maturity and stub dates (if specified)',
						'Interest_Frequency':	'Interest frequency, which is the length of interest accrual periods. If this property is set to 0M, then the length of the interest accrual periods will be the payment frequency. Deals with compounding require this property to be less than payment frequency',
						'Reset_Frequency':		'The reset frequency, which is the period between resets. This is required if the deal uses averaging and must not be greater than the payment frequency',
						'Payment_Offset':		'Specifies the number of business days between accrual date and payment date. For example, if this property is set to 1 then the payment dates are shifted forward by one business day',
						'Margin':				'Margin rate added to the generalized cashflow. Rates are entered in basis points. For example, enter 50 for 50 basis points',
						'Margin_Schedule':		'Schedule of margin rates and payment dates. Overrides Margin/Floating_Margin for the payment dates in the schedule. Rates are entered in basis points',
						'Rate_Multiplier':		'Rate multiplier of the floating interest rate',
						'Rate_Constant':		'Rate constant of the floating interest rate. Values are entered as a decimal',
						'Principal':			'Principal amount in units of settlement currency',
						'Principal_Exchange':	'Type of principal exchange cashflows: no principal cashflows (None), principal cashflows at the effective date (Start), principal cashflows at the maturity date (Maturity), or principal cashflows at both the effective date and the maturity date (Start_Maturity)'
						}
						
	#cuda code required to price this instrument - none - as it's broken up into an addfixedleg and an addfloatleg
	cudacodetemplate = ''
	def __init__(self, params):
		super(FloatingSwapLegDeal, self).__init__(params)
		
	def reset(self, calendars):
		super(FloatingSwapLegDeal, self).reset()
		self.dates		= generatedatesBackward ( self.field['Maturity_Date'], self.field['Effective_Date'], self.field['Payment_Frequency'] )
		self.add_reval_dates ( self.dates, self.field['Currency'] )
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 		= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']	= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Forecast_Rate']	= Utils.CheckRateName(self.field['Forecast_Rate']) if self.field['Forecast_Rate'] else field['Discount_Rate']

		field_index		= {}		
		field_index['Forward']  		= getInterestFactor ( field['Forecast_Rate'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Discount'] 		= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Currency']			= getFXRateFactor 	( field['Currency'], static_offsets, stochastic_offsets )
		field_index['InterestYieldVol'] = np.zeros(1, dtype=np.int32)

		Nominal_Principal 		  			= ( 1.0 if self.field['Buy_Sell']=='Buy' else -1.0 ) * self.field['Principal']
		field_index['CompoundingMethod'] 	= Utils.CASHFLOW_CompoundingMethodLookup [ self.field['Compounding_Method'] ]
		field_index['Cashflows']  			= Utils.GenerateFloatCashflows ( base_date, time_grid, self.dates,  Nominal_Principal, self.field['Amortisation'], self.field['Known_Rates'], self.field['Interest_Frequency'], self.field['Index_Tenor'], Utils.GetDayCount( self.field['Accrual_Day_Count'] ), self.field['Margin']/10000.0 )
		
		field_index['Cashflows'].AddFixedPayments(base_date, self.field['Principal_Exchange'], self.field['Effective_Date'], self.field['Accrual_Day_Count'], Nominal_Principal)
		field_index['StartIndex'] 			= field_index['Cashflows'].GetCashflowStartIndex(time_grid)
		
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)

		AddFloatCashflow 	= module.get_function ('AddFloatCashflow')
		AddFloatCashflow( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32 (  Utils.SCENARIO_CASHFLOWS_FloatLeg ),
							np.int32 (  field_index['CompoundingMethod'] ),
							np.int32( 1 ),
							np.int32  ( field_index['Cashflows'].Count() ),
							drv.In    ( field_index['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['StartIndex'] ),
							drv.In 	  ( field_index['Cashflows'].Resets.GetSchedule(precision).ravel() ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['InterestYieldVol'] ),
							drv.In    ( field_index['Forward'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )
		
class CapDeal(Deal):
	#dependent price factors for this instrument
	factor_fields = {'Currency':				['FxRate'],
					 'Discount_Rate':			['DiscountRate'],
					 'Forecast_Rate':			['InterestRate'],
					 'Forecast_Rate_Volatility':['InterestRateVol','InterestYieldVol'],
					 'Discount_Rate_Volatility':['InterestRateVol','InterestYieldVol']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Forecast_Rate':		"ID of the interest rate price factor used to calculate forward interest rates. For example, USD.AAA. A quanto variant of the deal can be obtained by ensuring the currency of Forecast_Rate (USD in the case of the USD.AAA example), also called the interest rate currency, is different from the settlement currency given by Currency. If left blank, the deal Discount_Rate interest rate price factor will be used. An entry in this property requires the corresponding Interest_Rate price factor. Quanto deals also require: the FX rate price factor of the FX rate given by the interest rate currency and base currency; the FX volatility price factor of the FX rate given by the interest rate currency and the settlement currency; and the implied correlations price factor between the FX rate given by the interest rate currency and the settlement currency, the interest rate currency's interest rate and the settlement currency's interest rate",
						'Effective_Date':		'The contract start date',
						'Maturity_Date':		'The contract end date',
						'Principal':			'Principal amount in units of settlement currency',
						'Amortisation':			'Permits an amortisation schedule for the principal to be specified as a list of (Date, Amount) pairs, where Date represents the date of this Amortisation and Amount represents the amount by which the principal is reduced by the Amortisation. A positive value for Amount represents a fall in the principal amount while a negative value for Amount represents a rise in the principal amount',
						'Payment_Interval':		'The payment frequency, which is the period between payments. This property must not be less than the reset frequency',
						'Reset_Frequency':		'The reset frequency, which is the period between resets. This is required if the deal uses averaging and must not be greater than the payment frequency. If this property entry is set to 0M, the reset frequency will be assumed to be the interest frequency',
						'Payment_Offset':		'Specifies the number of business days between accrual date and payment date. For example, if this property is set to 1 then the payment dates are shifted forward by one business day',
						'Known_Rates':			'List of reset dates and corresponding known interest rates specified in the form of (date, value) pairs. Rates are entered in percentage. For example, enter 5 for 5%',
						'Cap_Rate':				'Cap strike rate. Rates are entered in percentage. For example, enter 5 for 5%'
						}
	
	#cuda code required to price this instrument - none - as it's in addfloatcashflow
	cudacodetemplate = ''
	
	def __init__(self, params):
		super(CapDeal, self).__init__(params)

	def reset(self, calendars):
		super(CapDeal, self).reset()
		self.resetdates	= generatedatesBackward ( self.field['Maturity_Date'], self.field['Effective_Date'], self.field['Payment_Interval'] )
		self.add_reval_dates ( self.resetdates, self.field['Currency'] )
		#this swap could be quantoed
		self.isQuanto 		= None

	def PostProcess( self, module, CudaMem, precision, batch_size, revals_per_batch, deal_data, child_dependencies, partition, mtm_offset ):
		logging.warning('CapDeal {0} - TODO'.format(self.field['Reference']))
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 		= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']	= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Forecast_Rate']	= Utils.CheckRateName(self.field['Forecast_Rate']) if self.field['Forecast_Rate'] else field['Discount_Rate']
		field['Forecast_Rate_Volatility']	= Utils.CheckRateName(self.field['Forecast_Rate_Volatility']) 

		field_index		= {}		
		self.isQuanto 	= getInterestRateCurrency (field['Forecast_Rate'],  all_factors) != field['Currency']
		if self.isQuanto:
			#TODO - Deal with Quanto Interest Rate swaps
			pass
		else:
			field_index['Forward']  	= getInterestFactor ( field['Forecast_Rate'], static_offsets, stochastic_offsets, all_tenors )
			field_index['Discount'] 	= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
			field_index['Currency']		= getFXRateFactor ( field['Currency'], static_offsets, stochastic_offsets )
			field_index['VolSurface']	= getInterestVolFactor ( field['Forecast_Rate_Volatility'], self.field['Payment_Interval'], static_offsets, stochastic_offsets, all_tenors )
			field_index['Cashflows']    = Utils.GenerateFloatCashflows ( base_date, time_grid, self.resetdates, ( 1.0 if self.field['Buy_Sell']=='Buy' else -1.0 ) * self.field['Principal'], self.field['Amortisation'], self.field['Known_Rates'], self.field['Index_Tenor'], self.field['Reset_Frequency'], Utils.GetDayCount( self.field['Accrual_Day_Count'] ), self.field['Cap_Rate']/100.0 )
		
		field_index['StartIndex'] 		= field_index['Cashflows'].GetCashflowStartIndex(time_grid)
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)

		AddFloatCashflow 	= module.get_function ('AddFloatCashflow')
		AddFloatCashflow( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32 (  Utils.SCENARIO_CASHFLOWS_Cap ),
							np.int32 (  Utils.CASHFLOW_METHOD_Average_Interest ),
							np.int32 ( 1 ),
							np.int32  ( field_index['Cashflows'].Count() ),
							drv.In    ( field_index['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['StartIndex'] ),
							drv.In 	  ( field_index['Cashflows'].Resets.GetSchedule(precision).ravel() ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['VolSurface'] ),
							drv.In    ( field_index['Forward'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )

class FloorDeal(Deal):
	#dependent price factors for this instrument
	factor_fields = {'Currency':				['FxRate'],
					 'Discount_Rate':			['DiscountRate'],
					 'Forecast_Rate':			['InterestRate'],
					 'Forecast_Rate_Volatility':['InterestRateVol','InterestYieldVol'],
					 'Discount_Rate_Volatility':['InterestRateVol','InterestYieldVol']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Forecast_Rate':		"ID of the interest rate price factor used to calculate forward interest rates. For example, USD.AAA. A quanto variant of the deal can be obtained by ensuring the currency of Forecast_Rate (USD in the case of the USD.AAA example), also called the interest rate currency, is different from the settlement currency given by Currency. If left blank, the deal Discount_Rate interest rate price factor will be used. An entry in this property requires the corresponding Interest_Rate price factor. Quanto deals also require: the FX rate price factor of the FX rate given by the interest rate currency and base currency; the FX volatility price factor of the FX rate given by the interest rate currency and the settlement currency; and the implied correlations price factor between the FX rate given by the interest rate currency and the settlement currency, the interest rate currency's interest rate and the settlement currency's interest rate",
						'Effective_Date':		'The contract start date',
						'Maturity_Date':		'The contract end date',
						'Principal':			'Principal amount in units of settlement currency',
						'Amortisation':			'Permits an amortisation schedule for the principal to be specified as a list of (Date, Amount) pairs, where Date represents the date of this Amortisation and Amount represents the amount by which the principal is reduced by the Amortisation. A positive value for Amount represents a fall in the principal amount while a negative value for Amount represents a rise in the principal amount',
						'Payment_Interval':		'The payment frequency, which is the period between payments. This property must not be less than the reset frequency',
						'Reset_Frequency':		'The reset frequency, which is the period between resets. This is required if the deal uses averaging and must not be greater than the payment frequency. If this property entry is set to 0M, the reset frequency will be assumed to be the interest frequency',
						'Payment_Offset':		'Specifies the number of business days between accrual date and payment date. For example, if this property is set to 1 then the payment dates are shifted forward by one business day',
						'Known_Rates':			'List of reset dates and corresponding known interest rates specified in the form of (date, value) pairs. Rates are entered in percentage. For example, enter 5 for 5%',
						'Floor_Rate':			'Floor strike rate. Rates are entered in percentage. For example, enter 5 for 5%'
						}
	
	#cuda code required to price this instrument - none - as it's in addfloatcashflow
	cudacodetemplate = ''
	
	def __init__(self, params):
		super(FloorDeal, self).__init__(params)

	def reset(self, calendars):
		super(FloorDeal, self).reset()
		self.resetdates	= generatedatesBackward ( self.field['Maturity_Date'], self.field['Effective_Date'], self.field['Payment_Interval'] )
		self.add_reval_dates ( self.resetdates, self.field['Currency'] )
		#this swap could be quantoed
		self.isQuanto 		= None

	def PostProcess( self, module, CudaMem, precision, batch_size, revals_per_batch, deal_data, child_dependencies, partition, mtm_offset ):
		logging.warning('FloorDeal {0} - TODO'.format(self.field['Reference']))
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 		= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']	= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Forecast_Rate']	= Utils.CheckRateName(self.field['Forecast_Rate']) if self.field['Forecast_Rate'] else field['Discount_Rate']
		field['Forecast_Rate_Volatility']	= Utils.CheckRateName(self.field['Forecast_Rate_Volatility']) 

		field_index		= {}		
		self.isQuanto 	= getInterestRateCurrency (field['Forecast_Rate'],  all_factors) != field['Currency']
		if self.isQuanto:
			#TODO - Deal with Quanto Interest Rate swaps
			pass
		else:
			field_index['Forward']  	= getInterestFactor ( field['Forecast_Rate'], static_offsets, stochastic_offsets, all_tenors )
			field_index['Discount'] 	= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
			field_index['Currency']		= getFXRateFactor ( field['Currency'], static_offsets, stochastic_offsets )
			field_index['VolSurface']	= getInterestVolFactor ( field['Forecast_Rate_Volatility'], self.field['Payment_Interval'], static_offsets, stochastic_offsets, all_tenors )
			field_index['Cashflows']    = Utils.GenerateFloatCashflows ( base_date, time_grid, self.resetdates, ( 1.0 if self.field['Buy_Sell']=='Buy' else -1.0 ) * self.field['Principal'], self.field['Amortisation'], self.field['Known_Rates'], self.field['Index_Tenor'], self.field['Reset_Frequency'], Utils.GetDayCount( self.field['Accrual_Day_Count'] ), self.field['Floor_Rate']/100.0 )
										 
		field_index['StartIndex'] 		= field_index['Cashflows'].GetCashflowStartIndex(time_grid)
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)
		
		AddFloatCashflow 	= module.get_function ('AddFloatCashflow')
		AddFloatCashflow( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32 ( Utils.SCENARIO_CASHFLOWS_Floor ),
							np.int32 ( Utils.CASHFLOW_METHOD_Average_Interest ),
							np.int32 ( 1 ),
							np.int32  ( field_index['Cashflows'].Count() ),
							drv.In    ( field_index['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['StartIndex'] ),
							drv.In 	  ( field_index['Cashflows'].Resets.GetSchedule(precision).ravel() ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['VolSurface'] ),
							drv.In    ( field_index['Forward'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )

class SwaptionDeal(Deal):
	factor_fields = {'Currency':				['FxRate'],
					 'Discount_Rate':			['DiscountRate'],
					 'Forecast_Rate':			['InterestRate'],
					 'Forecast_Rate_Volatility':['InterestRateVol','InterestYieldVol']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Forecast_Rate':		"ID of the interest rate price factor used to calculate forward interest rates. For example, USD.AAA. A quanto variant of the deal can be obtained by ensuring the currency of Forecast_Rate (USD in the case of the USD.AAA example), also called the interest rate currency, is different from the settlement currency given by Currency. If left blank, the deal Discount_Rate interest rate price factor will be used. An entry in this property requires the corresponding Interest_Rate price factor. Quanto deals also require: the FX rate price factor of the FX rate given by the interest rate currency and base currency; the FX volatility price factor of the FX rate given by the interest rate currency and the settlement currency; and the implied correlations price factor between the FX rate given by the interest rate currency and the settlement currency, the interest rate currency's interest rate and the settlement currency's interest rate",
						'Payer_Receiver':		'Indicates whether the holder pays fixed (Payer) or receives fixed (Receiver) leg of the deal',
						'Principal':			'Principal amount in units of settlement currency',
						'Pay_Amortisation':		'Permits an amortisation schedule for the principal to be specified as a list of (Amount, Date) pairs, where Amount represents the amount by which the principal is reduced by the Amortisation and Date represents the date of this Amortisation. A positive value for Amount represents a fall in the principal amount while a negative value for Amount represents a rise in the principal amount. For legs with Principal Exchange at meturity each amortisation generates a cashflow equal to the amortised amount as the amortisation date. For legs with a payment offset, this flow will be delayed by the payment offset',
						'Receive_Amortisation': 'Permits an amortisation schedule for the principal to be specified as a list of (Amount, Date) pairs, where Amount represents the amount by which the principal is reduced by the Amortisation and Date represents the date of this Amortisation. A positive value for Amount represents a fall in the principal amount while a negative value for Amount represents a rise in the principal amount. For legs with Principal Exchange at maturity each amortisation generates a cashflow equal to the amortised amount as the amortisation date. For legs with a payment offset, this flow will be delayed by the Payment offset',
						'Option_Expiry_Date':	'Option expiry date',
						'Settlement_Date':		'Settlement date',
						'Swap_Maturity_Date':	'Maturity date of underlying swap',
						'Pay_Frequency':		'The payment frequency, which is the period between payments of the pay leg',
						'Receive_Frequency':	'The payment frequency, which is the period between payments of the receive leg',
						'Swap_Rate':			'Fixed interest rate. Rates are entered in percentage. For example, enter 5 for 5%',
						'Floating_Margin':		'Margin rate added to the floating interest rate. Rates are entered in basis points. For example, enter 50 for 50 basis points',
						'Margin_Schedule':		'Schedule of margin rates and payment dates. Overrides Floating_Margin for the payment dates in the schedule. Rates are entered in basis points. For example, enter 50 for 50 basis points'
						}
	
	#cuda code required to price this instrument
	cudacodetemplate = '''
		__global__ void SwaptionDeal	(	const REAL* __restrict__ stoch_factors,
											const REAL* __restrict__ static_factors,
											const int*  __restrict__ deal_time_index,
											const REAL* __restrict__ time_grid,
											const int*  __restrict__ cashflow_index,
											REAL* all_cashflows,
											REAL* Output,
											REAL  maturity_in_days,
											REAL  underlyingSwapMaturity,
											REAL  strike,
											REAL  callorput,
											REAL  buyorsell,
											int num_fixed_cashflow,
											const REAL* __restrict__ fixed_cashflow,
											const int*  __restrict__ fixed_starttime_index,
											int   num_float_cashflow,
											const REAL* __restrict__ float_cashflow,
											const int*  __restrict__ float_starttime_index,
											const REAL* __restrict__ resets,
											const REAL* __restrict__ expiry_sample,
											const int* __restrict__ currency,
											const int* __restrict__ interest_rate_vol,
											const int* __restrict__ forward,
											const int* __restrict__ discount )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];			
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
			OFFSET scenario_index		= scenario_number * ScenarioTimeSteps;
			OFFSET scenario_prior_index = scenario_index + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			
			//set the mtm of this deal
			REAL mtm 					= 0.0;
			//FX rate for this deal
			REAL FX_Base				= ScenarioRate ( t, currency, scenario_prior_index, static_factors, stoch_factors ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
			//Fixed and float legs
			REAL fixed_leg, float_leg;
			
			//dummy variabe needed to call the cashflow pv functions
			REAL settlement_cashflow    = 0.0;
			
			//work out the remaining term
			REAL remainingMaturityinDays = maturity_in_days - t[TIME_GRID_MTM];
			
			if ( remainingMaturityinDays>= 0.0 )
			{
				REAL expiry 	= calcDayCountAccrual(remainingMaturityinDays, discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
				//Get the reporting currency
				
				//PV the fixed leg
				fixed_leg	= ScenarioPVFixedLeg ( t, CASHFLOW_METHOD_Ignore_Fixed_Rate, CASHFLOW_METHOD_Fixed_Compounding_No, 0, num_fixed_cashflow, settlement_cashflow, fixed_cashflow,
															discount, scenario_prior_index, static_factors, stoch_factors );
				//PV the float leg
				float_leg	= ScenarioPVFloatLeg ( t, CASHFLOW_METHOD_Average_Interest, 0, num_float_cashflow, settlement_cashflow, NULL, float_cashflow, resets,
														interest_rate_vol, forward, discount, scenario_prior_index, static_factors, stoch_factors );

				REAL swap_rate  = - float_leg / fixed_leg;
				REAL vol 		= ScenarioSurface3D   ( t, swap_rate - strike, expiry, underlyingSwapMaturity, interest_rate_vol, scenario_prior_index, static_factors, stoch_factors );
				REAL payoff		= blackEuropeanOption ( swap_rate, strike, 0.0, vol, expiry, buyorsell, callorput );
				
				//now calculate the MTM
				mtm 			= fabs ( fixed_leg ) * payoff * FX_Base;
				
				//Settle the currency_cashflows
				if ((remainingMaturityinDays==0) && (!expiry_sample[RESET_INDEX_Start_Day]))
					ScenarioSettle ( mtm/FX_Base, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows );
			}
			else if ( expiry_sample[RESET_INDEX_Start_Day] )
			{
				//Scenario Offset at time of expiry
				const OFFSET scenario_offset 	= reinterpret_cast<const OFFSET &>(expiry_sample[RESET_INDEX_Scenario]);
				
				//Instrument was physically settled - need to check that the swap was in the money when it expired
				fixed_leg	= ScenarioPVFixedLeg ( expiry_sample + RESET_INDEX_Time_Grid, CASHFLOW_METHOD_Fixed_Rate_Standard, CASHFLOW_METHOD_Fixed_Compounding_No, 0, num_fixed_cashflow, settlement_cashflow,
													fixed_cashflow, discount, scenario_index + scenario_offset, static_factors, stoch_factors );
				//PV the float leg
				float_leg	= ScenarioPVFloatLeg ( expiry_sample + RESET_INDEX_Time_Grid, CASHFLOW_METHOD_Average_Interest, 0, num_float_cashflow, settlement_cashflow, NULL, float_cashflow, resets,
													interest_rate_vol, forward, discount, scenario_index + scenario_offset, static_factors, stoch_factors );
														
				if ( callorput * ( fabs ( float_leg ) - fabs ( fixed_leg ) ) > 0.0 )
				{
					//If the swap was in the money at expiry, assume the swaption is now a swap					
					fixed_leg	= ScenarioPVFixedLeg ( t, CASHFLOW_METHOD_Fixed_Rate_Standard, CASHFLOW_METHOD_Fixed_Compounding_No, fixed_starttime_index[mtm_time_index], num_fixed_cashflow,
														settlement_cashflow, fixed_cashflow, discount, scenario_prior_index, static_factors, stoch_factors );
														
					//Settle any cashflow									
					if (settlement_cashflow!=0)
						ScenarioSettle ( settlement_cashflow, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows );
						
					float_leg	= ScenarioPVFloatLeg ( t, CASHFLOW_METHOD_Average_Interest, mtm_time_index, num_float_cashflow, settlement_cashflow, float_starttime_index, float_cashflow, resets,
														interest_rate_vol, forward, discount, scenario_prior_index, static_factors, stoch_factors );
					//Settle any cashflow
					if (settlement_cashflow!=0)
						ScenarioSettle ( settlement_cashflow, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows );

					//convert to the settlement currency
					mtm 			= buyorsell * callorput * ( fabs ( float_leg ) - fabs ( fixed_leg ) ) * FX_Base;
					
					/*
					if (threadIdx.x==0 && blockIdx.x==0)
					{
						printf("Swaption,mtm_time_index,%d,float,%.6f,fixed,%.6f,mtm,%.6f\\n",mtm_time_index,float_leg,fixed_leg,mtm);
					}
					*/
				}
			}
			
			Output [ index ] = mtm ;
		}
		'''
	
	def __init__(self, params):
		super(SwaptionDeal, self).__init__(params)

	def reset(self, calendars):
		super(SwaptionDeal, self).reset()
		self.add_reval_dates ( set([self.field['Option_Expiry_Date']]), self.field['Currency'] )
		#calc the underlying swap dates
		self.paydates	= generatedatesBackward ( self.field['Swap_Maturity_Date'], self.field['Swap_Effective_Date'], self.field['Pay_Frequency'] )
		self.recdates	= generatedatesBackward ( self.field['Swap_Maturity_Date'], self.field['Swap_Effective_Date'], self.field['Receive_Frequency'] )
		
		if self.field['Settlement_Style']=='Physical':
			self.add_reval_dates ( self.paydates, self.field['Currency'] )
			self.add_reval_dates ( self.recdates, self.field['Currency'] )
		else:
			self.add_reval_dates ( set([self.field['Settlement_Date']]), self.field['Currency'] )

	def finalize_dates ( self, parser, base_date, grid, node_children, node_resets, node_settlements ):
		#have to reset the original instrument and let the child node decide
		super(SwaptionDeal, self).reset()
		self.add_reval_dates ( set([self.field['Option_Expiry_Date']]), self.field['Currency'] )
		
		if self.field['Settlement_Style']=='Cash':
			#cash settle - do not include node cashflows
			node_resets.clear()
			node_settlements.clear()
			self.add_reval_dates ( set([self.field['Settlement_Date']]), self.field['Currency'] )
		else:
			self.add_reval_dates ( node_resets, self.field['Currency'] )

		return super ( SwaptionDeal, self ).finalize_dates ( parser, base_date, grid, node_children, node_resets, node_settlements )
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 		= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']	= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Forecast_Rate']	= Utils.CheckRateName(self.field['Forecast_Rate']) if self.field['Forecast_Rate'] else field['Discount_Rate']
		field['Forecast_Rate_Volatility']	= Utils.CheckRateName(self.field['Forecast_Rate_Volatility']) 

		field_index		= {}
		field_index['Forward']  	= getInterestFactor ( field['Forecast_Rate'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Discount'] 	= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Currency']		= getFXRateFactor ( field['Currency'], static_offsets, stochastic_offsets )
		#hard coding a tenor of 2 years to force using an interest yield vol price factor. 
		field_index['VolSurface']	= getInterestVolFactor ( field['Forecast_Rate_Volatility'], pd.DateOffset(years=2), static_offsets, stochastic_offsets, all_tenors)
		field_index['Expiry'] 		= (self.field['Option_Expiry_Date'] - base_date).days
		
		#need to check defaults - f**k you adaptiv
		Principal 					= self.field.get('Principal', 1000000.0)
		Pay_Amortisation			= self.field.get('Pay_Amortisation')
		Receive_Amortisation		= self.field.get('Receive_Amortisation')
		Receive_Day_Count			= self.field.get('Receive_Day_Count', 'ACT_365' )
		Pay_Day_Count				= self.field.get('Pay_Day_Count', 'ACT_365' )
		Floating_Margin				= self.field.get('Floating_Margin', 0.0 )
		Index_Day_Count				= self.field.get('Index_Day_Count', 'ACT_365' )
		
		if self.field['Payer_Receiver']=='Payer':
			field_index['FixedCashflows'] = Utils.GenerateFixedCashflows ( base_date, self.paydates, -Principal, Pay_Amortisation, Utils.GetDayCount( Pay_Day_Count ), self.field['Swap_Rate']/100.0 )
			field_index['FloatCashflows'] = Utils.GenerateFloatCashflows ( base_date, time_grid, self.recdates, Principal, Receive_Amortisation, None, self.field['Receive_Frequency'], self.field['Index_Tenor'], Utils.GetDayCount( Receive_Day_Count ), Floating_Margin/10000.0 )
		else:
			field_index['FixedCashflows'] = Utils.GenerateFixedCashflows ( base_date, self.recdates, Principal, Receive_Amortisation, Utils.GetDayCount ( Receive_Day_Count ), self.field['Swap_Rate']/100.0 )
			field_index['FloatCashflows'] = Utils.GenerateFloatCashflows ( base_date, time_grid, self.paydates, -Principal, Pay_Amortisation, None, self.field['Pay_Frequency'], self.field['Index_Tenor'], Utils.GetDayCount ( Pay_Day_Count ), Floating_Margin/10000.0 )

		if self.field['Settlement_Style']=='Physical':
			#remember to potentially deliver the underlying swap if it's in the money
			field_index['FixedStartIndex'] 	= field_index['FixedCashflows'].GetCashflowStartIndex(time_grid)
			field_index['FloatStartIndex'] 	= field_index['FloatCashflows'].GetCashflowStartIndex(time_grid)
			field_index['SampleExpiryDate']	= Utils.MakeSamplingData ( base_date, time_grid, [ [self.field['Option_Expiry_Date'], 0.0, 1.0] ] )
		else:
			field_index['FixedStartIndex'] 	= np.zeros(1, dtype=np.int32)
			field_index['FloatStartIndex']  = np.zeros(1, dtype=np.int32)
			#notice the base date in place of the expiry date - a hacky way to tell the pricing function not to settle physically
			field_index['SampleExpiryDate'] = Utils.MakeSamplingData ( base_date, time_grid, [ [base_date, 0.0, 1.0] ] )
		
		#might want to change this
		field_index['Underlying_Swap_maturity'] = Utils.GetDayCountAccrual ( base_date, (self.field['Swap_Maturity_Date']-self.field['Swap_Effective_Date']).days, Utils.GetDayCount( Index_Day_Count ) )
		
		return field_index

	def Aggregate(self, CudaMem, parent_partition, index, partition):
		parent_partition.DealMTMs[ index ] += partition.DealMTMs [ index ]
		drv.memcpy_htod ( CudaMem.d_MTM_Accum_Buffer.ptr, parent_partition.DealMTMs [index] )
		
		#reset the cashflow buffer
		if CudaMem.d_Cashflows:
			parent_partition.Cashflows [ index ] += partition.Cashflows [ index ]
			drv.memcpy_htod ( CudaMem.d_Cashflows.ptr, parent_partition.Cashflows [ index ] )

	def PostProcess( self, module, CudaMem, precision, batch_size, revals_per_batch, deal_data, child_dependencies, partition, mtm_offset ):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, deal_data.Time_dep.deal_time_grid.size)
		dependencies		= deal_data.Factor_dep
		#child_map			= { x.Instrument.field['Object']: x.Factor_dep for x in child_dependencies }
		child_map			= dict( ( x.Instrument.field['Object'], x.Factor_dep ) for x in child_dependencies )
		
		SwaptionDeal	= module.get_function ('SwaptionDeal')
		SwaptionDeal     (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(deal_data.Time_dep.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							precision ( dependencies['Expiry'] ),
							precision ( dependencies['Underlying_Swap_maturity'] ),
							precision ( self.field['Swap_Rate']/100.0 ),
							precision ( 1.0 if self.field['Payer_Receiver']=='Payer' else -1.0 ),
							precision ( 1.0 if self.field['Buy_Sell']=='Buy' else -1.0 ),
							np.int32  ( child_map['CFFixedInterestListDeal']['Cashflows'].Count() ),
							drv.In    ( child_map['CFFixedInterestListDeal']['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( child_map['CFFixedInterestListDeal']['StartIndex'] ),
							np.int32  ( child_map['CFFloatingInterestListDeal']['Cashflows'].Count() ),
							drv.In    ( child_map['CFFloatingInterestListDeal']['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( child_map['CFFloatingInterestListDeal']['StartIndex'] ),
							drv.In 	  ( child_map['CFFloatingInterestListDeal']['Cashflows'].Resets.GetSchedule(precision).ravel() ),
							drv.In ( dependencies['SampleExpiryDate'].GetSchedule(precision).ravel() ),
							drv.In ( dependencies['Currency'] ),
							drv.In ( dependencies['VolSurface'] ),
							drv.In ( dependencies['Forward'] ),
							drv.In ( dependencies['Discount'] ),
							block=block, grid=grid )
		
		#interpolate the Theo price
		self.InterpolateBlock ( module, CudaMem, deal_data, batch_size, revals_per_batch )
		
		#copy the cashflows
		if CudaMem.d_Cashflows:
			partition.Cashflows [ mtm_offset ] = CudaMem.d_Cashflows.get()[:batch_size*revals_per_batch]
			
		#copy the mtm
		partition.DealMTMs  [ mtm_offset ] = CudaMem.d_MTM_Buffer.get()[:batch_size*revals_per_batch]
		
	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)

		SwaptionDeal	= module.get_function ('SwaptionDeal')
		SwaptionDeal     (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							precision ( field_index['Expiry'] ),
							precision ( field_index['Underlying_Swap_maturity'] ),
							precision ( self.field['Swap_Rate']/100.0 ),
							precision ( 1.0 if self.field['Payer_Receiver']=='Payer' else -1.0 ),
							precision ( 1.0 if self.field['Buy_Sell']=='Buy' else -1.0 ),
							np.int32  ( field_index['FixedCashflows'].Count() ),
							drv.In    ( field_index['FixedCashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['FixedStartIndex'] ),
							np.int32  ( field_index['FloatCashflows'].Count() ),
							drv.In    ( field_index['FloatCashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['FloatStartIndex'] ),
							drv.In 	  ( field_index['FloatCashflows'].Resets.GetSchedule(precision).ravel() ),
							drv.In ( field_index['SampleExpiryDate'].GetSchedule(precision).ravel() ),
							drv.In ( field_index['Currency'] ),
							drv.In ( field_index['VolSurface'] ),
							drv.In ( field_index['Forward'] ),
							drv.In ( field_index['Discount'] ),
							block=block, grid=grid )

class EquityDiscreteExplicitAsianOption(Deal):
	factor_fields = {'Currency':				['FxRate'],
					 'Equity':					['EquityPrice','DividendRate'],
					 'Discount_Rate':			['DiscountRate'],
					 'Equity_Volatility':		['EquityPriceVol']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Expiry_Date':			'Expiry date of the option',
						'Units':				'Number of units of the option held',
						'Strike_Price':			'Strike price of the option in asset currency (or settlement currency for compo options)',
						'Payoff_Currency':		'ID of the FX rate price factor used to define the settlement currency. This property is required for deals with a quanto or compo Payoff_Type. For deals with a standard Payoff_Type, this property must be the same as Currency or left empty. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor. Quanto or Compo deals also require the FX volatility price factor of the FX rate given by the asset currency and this settlement currency and the Implied Correlations price factor between that FX rate and the underlying asset',
						'Equity':				'Underlying equity name. For example, IBM. This requires the Equity Price price factor and the Equity Dividend Rate price factor',
						'Equity_Volatility':	"ID of the equity volatility price factor used to value the deal. This property must be set to either 'Equity' or 'Equity.Curreny', where Equity is the underlying equity name and Currency is the ID of the currency of the equity. For example, 'IBM' or 'IBM.USD'. To use an equity volatility spread price factor, volatility spreads must be enabled and this property must use a tilde ('~') character and be set to 'Base~Spread', where 'Base' must be either 'Equity' or 'Equity.Currency' as described above and 'Spread' is the ID of the spread. For example, 'IBM~BARRIERKOCALL' or 'IBM.USD~BARRIERKOCALL'. If left blank, the underlying equity name will be used. This requires the equity price volatility price factor and, if volatility spreads are enabled and a tilde ('~') character is used, the equity price volatility spread price factor"
						}
						
	cudacodetemplate = '''
		//all pricing functions will have the same 5 initial parameters
		//Note that we assume Term Structured is No - ie. constant cost of carry over the averaging period
		__global__ void EquityDiscreteExplicitAsianOption(	const REAL* __restrict__ stoch_factors,
															const REAL* __restrict__ static_factors,
															const int*  __restrict__ deal_time_index,
															const REAL* __restrict__ time_grid,
															const int*  __restrict__ cashflow_index,
															REAL* all_cashflows,
															REAL* Output,
															REAL maturity_in_days,
															REAL strike,
															REAL callorput,
															REAL buyorsell,
															REAL units,											
															const int* __restrict__ currency,
															const int* __restrict__ equity,
															const int* __restrict__ equity_zero,
															const int* __restrict__ dividend_yield,
															const int* __restrict__ discount,
															int   num_samples,
															const int*  __restrict__ starttime_index,
															const REAL* __restrict__ samples,
															const int* __restrict__ equityvolsurface )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];			
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
			OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );

			//set the mtm of this deal to 0
			REAL mtm 					= 0.0;

			//work out the remaining term
			REAL remainingMaturityinDays = maturity_in_days - t[TIME_GRID_MTM];

			if (remainingMaturityinDays>= 0.0)
			{
				//assume the daycount is stored in the first factor 
				REAL expiry 				= calcDayCountAccrual(remainingMaturityinDays, discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
				REAL baseFX				    = ScenarioRate ( t, currency, scenario_prior_index, static_factors, stoch_factors ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
													
				//reconstruct (and interpolate) the discount curves
				REAL rd  		 = ScenarioCurve ( t, expiry, discount, scenario_prior_index, static_factors, stoch_factors );
				REAL r  	 	 = ScenarioCurve ( t, expiry, equity_zero, scenario_prior_index, static_factors, stoch_factors );
				REAL q 			 = ScenarioDividendYield ( t, expiry, dividend_yield, scenario_prior_index, static_factors, stoch_factors );
				REAL spot        = ScenarioRate  ( t, equity, scenario_prior_index, static_factors, stoch_factors );
				
				REAL forward	 = 0.0;
				REAL normalize   = 0.0;
				
				for ( int si=starttime_index[mtm_time_index]; si < num_samples; si++ )
				{
					const REAL* sample_i	=  samples+si*RESET_INDEX_Size;
					
					REAL t_s				= calcDayCountAccrual ( sample_i[RESET_INDEX_End_Day] - t[TIME_GRID_MTM], equity_zero[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
					forward 				+= sample_i[RESET_INDEX_Weight] * spot * exp ( ( r - q ) * t_s );
					normalize				+= sample_i[RESET_INDEX_Weight];
				}
				
				REAL average     = AverageSample ( starttime_index[mtm_time_index], equity, samples, scenario_number * ScenarioTimeSteps, static_factors, stoch_factors );
				REAL strike_bar	 = strike - average ;
				REAL moneyness 	 = spot / ( strike_bar/normalize) ;
				
				REAL implied_vol = ScenarioSurface2D ( t, moneyness, expiry, equityvolsurface, scenario_prior_index, static_factors, stoch_factors );
				REAL vol 		= sqrt ( Calc_SingleCurrencyAssetVariance ( t, forward, r - q, implied_vol*implied_vol, spot, starttime_index[mtm_time_index], num_samples, samples, scenario_prior_index, static_factors, stoch_factors ) );
				//Tenor is set to 1.0 as vol has already been scaled using the correct tenors
				REAL black       = blackEuropeanOption ( forward, strike_bar, 0.0, vol, 1.0, buyorsell, callorput ) * exp ( -rd * expiry );
				/*
				if (threadIdx.x==0 && blockIdx.x==0)
				{
					printf("EquityAsianOption,mtm_time_index,%d,start_time,%d,average,%.6f,div_yield,%.6f,spot,%.6f,repo,%.6f,forward,%.4f,strike,%.6f,strike_bar,%.6f,expiry,%.6f,implied_vol,%.6f,vol,%.6f,moneyness,%.6f,black,%.6f\\n",mtm_time_index,starttime_index[mtm_time_index],average,q,spot,r,forward,strike,strike_bar,expiry,implied_vol,vol,moneyness,black);
				}
				*/
				//now calculate the MTM
				mtm 			= units * black * baseFX;
				
				//settle the currency_cashflows
				if (remainingMaturityinDays==0)
					ScenarioSettle(mtm/baseFX, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows);
			}
			
			Output [ index ] = mtm ;
		}		
	'''
	
	def __init__(self, params):
		super(EquityDiscreteExplicitAsianOption, self).__init__(params)

	def reset(self, calendars):
		super(EquityDiscreteExplicitAsianOption, self).reset()
		self.add_reval_dates ( set([self.field['Expiry_Date']]), self.field['Currency'] )
		
	def calc_dependancies ( self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars ):
		field 							= {}
		field['Currency'] 				= Utils.CheckRateName(self.field['Currency'])
		field['Equity'] 				= Utils.CheckRateName(self.field['Equity'])
		field['Discount_Rate']			= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Equity_Volatility']		= Utils.CheckRateName(self.field['Equity_Volatility'])

		field_index		= {}		
		field_index['Currency']				= getFXRateFactor 			( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 			= getDiscountFactor 		( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Equity']				= getEquityRateFactor 		( field['Equity'], static_offsets, stochastic_offsets )
		field_index['Equity_Zero']			= getEquityZeroRateFactor 	( field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Dividend_Yield']		= getDividendRateFactor		( field['Equity'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Equity_Volatility']	= getEquityPriceVolFactor 	( field['Equity_Volatility'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Expiry'] 				= (self.field['Expiry_Date'] - base_date).days
		
		#map the past fixings
		field_index['Samples']				= Utils.MakeSamplingData	( base_date, time_grid, self.field['Sampling_Data'] )
		field_index['StartIndex'] 			= field_index['Samples'].GetStartIndex(time_grid)
		return field_index
	
	def Generate ( self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid ):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)

		EquityDiscreteExplicitAsianOption	= module.get_function ('EquityDiscreteExplicitAsianOption')
		EquityDiscreteExplicitAsianOption ( CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
											precision ( field_index['Expiry'] ),
											precision ( self.field['Strike_Price'] ),
											precision ( 1.0 if self.field['Option_Type']=='Call' else -1.0 ),
											precision ( 1.0 if self.field['Buy_Sell']=='Buy' else -1.0 ),
											precision ( self.field['Units'] ),
											drv.In ( field_index['Currency'] ),
											drv.In ( field_index['Equity'] ),
											drv.In ( field_index['Equity_Zero'] ),
											drv.In ( field_index['Dividend_Yield'] ),
											drv.In ( field_index['Discount'] ),
											np.int32 ( field_index['Samples'].Count() ),
											drv.In ( field_index['StartIndex'] ),
											drv.In ( field_index['Samples'].GetSchedule(precision).ravel() ),
											drv.In ( field_index['Equity_Volatility'] ),
											block=block, grid=grid )

class EquityOptionDeal(Deal):
	factor_fields = {'Currency':				['FxRate'],
					 'Equity':					['EquityPrice','DividendRate'],
					 'Discount_Rate':			['DiscountRate'],
					 'Equity_Volatility':		['EquityPriceVol']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Expiry_Date':			'Expiry date of the option',
						'Units':				'Number of units of the option held',
						'Strike_Price':			'Strike price of the option in asset currency (or settlement currency for compo options)',
						'Payoff_Currency':		'ID of the FX rate price factor used to define the settlement currency. This property is required for deals with a quanto or compo Payoff_Type. For deals with a standard Payoff_Type, this property must be the same as Currency or left empty. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor. Quanto or Compo deals also require the FX volatility price factor of the FX rate given by the asset currency and this settlement currency and the Implied Correlations price factor between that FX rate and the underlying asset',
						'Equity':				'Underlying equity name. For example, IBM. This requires the Equity Price price factor and the Equity Dividend Rate price factor',
						'Equity_Volatility':	"ID of the equity volatility price factor used to value the deal. This property must be set to either 'Equity' or 'Equity.Curreny', where Equity is the underlying equity name and Currency is the ID of the currency of the equity. For example, 'IBM' or 'IBM.USD'. To use an equity volatility spread price factor, volatility spreads must be enabled and this property must use a tilde ('~') character and be set to 'Base~Spread', where 'Base' must be either 'Equity' or 'Equity.Currency' as described above and 'Spread' is the ID of the spread. For example, 'IBM~BARRIERKOCALL' or 'IBM.USD~BARRIERKOCALL'. If left blank, the underlying equity name will be used. This requires the equity price volatility price factor and, if volatility spreads are enabled and a tilde ('~') character is used, the equity price volatility spread price factor"
						}
						
	cudacodetemplate = '''
		//all pricing functions will have the same 5 initial parameters
		__global__ void EquityOptionDeal(	const REAL* __restrict__ stoch_factors,
											const REAL* __restrict__ static_factors,
											const int*  __restrict__ deal_time_index,
											const REAL* __restrict__ time_grid,
											const int*  __restrict__ cashflow_index,
											REAL* all_cashflows,
											REAL* Output,
											REAL maturity_in_days,
											REAL strike,
											REAL callorput,
											REAL buyorsell,
											REAL units,
											REAL isforward,
											const int* __restrict__ currency,
											const int* __restrict__ equity,
											const int* __restrict__ equity_zero,
											const int* __restrict__ dividend_yield,
											const int* __restrict__ discount,
											const int* __restrict__ equityvolsurface )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];			
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
			OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );

			//set the mtm of this deal to 0
			REAL mtm 					= 0.0;

			//work out the remaining term
			REAL remainingMaturityinDays = maturity_in_days - t[TIME_GRID_MTM];

			if (remainingMaturityinDays>= 0.0)
			{
				//assume the daycount is stored in the first factor 
				REAL expiry 				= calcDayCountAccrual(remainingMaturityinDays, discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
				REAL baseFX				    = ScenarioRate ( t, currency, scenario_prior_index, static_factors, stoch_factors ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
													
				//reconstruct (and interpolate) the discount curves
				REAL rd  		= ScenarioCurve ( t, expiry, discount, scenario_prior_index, static_factors, stoch_factors );
				REAL spot       = ScenarioRate  ( t, equity, scenario_prior_index, static_factors, stoch_factors );
				REAL moneyness 	= spot/strike;
				REAL forward	= Calc_EquityForward ( t, maturity_in_days, equity, equity_zero, dividend_yield, scenario_prior_index, static_factors, stoch_factors );
				REAL vol 		= ScenarioSurface2D ( t, moneyness, expiry, equityvolsurface, scenario_prior_index, static_factors, stoch_factors );

				//now calculate the MTM
				mtm 			= units * blackEuropeanOption ( forward, strike, rd, vol, expiry, buyorsell, callorput ) * baseFX;
				
				/*
				if (threadIdx.x==0 && blockIdx.x==0)
				{
					printf("Equity Option,div_yield,%.6f,repo,%.6f,spot,%.4f,expiry,%.6f,vol,%.6f,moneyness,%.6f,forward,%.6f\\n",q,r,spot,expiry,vol,moneyness,forward);
				}
				*/
				
				//settle the currency_cashflows
				if (remainingMaturityinDays==0)
					ScenarioSettle ( mtm/baseFX, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows );
			}
			
			Output [ index ] = mtm ;
		}		
	'''
	
	def __init__(self, params):
		super(EquityOptionDeal, self).__init__(params)

	def reset(self, calendars):
		super(EquityOptionDeal, self).reset()
		self.add_reval_dates ( set([self.field['Expiry_Date']]), self.field['Currency'] )
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 				= Utils.CheckRateName(self.field['Currency'])
		field['Equity'] 				= Utils.CheckRateName(self.field['Equity'])
		field['Discount_Rate']			= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Equity_Volatility']		= Utils.CheckRateName(self.field['Equity_Volatility'])

		field_index		= {}		
		field_index['Currency']				= getFXRateFactor 			( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 			= getDiscountFactor 		( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Equity']				= getEquityRateFactor 		( field['Equity'], static_offsets, stochastic_offsets )
		field_index['Equity_Zero']			= getEquityZeroRateFactor 	( field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Dividend_Yield']		= getDividendRateFactor		( field['Equity'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Equity_Volatility']	= getEquityPriceVolFactor 	( field['Equity_Volatility'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Expiry'] 				= (self.field['Expiry_Date'] - base_date).days
		
		return field_index
	
	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)

		EquityOptionDeal	= module.get_function ('EquityOptionDeal')
		EquityOptionDeal (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							precision ( field_index['Expiry'] ),
							precision ( self.field['Strike_Price'] ),
							precision ( 1.0 if self.field['Option_Type']=='Call' else -1.0 ),
							precision ( 1.0 if self.field['Buy_Sell']=='Buy' else -1.0 ),
							precision ( self.field['Units'] ),
							precision ( 1.0 if self.field['Option_On_Forward'] else 0.0 ),
							drv.In ( field_index['Currency'] ),
							drv.In ( field_index['Equity'] ),
							drv.In ( field_index['Equity_Zero'] ),
							drv.In ( field_index['Dividend_Yield'] ),
							drv.In ( field_index['Discount'] ),
							drv.In ( field_index['Equity_Volatility'] ),
							block=block, grid=grid )

class EquityBarrierOption(Deal):
	factor_fields = {'Currency':				['FxRate'],
					 'Equity':					['EquityPrice','DividendRate'],
					 'Discount_Rate':			['DiscountRate'],
					 'Equity_Volatility':		['EquityPriceVol']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Expiry_Date':			'Expiry date of the option',
						'Units':				'Number of units of the option held',
						'Strike_Price':			'Strike price of the option in asset currency (or settlement currency for compo options)',
						'Payoff_Currency':		'ID of the FX rate price factor used to define the settlement currency. This property is required for deals with a quanto or compo Payoff_Type. For deals with a standard Payoff_Type, this property must be the same as Currency or left empty. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor. Quanto or Compo deals also require the FX volatility price factor of the FX rate given by the asset currency and this settlement currency and the Implied Correlations price factor between that FX rate and the underlying asset',
						'Equity':				'Underlying equity name. For example, IBM. This requires the Equity Price price factor and the Equity Dividend Rate price factor',
						'Equity_Volatility':	"ID of the equity volatility price factor used to value the deal. This property must be set to either 'Equity' or 'Equity.Curreny', where Equity is the underlying equity name and Currency is the ID of the currency of the equity. For example, 'IBM' or 'IBM.USD'. To use an equity volatility spread price factor, volatility spreads must be enabled and this property must use a tilde ('~') character and be set to 'Base~Spread', where 'Base' must be either 'Equity' or 'Equity.Currency' as described above and 'Spread' is the ID of the spread. For example, 'IBM~BARRIERKOCALL' or 'IBM.USD~BARRIERKOCALL'. If left blank, the underlying equity name will be used. This requires the equity price volatility price factor and, if volatility spreads are enabled and a tilde ('~') character is used, the equity price volatility spread price factor"
						}
	
	cudacodetemplate = '''
		//all pricing functions will have the same 5 initial parameters
		__global__ void EquityBarrierOption(	const REAL* __restrict__ stoch_factors,
												const REAL* __restrict__ static_factors,
												const int*  __restrict__ deal_time_index,
												const REAL* __restrict__ time_grid,
												const int*  __restrict__ cashflow_index,
												REAL* all_cashflows,
												REAL* Output,
												int   DealTimeGridSize,
												REAL maturity_in_days,
												REAL strike,
												REAL phi, //call or put
												REAL buyorsell,
												REAL nominal,
												REAL barrier,
												REAL cash_rebate,
												REAL eta, //up or down
												REAL direction, //in or out
												REAL barrier_monitoring, //handle discrete monitoring
												const int* __restrict__ currency,
												const int* __restrict__ equity,
												const int* __restrict__ equity_zero,
												const int* __restrict__ dividend_yield,
												const int* __restrict__ discount,
												const int* __restrict__ equityvolsurface )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			
			//work out what the barrier payoff is
			int  barrier_payoff_type    = barrierPayoffType ( direction, eta, phi, strike, barrier );
			
			//state for path dependency
			REAL  previous_spot			= 0.0;
			int   barrier_touched 		= 0;

			for (int mtm_time=0; mtm_time<DealTimeGridSize; mtm_time++)
			{
				int mtm_time_index 			= deal_time_index[mtm_time];
				const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
				int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
				OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );

				//set the mtm of this deal to 0
				REAL mtm 					= 0.0;
			
				//work out the remaining term
				REAL remainingMaturityinDays = maturity_in_days - t[TIME_GRID_MTM];				

				if (remainingMaturityinDays>= 0.0)
				{
					//assume the daycount is stored in the first factor 
					REAL expiry 				= calcDayCountAccrual(remainingMaturityinDays, discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
					REAL baseFX				    = ScenarioRate ( t, currency, scenario_prior_index, static_factors, stoch_factors );					

					//reconstruct (and interpolate) the discount curves
					REAL r 			 = ScenarioCurve ( t, expiry, discount, scenario_prior_index, static_factors, stoch_factors );
					REAL spot        = ScenarioRate  ( t, equity, scenario_prior_index, static_factors, stoch_factors );
					REAL moneyness 	 = spot/strike;
					REAL b           = ScenarioCurve ( t, expiry, equity_zero, scenario_prior_index, static_factors, stoch_factors ) -
									   ScenarioDividendYield ( t, expiry, dividend_yield, scenario_prior_index, static_factors, stoch_factors );
					
					//option pricing variables
					REAL payoff 	 = 0.0;
					REAL sigma 		 = ScenarioSurface2D ( t, moneyness, expiry, equityvolsurface, scenario_prior_index, static_factors, stoch_factors );
					REAL adj_barrier = barrier * exp ( ( ( barrier>spot ) ? 1.0 : -1.0 ) * sigma * barrier_monitoring ); 
					
					if ( !barrier_touched )
					{						
						if ( ( eta==BARRIER_UP && previous_spot<barrier && spot>barrier ) ||
							 ( eta==BARRIER_DOWN && previous_spot>barrier && spot<barrier ) )
						{
							barrier_touched = 1;
							if ( direction==BARRIER_OUT )
							{
								payoff			= buyorsell * cash_rebate ;
								if (cash_rebate!=0.0)
									ScenarioSettle ( payoff, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows );
							}
							else
							{
								payoff			= nominal * blackEuropeanOption ( spot * exp ( b * expiry ), strike, r, sigma, expiry, buyorsell, phi );
							}
							mtm 			= ( payoff * baseFX ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
						}
						else
						if (remainingMaturityinDays==0)
						{
							payoff			= buyorsell * ( ( direction==BARRIER_IN ) ? cash_rebate : nominal * max ( phi * (spot - strike), 0.0 ) );
							mtm 			= ( payoff * baseFX ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
							ScenarioSettle (  payoff, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows );
						}
						else
						{
							payoff          = barrierOption ( barrier_payoff_type, eta, phi, sigma, expiry, cash_rebate/nominal, b, r, spot, strike, adj_barrier );
							//now calculate the MTM
							mtm 			= ( nominal * buyorsell * payoff * baseFX ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
						}
					}
					else
					if ( direction==BARRIER_IN )
					{
						if (remainingMaturityinDays>0)
						{
							payoff			= nominal * blackEuropeanOption ( spot * exp ( b * expiry ), strike, r, sigma, expiry, buyorsell, phi ) ;
						}
						else
						{
							payoff			= nominal * buyorsell * max ( phi * (spot - strike), 0.0 ) ;
							ScenarioSettle (  payoff, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows );
						}
						mtm = ( payoff * baseFX ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
					}

					/*
					if (threadIdx.x==0 && blockIdx.x==0)
					{
						printf("barrier_payoff_type,%d,volume,%.4f,spot,%.4f,moneyness,%.5f,barrier,%.5f,expiry,%.6f,sigma,%.8f,b,%.8f,baseFX,%.4f,payoff,%.8f,mtm,%.4f\\n",barrier_payoff_type,nominal,spot,moneyness,barrier,expiry,sigma,b,baseFX,payoff,mtm);
					}
					*/

					//store the spot price
					previous_spot = spot;
				}
				
				Output [ index ] = mtm ;
			}
		}
	'''

	def __init__(self, params):
		super(EquityBarrierOption, self).__init__(params)
		self.path_dependent = True

	def reset(self, calendars):
		super(EquityBarrierOption, self).reset()
		self.add_reval_dates ( set([self.field['Expiry_Date']]), self.field['Payoff_Currency'] )

	def add_grid_dates(self, parser, base_date, grid):
		#a cash rebate is paid on touch if the option knocks out
		if self.field['Cash_Rebate']:
			grid_dates = parser ( base_date, self.field['Expiry_Date'], grid )
			self.reval_dates.update ( grid_dates )
			self.settlement_currencies.setdefault ( self.field[ self.field['Payoff_Currency'] ], set() ).update(grid_dates)
					   
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 				= Utils.CheckRateName(self.field['Currency'])
		field['Equity'] 				= Utils.CheckRateName(self.field['Equity'])
		field['Discount_Rate']			= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Equity_Volatility']		= Utils.CheckRateName(self.field['Equity_Volatility'])

		field_index		= {}		
		field_index['Currency']				= getFXRateFactor 			( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 			= getDiscountFactor 		( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Equity']				= getEquityRateFactor 		( field['Equity'], static_offsets, stochastic_offsets )
		field_index['Equity_Zero']			= getEquityZeroRateFactor 	( field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Dividend_Yield']		= getDividendRateFactor		( field['Equity'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Equity_Volatility']	= getEquityPriceVolFactor 	( field['Equity_Volatility'], static_offsets, stochastic_offsets, all_tenors )
		
		#discrete barrier monitoring requires adjusting the barrier by 0.58 ( -(scipy.special.zetac(0.5)+1)/np.sqrt(2.0*np.pi) ) * sqrt (monitoring freq)
		field_index['Barrier_Monitoring']   = 0.5826 * np.sqrt( (base_date+self.field['Barrier_Monitoring_Frequency']-base_date).days/365.0 )
		field_index['Expiry'] 				= (self.field['Expiry_Date'] - base_date).days

		return field_index
	
	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, 1)

		EquityBarrierOption	= module.get_function ('EquityBarrierOption')
		EquityBarrierOption (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
								np.int32(time_grid.deal_time_grid.size),
								precision ( field_index['Expiry'] ),
								precision ( self.field['Strike_Price'] ),
								precision ( 1.0 if self.field['Option_Type']=='Call' else -1.0 ),
								precision ( 1.0 if self.field['Buy_Sell']=='Buy' else -1.0 ),
								precision ( self.field['Units'] ),
								precision ( self.field['Barrier_Price'] ),
								precision ( self.field['Cash_Rebate'] ),
								precision ( 1.0 if 'Down' in self.field['Barrier_Type'] else -1.0 ),
								precision ( 1.0 if 'Out' in self.field['Barrier_Type'] else -1.0 ),
								precision ( field_index['Barrier_Monitoring'] ),
								drv.In ( field_index['Currency'] ),
								drv.In ( field_index['Equity'] ),
								drv.In ( field_index['Equity_Zero'] ),
								drv.In ( field_index['Dividend_Yield'] ),
								drv.In ( field_index['Discount'] ),
								drv.In ( field_index['Equity_Volatility'] ),
								block=block, grid=grid )

class EquityForwardDeal(Deal):
	factor_fields = {'Currency':				['FxRate'],
					 'Equity':					['EquityPrice','DividendRate'],
					 'Discount_Rate':			['DiscountRate']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Maturity_Date':		'Expiry date of the option',
						'Units':				'Number of units of the option held',
						'Forward_Price':		'Strike price of the option in asset currency (or settlement currency for compo options)',
						'Payoff_Currency':		'ID of the FX rate price factor used to define the settlement currency. This property is required for deals with a quanto or compo Payoff_Type. For deals with a standard Payoff_Type, this property must be the same as Currency or left empty. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor. Quanto or Compo deals also require the FX volatility price factor of the FX rate given by the asset currency and this settlement currency and the Implied Correlations price factor between that FX rate and the underlying asset',
						'Equity':				'Underlying equity name. For example, IBM. This requires the Equity Price price factor and the Equity Dividend Rate price factor'
						}
						
	cudacodetemplate = '''
		//all pricing functions will have the same 5 initial parameters
		__global__ void EquityForwardDeal(	const REAL* __restrict__ stoch_factors,
											const REAL* __restrict__ static_factors,
											const int*  __restrict__ deal_time_index,
											const REAL* __restrict__ time_grid,
											const int*  __restrict__ cashflow_index,
											REAL* all_cashflows,
											REAL* Output,
											REAL maturity_in_days,
											REAL strike,
											REAL buyorsell,
											REAL units,
											const int* __restrict__ currency,
											const int* __restrict__ equity,
											const int* __restrict__ equity_zero,
											const int* __restrict__ dividend_yield,
											const int* __restrict__ discount )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];			
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
			OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );

			//set the mtm of this deal to 0
			REAL mtm 					= 0.0;

			if ( maturity_in_days  >= t[TIME_GRID_MTM] )
			{
				REAL baseFX		= ScenarioRate ( t, currency, scenario_prior_index, static_factors, stoch_factors ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
				REAL forward	= Calc_EquityForward ( t, maturity_in_days, equity, equity_zero, dividend_yield, scenario_prior_index, static_factors, stoch_factors );
				REAL DtT  		= CalcDiscount ( t, maturity_in_days, discount, scenario_prior_index, static_factors, stoch_factors );
				
				//now calculate the MTM
				mtm 			= units * buyorsell * ( forward - strike ) * DtT * baseFX;
				
				/*
				if (threadIdx.x==0 && blockIdx.x==0)
				{
					printf("Equity Option,div_yield,%.6f,repo,%.6f,spot,%.4f,expiry,%.6f,vol,%.6f,moneyness,%.6f,forward,%.6f\\n",q,r,spot,expiry,vol,moneyness,forward);
				}
				*/
				
				//settle the currency_cashflows
				if ( maturity_in_days == t[TIME_GRID_MTM] )
					ScenarioSettle ( mtm/baseFX, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows );
			}
			
			Output [ index ] = mtm ;
		}
	'''
	
	def __init__(self, params):
		super(EquityForwardDeal, self).__init__(params)

	def reset(self, calendars):
		super(EquityForwardDeal, self).reset()
		self.add_reval_dates ( set([self.field['Maturity_Date']]), self.field['Currency'] )
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 				= Utils.CheckRateName(self.field['Currency'])
		field['Equity'] 				= Utils.CheckRateName(self.field['Equity'])
		field['Discount_Rate']			= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		#field['Equity_Volatility']		= Utils.CheckRateName(self.field['Equity_Volatility'])

		field_index		= {}		
		field_index['Currency']				= getFXRateFactor 			( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 			= getDiscountFactor 		( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Equity']				= getEquityRateFactor 		( field['Equity'], static_offsets, stochastic_offsets )
		field_index['Equity_Zero']			= getEquityZeroRateFactor 	( field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Dividend_Yield']		= getDividendRateFactor		( field['Equity'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Expiry'] 				= (self.field['Maturity_Date'] - base_date).days
		
		return field_index
	
	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)

		EquityForwardDeal	= module.get_function ('EquityForwardDeal')
		EquityForwardDeal (  	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
								precision ( field_index['Expiry'] ),
								precision ( self.field['Forward_Price'] ),
								precision ( 1.0 if self.field['Buy_Sell']=='Buy' else -1.0 ),
								precision ( self.field['Units'] ),
								drv.In ( field_index['Currency'] ),
								drv.In ( field_index['Equity'] ),
								drv.In ( field_index['Equity_Zero'] ),
								drv.In ( field_index['Dividend_Yield'] ),
								drv.In ( field_index['Discount'] ),
								block=block, grid=grid )

class EquitySwapletListDeal(Deal):
	#dependent price factors for this instrument
	factor_fields = {'Currency':				['FxRate'],
					 'Equity_Currency':			['FxRate'],
					 'Discount_Rate':			['DiscountRate'],
					 'Equity':					['EquityPrice','DividendRate'],
					 'Equity_Volatility':		['EquityPriceVol']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Equity':				'Underlying equity name. For example, IBM. This requires the Equity Price price factor and the Equity Dividend Rate price factor',
						'Equity_Volatility':	"ID of the equity volatility price factor used to value the deal. This property must be set to either 'Equity' or 'Equity.Curreny', where Equity is the underlying equity name and Currency is the ID of the currency of the equity. For example, 'IBM' or 'IBM.USD'. To use an equity volatility spread price factor, volatility spreads must be enabled and this property must use a tilde ('~') character and be set to 'Base~Spread', where 'Base' must be either 'Equity' or 'Equity.Currency' as described above and 'Spread' is the ID of the spread. For example, 'IBM~BARRIERKOCALL' or 'IBM.USD~BARRIERKOCALL'. If left blank, the underlying equity name will be used. This requires the equity price volatility price factor and, if volatility spreads are enabled and a tilde ('~') character is used, the equity price volatility spread price factor"
						}
	
	#cuda code required to price this instrument - none - as it's in addfloatcashflow
	cudacodetemplate = ''
	
	def __init__(self, params):
		super(EquitySwapletListDeal, self).__init__(params)

	def reset(self, calendars):
		super(EquitySwapletListDeal, self).reset()
		self.paydates	= set ( [x['Payment_Date'] for x in self.field['Cashflows']['Items']] )
		self.add_reval_dates ( self.paydates, self.field['Currency'] )
		#this swap could be quantoed
		self.isQuanto 		= None	
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 			= Utils.CheckRateName(self.field['Currency'])
		field['Equity'] 			= Utils.CheckRateName(self.field['Equity'])
		field['Discount_Rate']		= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Equity_Currency'] 	= Utils.CheckRateName(self.field['Equity_Currency'])
		#field['Equity_Volatility']	= Utils.CheckRateName(self.field['Equity_Volatility'])

		field_index		= {}		
		self.isQuanto 	= field['Equity_Currency'] != field['Currency']
		if self.isQuanto:
			#TODO - Deal with Quanto Interest Rate swaps
			pass
		else:
			field_index['Currency']				= getFXRateFactor 			( field['Currency'], static_offsets, stochastic_offsets )
			field_index['Discount'] 			= getDiscountFactor 		( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
			field_index['Equity']				= getEquityDividentFactors	( field['Equity'], static_offsets, stochastic_offsets, all_tenors)
			field_index['Equity_Zero']			= getEquityZeroRateFactor 	( field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors )
			#field_index['Equity_Volatility']	= getEquityPriceVolFactor 	( field['Equity_Volatility'], static_offsets, stochastic_offsets, all_tenors )
			
			field_index['Cashflows']    		= Utils.MakeEquitySwapletCashflows ( base_date, time_grid, 1 if self.field['Buy_Sell']=='Buy' else -1, self.field['Cashflows'] )
										 
		field_index['StartIndex'] 		= field_index['Cashflows'].GetCashflowStartIndex(time_grid)
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)
		
		AddFloatCashflow 	= module.get_function ('AddFloatCashflow')
		AddFloatCashflow( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32 ( Utils.SCENARIO_CASHFLOWS_Equity ),
							np.int32 ( Utils.CASHFLOW_METHOD_Average_Interest ),
							np.int32 ( 1 ),
							np.int32  ( field_index['Cashflows'].Count() ),
							drv.In    ( field_index['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['StartIndex'] ),
							drv.In 	  ( field_index['Cashflows'].Resets.GetSchedule(precision).ravel() ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['Equity'] ),
							drv.In    ( field_index['Equity_Zero'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )

class EquitySwapLeg(Deal):
	#dependent price factors for this instrument
	factor_fields = {'Currency':				['FxRate'],
					 'Equity_Currency':			['FxRate'],
					 'Discount_Rate':			['DiscountRate'],
					 'Equity':					['EquityPrice','DividendRate']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Equity':				'Underlying equity name. For example, IBM. This requires the Equity Price price factor and the Equity Dividend Rate price factor'
						}
	
	#cuda code required to price this instrument - none - as it's in addfloatcashflow
	cudacodetemplate = ''
	
	def __init__(self, params):
		super(EquitySwapLeg, self).__init__(params)

	def reset(self, calendars):
		super(EquitySwapLeg, self).reset()
		self.bus_pay_day = calendars.get(self.field.get('Payment_Calendars', self.field['Accrual_Calendars']), {'businessday':pd.offsets.BDay(1)})['businessday']
		paydates	 = set ( [self.field['Maturity_Date']+self.bus_pay_day*self.field['Payment_Offset']] )
		self.add_reval_dates ( paydates, self.field['Currency'] )
		#this swap could be quantoed
		self.isQuanto 		= None	
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 				= Utils.CheckRateName(self.field['Currency'])
		field['Equity'] 				= Utils.CheckRateName(self.field['Equity'])
		field['Discount_Rate']			= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Payoff_Currency'] 		= Utils.CheckRateName(self.field['Payoff_Currency'])
		
		#Implicitly we assume that units is the number of shares (Principle is 0) and that Dividend Timing is "Terminal" - need to add support for dividends..
		start_prices					= self.field['Equity_Known_Prices'].data.get(self.field['Effective_Date'], (None, None)) if self.field['Equity_Known_Prices'] else (None, None)
		end_prices						= self.field['Equity_Known_Prices'].data.get(self.field['Maturity_Date'], (None, None)) if self.field['Equity_Known_Prices'] else (None, None)
		start_dividend_sum				= self.field['Known_Dividends'].SumRange (base_date, self.field['Effective_Date'], 0) if self.field['Known_Dividends'] else 0.0
		
		#sometimes the equity price is listed but in the past - need to check for this
		if self.field['Equity_Known_Prices'] and start_prices == (None, None):
			earlier_dates = [x for x in self.field['Equity_Known_Prices'].data.keys() if x<self.field['Effective_Date']]
			start_prices = self.field['Equity_Known_Prices'].data [ max(earlier_dates) ] if earlier_dates else (None, None)
		
		field['cashflow']				= 	{'Items': [
												 {'Payment_Date': 			self.field['Maturity_Date']+self.bus_pay_day*self.field['Payment_Offset'],
												   'Start_Date': 			self.field['Effective_Date'],
												   'End_Date': 				self.field['Maturity_Date'],
												   'Start_Multiplier':		1.0,
												   'End_Multiplier':		1.0,
												   'Known_Start_Price':		start_prices[0],
												   'Known_StartFX_Rate':	start_prices[1],
												   'Known_End_Price':		end_prices[0],
												   'Known_EndFX_Rate':		end_prices[1],
												   'Known_Dividend_Sum':	start_dividend_sum,
												   'Dividend_Multiplier':	1.0 if self.field['Include_Dividends']=='Yes' else 0.0,
												   'Amount':   				self.field['Units'] if self.field['Principal_Fixed_Variable']=='Variable' else self.field['Principal']
												   }
												 ]
											}

		field_index		= {}		
		self.isQuanto 	= field['Payoff_Currency'] != field['Currency']
		if self.isQuanto:
			#TODO - Deal with Quanto Equity Swaps
			raise Exception("EquitySwapLeg Compo deal - TODO")
		else:
			field_index['Currency']				= getFXRateFactor 			( field['Currency'], static_offsets, stochastic_offsets )
			field_index['Discount'] 			= getDiscountFactor 		( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
			field_index['Equity']				= getEquityDividentFactors	( field['Equity'], static_offsets, stochastic_offsets, all_tenors )
			field_index['Equity_Zero']			= getEquityZeroRateFactor 	( field['Equity'], static_offsets, stochastic_offsets, all_tenors, all_factors )
			
			field_index['Cashflows']    		= Utils.MakeEquitySwapletCashflows ( base_date, time_grid, 1 if self.field['Buy_Sell']=='Buy' else -1, field['cashflow'] )
										 
		field_index['StartIndex'] 		= field_index['Cashflows'].GetCashflowStartIndex(time_grid)
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)
		
		AddFloatCashflow 	= module.get_function ('AddFloatCashflow')
		AddFloatCashflow( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32 ( Utils.SCENARIO_CASHFLOWS_Equity ),
							np.int32 ( Utils.CASHFLOW_METHOD_Equity_Shares ),
							np.int32 ( 1 ),
							np.int32  ( field_index['Cashflows'].Count() ),
							drv.In    ( field_index['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['StartIndex'] ),
							drv.In 	  ( field_index['Cashflows'].Resets.GetSchedule(precision).ravel() ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['Equity'] ),
							drv.In    ( field_index['Equity_Zero'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )
		
class FXOneTouchOption(Deal):
	factor_fields = {'Currency':				['FxRate'],
					 'Underlying_Currency':		['FxRate'],
					 'Discount_Rate':			['DiscountRate'],
					 'FX_Volatility':			['FXVol']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Expiry_Date':			'Expiry date of the option',
						'Underlying_Currency':	'ID of the FX rate price factor used to define the foreign currency name, for example, USD. The underlying exchange rate asset used by the deal is the amount of Currency per unit of Underlying_Currency. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'FX_Volatility':		"ID of the FX volatility price factor used to value the deal. This must be set to either 'UnderlyingCurrency' or 'Currency1.Curreny2', where UnderlyingCurrency is the foreign currency name and Currency1 and Currency2 are the IDs of the domestic and foreign currencies in alphabetical order. For example, 'EUR' or 'EUR.USD'. To use an FX volatility spread price factor, volatility spreads must be enabled and this property must use a tilde ('~') character and be set to 'Base~Spread', where 'Base' must be either 'UnderlyingCurrency' or 'Currency1.Currency2' as described above and 'Spread' is the ID of the spread. If left blank, the foreign currency name will be used. For example, 'EUR~BARRIERKOCALL' or 'EUR.USD~BARRIERKOCALL'. This requires the FX volatility price factor and, if volatility spreads are enabled and a tilde ('~') character is used, the FX volatility spread price factor",
						'Cash_Payoff':			'Payoff amount entered in the settlement currency',
						'Barrier_Price':		'Barrier price entered in the form of number of units of domestic currency per unit of foreign currency',
						'Payment_Timing':		'Timing of the payment of the option. Payment when the barrier is touched (Touch) or at the option expiry date (Expiry)'
						}
	
	cudacodetemplate = '''
		//all pricing functions will have the same 5 initial parameters
		__global__ void FXOneTouchOption(	const REAL* __restrict__ stoch_factors,
											const REAL* __restrict__ static_factors,
											const int*  __restrict__ deal_time_index,
											const REAL* __restrict__ time_grid,
											const int*  __restrict__ cashflow_index,
											REAL* all_cashflows,
											REAL* Output,
											int   DealTimeGridSize,
											REAL maturity_in_days,
											REAL buyorsell,
											REAL nominal,
											REAL barrier,
											REAL eta,
											int  invert_moneyness,
											int  payment_timing,
											const int* __restrict__ currency,
											const int* __restrict__ underlying_currency,
											const int* __restrict__ discount,
											const int* __restrict__ fxvolsurface )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			
			//get the currency repo curves
			const int* curr_curve 		= CurrencyRepoCurve ( currency );
			const int* und_curr_curve 	= CurrencyRepoCurve ( underlying_currency );

			//state for path dependency
			REAL  previous_spot			= 0.0;
			int   barrier_touched 		= 0;
			
			for ( int mtm_time=0; mtm_time<DealTimeGridSize; mtm_time++ )
			{
				int mtm_time_index 			= deal_time_index[mtm_time];
				const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
				int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
				OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
				
				//set the mtm of this deal to 0
				REAL mtm 					= 0.0;

				//work out the remaining term
				REAL remainingMaturityinDays = maturity_in_days - t[TIME_GRID_MTM];

				if (remainingMaturityinDays>= 0.0)
				{
					//assume the daycount is stored in the first factor 
					REAL expiry 				= calcDayCountAccrual(remainingMaturityinDays, discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
					REAL baseFX				    = ScenarioRate ( t, currency, scenario_prior_index, static_factors, stoch_factors ) ;					

					//reconstruct (and interpolate) the discount curves
					REAL r 			= ScenarioCurve ( t, expiry, discount, scenario_prior_index, static_factors, stoch_factors );
					REAL rd 		= ScenarioCurve ( t, expiry, curr_curve, scenario_prior_index, static_factors, stoch_factors );
					REAL rf 		= ScenarioCurve ( t, expiry, und_curr_curve, scenario_prior_index, static_factors, stoch_factors );
					REAL spot       = ScenarioRate  ( t, underlying_currency, scenario_prior_index, static_factors, stoch_factors ) / baseFX;
					
					if ( !barrier_touched )
					{
						if ( (eta==BARRIER_UP && previous_spot<barrier && spot>barrier) ||
							 (eta==BARRIER_DOWN && previous_spot>barrier && spot<barrier) )
						{
							barrier_touched	= 1;
							if (payment_timing==PAYMENT_TOUCH)
							{
								mtm 			= ( nominal * buyorsell * baseFX ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
								ScenarioSettle ( nominal * buyorsell, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows);
							}
							/*
							if (threadIdx.x==0 && blockIdx.x==0)
							{
								printf("BARRIER CROSSED,barrier,%.4f,previous_spot,%.4f,spot,%.4f,expiry,%.6f\\n",barrier,previous_spot,spot,expiry);
							}
							*/
						}
						else
						{
							//price the option
							REAL moneyness 	= invert_moneyness ? ( barrier/spot ) : ( spot/barrier );
							REAL vol 		= ScenarioSurface2D ( t, moneyness, expiry, fxvolsurface, scenario_prior_index, static_factors, stoch_factors );
							
							REAL root_tau   = sqrt(expiry);
							REAL mu 		= (rd-rf) / vol - 0.5 * vol;
							REAL log_vol	= log(barrier/spot)/vol;
							REAL barrovert  = log_vol/root_tau;
							
							REAL payoff 	= 0.0;

							if (payment_timing==PAYMENT_EXPIRY)
							{
								REAL muroot     = mu*root_tau;
								REAL d1 		= muroot - barrovert;
								REAL d2 		= -muroot - barrovert;
								
								payoff = exp(-r*expiry) * ( CND ( -eta * d1 ) + exp ( 2.0 * mu * log_vol ) * CND ( -eta * d2 ) );
							}
							else
							{
								REAL lambda 	= sqrt ( mu*mu + (2.0*r) );
								REAL lambdaroot = lambda*root_tau;
								REAL d1 		= lambdaroot - barrovert;
								REAL d2 		= -lambdaroot - barrovert;
							
								payoff 			= exp( (mu - lambda) * log_vol ) * CND ( -eta * d1 ) + exp( (mu + lambda) * log_vol ) * CND ( -eta * d2 );
								/*
								if (threadIdx.x==0 && blockIdx.x==0)
								{
									printf("volume,%.4f,spot,%.4f,moneyness,%.5f,barrier,%.5f,eta,%.2f,expiry,%.6f,b,%.8f,barrovert,%.8f,lambda,%.8f,mu,%.8f,vol,%.8f,d1,%.8f,d2,%.8f,baseFX,%.4f,payoff,%.8f,mtm,%.4f\\n",nominal,spot,moneyness,barrier,eta,expiry,rd-rf,barrovert,lambda,mu,vol,d1,d2,baseFX,payoff,mtm);
								}
								*/
							}
							
							//now calculate the MTM
							mtm 			= ( nominal * buyorsell * payoff * baseFX ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
						}
					}
					else
					if ( payment_timing==PAYMENT_EXPIRY )
					{
						//calculate the MTM
						mtm 			= ( nominal * exp(-r*expiry) * buyorsell * baseFX ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
						
						//note also that Im implicitly assuming that the payoff currency is the currency (if not, then rebook the deal sothat the currency is swapped with the underlying)
						if (remainingMaturityinDays==0)
							ScenarioSettle ( nominal * buyorsell, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows);
					}
					
					//store the spot price
					previous_spot = spot;
				}
				
				//store the mtm
				Output [ index ] = mtm ;
			}
		}
	'''

	def __init__(self, params):
		super(FXOneTouchOption, self).__init__(params)
		self.path_dependent = True

	def reset(self, calendars):
		super(FXOneTouchOption, self).reset()
		self.add_reval_dates ( set([self.field['Expiry_Date']]), self.field[ self.field['Payoff_Currency'] ] )
		
	def add_grid_dates(self, parser, base_date, grid):
		#only if the payoff is american (Touch) should we add potential payoff dates
		if self.field['Payment_Timing']=='Touch':
			grid_dates = parser ( base_date, self.field['Expiry_Date'], grid )
			self.reval_dates.update ( grid_dates )
			self.settlement_currencies.setdefault ( self.field[ self.field['Payoff_Currency'] ], set() ).update(grid_dates)
	
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field 							= {}
		field['Currency'] 				= Utils.CheckRateName(self.field['Currency'])
		field['Underlying_Currency'] 	= Utils.CheckRateName(self.field['Underlying_Currency'])
		field['Discount_Rate']			= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['FX_Volatility']			= Utils.CheckRateName(self.field['FX_Volatility'])

		field_index							= {}		
		field_index['Currency']				= getFXRateFactor 	( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 			= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Underlying_Currency']	= getFXRateFactor 	( field['Underlying_Currency'], static_offsets, stochastic_offsets )
		field_index['FX_Volatility'] 		= getFXVolFactor 	( field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Expiry'] 				= (self.field['Expiry_Date'] - base_date).days
		field_index['Invert_Moneyness']		= 1 if field['Currency'][0]==field['FX_Volatility'][0] else 0
		
		return field_index
	
	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, 1)

		FXOneTouchOption	= module.get_function ('FXOneTouchOption')
		FXOneTouchOption (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32(time_grid.deal_time_grid.size),
							precision ( field_index['Expiry'] ),
							precision ( 1.0 if self.field['Buy_Sell']=='Buy' else -1.0 ),
							precision ( self.field['Cash_Payoff'] ),
							precision ( self.field['Barrier_Price'] ),
							precision ( 1.0 if self.field['Barrier_Type']=='Down' else -1.0 ),
							np.int32 ( field_index['Invert_Moneyness'] ),
							np.int32 ( 0 if self.field['Payment_Timing']=='Touch' else 1 ),
							drv.In ( field_index['Currency'] ),
							drv.In ( field_index['Underlying_Currency'] ),
							drv.In ( field_index['Discount'] ),
							drv.In ( field_index['FX_Volatility'] ),
							block=block, grid=grid )

class FXBarrierOption(Deal):
	factor_fields = {'Currency':				['FxRate'],
					 'Underlying_Currency':		['FxRate'],
					 'Discount_Rate':			['DiscountRate'],
					 'FX_Volatility':			['FXVol']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Expiry_Date':			'Expiry date of the option',
						'Underlying_Amount':	'Number of units of the option held',
						'Strike_Price':			'Option exercise price, entered in the form of number of units of domestic currency per unit of foreign currency',
						'Barrier_Price':		'Barrier price entered in the form of number of units of domestic currency per unit of foreign currency',
						'Cash_Rebate':			'Rebate amount entered in the settlement currency. If the option is knocked out, it is paid at touch. If the option is never knocked in, it is paid at expiry',
						'Payoff_Currency':		'Settlement currency of the option',
						'Underlying_Currency':	'ID of the FX rate price factor used to define the foreign currency name, for example, USD. The underlying exchange rate asset used by the deal is the amount of Currency per unit of Underlying_Currency. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'FX_Volatility':		"ID of the FX volatility price factor used to value the deal. This must be set to either 'UnderlyingCurrency' or 'Currency1.Curreny2', where UnderlyingCurrency is the foreign currency name and Currency1 and Currency2 are the IDs of the domestic and foreign currencies in alphabetical order. For example, 'EUR' or 'EUR.USD'. To use an FX volatility spread price factor, volatility spreads must be enabled and this property must use a tilde ('~') character and be set to 'Base~Spread', where 'Base' must be either 'UnderlyingCurrency' or 'Currency1.Currency2' as described above and 'Spread' is the ID of the spread. If left blank, the foreign currency name will be used. For example, 'EUR~BARRIERKOCALL' or 'EUR.USD~BARRIERKOCALL'. This requires the FX volatility price factor and, if volatility spreads are enabled and a tilde ('~') character is used, the FX volatility spread price factor"
						}
	
	cudacodetemplate = '''
		//all pricing functions will have the same 5 initial parameters
		__global__ void FXBarrierOption(	const REAL* __restrict__ stoch_factors,
											const REAL* __restrict__ static_factors,
											const int*  __restrict__ deal_time_index,
											const REAL* __restrict__ time_grid,
											const int*  __restrict__ cashflow_index,
											REAL* all_cashflows,
											REAL* Output,
											int   DealTimeGridSize,
											REAL maturity_in_days,
											REAL strike,
											REAL phi, //call or put
											REAL buyorsell,
											REAL nominal,
											REAL barrier,
											REAL cash_rebate,
											REAL eta, //up or down
											REAL direction, //in or out
											REAL barrier_monitoring, //handle discrete monitoring
											int  invert_moneyness,
											int  settlement_currency,
											const int* __restrict__ currency,
											const int* __restrict__ underlying_currency,
											const int* __restrict__ discount,
											const int* __restrict__ fxvolsurface )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			
			//get the currency repo curves
			const int* curr_curve 		= CurrencyRepoCurve ( currency );
			const int* und_curr_curve 	= CurrencyRepoCurve ( underlying_currency );
					
			//work out what the barrier payoff is
			int  barrier_payoff_type    = barrierPayoffType ( direction, eta, phi, strike, barrier );
			
			//state for path dependency
			REAL  previous_spot			= 0.0;
			int   barrier_touched 		= 0;

			for (int mtm_time=0; mtm_time<DealTimeGridSize; mtm_time++)
			{
				int mtm_time_index 			= deal_time_index[mtm_time];
				const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
				int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
				OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );

				//set the mtm of this deal to 0
				REAL mtm 					= 0.0;
			
				//work out the remaining term
				REAL remainingMaturityinDays = maturity_in_days - t[TIME_GRID_MTM];				

				if (remainingMaturityinDays>= 0.0)
				{
					//assume the daycount is stored in the first factor 
					REAL expiry 				= calcDayCountAccrual(remainingMaturityinDays, discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
					REAL baseFX				    = ScenarioRate ( t, currency, scenario_prior_index, static_factors, stoch_factors );					

					//reconstruct (and interpolate) the discount curves
					REAL r 			 = ScenarioCurve ( t, expiry, discount, scenario_prior_index, static_factors, stoch_factors );
					REAL b           = ScenarioCurve ( t, expiry, curr_curve, scenario_prior_index, static_factors, stoch_factors ) -
									   ScenarioCurve ( t, expiry, und_curr_curve, scenario_prior_index, static_factors, stoch_factors );
					REAL spot        = ScenarioRate  ( t, underlying_currency, scenario_prior_index, static_factors, stoch_factors ) / baseFX;
					
					//option pricing variables
					REAL payoff 	 = 0.0;
					REAL moneyness 	 = invert_moneyness ? ( strike/spot ) : ( spot/strike );
					REAL sigma 		 = ScenarioSurface2D ( t, moneyness, expiry, fxvolsurface, scenario_prior_index, static_factors, stoch_factors );
					REAL adj_barrier = barrier * exp ( ( ( barrier>spot ) ? 1.0 : -1.0 ) * sigma * barrier_monitoring ); 
					
					if ( !barrier_touched )
					{						
						if ( ( eta==BARRIER_UP && previous_spot<barrier && spot>barrier ) ||
							 ( eta==BARRIER_DOWN && previous_spot>barrier && spot<barrier ) )
						{
							barrier_touched = 1;
							if ( direction==BARRIER_OUT )
							{
								payoff			= buyorsell * cash_rebate ;
								if (cash_rebate!=0.0)
									ScenarioSettle ( payoff, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows );
							}
							else
							{
								payoff			= nominal * blackEuropeanOption ( spot * exp ( b * expiry ), strike, r, sigma, expiry, buyorsell, phi );
							}
							mtm 			= ( payoff * baseFX ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
						}
						else
						if (remainingMaturityinDays==0)
						{
							payoff			= buyorsell * ( ( direction==BARRIER_IN ) ? cash_rebate : nominal * max ( phi * (spot - strike), 0.0 ) );
							mtm 			= ( payoff * baseFX ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
							ScenarioSettle (  payoff, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows );
						}
						else
						{
							payoff          = barrierOption ( barrier_payoff_type, eta, phi, sigma, expiry, cash_rebate/nominal, b, r, spot, strike, adj_barrier );							
							//now calculate the MTM
							mtm 			= ( nominal * buyorsell * payoff * baseFX ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
						}
					}
					else
					if ( direction==BARRIER_IN )
					{
						if (remainingMaturityinDays>0)
						{
							payoff			= nominal * blackEuropeanOption ( spot * exp ( b * expiry ), strike, r, sigma, expiry, buyorsell, phi ) ;
						}
						else
						{
							payoff			= nominal * buyorsell * max ( phi * (spot - strike), 0.0 ) ;
							ScenarioSettle (  payoff, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows );
						}
						mtm = ( payoff * baseFX ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
					}

					/*
					if (threadIdx.x==0 && blockIdx.x==0)
					{
						printf("barrier_payoff_type,%d,volume,%.4f,spot,%.4f,moneyness,%.5f,barrier,%.5f,expiry,%.6f,vol,%.8f,rf,%.8f,rd,%.8f,baseFX,%.4f,payoff,%.8f,mtm,%.4f\\n",barrier_payoff_type,nominal,spot,moneyness,barrier,expiry,vol,rf,rd,baseFX,payoff,mtm);
					}
					*/

					//store the spot price
					previous_spot = spot;
				}
				
				Output [ index ] = mtm ;
			}
		}
	'''

	def __init__(self, params):
		super(FXBarrierOption, self).__init__(params)
		self.path_dependent = True

	def reset(self, calendars):
		super(FXBarrierOption, self).reset()
		self.add_reval_dates ( set([self.field['Expiry_Date']]), self.field[ self.field['Payoff_Currency'] ] )

	def add_grid_dates(self, parser, base_date, grid):
		#a cash rebate is paid on touch if the option knocks out
		if self.field['Cash_Rebate']:
			grid_dates = parser ( base_date, self.field['Expiry_Date'], grid )
			self.reval_dates.update ( grid_dates )
			self.settlement_currencies.setdefault ( self.field[ self.field['Payoff_Currency'] ], set() ).update(grid_dates)
					   
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 				= Utils.CheckRateName(self.field['Currency'])
		field['Underlying_Currency'] 	= Utils.CheckRateName(self.field['Underlying_Currency'])
		field['Discount_Rate']			= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['FX_Volatility']			= Utils.CheckRateName(self.field['FX_Volatility'])

		field_index		= {}		
		field_index['Currency']				= getFXRateFactor 	( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 			= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Underlying_Currency']	= getFXRateFactor 	( field['Underlying_Currency'], static_offsets, stochastic_offsets )
		field_index['FX_Volatility'] 		= getFXVolFactor 	( field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors )
		
		#discrete barrier monitoring requires adjusting the barrier by 0.58 ( -(scipy.special.zetac(0.5)+1)/np.sqrt(2.0*np.pi) ) * sqrt (monitoring freq)
		field_index['Barrier_Monitoring']   = 0.5826 * np.sqrt( (base_date+self.field['Barrier_Monitoring_Frequency']-base_date).days/365.0 )
		field_index['Expiry'] 				= (self.field['Expiry_Date'] - base_date).days
		field_index['Invert_Moneyness']		= 1 if field['Currency'][0]==field['FX_Volatility'][0] else 0
		field_index['settlement_currency']	= 1.0
		
		return field_index
	
	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, 1)

		FXBarrierOption	= module.get_function ('FXBarrierOption')
		FXBarrierOption (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32(time_grid.deal_time_grid.size),
							precision ( field_index['Expiry'] ),
							precision ( self.field['Strike_Price'] ),
							precision ( 1.0 if self.field['Option_Type']=='Call' else -1.0 ),
							precision ( 1.0 if self.field['Buy_Sell']=='Buy' else -1.0 ),
							precision ( self.field['Underlying_Amount'] ),
							precision ( self.field['Barrier_Price'] ),
							precision ( self.field['Cash_Rebate'] ),
							precision ( 1.0 if 'Down' in self.field['Barrier_Type'] else -1.0 ),
							precision ( 1.0 if 'Out' in self.field['Barrier_Type'] else -1.0 ),
							precision ( field_index['Barrier_Monitoring'] ),
							np.int32 ( field_index['Invert_Moneyness'] ),
							np.int32( field_index['settlement_currency'] ),
							drv.In ( field_index['Currency'] ),
							drv.In ( field_index['Underlying_Currency'] ),
							drv.In ( field_index['Discount'] ),
							drv.In ( field_index['FX_Volatility'] ),
							block=block, grid=grid )
		
class FXOptionDeal(Deal):
	factor_fields = {'Currency':				['FxRate'],
					 'Underlying_Currency':		['FxRate'],
					 'Discount_Rate':			['DiscountRate'],					 
					 'FX_Volatility':			['FXVol']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Expiry_Date':			'Expiry date of the option',
						'Underlying_Amount':	'Number of units of the option held',
						'Strike_Price':			'Option exercise price, entered in the form of number of units of domestic currency per unit of foreign currency',
						'Underlying_Currency':	'ID of the FX rate price factor used to define the foreign currency name, for example, USD. The underlying exchange rate asset used by the deal is the amount of Currency per unit of Underlying_Currency. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'FX_Volatility':		"ID of the FX volatility price factor used to value the deal. This must be set to either 'UnderlyingCurrency' or 'Currency1.Curreny2', where UnderlyingCurrency is the foreign currency name and Currency1 and Currency2 are the IDs of the domestic and foreign currencies in alphabetical order. For example, 'EUR' or 'EUR.USD'. To use an FX volatility spread price factor, volatility spreads must be enabled and this property must use a tilde ('~') character and be set to 'Base~Spread', where 'Base' must be either 'UnderlyingCurrency' or 'Currency1.Currency2' as described above and 'Spread' is the ID of the spread. If left blank, the foreign currency name will be used. For example, 'EUR~BARRIERKOCALL' or 'EUR.USD~BARRIERKOCALL'. This requires the FX volatility price factor and, if volatility spreads are enabled and a tilde ('~') character is used, the FX volatility spread price factor"
						}
	
	cudacodetemplate = '''
		//all pricing functions will have the same 5 initial parameters
		__global__ void FXOptionDeal(	const REAL* __restrict__ stoch_factors,
										const REAL* __restrict__ static_factors,
										const int*  __restrict__ deal_time_index,
										const REAL* __restrict__ time_grid,
										const int*  __restrict__ cashflow_index,
										REAL* all_cashflows,
										REAL* Output,
										REAL maturity_in_days,
										REAL strike,
										REAL callorput,
										REAL buyorsell,
										REAL nominal,
										REAL isforward,
										int  invert_moneyness,
										const int* __restrict__ currency,
										const int* __restrict__ underlying_currency,
										const int* __restrict__ discount,
										const int* __restrict__ fxvolsurface )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];			
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
			OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			
			//set the mtm of this deal to 0
			REAL mtm 					= 0.0;

			//work out the remaining term
			REAL remainingMaturityinDays = maturity_in_days - t[TIME_GRID_MTM];

			if (remainingMaturityinDays>= 0.0)
			{
				//assume the daycount is stored in the first factor 
				REAL expiry 				= calcDayCountAccrual(remainingMaturityinDays, discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
				REAL baseFX				    = ScenarioRate ( t, currency, scenario_prior_index, static_factors, stoch_factors ) ;

				//get the currency repo curves
				const int* curr_curve 		= CurrencyRepoCurve ( currency );
				const int* und_curr_curve 	= CurrencyRepoCurve ( underlying_currency );

				//reconstruct (and interpolate) the discount curves
				REAL forward    = ScenarioFXForward ( t, remainingMaturityinDays, underlying_currency, currency, und_curr_curve, curr_curve, scenario_prior_index, static_factors, stoch_factors );
				REAL r 		    = ScenarioCurve ( t, expiry, discount, scenario_prior_index, static_factors, stoch_factors );
				REAL spot       = ScenarioRate  ( t, underlying_currency, scenario_prior_index, static_factors, stoch_factors ) / baseFX;
				REAL moneyness 	= invert_moneyness ? ( strike/spot ) : ( spot/strike );
				REAL vol 		= ScenarioSurface2D ( t, moneyness, expiry, fxvolsurface, scenario_prior_index, static_factors, stoch_factors );
				
				//now calculate the MTM
				mtm 			= ( nominal * blackEuropeanOption ( forward, strike, r, vol, expiry, buyorsell, callorput ) * baseFX ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
				
				/*
				if (threadIdx.x==0 && blockIdx.x==0)
				{
					printf("FXOption,mtm_time_index,%d,rd,%.6f,spot,%.6f,forward,%.6f,rf,%.6f,r,%.6f,strike,%.6f,expiry,%.6f,vol,%.6f,moneyness,%.6f,black,%.6f,mtm,%.6f\\n",mtm_time_index,rd,spot,forward,rf,r,strike,expiry,vol,moneyness,black,mtm);
				}
				*/
				
				//settle the currency_cashflows
				if (remainingMaturityinDays==0)
					ScenarioSettle( buyorsell * nominal * max ( callorput * (spot - strike), 0.0 ), currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows );
			}
			
			Output [ index ] = mtm ;
		}		
	'''

	def __init__(self, params):
		super(FXOptionDeal, self).__init__(params)

	def reset(self, calendars):
		super(FXOptionDeal, self).reset()
		self.add_reval_dates ( set([self.field['Expiry_Date']]), self.field['Currency'] )
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 				= Utils.CheckRateName(self.field['Currency'])
		field['Underlying_Currency'] 	= Utils.CheckRateName(self.field['Underlying_Currency'])
		field['Discount_Rate']			= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['FX_Volatility']			= Utils.CheckRateName(self.field['FX_Volatility'])

		field_index		= {}		
		field_index['Currency']				= getFXRateFactor 	( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 			= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Underlying_Currency']	= getFXRateFactor 	( field['Underlying_Currency'], static_offsets, stochastic_offsets )
		field_index['FX_Volatility'] 		= getFXVolFactor 	( field['FX_Volatility'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Expiry'] 				= (self.field['Expiry_Date'] - base_date).days
		field_index['Invert_Moneyness']		= 1 if field['Currency'][0]==field['FX_Volatility'][0] else 0
		
		return field_index
	
	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)

		FXOptionDeal	= module.get_function ('FXOptionDeal')
		FXOptionDeal (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
						precision ( field_index['Expiry'] ),
						precision ( self.field['Strike_Price'] ),
						precision ( 1.0 if self.field['Option_Type']=='Call' else -1.0 ),
						precision ( 1.0 if self.field['Buy_Sell']=='Buy' else -1.0 ),
						precision ( self.field['Underlying_Amount'] ),
						precision ( 1.0 if self.field['Option_On_Forward'] else 0.0 ),
						np.int32 ( field_index['Invert_Moneyness'] ),
						drv.In ( field_index['Currency'] ),
						drv.In ( field_index['Underlying_Currency'] ),
						drv.In ( field_index['Discount'] ),
						drv.In ( field_index['FX_Volatility'] ),
						block=block, grid=grid )

class DealDefaultSwap(Deal):
	factor_fields = {'Currency':				['FxRate'],
					 'Discount_Rate':			['DiscountRate'],
					 'Name':					['SurvivalProb']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Name':					"Issuer name and the ID of the survival probability price factor used to obtain the issuer's survival probability. For example, IBM. If the Respect_Default property of the deal's valuation model is set to 'Yes', this is also the ID of the credit rating price factor used to simulate the issuer's credit rating",
						'Principal':			'Principal amount in units of settlement currency',
						'Effective_Date':		'The contract start date',
						'Maturity_Date':		'The contract end date',
						'Pay_Frequency':		'Period between payments',
						'Amortisation':			'Permits an amortisation schedule for the principal to be specified as a list of (Date, Amount) pairs, where Date represents the date of this Amortisation and Amount represents the amount by which the principal is reduced by the Amortisation. A positive value for Amount represents a fall in the principal amount while a negative value for Amount represents a rise in the principal amount',
						'Pay_Rate':				'The fixed payment rate. Values are entered in percentage. For example, enter 5 for 5%'
						}

	#cuda code required to price this instrument
	cudacodetemplate = '''
		__global__ void DealDefaultSwap(const REAL* __restrict__ stoch_factors,
										const REAL* __restrict__ static_factors,
										const int*  __restrict__ deal_time_index,
										const REAL* __restrict__ time_grid,
										const int*  __restrict__ cashflow_index,
										REAL* all_cashflows,
										REAL* Output,
										REAL recovery_rate,
										int num_cashflow,
										const REAL* __restrict__ cashflows,
										const int*  __restrict__ starttime_index,
										const int* __restrict__ currency,
										const int* __restrict__ discount,
										const int* __restrict__ survival )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];			
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
			OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );

			//set the mtm of this deal
			REAL mtm 					= 0.0;
			
			//work out the remaining term
			int from_index = starttime_index[mtm_time_index];

			if ( from_index < num_cashflow )
			{
				//Get the settlement currency
				REAL FX_Base	= ScenarioRate ( t, currency, scenario_prior_index, static_factors, stoch_factors ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
				
				//PV the legs
				REAL premium = 0.0;
				REAL credit  = 0.0;
				
				for (int i=from_index; i<num_cashflow; i++)
				{
					const REAL* cashflow = cashflows+i*CASHFLOW_INDEX_Size;
					REAL start_time      = cashflow[CASHFLOW_INDEX_Start_Day] < t[TIME_GRID_MTM] ? t[TIME_GRID_MTM] : cashflow[CASHFLOW_INDEX_Start_Day];

					//Get the discount factors
					REAL Dt_T		     = CalcDiscount ( t, cashflow[CASHFLOW_INDEX_Pay_Day], discount, scenario_prior_index, static_factors, stoch_factors );
					REAL Dt_Tm1		     = CalcDiscount ( t, start_time, discount, scenario_prior_index, static_factors, stoch_factors );

					//survival factor daycount accruals
					REAL sT 			 = calcDayCountAccrual ( cashflow[CASHFLOW_INDEX_Pay_Day] - t[TIME_GRID_MTM], survival[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
					REAL st 			 = calcDayCountAccrual ( start_time - t[TIME_GRID_MTM], survival[FACTOR_INDEX_START+FACTOR_INDEX_Daycount] );
					
					REAL log_survival_T	 		= ScenarioCurve ( t, sT, survival, scenario_prior_index, static_factors, stoch_factors );
					REAL log_survival_t	 		= ScenarioCurve ( t, st, survival, scenario_prior_index, static_factors, stoch_factors );

					REAL premium_cashflow = cashflow[CASHFLOW_INDEX_FixedRate] * cashflow[CASHFLOW_INDEX_Year_Frac] * cashflow[CASHFLOW_INDEX_Nominal] * Dt_T * exp ( - log_survival_T );
					REAL credit_cashflow  = ( 1.0 - recovery_rate ) * cashflow[CASHFLOW_INDEX_Nominal] * 0.5 * ( Dt_T + Dt_Tm1 ) * ( exp (-log_survival_t) - exp(-log_survival_T) );
					
					premium += premium_cashflow ;
					credit  += credit_cashflow  ;
					/*
					if (threadIdx.x==0 && blockIdx.x==0)
					{
						printf("cashflow,%d,start_day,%.4f,pay_day,%.4f,premium_cashflow,%.4f,credit_cashflow,%.5f\\n",i,start_time,cashflow[CASHFLOW_INDEX_Pay_Day],premium_cashflow,credit_cashflow);
					}
					*/
					//settle the cashflows
					if (cashflow[CASHFLOW_INDEX_Pay_Day]==t[TIME_GRID_MTM])
						ScenarioSettle(credit_cashflow - premium_cashflow, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows);
				}

				//convert to the settlement currency
				mtm				= FX_Base * (credit - premium);
			}
			
			Output [ index ] = mtm ;
		}
	'''
	
	def __init__(self, params):
		super(DealDefaultSwap, self).__init__(params)		

	def reset(self, calendars):
		super(DealDefaultSwap, self).reset()
		bus_day 		= calendars.get(self.field['Calendars'], {'businessday':pd.offsets.Day(1)} )['businessday']
		self.resetdates	= generatedatesBackward ( self.field['Maturity_Date'], self.field['Effective_Date'], self.field['Pay_Frequency'], bus_day=bus_day )
		self.add_reval_dates ( self.resetdates, self.field['Currency'] )

	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 				= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']			= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Name']					= Utils.CheckRateName(self.field['Name'])

		field_index		= {}
		field_index['Currency']			= getFXRateFactor 	( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 		= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Name']				= getSurvivalFactor	( field['Name'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Recovery_Rate']	= getRecoveryRate   ( field['Name'], all_factors )
		
		pay_rate 						= self.field['Pay_Rate']/100.0 if isinstance(self.field['Pay_Rate'], float) else self.field['Pay_Rate'].amount

		field_index['Cashflows'] 		= Utils.GenerateFixedCashflows ( base_date, self.resetdates, (1 if self.field['Buy_Sell']=='Buy' else -1) * self.field['Principal'], self.field['Amortisation'], Utils.GetDayCount( self.field['Accrual_Day_Count'] ), pay_rate )
		
		#include the maturity date in the daycount
		field_index['Cashflows'].AddMaturityAccrual( base_date, Utils.GetDayCount( self.field['Accrual_Day_Count'] ) )
		
		field_index['StartIndex']  		= field_index['Cashflows'].GetCashflowStartIndex(time_grid)
			
		return field_index
	
	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)
		
		DealDefaultSwap 	= module.get_function ('DealDefaultSwap')
		DealDefaultSwap( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							precision ( field_index['Recovery_Rate'] ),
							np.int32  ( field_index['Cashflows'].Count() ),
							drv.In    ( field_index['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['StartIndex'] ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['Discount'] ),
							drv.In    ( field_index['Name'] ),
							block=block, grid=grid )
		
class FRADeal(Deal):
	factor_fields = {'Currency':				['FxRate'],
					 'Discount_Rate':			['DiscountRate'],
					 'Interest_Rate':			['InterestRate']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Interest_Rate':		'ID of the interest rate price factor used to calculate forward interest rates. For example, USD.AAA',
						'Principal':			'Principal amount in units of settlement currency',
						'FRA_Rate':				'Fixed interest rate. Rates are entered in percentage. For example, enter 5 for 5%',
						'Borrower_Lender':		'Determines whether the holder borrows (Borrower) or lends (Lender) the security leg'
						}
	
	#cuda code required to price this instrument - None - reusing the AddFloatCashflow
	cudacodetemplate = ''
	
	def __init__(self, params):
		super(FRADeal, self).__init__(params)

	def reset(self, calendars):
		super(FRADeal, self).reset()
		self.pay_date = self.field['Maturity_Date'] if self.field['Payment_Timing']=='End' else  self.field['Effective_Date']
		self.add_reval_dates ( set( [ self.pay_date ] ), self.field['Currency'] )
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 		= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']	= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Interest_Rate']	= Utils.CheckRateName(self.field['Interest_Rate']) if self.field['Interest_Rate'] else field['Discount_Rate']

		field_index		= {}		
		field_index['Currency']		= getFXRateFactor ( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount']		= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Forward']  	= getInterestFactor ( field['Interest_Rate'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Daycount']		= Utils.GetDayCount ( self.field['Day_Count'] )

		Accrual_fraction            = Utils.GetDayCountAccrual ( base_date, (self.field['Maturity_Date']-self.field['Effective_Date']).days, field_index['Daycount'] )
		cashflows                   = {'Items':
										[{
										   'Payment_Date': 			    self.pay_date,
										   'Accrual_Start_Date': 	    self.field['Effective_Date'],
										   'Accrual_End_Date': 		    self.field['Maturity_Date'],
										   'Accrual_Year_Fraction':		Accrual_fraction,
										   'Notional':			        self.field['Principal'],
										   'Margin':		            Utils.Basis( -100.0*self.field['FRA_Rate'] ),
										   'Resets':   	                [ [self.field['Reset_Date'], self.field['Reset_Date'], self.field['Maturity_Date'], Accrual_fraction, self.field['Use_Known_Rate'], self.field['Known_Rate'] ] ]
										}]
									}

		field_index['VolSurface'] 	= np.zeros(1, dtype=np.int32)
		field_index['Cashflows']  	= Utils.MakeFloatCashflows ( base_date, time_grid, 1 if self.field['Borrower_Lender']=='Borrower' else -1, cashflows )
		field_index['StartIndex'] 	= field_index['Cashflows'].GetCashflowStartIndex(time_grid)

		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)
		
		AddFloatCashflow 	= module.get_function ( 'AddFloatCashflow' )
		AddFloatCashflow( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32 ( Utils.SCENARIO_CASHFLOWS_FloatLeg ),
							np.int32 ( Utils.CASHFLOW_METHOD_Compounding_None ),
							np.int32 ( 1 ),
							np.int32  ( field_index['Cashflows'].Count() ),
							drv.In    ( field_index['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['StartIndex'] ),
							drv.In    ( field_index['Cashflows'].Resets.GetSchedule(precision).ravel() ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['VolSurface'] ),
							drv.In    ( field_index['Forward'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )

class FloatingEnergyDeal(Deal):
	#dependent price factors for this instrument 
	factor_fields = {'Currency':							['FxRate'],
					 'Discount_Rate':						['DiscountRate'],
					 'Sampling_Type':						['ForwardPriceSample'],
					 'FX_Sampling_Type':					['ForwardPriceSample'],
					 'Reference_Type':						['ReferencePrice'],
					 'Payoff_Currency':						['FxRate']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Sampling_Type':		'ID of the forward price sample price factor used to define reference price sampling convention. For example, DAILY-NEXT',
						'FX_Sampling_Type':		'ID of the forward price sample price factor used to define FX sampling convention. For example, DAILY-NEXT',
						'Payments':				'List of forward contracts, where each contract is linked to a future energy price',
						'Reference_Type':		'ID of the reference price price factor used to define the energy reference price. For example, WTI. This requires the reference price price factor and the forward price price factor given by the ForwardPrice property of that reference price price factor',
						'Payoff_Currency':		"ID of the FX rate price factor used to define the settlement currency. This property is required for quanto deals. For deals with a standard or compo payoff type, this property must be the same as the deal currency. Otherwise, the deal will be have a quanto payoff type. If left blank, the ID of the deal currency will be used. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor. Quanto deals also require the FX volatility price factor of the FX rate given by the underlying asset's price factor currency and this settlement currency and the Implied Correlations price factor between that FX rate and the underlying asset",
						}
	
	#cuda code required to price this instrument - none - as it's just AddFloatCashflow
	cudacodetemplate = ''
	
	def __init__(self, params):
		super(FloatingEnergyDeal, self).__init__(params)
		
	def reset(self, calendars):
		super(FloatingEnergyDeal, self).reset()
		paydates	= set( [x['Payment_Date'] for x in self.field['Payments']['Items']] )
		self.add_reval_dates ( paydates, self.field['Payoff_Currency'] if self.field['Payoff_Currency'] else field['Currency'] )
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 			= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']		= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Sampling_Type']		= Utils.CheckRateName(self.field['Sampling_Type'])
		field['FX_Sampling_Type']	= Utils.CheckRateName(self.field['FX_Sampling_Type']) if self.field['FX_Sampling_Type'] else None
		field['Reference_Type']		= Utils.CheckRateName(self.field['Reference_Type'])
		field['Payoff_Currency'] 	= Utils.CheckRateName(self.field['Payoff_Currency']) if self.field['Payoff_Currency'] else field['Currency']

		field_index							= {}
		reference_factor, forward_factor	= getReferenceFactorObjects ( field['Reference_Type'], all_factors )
		forward_sample 						= getForwardPriceSampling ( field['Sampling_Type'], all_factors )
		fx_sample 							= getForwardPriceSampling ( field['FX_Sampling_Type'], all_factors ) if field['FX_Sampling_Type'] else None
		
		#need to give cuda the delta from the base date
		field_index['ForwardPrice']			= getForwardPriceFactor   ( field['Reference_Type'], field['Currency'], static_offsets, stochastic_offsets, all_tenors, reference_factor, forward_factor, base_date )
		field_index['Payoff_Currency'] 		= getFXRateFactor ( field['Payoff_Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 			= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Cashflows']  			= Utils.MakeEnergyCashflows ( base_date, time_grid, -1.0 if self.field['Payer_Receiver']=='Payer' else 1.0, self.field['Payments'], reference_factor, forward_sample, fx_sample, calendars )
		field_index['StartIndex']  			= field_index['Cashflows'].GetCashflowStartIndex(time_grid)
		
		#zero dummy ir vol
		field_index['InterestYieldVol'] 	= np.zeros(1, dtype=np.int32)
		
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)

		AddFloatCashflow 	= module.get_function ('AddFloatCashflow')
		AddFloatCashflow( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32 ( Utils.SCENARIO_CASHFLOWS_Energy ),
							np.int32 ( Utils.CASHFLOW_METHOD_Average_Interest ),
							np.int32 ( 1 ),
							np.int32 ( field_index['Cashflows'].Count() ),
							drv.In   ( field_index['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In   ( field_index['StartIndex'] ),
							drv.In   ( field_index['Cashflows'].Resets.GetSchedule(precision).ravel() ),
							drv.In   ( field_index['Payoff_Currency'] ),
							drv.In   ( field_index['InterestYieldVol'] ),
							drv.In   ( field_index['ForwardPrice'] ),
							drv.In   ( field_index['Discount'] ),
							block=block, grid=grid )

class FixedEnergyDeal(Deal):
	#dependent price factors for this instrument 
	factor_fields = {'Currency':							['FxRate'],
					 'Discount_Rate':						['DiscountRate']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor'
						}
	
	#cuda code required to price this instrument - none - as it's just AddFixedCashflow
	cudacodetemplate = ''
	
	def __init__(self, params):
		super(FixedEnergyDeal, self).__init__(params)
		
	def reset(self, calendars):
		super(FixedEnergyDeal, self).reset()
		self.paydates	= set( [x['Payment_Date'] for x in self.field['Payments']['Items']] )
		self.add_reval_dates ( self.paydates, self.field['Currency'] )
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 			= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']		= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']

		field_index							= {}
		field_index['Currency']				= getFXRateFactor ( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 			= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['Cashflows']  			= Utils.MakeEnergyFixedCashflows( base_date, -1.0 if self.field['Payer_Receiver']=='Payer' else 1.0, self.field['Payments'] )
		field_index['StartIndex']  			= field_index['Cashflows'].GetCashflowStartIndex(time_grid)
		
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)

		AddFixedCashflow	= module.get_function ('AddFixedCashflow')
		AddFixedCashflow (  CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32  ( 1 ),
							np.int32  ( Utils.CASHFLOW_METHOD_Fixed_Compounding_No ),
							np.int32  ( field_index['Cashflows'].Count() ),
							drv.In    ( field_index['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In    ( field_index['StartIndex'] ),
							drv.In    ( field_index['Currency'] ),
							drv.In    ( field_index['Discount'] ),
							block=block, grid=grid )
		
class EnergySingleOption(Deal):
	#dependent price factors for this instrument 
	factor_fields = {'Currency':							['FxRate'],
					 'Discount_Rate':						['DiscountRate'],
					 'Sampling_Type':						['ForwardPriceSample'],
					 'FX_Sampling_Type':					['ForwardPriceSample'],
					 'Reference_Type':						['ReferencePrice'],
					 'Reference_Volatility':				['ReferenceVol'],
					 'Payoff_Currency':						['FxRate']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Sampling_Type':		'ID of the forward price sample price factor used to define reference price sampling convention. For example, DAILY-NEXT',
						'FX_Sampling_Type':		'ID of the forward price sample price factor used to define FX sampling convention. For example, DAILY-NEXT',
						'Settlement_Date':		'Settlement date',
						'Period_Start':			'Sampling period start date',
						'Period_End':			'Sampling period end date',
						'Strike':				'Strike price',
						'Realized_Average':		'Average of realized sample prices up to the base valuation date. The realized average must be in the forward price factor currency unless the deal is compo, where this average must be given in payoff currency. This property is required if the base calculation date is after the sampling period start date and ignored otherwise',
						'FX_Period_Start':		'FX averaging period start date. This property is required if Average_FX is Standard or Inverted and ignored otherwise',
						'FX_Period_End':		'FX averaging period end date. This property is required if Average_FX is Standard or Inverted and ignored otherwise',
						'FX_Realized_Average':	'Average of realized FX rates up to the base valuation date. This property is required if Average_FX is Standard or Inverted and the base valuation date is after the FX averaging period start date and ignored otherwise',
						'Volume':				'Volume of energy specified in the contract',
						'Reference_Type':		'ID of the reference price price factor used to define the energy reference price. For example, WTI. This requires the reference price price factor and the forward price price factor given by the ForwardPrice property of that reference price price factor',
						'Reference_Volatility':	"ID of the reference volatility price factor used to value the deal. This property must be set to either 'ReferenceType' or 'ReferenceType~Spread', where 'ReferenceType' is the ID of the reference price price factor and 'Spread' has no restrictions. For example, 'WTI' or 'WTI~BARRIERKOCALL'. If left blank, the ID of the reference price price factor will be used. This requires the reference volatility price factor, the reference price price factor, the forward price price factor given by the ReferencePrice property of the reference volatility price factor and the forward price volatility price factor (and optionally the forward price volatility spread price factor) given by the ForwardPriceVol property of the reference volatility",
						'Payoff_Currency':		"ID of the FX rate price factor used to define the settlement currency. This property is required for quanto deals. For deals with a standard or compo payoff type, this property must be the same as the deal currency. Otherwise, the deal will be have a quanto payoff type. If left blank, the ID of the deal currency will be used. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor. Quanto deals also require the FX volatility price factor of the FX rate given by the underlying asset's price factor currency and this settlement currency and the Implied Correlations price factor between that FX rate and the underlying asset"
						}
	
	#cuda code required to price this instrument 
	cudacodetemplate = '''
	__global__ void EnergySingleOption(	const REAL* __restrict__ stoch_factors,
										const REAL* __restrict__ static_factors,
										const int*  __restrict__ deal_time_index,
										const REAL* __restrict__ time_grid,
										const int*  __restrict__ cashflow_index,
										REAL* all_cashflows,
										REAL* Output,
										REAL maturity_in_days,
										REAL strike,
										REAL callorput,
										REAL buyorsell,
										REAL volume,
										REAL correlation,
										const int* __restrict__ forward,
										const int* __restrict__ forward_vol,
										const int* __restrict__ fx_vol,
										const int* __restrict__ currency,
										const int* __restrict__ discount,
										int   num_samples,
										const int*  __restrict__ starttime_index,
										const REAL* __restrict__ samples )
		{
			int scenario_number 		= blockIdx.x*blockDim.x + threadIdx.x;
			int mtm_time_index 			= deal_time_index[blockIdx.y];			
			const REAL* t 				= time_grid + TIME_GRID_Size*mtm_time_index;
			int index  					= scenario_number * MTMTimeSteps + mtm_time_index;
			OFFSET scenario_prior_index = scenario_number * ScenarioTimeSteps + reinterpret_cast <const OFFSET &> ( t[TIME_GRID_ScenarioPriorIndex] );
			
			//set the mtm of this deal to 0
			REAL mtm 					= 0.0;

			//work out the remaining term
			REAL remainingMaturityinDays = maturity_in_days - t[TIME_GRID_MTM];

			if (remainingMaturityinDays>= 0.0)
			{
				//assume the daycount is stored in the first factor 
				REAL expiry 				= calcDayCountAccrual(remainingMaturityinDays, discount[FACTOR_INDEX_START+FACTOR_INDEX_Daycount]);
				REAL baseFX				    = ScenarioRate ( t, currency, scenario_prior_index, static_factors, stoch_factors ) ;

				//reconstruct (and interpolate) the discount curve
				REAL rd 		 = ScenarioCurve ( t, expiry, discount, scenario_prior_index, static_factors, stoch_factors );
				REAL normalize   = 0.0;
				REAL forward_p	 = Calc_ForwardPrice ( t, starttime_index[mtm_time_index], num_samples, normalize, forward, samples, scenario_prior_index, static_factors, stoch_factors );
				REAL average     = Calc_PastPrice ( t, starttime_index[mtm_time_index], forward, samples, scenario_prior_index, static_factors, stoch_factors );
				REAL strike_bar	 = strike - average;
				REAL moneyness 	 = forward_p / ( strike_bar/normalize) ;
				REAL vol 		 = sqrt ( Calc_SinglePriceVariance( t, forward_p, correlation, forward, forward_vol, fx_vol, moneyness, starttime_index[mtm_time_index], num_samples, samples, scenario_prior_index, static_factors, stoch_factors ) );
				
				//Tenor is set to 1.0 as vol has already been scaled using the correct tenors
				REAL black      = blackEuropeanOption ( forward_p, strike_bar, 0.0, vol, 1.0, buyorsell, callorput ) * exp ( -rd * expiry );

				//now calculate the MTM
				mtm 			= ( volume * black * baseFX ) / ScenarioRate ( t, ReportCurrency, scenario_prior_index, static_factors, stoch_factors );
				/*				
				if (threadIdx.x==0 && blockIdx.x==0)
				{
					printf("volume,%.4f,forward_p,%.4f,moneyness,%.5f,strike,%.5f,strike_bar,%.5f,expiry,%.4f,vol,%.5f,baseFX,%.4f,black,%.8f,mtm,%.4f\\n",volume,forward_p,moneyness,strike,strike_bar,expiry,vol,baseFX,black,mtm);
				}
				*/
				//settle the currency_cashflows
				if (remainingMaturityinDays==0)
					ScenarioSettle ( volume * black, currency, scenario_number, mtm_time_index, cashflow_index, all_cashflows);
			}
			
			Output [ index ] = mtm ;
		}
	'''
	
	def __init__(self, params):
		super(EnergySingleOption, self).__init__(params)
		
	def reset(self, calendars):
		super(EnergySingleOption, self).reset()
		self.paydates	= set( [ self.field['Settlement_Date'] ] )
		self.add_reval_dates ( self.paydates, self.field['Payoff_Currency'] if self.field['Payoff_Currency'] else field['Currency'] )
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 				= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']			= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Sampling_Type']			= Utils.CheckRateName(self.field['Sampling_Type'])
		field['FX_Sampling_Type']		= Utils.CheckRateName(self.field['FX_Sampling_Type']) if self.field['FX_Sampling_Type'] else None
		field['Reference_Type']			= Utils.CheckRateName(self.field['Reference_Type'])
		field['Reference_Volatility']	= Utils.CheckRateName(self.field['Reference_Volatility'])
		field['Payoff_Currency'] 		= Utils.CheckRateName(self.field['Payoff_Currency']) if self.field['Payoff_Currency'] else field['Currency']
		
		field['cashflow']				= {'Payment_Date': 			self.field['Settlement_Date'],
										   'Period_Start': 			self.field['Period_Start'],
										   'Period_End': 			self.field['Period_End'],
										   'Volume':				1.0,
										   'Fixed_Price':			self.field['Strike'],
										   'Realized_Average':		self.field['Realized_Average'],
										   'FX_Period_Start':   	self.field['FX_Period_Start'],
										   'FX_Period_End':			self.field['FX_Period_End'],
										   'FX_Realized_Average':	self.field['FX_Realized_Average']}

		field_index							= {}
		reference_factor, forward_factor	= getReferenceFactorObjects ( field['Reference_Type'], all_factors )
		forward_sample 						= getForwardPriceSampling ( field['Sampling_Type'], all_factors )
		fx_sample 							= getForwardPriceSampling ( field['FX_Sampling_Type'], all_factors ) if field['FX_Sampling_Type'] else None
		forward_price_vol					= getForwardPriceVol ( field['Reference_Volatility'], all_factors )

		if field['Currency'] != forward_factor.GetCurrency():
			fx_lookup							= tuple ( sorted ( [field['Currency'][0], forward_factor.GetCurrency()[0] ] ) )
			field_index['FXCompoVol'] 			= getFXVolFactor( fx_lookup, static_offsets, stochastic_offsets, all_tenors )
			field_index['ImpliedCorrelation']	= getImpliedCorrelation ( ('FxRate',)+fx_lookup, ('ReferencePrice',)+ forward_price_vol, all_factors )
			#field_index['ImpliedCorrelation']	= 0.0
		else:
			field_index['FXCompoVol'] 			= np.zeros(1, dtype=np.int32)
			field_index['ImpliedCorrelation']	= 0.0
			
		#make a pricing cashflow
		Cashflow							= Utils.MakeEnergyCashflows ( base_date, time_grid, 1, {'Items':[field['cashflow']]}, reference_factor, forward_sample, fx_sample, calendars )
		#turn it into a sampling object
		field_index['Samples']				= Cashflow.Resets
		field_index['StartIndex']			= Cashflow.Resets.GetStartIndex ( time_grid, (base_date-Utils.excel_offset).days )
		field_index['ForwardPrice']			= getForwardPriceFactor   ( field['Reference_Type'], field['Currency'], static_offsets, stochastic_offsets, all_tenors, reference_factor, forward_factor, base_date )
		field_index['Payoff_Currency'] 		= getFXRateFactor ( field['Payoff_Currency'], static_offsets, stochastic_offsets )
		field_index['Discount'] 			= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		field_index['ReferenceVol'] 		= getForwardPriceVolFactor ( forward_price_vol, static_offsets, stochastic_offsets, all_tenors )
		field_index['Expiry'] 				= (self.field['Settlement_Date'] - base_date).days
		
		return field_index

	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)

		EnergySingleOption 	= module.get_function ('EnergySingleOption')
		EnergySingleOption( CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							precision ( field_index['Expiry'] ),
							precision ( self.field['Strike'] ),
							precision ( 1.0 if self.field['Option_Type']=='Call' else -1.0 ),
							precision ( 1.0 if self.field['Buy_Sell']=='Buy' else -1.0 ),
							precision ( self.field['Volume'] ),
							precision ( field_index['ImpliedCorrelation'] ),
							drv.In ( field_index['ForwardPrice'] ),
							drv.In ( field_index['ReferenceVol'] ),
							drv.In ( field_index['FXCompoVol'] ),
							drv.In ( field_index['Payoff_Currency'] ),
							drv.In   ( field_index['Discount'] ),
							np.int32 ( field_index['Samples'].Count() ),
							drv.In ( field_index['StartIndex'] ),
							drv.In ( field_index['Samples'].GetSchedule(precision).ravel() ),
							block=block, grid=grid )

class SwapBasisDeal(Deal):
	factor_fields = {'Currency':						['FxRate'],
					 'Discount_Rate':					['DiscountRate'],
					 'Pay_Rate':						['InterestRate'],
					 'Receive_Rate':					['InterestRate'],
					 'Pay_Rate_Volatility':				['InterestRateVol','InterestYieldVol'],
					 'Pay_Discount_Rate_Volatility':	['InterestRateVol','InterestYieldVol'],
					 'Receive_Rate_Volatility':			['InterestRateVol','InterestYieldVol'],
					 'Receive_Discount_Rate_Volatility':['InterestRateVol','InterestYieldVol']}
	
	required_fields = {	'Currency':				'ID of the FX rate price factor used to define the settlement currency. For example, USD. This property requires the corresponding FX rate price factor and the interest rate price factor specified by the Interest_Rate property (or, if empty, the ID) of that FX rate price factor',
						'Discount_Rate':		'ID of the discount rate price factor used to discount the cashflows. For example, USD.AAA. The currency of this price factor must be the settlement currency. If left blank, the ID of the settlement currency will be used. This property requires the corresponding discount rate price factor and the interest rate price factor specified by the Interest_Rate property of the discount rate price factor. If a discount rate price factor with this ID has not been provided, a new discount factor with this ID will be created with its Interest_Rate property set by default to the ID of this discount rate price factor',
						'Effective_Date':		'The contract start date',
						'Maturity_Date':		'The contract end date',
						'Pay_Frequency':		'The payment frequency, which is the period between payments of the pay leg',
						'Pay_Margin':			'Margin rate added to the pay leg generalized cashflows. Rates are entered in basis points. For example, enter 50 for 50 basis points',
						'Pay_Known_Rates':		'List of reset dates and corresponding known interest rates specified in the form of (date, value) pairs. Rates are entered in percentage. For example, enter 5 for 5%',
						'Receive_Frequency':	'The payment frequency, which is the period between payments of the receive leg. If this property is set to 0M, then the receive leg payment periods will be determined by the trade effective, maturity and receive leg stub dates (if specified)',
						'Receive_Margin':		'Margin rate added to the receive leg generalized cashflows. Rates are entered in basis points. For example, enter 50 for 50 basis points',
						'Receive_Known_Rates':	'List of reset dates and corresponding known interest rates specified in the form of (date, value) pairs. Rates are entered in percentage. For example, enter 5 for 5%',
						'Pay_Rate':				'ID of the interest rate price factor used to calculate forward interest rates. For example, USD.AAA',
						'Receive_Rate':			'ID of the interest rate price factor used to calculate forward interest rates. For example, USD.AAA',
						'Principal':			'Principal amount in units of settlement currency',
						'Amortisation':			'Permits an amortisation schedule for the principal to be specified as a list of (Date, Amount) pairs, where Date represents the date of this Amortisation and Amount represents the amount by which the principal is reduced by the Amortisation. A positive value for Amount represents a fall in the principal amount while a negative value for Amount represents a rise in the principal amount'
						}
	
	#cuda code required to price this instrument - none at this moment - it calls the standard AddFloatCashflow function
	cudacodetemplate = ''
	
	def __init__(self, params):
		super(SwapBasisDeal, self).__init__(params)

	def reset(self, calendars):
		super(SwapBasisDeal, self).reset()
		self.paydates	= generatedatesBackward ( self.field['Maturity_Date'], self.field['Effective_Date'], self.field['Pay_Frequency'] )
		self.recdates	= generatedatesBackward ( self.field['Maturity_Date'], self.field['Effective_Date'], self.field['Receive_Frequency'] )
		self.add_reval_dates ( self.paydates, self.field['Currency'] )
		self.add_reval_dates ( self.recdates, self.field['Currency'] )
		
	def PostProcess( self, module, CudaMem, precision, batch_size, revals_per_batch, deal_data, child_dependencies, partition, mtm_offset ):
		logging.warning('SwapBasisDeal {0} - TODO'.format(self.field['Reference']))
		
	def calc_dependancies(self, base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid, calendars):
		field = {}
		field['Currency'] 		= Utils.CheckRateName(self.field['Currency'])
		field['Discount_Rate']	= Utils.CheckRateName(self.field['Discount_Rate']) if self.field['Discount_Rate'] else field['Currency']
		field['Pay_Rate']		= Utils.CheckRateName(self.field['Pay_Rate']) if self.field['Pay_Rate'] else field['Currency']
		field['Receive_Rate']	= Utils.CheckRateName(self.field['Receive_Rate']) if self.field['Receive_Rate'] else field['Currency']

		field_index							= {'Pay':{}, 'Receive':{}}
		field_index['Pay']['Forward']  		= getInterestFactor ( field['Pay_Rate'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Receive']['Forward']  	= getInterestFactor ( field['Receive_Rate'], static_offsets, stochastic_offsets, all_tenors )
		field_index['Currency']				= getFXRateFactor   ( field['Currency'], static_offsets, stochastic_offsets )
		field_index['Discount']				= getDiscountFactor ( field['Discount_Rate'], static_offsets, stochastic_offsets, all_tenors, all_factors )
		
		field_index['Pay']['CompoundingMethod'] 	= Utils.CASHFLOW_CompoundingMethodLookup [ self.field['Pay_Compounding_Method'] ]
		field_index['Receive']['CompoundingMethod'] = Utils.CASHFLOW_CompoundingMethodLookup [ self.field['Receive_Compounding_Method'] ]
		
		#cashflows
		field_index['Pay']['Cashflows']			= Utils.GenerateFloatCashflows ( base_date, time_grid, self.paydates, -self.field['Principal'], self.field['Amortisation'], self.field['Pay_Known_Rates'], self.field['Pay_Index_Tenor'], self.field['Pay_Interest_Frequency'], Utils.GetDayCount( self.field['Pay_Day_Count'] ), self.field['Pay_Margin']/10000.0 )
		field_index['Receive']['Cashflows']		= Utils.GenerateFloatCashflows ( base_date, time_grid, self.recdates, self.field['Principal'], self.field['Amortisation'], self.field['Receive_Known_Rates'], self.field['Receive_Index_Tenor'], self.field['Receive_Interest_Frequency'], Utils.GetDayCount( self.field['Receive_Day_Count'] ), self.field['Receive_Margin']/10000.0 )
			
		#cashflow start indexes
		field_index['Pay']['StartIndex'] 	 = field_index['Pay']['Cashflows'].GetCashflowStartIndex(time_grid)
		field_index['Receive']['StartIndex'] = field_index['Receive']['Cashflows'].GetCashflowStartIndex(time_grid)
		
		#zero dummy ir vol
		field_index['InterestYieldVol'] = np.zeros(1, dtype=np.int32)
		
		return field_index
	
	def Generate(self, module, CudaMem, precision, batch_size, revals_per_batch, field_index, time_grid):
		block       		= (revals_per_batch,1,1)
		grid        		= (batch_size, time_grid.deal_time_grid.size)
		
		AddFloatCashflow 	= module.get_function ('AddFloatCashflow')
		AddFloatCashflow( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32 (  Utils.SCENARIO_CASHFLOWS_FloatLeg ),
							np.int32 (  field_index['Pay']['CompoundingMethod'] ),
							np.int32 ( 1 ),
							np.int32 ( field_index['Pay']['Cashflows'].Count() ),
							drv.In   ( field_index['Pay']['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In   ( field_index['Pay']['StartIndex'] ),
							drv.In   ( field_index['Pay']['Cashflows'].Resets.GetSchedule(precision).ravel() ),
							drv.In   ( field_index['Currency'] ),
							drv.In   ( field_index['InterestYieldVol'] ),
							drv.In   ( field_index['Pay']['Forward'] ),
							drv.In   ( field_index['Discount'] ),
							block=block, grid=grid )
		
		AddFloatCashflow( 	CudaMem.d_Scenario_Buffer, CudaMem.d_Static_Buffer, drv.In(time_grid.deal_time_grid), CudaMem.d_Time_Grid, CudaMem.d_Cashflow_Index, CudaMem.d_Cashflows, CudaMem.d_MTM_Buffer,
							np.int32 (  Utils.SCENARIO_CASHFLOWS_FloatLeg ),
							np.int32 (  field_index['Receive']['CompoundingMethod'] ),
							np.int32 ( 0 ),
							np.int32 ( field_index['Receive']['Cashflows'].Count() ),
							drv.In   ( field_index['Receive']['Cashflows'].GetSchedule(precision).ravel() ),
							drv.In   ( field_index['Receive']['StartIndex'] ),
							drv.In   ( field_index['Receive']['Cashflows'].Resets.GetSchedule(precision).ravel() ),
							drv.In   ( field_index['Currency'] ),
							drv.In   ( field_index['InterestYieldVol'] ),
							drv.In   ( field_index['Receive']['Forward'] ),
							drv.In   ( field_index['Discount'] ),
							block=block, grid=grid )

def ConstructInstrument(param):
	if param.get('Object') not in globals():
		print "!!! MAPPING TODO !!!!", param.get('Object')
	return globals().get(param.get('Object'))(param)

def GetAllInstrumentCudaFunctions():
	return set([cls.cudacodetemplate for cls in globals().values() if isinstance(cls, type) and hasattr(cls, 'cudacodetemplate')])

if __name__=='__main__':
	import cPickle
	
	#field_index = cPickle.load(file('field_index.obj','rb'))	
	#generatedatesBackward(pd.datetime(2017,1,1), pd.datetime(2014,1,1), pd.DateOffset(months=3))
