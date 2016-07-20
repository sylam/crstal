#import standard libraries
import os
import sys
import json
import calendar
import operator
import logging

#import parsing libraries
from pyparsing import *
from xml.etree.ElementTree import ElementTree, Element, iterparse

#useful types
if sys.version[:3]>'2.6':
    from collections import OrderedDict
else:
    from ordereddict import OrderedDict

#needed types
import Utils
from Instruments import ConstructInstrument, Deal

try:
    #load up extra libraries
    #Note that PyPy does not have access to numpy and pandas (by default) - but it's stupidly fast at parsing
    import pandas as pd
    import numpy as np

    #import pricing library
    from StochasticProcess import ConstructCalibrationConfig, ConstructProcess
    from RiskFactors import ConstructFactor
    
    #define datetime routines
    Timestamp           = pd.Timestamp
    DateOffset          = pd.DateOffset
    array_transform     = lambda x:x.tolist()
    
except:
    #define dummy objects for parsing with PyPy
    class DateOffset(object):
        def __init__(self, **kwds):
            self.kwds = kwds
        
    class Timestamp(object):
        def __init__(self, date):
            self.date = date
            
        def strftime(self, fmt):
            return self.date
        
    array_transform     = lambda x:x

def copy_dict(source_dict, diffs):
    """Returns a copy of source_dict, updated with the new key-value pairs in diffs."""
    result=OrderedDict(source_dict)
    result.update(diffs)
    return result

def drawobj(obj):
    '''I really don't like the Adaptiv Analytic file format - it's really crap - hence this code - Note the insane special case for Eigenvectors/Resets - who does this? Really.'''
    buffer = []
    if isinstance(obj, list):
        for value in obj:
            if isinstance(value, tuple):
                buffer += ['.'.join(value)]
            elif isinstance(value, DateOffset):
                buffer += [''.join(['%d%s' % (v,Parser.reverse_offset[k]) for k,v in value.kwds.items()])]
            elif isinstance(value, float):
                buffer += ['%.12g' % value]
            elif isinstance(value, Timestamp):
                buffer += ['%02d%s%04d' % (value.day, calendar.month_abbr[value.month], value.year)]
            else:
                buffer += [str(value) if value else '']
    else:
        for key, value in obj.items():
            if isinstance(value, dict):
                buffer += ['='.join([key, '[%s]' % drawobj(value)])]
            elif isinstance(value, list):
                temp = []
                for sub_value in value:
                    if isinstance(sub_value,str):
                        temp += [sub_value]
                    elif isinstance(sub_value, Utils.Curve):
                        temp += [str(sub_value)]
                    else:
                        temp += ['['+drawobj(sub_value)+']']
                buffer += ['='.join([key, '[%s]' % ( ','.join(temp) if key in ['InstantaneousDrift','InstantaneousVols','Eigenvectors','Resets','Points'] else ''.join(temp)) ])]
            elif isinstance(value, tuple):
                buffer += ['='.join([key, '.'.join(value)])]
            elif isinstance(value, DateOffset):
                buffer += ['='.join([key, ''.join(['%d%s' % (v,Parser.reverse_offset[k]) for k,v in value.kwds.items()]) ])] 
            elif isinstance(value, float):
                buffer += ['='.join([key, '%.12g' % value])]
            elif isinstance(value, Timestamp):
                buffer += ['='.join([key, '%02d%s%04d' % (value.day, calendar.month_abbr[value.month], value.year)])]
            else:
                buffer += ['='.join([key, str(value) if value else ''])]

    return ','.join(buffer)

class ModelParams(object):
    def __init__(self, state=None):
        self.valid_subtype = {'BasisSpread':'BasisSpread'}
        self.modeldefaults = OrderedDict() 
        self.modelfilters  = OrderedDict()
        if state:
            defaults, filters = state
            for factor, model in defaults.items():
                self.Append(factor, (), model )
            for factor, mappings in filters.items():
                for condition, model in mappings:
                    self.Append ( factor, tuple(condition), model )
        
    def Append(self, price_factor, price_filter, stoch_proc):
        if price_filter==():
            self.modeldefaults.setdefault(price_factor,stoch_proc)
        else:
            self.modelfilters.setdefault(price_factor, []).append((price_filter, stoch_proc))

    def WriteToFile(self, filehandle):
        #write out the defaults first, then any filters that apply
        filters_written = set()
        for factor, model_name in self.modeldefaults.items():
            rules = self.modelfilters.get(factor)
            if rules:
                for (attrib, value), model in rules:
                    filehandle.write( '{0}={1} where {2} = "{3}"\n'.format(factor,model,attrib,value) )
                filters_written.add(factor)
            filehandle.write( '{0}={1}\n'.format(factor, model_name) )
                
        #make sure we write all the filters out		
        for factor, rules in self.modelfilters.items():
            if factor not in filters_written:
                for (attrib, value), model in rules:
                    filehandle.write( '{0}={1} where {2} = "{3}"\n'.format(factor,model,attrib,value) )
                            
    def Search (self, factor, actual_factor ):
        price_factor_type = factor.type+self.valid_subtype.get( actual_factor.get('Sub_Type'), '' )
        #look for a filter rule
        rule = self.modelfilters.get(price_factor_type)
        if rule:
            factor_attribs = dict(actual_factor, **{'ID':'.'.join(factor.name)})
            for (attrib, value), model in rule:
                if (factor_attribs.get(attrib.strip()) == value.strip()):
                    return model
        return self.modeldefaults.get(price_factor_type)
    
class AAJsonEncoder(json.JSONEncoder):
            
    def default(self, obj):
        return_value = {'.Unknown':str(type(obj))}
        if isinstance(obj,Utils.Curve):
            return_value = {'.Curve': {'meta':obj.meta, 'data':array_transform(obj.array)}}
        elif isinstance(obj, Deal):
            return_value  = {'.Deal': obj.field }        
        elif isinstance(obj, ModelParams):
            return_value = {'.ModelParams': {'modeldefaults':obj.modeldefaults, 'modelfilters':obj.modelfilters}}
        elif isinstance(obj, Utils.Descriptor):
            return_value  = {'.Descriptor': obj.data }
        elif isinstance(obj,Utils.Percent):
            return_value = {'.Percent': 100.0 * obj.amount }
        elif isinstance(obj,Utils.Basis):
            return_value = {'.Basis': 10000.0 * obj.amount}
        elif isinstance(obj,Utils.Offsets):
            return_value = {'.Offsets':obj.data}
        elif isinstance(obj,Utils.CreditSupportList):
            return_value = {'.CreditSupportList':[ [rating,value] for rating, value in obj.data.items() ]}
        elif isinstance(obj,Utils.DateEqualList):
            return_value = {'.DateEqualList': [ [date.strftime('%Y-%m-%d')]+list(value) for date, value in obj.data.items() ] }
        elif isinstance(obj,Utils.DateList):
            return_value = {'.DateList': [ [date.strftime('%Y-%m-%d'),value] for date, value in obj.data.items() ] }
        elif isinstance(obj, DateOffset):
            return_value = {'.DateOffset': obj.kwds}
        elif isinstance(obj, Timestamp):
            return_value = {'.Timestamp': obj.strftime("%Y-%m-%d")}
        return return_value

class Parser(object):
    '''
    Parser(config) - initialized with a global state config object.
    Reads (parses) an Adaptiv Analytics marketdata and deals file.
    Also writes out these files once the data has been modified.
    Provides support for parsing grids of dates and working out dynamic dates for the
    given portfolio of deals.
    '''
    
    #class level lookups
    month_lookup   = dict((m,i) for i,m in enumerate(calendar.month_abbr) if m)
    offset_lookup  = {'M':'months', 'D':'days', 'Y':'years', 'W':'weeks' }
    reverse_offset = {'months':'M', 'days':'D', 'years':'Y', 'weeks':'W'}
    
    def __init__(self, config):
        self.parent          = config        
        self.deals           = {'Deals':{'Children':[]}}
        self.calibrations    = ElementTree(Element(tag='CalibrationConfig'))
        self.calendars       = ElementTree(Element(tag='Calendars'))
        self.holidays        = {}
        self.archive         = None
        self.archive_columns = {}
        
        #setup an empty calibration config
        for elem_tag in ['MarketDataArchiveFile','DealsFile','Calibrations','BootstrappingPriceFactorSelections']:
            self.calibrations.getroot().append( Element(tag=elem_tag) )
        
        #the default state of the system
        self.version         = None
        self.params          = {
                                'System Parameters' :
                                     {'Base_Currency':'USD',
                                      'Description': '',
                                      'Base_Date': '',
                                      'Exclude_Deals_With_Missing_Market_Data':'Yes',
                                      'Proxying_Rules_File': '',
                                      'Script_Base_Scenario_Multiplier': 1,
                                      'Correlations_Healing_Method':'Eigenvalue_Raising',
                                      'Grouping_File': ''
                                      },
                                 'Model Configuration': ModelParams(),
                                 'Price Factors': {},
                                 'Price Factor Interpolation': ModelParams(),
                                 'Price Models': {},
                                 'Correlations': {},
                                 'Market Prices': {}
                               }

        #make sure that there are no default calibration mappings
        self.calibration_process_map = {}

        self.parser, self.lineparser, self.gridparser, self.periodparser, self.assignparser = self.Grammer()

    def ParseGrid(self, run_date, max_date, grid, past_max_date=False):
        '''
        Construct a set of dates (NOT adjusted to the next business day) as specified in the grid.
        Dates are capped at max_date (but may include the next date after if past_max_date is True)
        '''
        
        offsets     = [self.gridparser.parseString(offset)[0] for offset in grid.upper().split()]
        fixed_dates = [ (run_date+code[0],code[1] if len(code)>1 else None) for code in offsets ] + [(Timestamp.max, None)]
        dates       = set()
        finish      = False
        
        for date_rule, next_date_rule in zip(fixed_dates[:-1], fixed_dates[1:]):
            next_date = date_rule[0]
            if next_date>max_date:
                break
            else:
                dates.add(next_date)
            if date_rule[1]:
                while True:
                    next_date= next_date+date_rule[1]
                    if next_date>max_date:
                        finish=True
                        break
                    if next_date>next_date_rule[0]:
                        break
                    dates.add(next_date)
                    
            if finish:
                break
                    
        if past_max_date:
            dates.add(next_date)

        #bus_dates = set()
        #bus_day   = pd.offsets.BDay(1)
        #for date in dates:
        #    bus_dates.add(bus_day.rollforward(date))
        return dates

    def ParseCalendarfile(self, filename):
        '''
        Parses the xml calendar file in filename
        '''
        self.holidays  = {}
        self.calendars = ElementTree(file=filename)
        for elem in self.calendars.getiterator():
            if elem.attrib.get('Location'):
                holidays = dict(tuple(x.split("|")) for x in elem.attrib['Holidays'].split(', '))
                self.holidays[ elem.attrib['Location'] ] = {'businessday':pd.tseries.offsets.CustomBusinessDay(holidays=holidays.viewkeys(), weekmask=Utils.WeekendMap[elem.attrib['Weekends']]), 'holidays':holidays }
        
    def ParseCalibrationfile(self, filename):
        '''
        Parses the xml calibration file in filename (and loads up the .ada file)
        '''
        self.calibrations = ElementTree(file=filename)
        for elem in self.calibrations.getroot():
            if elem.tag=='MarketDataArchiveFile':
                self.archive = pd.read_csv(elem.text, skiprows=3, sep='\t', index_col=0)
            elif elem.tag=='Calibrations':
                for calibration in elem:
                    param = OrderedDict(self.lineparser.parseString(calibration.text).asList())
                    calibration.calibration = ConstructCalibrationConfig(calibration.attrib, param)
                    
        #self.calibration_process_map = { x.calibration.model : x.calibration for x in self.calibrations.find('Calibrations') }
        self.calibration_process_map = dict( (x.calibration.model, x.calibration) for x in self.calibrations.find('Calibrations') )
        #store a lookup to all columns
        self.archive_columns     = {}
        
        for col in self.archive.columns:
            self.archive_columns.setdefault(col.split(',')[0],[]).append(col)
                    
    def Fetch_AllCalibrationFactors(self):
        '''
        Assumes a valid marketdata.dat and calibration config xml file has been loaded (via ParseMarketfile and ParseCalibrationfile) and returns
        the list of price factors that have mapped price models.
        The return value of this method is suitable to pass to the Calibrate_factors method.
        '''
        model_factor = {}
        for factor in self.params.get('Price Factors',{}):
            price_factor = Utils.CheckRateName (factor)
            model = self.params['Model Configuration'].Search( Utils.Factor(price_factor[0], price_factor[1:]) , self.params['Price Factors'][factor] )
            if model:
                subtype                 = self.params['Price Factors'][factor].get('Sub_Type')
                model_name              = Utils.Factor( model, price_factor[1:] )
                archive_name            = Utils.Factor( price_factor[0]+(subtype if subtype and subtype!="None" else ''), price_factor[1:] )
                model_factor[factor]    = Utils.RateInfo ( Utils.CheckTupleName(model_name), Utils.CheckTupleName(archive_name), self.calibration_process_map.get(model) )
                
        remaining_factor = {}        
        remaining_rates = set([col.split(',')[0] for col in self.archive.columns]).difference(model_factor.keys())
        for factor in remaining_rates:
            price_factor = Utils.CheckRateName (factor)
            model = self.params['Model Configuration'].Search( Utils.Factor(price_factor[0], price_factor[1:]) , {} )
            if model:
                model_name                  = Utils.Factor( model, price_factor[1:] )
                archive_name                = Utils.Factor( price_factor[0], price_factor[1:] )
                remaining_factor[factor]    = Utils.RateInfo ( Utils.CheckTupleName(model_name), Utils.CheckTupleName(archive_name), self.calibration_process_map.get(model) )
                
        return {'present':model_factor,'absent':remaining_factor}

    def Calibrate_factors ( self, from_date, to_date, factors, smooth=0.0, correlation_cuttoff=0.2, overwrite_correlations=True ):
        '''
        Assumes a valid calibration configuration file is loaded first (via method ParseCalibrationfile), then proceeds to strip out only data between
        from_date and to_date. The calibration rules as specified by the calibration configuration file is then applied to the factors given.
        Note that factors needs to be a list of Util.RateInfo (obtained via a call to Fetch_AllCalibrationFactors). Also note that this method
        overwrites the Price Model section of the config file in memory. To save the changes, an explicit call to WriteMarketfile must be made.
        '''
        correlation_names   = []
        consolidated_DF     = None
        ak                  = []
        num_indexes         = 0
        num_factors         = 0
        factor_names        = set([col.split(',')[0] for col in self.archive.columns])
        total_rates         = reduce(operator.concat, [self.archive_columns[rate.archive_name] for rate in factors.values()], [])
        factor_data         = Utils.Filter_DF(self.archive, from_date, to_date)[ total_rates ]
        
        for rate_name, rate_value in sorted(factors.items()):
            df              = factor_data[ [col for col in factor_data.columns if col.split(',')[0]==rate_value.archive_name] ]
            #now remove spikes
            data_frame      = df[np.abs(df-df.mean())<=(smooth*df.std())].interpolate(method='index') if smooth else df
            #log it
            logging.info('Calibrating {0} (archive name {1}) with raw shape {2} and cleaned non-null shape {3}'.format(rate_name, rate_value.archive_name, str(df.shape), str(data_frame.dropna().shape)))
            #plot it
            #data_frame.plot(legend=False, title=rate_name)
            
            #calibrate
            try:
                result          = rate_value.calibration.calibrate ( data_frame, num_business_days=252.0 )
            except:
                logging.error('Data errors in factor {0} resulting in flawed calibration - skipping'.format(rate_value.archive_name))
                continue
            
            #check that it makes sense . . .
            if (np.array(result.correlation).max()>1) or (np.array(result.correlation).min()<-1) or (result.delta.std()==0).any():
                logging.error('Data errors in factor {0} resulting in incorrect correlations - skipping'.format(rate_value.archive_name))
                continue
                #raise Exception()

            model_tuple  = Utils.CheckRateName(rate_value.model_name)
            model_factor = Utils.Factor(model_tuple[0], model_tuple[1:])
            
            #get the correlation name
            process_name, addons = ConstructProcess(model_factor.type, None, result.param).CorrelationName()
            
            for sub_factors in addons:
                correlation_names.append ( Utils.CheckTupleName ( Utils.Factor ( process_name, model_factor.name+sub_factors ) ) )            
            
            consolidated_DF = result.delta if consolidated_DF is None else pd.concat( [consolidated_DF, result.delta], axis=1 )
            #store the correlation coefficients
            ak.append( result.correlation )
            
            #store result back in .dat file
            self.params['Price Models'] [ rate_value.model_name ] =  result.param
            
            num_indexes     += result.delta.shape[1]
            num_factors     += rate_value.calibration.num_factors

        a       = np.zeros ( ( num_factors, num_indexes ) )
        rho     = consolidated_DF.interpolate(method='index').bfill().corr()
        offset  = 0
        row_num = 0
        for coeff in ak:
            for factor_index, factor in enumerate(coeff):
                a[row_num+factor_index,offset:offset+len(factor)] = factor
            row_num+=len(coeff)
            offset+=len(factor)
            
        #cheating here    
        factor_correlations = a.dot(rho).dot(a.T).clip(-1.0, 1.0)
        
        #see if we need to delete the old correlations
        if overwrite_correlations:
            self.params['Correlations'] = {}
            
        for index1 in range(len(correlation_names)-1):
            for index2 in range(index1+1, len(correlation_names)):
                if np.fabs(factor_correlations[index1,index2])>correlation_cuttoff:
                    self.params['Correlations'][ ( correlation_names[index1], correlation_names[index2] ) ] = factor_correlations[index1,index2]
        
    def CalculateDependencies(self, report_currency, base_date, base_MTM_dates, calc_dates=True):
        '''
        Works out the risk factors (and risk models) in the given set of deals.
        These factors are cross referenced in the marketdata file and matched by
        name. This needs to be extended to Equity and Commodity deals.

        Returns the dependant factors, the stochastic models, the full list of
        reset dates and optionally the potential currency settlements
        '''
        def get_rates(factor, instrument):
            rates_to_add    = {factor:[]}
            factor_name     = Utils.CheckTupleName(factor)
            
            if factor.type in dependant_fields:
                linked_factors  = [ Utils.Factor ( dep_field[1], Utils.CheckRateName ( self.params['Price Factors'][factor_name][dep_field[0]] ) ) for dep_field in dependant_fields[factor.type] if self.params['Price Factors'][factor_name][dep_field[0]] ]
                for linked_factor in linked_factors:
                    #add it assuming no dependencies
                    rates_to_add.setdefault(linked_factor,[])
                    #update and dependencies
                    if linked_factor.type in dependant_fields:
                        rates_to_add.update ( get_rates(linked_factor, instrument) )
                    #link it to the original factor
                    rates_to_add[factor].append(linked_factor)
                    
                    #check that we include any nested factors
                    if linked_factor.type in nested_fields:
                        for i in range(1,len(linked_factor.name)+1):    
                            rates_to_add.update( { Utils.Factor(linked_factor.type, linked_factor.name[:i] ): [ Utils.Factor(linked_factor.type, linked_factor.name[:i-1]) ] if i>1 else [] } )
                    
            if factor.type in nested_fields:
                for i in range(1,len(factor.name)+1):
                    rates_to_add.update( { Utils.Factor(factor.type, factor.name[:i] ): [ Utils.Factor(factor.type, factor.name[:i-1]) ] if i>1 else [] } )
                        
            if factor.type in conditional_fields:
                for conditional_factor in conditional_fields[factor.type](instrument, self.params['Price Factors'][factor_name], self.params['Price Factors']):
                    rates_to_add[factor].append(conditional_factor)
                    rates_to_add.update ( {conditional_factor:[]} )
                
            return rates_to_add
            
        def walkGroups(deals, price_factors):            
                    
            def get_price_factors(rates_to_add, instrument):
                for field_name, factor_types in instrument.factor_fields.items():
                    field_value = Utils.GetFieldName(field_name, instrument.field)
                    if field_value:
                        for factor in [ Utils.Factor ( factor_type,  Utils.CheckRateName ( field_value ) ) for factor_type in factor_types ]:
                            if factor not in rates_to_add:
                                try:
                                    rates_to_add.update ( get_rates(factor, instrument) )
                                except KeyError,e:
                                    logging.warning('Price Factor {0} not found in market data'.format(e))
                                    if factor.type=='DiscountRate':
                                        logging.info( 'Creating default Discount {0}'.format(factor) )
                                        self.params['Price Factors'][ Utils.CheckTupleName(factor) ] = OrderedDict( [ ('Interest_Rate', '.'.join(factor.name) ), ('Property_Aliases', None) ] )
                                        #try again
                                        rates_to_add.update ( get_rates(factor, instrument) )
                                    else:
                                        logging.error('Skipping Price Factor')

            resets                  = set()
            children                = []
            settlement_currencies   = {}
            
            for node in deals:
                #get the instrument
                instrument = node['instrument']
                
                if node.get('Ignore')=='True':
                    continue
                
                #get a list of children ready to pass back to the parent
                children.append(instrument)
                #get it's price factors
                get_price_factors ( price_factors, instrument )
                    
                if node.get('Children'):
                    node_children, node_resets, node_settlements  = walkGroups ( node['Children'], price_factors )
                    
                    #sort out dates and calendars
                    instrument.reset(self.holidays)
                    if calc_dates:
                        instrument.finalize_dates ( self.ParseGrid, base_date, base_MTM_dates, node_children, node_resets, node_settlements )

                    #merge dates
                    resets.update( node_resets )
                    for key, value in node_settlements.items():
                       settlement_currencies.setdefault( key, set() ).update( value )
                else:
                    #sort out dates and calendars
                    instrument.reset(self.holidays)
                    if calc_dates:
                        instrument.finalize_dates ( self.ParseGrid, base_date, base_MTM_dates, None, resets, settlement_currencies )
                    
            return children, resets, settlement_currencies
                    
        #derived fields are fields that embed other risk factors 
        dependant_fields = {'FxRate':[('Interest_Rate','InterestRate')],
                            'DiscountRate':[('Interest_Rate','InterestRate')],
                            'ForwardPrice':[('Currency','FxRate')],
                            'ReferencePrice':[('ForwardPrice','ForwardPrice')],
                            'ReferenceVol':[('ForwardPriceVol','ForwardPriceVol'),('ReferencePrice','ReferencePrice')],
                            'InflationRate':[('Price_Index','PriceIndex')],
                            'EquityPrice':[('Interest_Rate','InterestRate'),('Currency','FxRate')]}
        
        #nested fields need to include all their children 
        nested_fields      = set(['InterestRate'])
        
        conditional_fields = {'ReferenceVol' : lambda instrument, factor_fields, params: [ Utils.Factor('Correlation', tuple( 'FxRate.{0}.{1}/ReferencePrice.{2}.{0}'.format(params[ Utils.CheckTupleName(Utils.Factor('ForwardPrice',(instrument.field['Reference_Type'],))) ]['Currency'],instrument.field['Currency'],instrument.field['Reference_Type']).split('.') ) ) ] if instrument.field['Currency']!=params[ Utils.CheckTupleName(Utils.Factor('ForwardPrice',(instrument.field['Reference_Type'],))) ]['Currency'] else [],
                              'ForwardPrice' : lambda instrument, factor_fields, params: [ Utils.Factor('FXVol', tuple ( sorted ( [instrument.field['Currency'], factor_fields['Currency']] ) ) ) ] if instrument.field['Currency'] != factor_fields['Currency'] else [] }
        
        #the list of returned factors
        dependend_factors  = set()
        stochastic_factors = OrderedDict()

        #complete list of reset dates referenced
        reset_dates = set()
        #complete list of currency settlement dates
        currency_settlement_dates = {}

        #only if we have a portfolio of trades can we calculate its dependencies
        if self.deals:
            #add the base Fx rate
            dependend_factors = get_rates ( Utils.Factor ( 'FxRate', Utils.CheckRateName ( self.params['System Parameters']['Base_Currency'] ) ), {} )
            
            #grab all the factor fields in the portfolio
            dependant_deals, reset_dates, currency_settlement_dates = walkGroups ( self.deals['Deals']['Children'], dependend_factors )
            
            #additional FX factors for the reporting currency
            dependend_factors.update ( get_rates ( Utils.Factor ( 'FxRate', Utils.CheckRateName ( report_currency ) ), {} ) )
            
            #now sort the factors taking any factor dependencies into account
            dependend_factors = Utils.Topolgical_Sort(dependend_factors)
            
            #now lookup the processes
            for factor in dependend_factors:
                stoch_proc		= self.params['Model Configuration'].Search ( factor, self.params['Price Factors'].get( Utils.CheckTupleName(factor), {} ) )
                if stoch_proc and factor.name[0] != self.params['System Parameters']['Base_Currency']:
                    factor_model = Utils.Factor (stoch_proc, factor.name) 
                    if Utils.CheckTupleName(factor_model) in self.params['Price Models']:
                        stochastic_factors.setdefault(factor_model,factor)
                    else:
                        #check if there's any market price data we need to process 
                        implied_params = self.params['Market Prices'].get('.'.join(('GBMTSModelPrices',)+factor.name))
                        if factor_model.type=='GBMAssetPriceTSModelImplied' and implied_params:
                            logging.info( 'Creating/Bootstrapping Implied model parameters {0}'.format(factor) )
                            
                            #note - this only works for fx vols at the moment - need to generalize to bootstrap other instruments
                            vol_factor = 'FXVol.{0}'.format(implied_params['instrument']['Asset_Price_Volatility'])
                            
                            #This is a hack to just set the implied fx vol the same as the calibrated (historical) one - this needs to be handled correctly
                            if vol_factor not in self.params['Price Factors']:
                                self.params['Price Factors'][vol_factor] = {'Surface':Utils.Curve([],[[1.0, 1.0, self.params['Price Models']['GBMAssetPriceModel.{0}'.format(factor_model.name[0])]['Vol'] ]])}
                            
                            fxvol      = ConstructFactor ( 'FXVol', self.params['Price Factors'][vol_factor] )
                            mn_ix      = np.searchsorted(fxvol.moneyness,1.0)
                            atm_vol    = [ np.interp ( 1, fxvol.moneyness[mn_ix-1:mn_ix+1], y ) for y in fxvol.GetVols()[:,mn_ix-1:mn_ix+1] ]
                            self.params['Price Models'][ Utils.CheckTupleName(factor_model) ] = OrderedDict ( [ ('Property_Aliases', None), ('Quanto_FX_Volatility', None ), ('Vol', Utils.Curve([],zip(fxvol.expiry, atm_vol)) ), ('Quanto_FX_Correlation', 0 ) ] )
                            
                            stochastic_factors.setdefault(factor_model,factor)
                        else:
                            logging.error( 'Risk Factor {0} using stochastic process {1} missing in Price Models section'.format( Utils.CheckTupleName(factor), stoch_proc ) )
                            
        return set(dependend_factors), stochastic_factors, reset_dates, currency_settlement_dates
        
    def ParseMarketfile(self, filename):
        '''Parses the AA marketdata .dat file in filename'''
        
        self.params = OrderedDict()
        self.parser.parseFile(filename)
        
    def ParseTradefile(self, filename):
        '''
        Parses the xml .aap file in filename
        '''
        self.deals  = {'Deals':{'Children':[]}}
        deals       = [self.deals['Deals']['Children']]
        path        = []
        attrib      = []
        
        for event, elem in iterparse(filename, events=('start', 'end')):
            
            if event == 'start':
                path.append (elem.tag)
                attrib.append (elem.attrib)
                
            elif event == 'end':
                # process the tag
                if (elem.text and elem.text.strip()!=''):
                    #get the field data
                    fields      = OrderedDict(self.lineparser.parseString(elem.text).asList())
                    if path[-2:]==['Deal', 'Properties']:                       
                        #make a new node
                        node    = { 'instrument':ConstructInstrument(fields), 'Children':[] }
                        #add it to the collection
                        deals[-1].append( node )
                        #add any modifiers
                        node.update(attrib[-2])
                        #go deeper
                        deals.append( node['Children'] )
                            
                    elif path[-2:]==['Deals', 'Deal']:
                        #make a new node    
                        node    = { 'instrument':ConstructInstrument(fields) }
                        #add it
                        deals[-1].append( node )
                        #add any modifiers
                        node.update(elem.attrib)
                        
                if path[-2:]==['Deal', 'Deals']:
                    #go back
                    deals.pop()

                path.pop()
                attrib.pop()
                
    def WriteTradefile(self, filename):
        def ammend(xmlparent, internal_deals):
            for node in internal_deals:
                instrument  = node['instrument']
                deal        = Element(tag='Deal')
                
                if node.get('Ignore'):
                    deal.attrib['Ignore'] = node['Ignore']
                    
                if node.get('Children'):
                    properties  = Element(tag='Properties')
                    properties.text = drawobj(instrument.field)
                    deal.append(properties)
                    deals       = Element(tag='Deals')
                    deal.append(deals)
                    ammend( deals, node['Children'] )
                else:
                    deal.text   = drawobj(instrument.field)
                    
                xmlparent.append(deal)
                
        '''Writes the state of these deals to filename (loosly compatible with adaptiv analytics)'''
        xml_deals  = ElementTree(Element(tag='Deals', attrib ={'AnalyticsVersion':'151.1.3.0'}))
        ammend(xml_deals.getroot(), self.deals['Deals']['Children'])
        xml_deals.write(filename)

    def WriteMarketfile(self, filename):
        '''Writes out the internal state of this config object out to filename'''
        
        with open(filename, 'wt') as f:
            f.write('='.join(self.version)+'\n\n')
            for k,v in self.params.items():
                f.write('<{0}>'.format(k)+'\n')
                if k=='Correlations':
                    for (f1, f2), corr in sorted(v.items()):
                        f.write(','.join([f1, f2, str(corr)])+'\n')
                elif k=='Market Prices':
                    for rate, data in v.items():
                        points = []
                        for point in data['Children']:
                            points.append ( copy_dict(point['quote'], {'Deal':point['instrument']}) )
                        #rewrite the madness
                        f.write( rate + ',' + drawobj ( copy_dict( data['instrument'], {'Points':points} ) ) + '\n')
                elif k in ['Price Factors', 'Price Models']:
                    for key, value in sorted(v.items()):
                        f.write( key + ',' + drawobj ( value ) + '\n')
                elif isinstance(v, dict):
                    for key, value in sorted(v.items()):
                        f.write(key+'='+('' if value is None else str(value))+'\n')
                elif v.__class__.__name__=='ModelParams':
                    v.WriteToFile(f)
                else:
                    raise Exception("Unknown model section {0} in writing market data {1}".format(k,filename))
                f.write('\n')
                
    def ParseJson(self, filename):
        
        def as_internal(dct):
            
            if '.Curve' in dct:
                return Utils.Curve ( dct['.Curve']['meta'], dct['.Curve']['data'] )
            elif '.Percent' in dct:
                return Utils.Percent ( dct['.Percent'] )
            elif '.Deal' in dct:              
                return ConstructInstrument(dct['.Deal'])
            elif '.Basis' in dct:
                return Utils.Basis(dct['.Basis'])
            elif '.Descriptor' in dct:
                return Utils.Descriptor(dct['.Descriptor'])
            elif '.DateList' in dct:
                return Utils.DateList ( OrderedDict([(Timestamp(date),val) for date,val in dct['.DateList']]) )
            elif '.DateEqualList' in dct:
                return Utils.DateEqualList ( [ [Timestamp(values[0])]+values[1:] for values in dct['.DateEqualList']] )
            elif '.CreditSupportList' in dct:   
                return Utils.CreditSupportList ( dct['.CreditSupportList'] )
            elif '.DateOffset' in dct:
                return DateOffset( **dct['.DateOffset'] )
            elif '.Offsets' in dct:
                return Utils.Offsets( dct['.Offsets'] )
            elif '.Timestamp' in dct:
                return Timestamp( dct['.Timestamp'] )
            elif '.ModelParams' in dct:
                return ModelParams( ( dct['.ModelParams']['modeldefaults'], dct['.ModelParams']['modelfilters'] ) )
            
            return dct

        with open(filename, 'rt') as f:
            data = json.load(f, object_hook=as_internal)
            
        if 'MarketData' in data:
            market_data = data['MarketData']
            correlations = {}
            for rate1,rate_list in market_data['Correlations'].items():
                for rate2, correlation in rate_list.items():
                    correlations.setdefault ( ( rate1, rate2 ), correlation )
                    
            #update the correlations
            market_data['Correlations']  = correlations
            self.params                  = market_data
            self.version                 = data['Version']
            
        elif 'Deals' in data:
            self.deals = data

    def ProcessHedgeCounterparties(self, hedgeparties):
        hedge   = []
        netting = []
        for party in hedgeparties:
            if party.endswith('.json'):
                self.ParseJson(party)
                if not hedge:
                    #make sure the hedge netting set has at least 1 deal 
                    hedge = self.deals['Deals']['Children'][0]['Children']
                    for deal_num, deal in enumerate(hedge):
                        deal['instrument'].field['Reference']='Hedge_{0}'.format(deal_num)
                else:
                    #name the netting set
                    self.deals['Deals']['Children'][0]['instrument'].field['Reference'] = os.path.split(party[:party.rfind('.')])[-1]
                    #add the hedge
                    self.deals['Deals']['Children'][0]['Children'].extend ( hedge )
                    #add it to the total netting sets
                    netting.extend( self.deals['Deals']['Children'] )
            else:
                logging.error('Skipping filename {0} - only .json files are supported for hedging'.format(party))
                
        #reset the deals
        self.deals['Deals']['Children'] = netting
                
    def WriteMarketdataJson(self, json_filename):
        #backup old data
        old_correlations  = self.params['Correlations']
        
        #need to serialize out new data
        correlations = {}
        for correlation, value in old_correlations.items():
            correlations.setdefault(correlation[0],{}).setdefault(correlation[1], value)
                
        #create new keys for json serialization
        self.params['Correlations']  = correlations
        
        with open(json_filename, 'wt') as f:
            f.write ( json.dumps ( {'MarketData':self.params, 'Version':self.version}, separators=(',', ':'), cls=AAJsonEncoder ) )
        
        #restore state
        self.params['Correlations']  =  old_correlations

    def WriteTradedataJson(self, json_filename):
        with open(json_filename, 'wt') as f:
            f.write ( json.dumps ( self.deals, separators=(',', ':'), cls=AAJsonEncoder ) )
                
    def ConvertMarketDataJson(self, dat_filename, json_filename):
        #read in the market_data_file
        self.ParseMarketfile(dat_filename)
        #write out a json file
        self.WriteMarketdataJson(json_filename)
            
    def ConvertTradeDataJson(self, aap_filename, json_filename):
        #read in the trade_data_file
        self.ParseTradefile(aap_filename)
        #write out a json file
        self.WriteTradedataJson(json_filename)
                
    def Grammer(self):
        '''
        Contains the grammer definition rules for parsing the market data file.
        Mostly complete but may need to be extended as needed.
        '''

        def pushDate(strg, loc, toks ):
            return Timestamp( '{0}-{1:02d}-{2:02d}'.format(toks[2], Parser.month_lookup[toks[1]], int(toks[0]) ) )
        
        def pushInt(strg, loc, toks ):
            return int(toks[0])
        
        def pushFloat(strg, loc, toks ):
            return float(toks[0])

        def pushPercent(strg, loc, toks ):
            return Utils.Percent(toks[0])
        
        def pushBasis(strg, loc, toks ):
            return Utils.Basis(toks[0])
        
        def pushIdent(strg, loc, toks ):
            return (toks[0], None if len(toks)==1 else toks[1])
        
        def pushChain(strg, loc, toks ):
            return OrderedDict({toks[0]: toks[1]})
        
        def pushPeriod(strg, loc, toks ):
            ofs = dict( [ (Parser.offset_lookup[toks[2*i+1]],toks[2*i]) for i in range(len(toks)/2) ] )
            return DateOffset(**ofs)
        
        def pushID(strg, loc, toks ):
            entry = OrderedDict()
            for k,v in toks[1:]:
                entry[k]=v
            return toks[0], entry
        
        def pushCurve(strg, loc, toks ):
            '''need to allow differentiation between adding a spread and scaling by a factor'''
            return Utils.Curve([], toks[0].asList()) if len(toks)==1 else Utils.Curve(toks[:-1], toks[-1].asList() )

        def pushList(strg, loc, toks ):
            return toks.asList()
        
        def pushOffset(strg, loc, toks ):
            return Utils.Offsets(toks[0].asList())
        
        def pushDateGrid(strg, loc, toks ):
            return toks[0] if len(toks)==1 else Utils.Offsets(toks.asList(), grid=True)
        
        def pushDescriptor(strg, loc, toks ):
            return Utils.Descriptor( toks[0].asList() )

        def pushDateList(strg, loc, toks ):
            return Utils.DateList(toks.asList())
        
        def pushDateEqualList(strg, loc, toks ):
            return Utils.DateEqualList(toks.asList())
        
        def pushCreditSupportList(strg, loc, toks ):
            return Utils.CreditSupportList(toks.asList())
        
        def pushObj(strg, loc, toks ):
            obj = OrderedDict()
            for token in toks.asList():
                if isinstance(token, OrderedDict) or isinstance(token, list) or isinstance(token, str) or isinstance(token, Utils.Curve):
                    obj.setdefault('_obj', []).append(token)
                else:
                    key, val = token
                    obj.setdefault(key, val)
            return [obj.get('_obj', obj)]

        def pushTuple(strg, loc, toks ):
            return tuple(toks[0])
        
        def pushName(strg, loc, toks ):
            return '.'.join(toks[0]) if len(toks[0])>1 else toks[0][0]
        
        def pushKeyVal(strg, loc, toks ):
            return (toks[0], toks[1].rstrip())
        
        def pushRule(strg, loc, toks ):
            return (toks[0], toks[1].strip()[1:-1])

        def pushMdlCfg(strg, loc, toks ):
            return (toks[0], toks[1].rstrip(), toks[2] if len(toks)>2 else ())
        
        def pushSection(strg, loc, toks ):
            if toks[0]=='<Correlations>':
                #self.params.setdefault( toks[0][1:-1], { (p1,p2):c for p1,p2,c in toks[1:] } )
                self.params.setdefault( toks[0][1:-1], dict( ( (p1,p2),c ) for p1,p2,c in toks[1:] ) )
            elif toks[0] in ['<Model Configuration>','<Price Factor Interpolation>']:
                #need to filter out paramters
                model_params = ModelParams()
                for elem in toks[1:]:
                    model_params.Append(elem[0], elem[2] if len(elem[-1])>0 else (), elem[1] )
                self.params.setdefault(toks[0][1:-1], model_params)
            elif toks[0]=='<Market Prices>':
                #change the way market prices are expressed due AA being fucking insane
                market_prices = {}
                for rate, data in OrderedDict(toks[1:]).items():
                    market_prices[rate] = {'instrument': OrderedDict((k,v) for k,v in data.items() if k<>'Points')}
                    children            = market_prices[rate].setdefault('Children', [])
                    for point in data.get('Points',[]):
                        children.append( {'quote':OrderedDict((k,v) for k,v in point.items() if k<>'Deal'), 'instrument':point['Deal'] } )
                self.params.setdefault(toks[0][1:-1], market_prices )
            else:
                self.params.setdefault(toks[0][1:-1], OrderedDict(toks[1:]))
            return toks[0][1:-1]
        
        def pushCorrel(strg, loc, toks ):
            return tuple(toks)

        def pushConfig(strg, loc, toks ):
            self.version = [x.rstrip() for x in toks[:2]]
        
        e		    = CaselessLiteral( "E" )
        undef       = Keyword( "<undefined>" )
        reference   = Keyword( "Reference" )
        
        #headings
        AAformat   = Keyword( "AnalyticsVersion" )
        correl_sec = Keyword( "<Correlations>" )
        market_p   = Keyword( "<Market Prices>" )
        bootstrap  = Keyword( "<Bootstrapper Configuration>" )
        valuation  = Keyword( "<Valuation Configuration>" )
        price_mod  = Keyword( "<Price Models>" )
        factor_int = Keyword( "<Price Factor Interpolation>" )
        price_fac  = Keyword( "<Price Factors>" )
        model_cfg  = Keyword( "<Model Configuration>" )
        sys_param  = Keyword( "<System Parameters>" )
        
        #reserved words
        where      = Keyword( "where" ).suppress()
        
        equals 	 = Literal( "=" ).suppress()
        null     = Empty().suppress()
        eol      = LineEnd().suppress()
        
        lapar  		= Literal( "<" ).suppress()
        rapar  		= Literal( ">" ).suppress()
        lsqpar  	= Literal( "[" ).suppress()
        rsqpar  	= Literal( "]" ).suppress()
        lpar  		= Literal( "(" ).suppress()
        rpar  		= Literal( ")" ).suppress()
        percent 	= Literal( "%" ).suppress()
        backslash	= Literal( '\\' ).suppress()
        decimal		= Literal( "." )
        comma 		= Literal( "," ).suppress()

        ident	    = Word(alphas, alphas+nums+"_-")
        desc	    = Word(alphas, alphas+nums+", %_-.").setName('Description')
        arbstring   = Word(alphas+nums+'_', alphas+nums+"/*_ :+-()#").setName('ArbString')
        namedId     = Group ( delimitedList ( arbstring, delim='.', combine=False) ).setName('namedId').setParseAction(pushName)
        integer     = ( Word ( "+-"+nums, nums ) + ~decimal ).setName('int').setParseAction(pushInt)
        fnumber	    = Combine( Word ( "+-"+nums, nums ) + Optional( decimal + Optional( Word ( nums ) ) ) + Optional( e + Word ( "+-"+nums, nums ) ) ).setName('float').setParseAction(pushFloat)
        date        = ( integer + oneOf( list(calendar.month_abbr)[1:] ) + integer ).setName('date').setParseAction(pushDate)
        dateitem    = Group ( date + equals + fnumber ).setParseAction(pushTuple)
        dateitemfx  = Group ( date + OneOrMore ( equals+fnumber ) ).setParseAction(pushTuple)
        credit_item = Group ( integer + equals + fnumber + backslash ).setParseAction(pushTuple)
        
        datelist    = ( backslash | delimitedList ( dateitem, delim=backslash ) + Optional(backslash) ).setParseAction(pushDateList)
        datelistdel = ( lsqpar + Optional ( delimitedList ( dateitemfx ) ) + rsqpar ).setParseAction(pushDateEqualList)
        creditlist  = ( backslash | OneOrMore ( credit_item ) ).setParseAction(pushCreditSupportList)
        
        #WHY DO I need to put a ' ' here?
        percentage  = ( fnumber + ZeroOrMore(' ') + percent ).setName('percent').setParseAction(pushPercent)
        basis_pts   = ( fnumber + ' bp' ).setName('bips').setParseAction(pushBasis)
        point       = (lpar + Group ( delimitedList(fnumber) ) + rpar).setParseAction(pushTuple)
        period      = OneOrMore( integer + oneOf(['D','M','Y','W'], caseless=True)).setName('period').setParseAction(pushPeriod)
        descriptor  = Group ( integer + Literal( 'X' ).suppress() + integer ).setParseAction(pushDescriptor)
        
        obj        = Forward()
        chain      = (ident + equals + obj).setParseAction(pushChain)
        listofstf  = delimitedList( lsqpar + Group( delimitedList (date|percentage|fnumber|namedId) ) + rsqpar ).setParseAction(pushList)
        
        curve   = ( lsqpar + Optional ( delimitedList (integer|fnumber|ident) + comma ) + Group ( delimitedList(point) ) + rsqpar ).setParseAction(pushCurve)
        tenors  = ( lsqpar + Group ( delimitedList(period) ) + rsqpar ).setParseAction(pushOffset)
        grid    = delimitedList(period, delim=' ').setParseAction(pushDateGrid)
        assign  = ( ( reference + equals + namedId ) | ( ident + equals + ( chain | creditlist | datelistdel | datelist | date | grid | percentage | basis_pts | descriptor | fnumber | namedId | undef | curve | tenors | obj | null ) ) ).leaveWhitespace().setParseAction(pushIdent)
        obj    << ( lsqpar + ( delimitedList(curve) | listofstf | delimitedList ( assign | OneOrMore(obj) ) | desc ) + rsqpar ).setParseAction(pushObj)
        line    = ( namedId + comma + delimitedList(assign) + eol ).setParseAction(pushID)
        correl  = ( namedId + comma + namedId + comma + fnumber ).setParseAction(pushCorrel)

        header   = AAformat + equals + restOfLine
        todo     = (~lapar + ident + equals + restOfLine ).setParseAction(pushKeyVal)
        rule     = (~lapar + ident + equals + restOfLine ).setParseAction(pushRule)
        modelcfg = (~lapar + ident + equals + ident + Optional( where + rule ) ).setParseAction(pushMdlCfg)
        
        section  = ( ( correl_sec + ZeroOrMore(correl) ) |
                     ( ( sys_param|bootstrap|valuation ) + ZeroOrMore(todo) ) |
                     ( ( model_cfg |factor_int ) + ZeroOrMore(modelcfg) ) |
                     ( ( price_fac|price_mod|market_p ) + ZeroOrMore(line) ) ).setParseAction(pushSection)

        marketfile = ( header + OneOrMore(section) ).setParseAction(pushConfig)

        marketfile.ignore(cppStyleComment)

        return marketfile, delimitedList(assign), Group ( period+Optional( lpar + period + rpar ) ), period, line
    
if __name__=='__main__':
##    test1=r'RAT=123,Amortisation=07Aug2015=555500.000000\09Nov2015=555500.000000\08Feb2016=555500.000000\09May2016=555500.000000\08Aug2016=556000.000000\07Nov2016=555500.000000\07Feb2017=555500.000000\08May2017=555500.000000,Is_Digital=No,Digital_Recovery=0%'
##    aa=Parser(None)
##    b=dict(aa.assignparser.parseString(test1).asList())
##    drawobj(b)
##    exit(0)
    
    if 1 or 'aa' not in locals():
        rundate = '2016-05-04'
        #path    = 'C:\\utils\\Synapse\\binary_trading\\notebooks\\CCR\\CRSTAL\\{0}\\{1}'
        path    = 'G:\\Credit Quants\\CRSTAL\\Arena\\{0}\\{1}'
        aa 		= Parser(None)
        #aa.ParseCalibrationfile(r'C:\Users\shuaib.osman\AppData\Local\SunGard\Adaptiv\Analytics\AAStudio\151\calibration.config')
        aa.ParseMarketfile(path.format(rundate, 'MarketDataTest.dat'))
        #aa.ProcessHedgeCounterparties([path.format(rundate, 'eskom.json'), path.format(rundate, 'CrB_JPMorgan_Chase_NYK_ISDA.json'), path.format(rundate, 'CrB_Citigroup_G_Mkt_Ldn_ISDA.json')])
        #aa.WriteTradefile(r'G:\Credit Quants\CRSTAL\Arena\2015-10-09\test.aap')
        #aa.ParseJson(path.format(rundate, 'eskom.json'))
        #aa.ParseJson(path.format(rundate, 'CrB_JPMorgan_Chase_NYK_ISDA.json'))
        
        #aa.ParseCalendarfile('G:\\Credit Quants\\CRSTAL\\Arena\\{0}'.format('calendars.cal'))
        #aa.ParseTradefile('Y:\\Treasury IT\\Arena CCR\\5. Profiles output\\Input AAJs\\{0}\\aaps\\{1}'.format(rundate, 'CrB_Bakwena_PCC_ISDA.aap'))
        #aa.ParseCalibrationfile(r'C:\Users\shuaib.osman\AppData\Local\SunGard\Adaptiv\Analytics\AAStudio\151\calibration.config')
        #aa.ParseMarketfile(r'C:\Users\shuaib.osman\Documents\AdaptivValidation\CRSTAL\Nov21.dat')
        #aa.ParseTradefile(r'C:\Users\shuaib.osman\Documents\AdaptivValidation\CRSTAL\rats2.aap')
        #aa.ParseMarketfile(r'C:\Users\shuaib.osman\Documents\AdaptivValidation\CRSTAL\March30.dat')
        #aa.ParseTradefile(r'C:\Users\shuaib.osman\Documents\AdaptivValidation\CRSTAL\inflation2.aap')
        #aa.WriteJson(r'C:\Users\shuaib.osman\Documents\AdaptivValidation\CRSTAL\energy.json')
        
        #res = json.dumps ( aa.deals.getroot()[0][1][2].instrument, cls=AAJsonEncoder )
        exit(0)

    Base_time_grid 		= '0d 2d 1w 2w 1m 3m(3m)'
    #Scenario_time_grid = '0d(1w) 1y(1m) 20y(3m)'
    Scenario_time_grid 	= '0d(1w) 6m(1m) 5y(3m)'
    
    dependend_factors, stochastic_factors, reset_dates, currency_settlement_dates = aa.CalculateDependencies ( 'ZAR', pd.Timestamp(rundate), Base_time_grid )