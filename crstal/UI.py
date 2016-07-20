import os
import sys
import json
import copy
import types
import operator
import logging
import itertools
import numpy as np
import pandas as pd
import Utils
import RiskFactors
import ipywidgets as widgets

from Config import Parser #useful parser
from Instruments import ConstructInstrument #needed to setup instruments
from StochasticProcess import ConstructProcess, ConstructCalibrationConfig #needed to setup Stochastic processes/Calibrations
from Calculation import  ConstructCalculation #needed to setup calculations
from Fields import FieldDefaults, FieldMappings

from traitlets import Unicode # Used to declare attributes of our widgets
from IPython.display import display # Used to display widgets in the notebook
from nbformat import v4 as nbf # Used to create notebooks containing documentation
from StringIO import StringIO
from collections import OrderedDict
from subprocess import call

#portfolio parent data
class DealCache:
    Json    = 0
    Deal    = 1
    Parent  = 2
    Count   = 3
    
#HandsonTable view
class Table(widgets.DOMWidget):
    _view_name = Unicode('TableView').tag(sync=True)
    _model_name = Unicode('TableModel').tag(sync=True)
    _view_module = Unicode('jupyter-crstal').tag(sync=True)
    _model_module = Unicode('jupyter-crstal').tag(sync=True)
    
    colTypes = Unicode().tag(sync=True)
    colHeaders = Unicode().tag(sync=True)
    value = Unicode().tag(sync=True)
    description = Unicode().tag(sync=True)
    
#jquery datepicker
class DateWidget(widgets.DOMWidget):
    _view_name = Unicode('DatePickerView').tag(sync=True)
    _model_name = Unicode('DatePickerModel').tag(sync=True)
    _view_module = Unicode('jupyter-crstal').tag(sync=True)
    _model_module = Unicode('jupyter-crstal').tag(sync=True)
    
    value = Unicode().tag(sync=True)
    description = Unicode().tag(sync=True)
    
#Fileloader
class FileWidget(widgets.DOMWidget):
    _view_name = Unicode('FilePickerView').tag(sync=True)
    _model_name = Unicode('FilePickerModel').tag(sync=True)
    _view_module = Unicode('jupyter-crstal').tag(sync=True)
    _model_module = Unicode('jupyter-crstal').tag(sync=True)
    
    value = Unicode().tag(sync=True)
    description = Unicode().tag(sync=True)
    filename = Unicode().tag(sync=True)

    def __init__(self, **kwargs):
        """Constructor"""
        widgets.DOMWidget.__init__(self, **kwargs) # Call the base.
        
        # Allow the user to register error callbacks with the following signatures:
        #    callback()
        #    callback(sender)
        self.errors = widgets.CallbackDispatcher(accepted_nargs=[0, 1])
        
        # Listen for custom msgs
        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, content):
        """Handle a msg from the front-end.

        Parameters
        ----------
        content: dict
            Content of the msg."""
        if 'event' in content and content['event'] == 'error':
            self.errors()
            self.errors(self)
    
#Flot graph viewer
class Flot(widgets.DOMWidget):
    _view_name = Unicode('FlotView').tag(sync=True)
    _model_name = Unicode('FlotModel').tag(sync=True)
    _view_module = Unicode('jupyter-crstal').tag(sync=True)
    _model_module = Unicode('jupyter-crstal').tag(sync=True)
    
    value = Unicode().tag(sync=True)
    description = Unicode().tag(sync=True)
    
#Interface for 3d graph viewing
class Three(widgets.DOMWidget):
    _view_name = Unicode('ThreeView').tag(sync=True)
    _model_name = Unicode('ThreeModel').tag(sync=True)
    _view_module = Unicode('jupyter-crstal').tag(sync=True)
    _model_module = Unicode('jupyter-crstal').tag(sync=True)
    
    value = Unicode().tag(sync=True)
    description = Unicode().tag(sync=True)
    
#base class for jstree
class Tree(widgets.DOMWidget):
    _view_name = Unicode('TreeView').tag(sync=True)
    _model_name = Unicode('TreeModel').tag(sync=True)
    _view_module = Unicode('jupyter-crstal').tag(sync=True)
    _model_module = Unicode('jupyter-crstal').tag(sync=True)
    
    value = Unicode('').tag(sync=True)
    type_data = Unicode('').tag(sync=True)
    selected = Unicode('').tag(sync=True)
    created = Unicode('').tag(sync=True)
    deleted = Unicode('').tag(sync=True)

#display a tree for examing profiles
class FlotTree(Tree):
    _view_name = Unicode('FlotTreeView').tag(sync=True)
    _model_name = Unicode('FlotTreeModel').tag(sync=True)

    description = Unicode().tag(sync=True)
    profiles = Unicode().tag(sync=True)

#Calculation tree		
class CalculationTree(Tree):
    _view_name = Unicode('CalculationTreeView').tag(sync=True)
    _model_name = Unicode('CalculationTreeModel').tag(sync=True)

    calculation_types = Unicode().tag(sync=True)

#Calibration tree		
class CalibrationTree(Tree):
    _view_name = Unicode('CalibrationTreeView').tag(sync=True)
    _model_name = Unicode('CalibrationTreeModel').tag(sync=True)

    calibration_types = Unicode().tag(sync=True)
    
#Riskfactor tree		
class FactorTree(Tree):
    _view_name = Unicode('FactorTreeView').tag(sync=True)
    _model_name = Unicode('FactorTreeModel').tag(sync=True)
    
    allselected = Unicode().tag(sync=True)
    risk_factors = Unicode().tag(sync=True)
    
#Portfolio tree		
class PortfolioTree(Tree):
    _view_name = Unicode('PortfolioTreeView').tag(sync=True)
    _model_name = Unicode('PortfolioTreeModel').tag(sync=True)

    instrument_def = Unicode().tag(sync=True)

#MarketPrice tree		
class MarketTree(Tree):
    _view_name = Unicode('MarketTreeView').tag(sync=True)
    _model_name = Unicode('MarketTreeModel').tag(sync=True)

    market_prices = Unicode().tag(sync=True)
    
#Code for custom pages go here..
class TreePanel(object):
    '''
    Base class for tree-based screens with a dynamic panel on the right to show data for the screen.
    '''
    def __init__(self, config):
        #load the config object - to load and store state
        self.config                  = config
        
        #load up relevant information from the config and define the tree type
        self.ParseConfig()
        
        #setup the view containters
        self.right_container         = widgets.Box()
        
        self.tree.selected = u"None"
        self.tree._dom_classes=['generictree']

        #event handlers
        self.tree.observe (self._on_selected_changed, 'selected')
        self.tree.observe (self._on_created_changed, 'created')
        self.tree.observe (self._on_deleted_changed, 'deleted')
        self.tree.on_displayed(self._on_displayed)
        
        #interface for the tree and the container
        self.main_container = widgets.HBox(_dom_classes=['mainframe'])
        self.main_container.children=[self.tree, self.right_container ]
        
    @staticmethod
    def load_fields(field_names, field_data):
        storage = OrderedDict()
        for k,v in field_names.items():
            storage[k] = OrderedDict()
            if isinstance(v, dict):
                storage[k] = TreePanel.load_fields(v, field_data)
            else:
                for property_name in v:
                    field_meta = field_data[property_name].copy()
                    if 'sub_fields' in field_meta:
                        field_meta.update( TreePanel.load_fields({'sub_fields':field_meta['sub_fields']}, field_data ) )
                    field_meta['default'] = field_meta['value']
                    storage[k].setdefault ( property_name, field_meta )
        return storage
            
    @staticmethod
    def GetValueforWidget(config, section_name, field_meta):
        '''Code for mapping between the library (model) and the UI (view) goes here.
           Handles both deal as well as factor data.
           
           For deals, config is the instrument object from the model and the section_name
           is usually the 'field' attribute. 
           
           For factors, config is the config section from the model and the section_name
           is the name of the factor/process.

            The field_meta contains the actual field names.
        '''
        
        def loadTablefromVol(vol_factor, vols):
            table = [ ['Term to maturity/Moneyness'] + list(vol_factor.GetMoneyness()) ]
            for term, vol in zip (vol_factor.GetExpiry(), vols):
                table.append( [term] + ( list(vol) if isinstance(vol, np.ndarray) else [vol] ) )
            return table
        
        def getRepr(obj, field_name, default_val):
            
            if isinstance(obj,Utils.Curve):
                if obj.meta:
                    #need to use a threeview
                    if obj.meta[0]==2:
                        t 	 	  = RiskFactors.Factor2D({'Surface':obj})
                        vols      = t.GetVols()
                        vol_space = loadTablefromVol(t, vols)
                    else:
                        t 			= RiskFactors.Factor3D({'Surface':obj})
                        vol_cube    = t.GetVols()
                        if len(vol_cube.shape)==2:
                            vol_space = { t.GetTenor()[0]:loadTablefromVol(t, vol_cube) }
                        elif len(vol_cube.shape)==3:
                            vol_space = {}
                            for index, tenor in enumerate(t.GetTenor()):
                                vol_space.setdefault ( tenor, loadTablefromVol(t, vol_cube[index] ) )
                        
                    return_value = json.dumps(vol_space)
                else:
                    return_value = json.dumps([{'label':'None', 'data':[[x,y] for x,y in obj.array]}])
            elif isinstance(obj,Utils.Percent):
                return_value = obj.amount
            elif isinstance(obj,Utils.Basis):
                return_value = obj.amount
            elif isinstance(obj,Utils.Descriptor):
                return_value = str(obj)
            elif isinstance(obj,Utils.DateEqualList):
                data = [ [getRepr(date, 'Date', default_val)] + [getRepr(sub_val, 'Value', default_val) for sub_val in value] for date, value in obj.data.items() ]
                return_value = json.dumps(data)
            elif isinstance(obj,Utils.DateList):
                data = [ [getRepr(date, 'Date', default_val), getRepr(value, 'Value', default_val)] for date, value in obj.data.items() ]
                return_value = json.dumps(data)
            elif isinstance(obj,Utils.CreditSupportList):
                data = [ [getRepr(value, 'Value', default_val), getRepr(rating, 'Value', default_val)] for rating, value in obj.data.items() ]
                return_value = json.dumps(data)
            elif isinstance(obj,list):
                if field_name == 'Eigenvectors':
                    madness = []
                    for i,element in enumerate(obj):
                        madness.append( {'label':str(element['Eigenvalue']), 'data':[[x,y] for x,y in element['Eigenvector'].array] } )
                    return_value = json.dumps(madness)
                elif field_name in ['Properties','Items','Cash_Collateral','Equity_Collateral']:
                    data = []
                    for flow in obj:
                        data.append( [getRepr(flow.get(field_name), field_name, FieldDefaults.get(widget_type,default_val) ) for field_name, widget_type in zip(field_meta['col_names'], field_meta['obj']) ] )
                    return_value = json.dumps(data)
                elif field_name == 'Resets':
                    headings = ['Reset_Date','Start_Date','End_Date','Year_Fraction','Use Known Rate','Known_Rate']
                    widgets  = ['DatePicker','DatePicker','DatePicker','Float','Text','Float']
                    data = []
                    for flow in obj:
                        data.append( [ getRepr(item, field, FieldDefaults.get(widget_type,default_val)) for field, item, widget_type in zip(headings,flow,widgets) ] )
                    return_value = json.dumps(data)
                elif field_name == 'Description':
                    return_value = json.dumps(obj)
                elif field_name == 'Sampling_Data':
                    headings = ['Date','Price','Weight']
                    widgets  = ['DatePicker','Float','Float']
                    data = []
                    for flow in obj:
                        data.append( [ getRepr(item, field, FieldDefaults.get(widget_type,default_val)) for field, item, widget_type in zip(headings,flow,widgets) ] )
                    return_value = json.dumps(data)
                else:
                    raise Exception('Unknown Array Field type {0}'.format(field_name))
            elif isinstance(obj, pd.DateOffset):
                return_value = ''.join(['%d%s' % (v,Parser.reverse_offset[k]) for k,v in obj.kwds.items()])
            elif isinstance(obj, pd.Timestamp):
                return_value = obj.strftime('%Y-%m-%d')
            elif obj is None:
                return_value = default_val
            else:
                return_value = obj
                
            #return the value
            return return_value
                        
        #update an existing factor
        field_name = field_meta['description'].replace(' ','_')
        if section_name in config and field_name in config[section_name]:
            #return the value in the config obj
            obj = config[section_name][field_name]
            return getRepr(obj, field_name, field_meta['value'])
        else:
            return field_meta['value']

    def ParseConfig(self):
        pass

    def GenerateHandler(self, field_name, widget_elements, label):
        pass

    def DefineInput(self, label, widget_elements):
        
        container = widgets.Box()
        
        #label this container
        wig = [widgets.HTML()]
        vals = [self.GetLabel(label)]
        
        for field_name, element in sorted(widget_elements.items()):
            #skip this element if its not visible
            if element.get('isvisible')=='False':
                continue
            
            if element['widget']=='Dropdown':
                w = widgets.Dropdown( options=element['values'], description=element['description'] )
                vals.append(element['value'])
            elif element['widget']=='Text':
                w = widgets.Text( description=element['description'] )
                vals.append(str(element['value']))
            elif element['widget']=='Container':
                new_label = label+[element['description']] if isinstance(label, list) else [element['description']]
                w, v = self.DefineInput( [x.replace(' ','_') for x in new_label], element['sub_fields'] )
                vals.append(v)
            elif element['widget']=='Flot':
                w = Flot(description=element['description'])
                vals.append(element['value'])
            elif element['widget']=='Three':
                w = Three(description=element['description'])
                vals.append(element['value'])
            elif element['widget']=='Integer':
                w = widgets.IntText( description=element['description'] )
                vals.append(element['value'])
            elif element['widget']=='TreeFlot':
                w = FlotTree( description=element['description'] )
                w.type_data = element['type_data']
                w.profiles  = element['profiles']
                vals.append(element['value'])
            elif element['widget']=='HTML':
                w = widgets.HTML() 
                vals.append(element['value'])
            elif element['widget']=='Table':
                w = Table( description=element['description'], colTypes=json.dumps(element['sub_types']), colHeaders=json.dumps(element['col_names']) )
                vals.append(element['value'])
            elif element['widget']=='Float':
                w = widgets.FloatText( description=element['description'] )
                vals.append(element['value'])
            elif element['widget']=='DatePicker':
                w = DateWidget( description=element['description'] )
                vals.append(element['value'])
            elif element['widget']=='BoundedFloat':
                w = widgets.BoundedFloatText( min=element['min'], max=element['max'], description=element['description'] )
                vals.append(element['value'])
            else:
                raise Exception('Unknown widget field')
            
            if element['widget']!='Container':
                w.on_trait_change(self.GenerateHandler(field_name, widget_elements, label), 'value')
                
            wig.append(w)

        container.children=wig        
        return container, vals
    
    def GetLabel(self, label):
        return ''

    def CalcFrames(self, selection):
        pass
    
    def _on_selected_changed(self, change):

        def update_frame(frame, values):
            #set the style
            frame._dom_classes=['genericframe']
            #update the values in the frame
            for index, child in enumerate(frame.children):
                #recursively update the frames if needed
                if isinstance(values[index], list):
                    update_frame (child, values[index])
                else:
                    child.value = values[index]
                
        frames = self.CalcFrames(change['new'])
        
        #update the style of the right container
        self.right_container._dom_classes=['rightframe']
        
        #refresh the main container
        self.main_container.children = [self.tree, self.right_container ]
        
        #set the value of all widgets in the frame . . .
        #need to do this last in case all widgets haven't fully rendered in the DOM
        for container, value in frames:
            update_frame(container, value)

    def Create(self, newval):
        pass
    
    def Delete(self, newval):
        pass
    
    def _on_created_changed(self, change):
        if self.tree.created:
            #print 'create',val
            self.Create(change['new'])
            #reset the created flag
            self.tree.created = ''
        
    def _on_deleted_changed(self, change):
        if self.tree.deleted:
            #print 'delete',val
            self.Delete(change['new'])
            #reset the deleted flag
            self.tree.deleted = ''
            
    def _on_displayed(self, e):
        self.tree.value = json.dumps(self.tree_data)
    
    def show(self):
        display(self.main_container)

class MarketPricePage(TreePanel):
    def __init__(self, config):
        super(MarketPricePage, self).__init__(config)
        
    def GetLabel(self, label):
        return '<h4>{0}:</h4><i>{1}</i>'.format( label[0], label[1] ) if type(label)==types.TupleType else '<h4>{0}</h4>'.format( '.'.join(label) )
    
    def SetValueFromWidget(self, instrument, field_meta, new_val):
        
        def checkArray(new_obj):
            return [x for x in new_obj if x[0] is not None] if new_obj is not None else None        

        def setRepr(obj, obj_type):
            
            if obj_type=='Percent':
                return Utils.Percent(100.0*obj)
            elif obj_type=='Basis':
                return Utils.Basis(10000.0*obj)
            elif obj_type=='Period':
                return self.config.periodparser.parseString(obj)[0]
            elif obj_type=='DateList':
                new_obj     = checkArray ( json.loads(obj) )
                return Utils.DateList ( OrderedDict([(pd.Timestamp(date),val) for date,val in new_obj]) ) if new_obj is not None else None
            elif obj_type=='DateEqualList':
                new_obj     = checkArray ( json.loads(obj) )
                return Utils.DateEqualList ( [ [ pd.Timestamp(item[0])] + item[1:] for item in new_obj ] ) if new_obj is not None else None
            elif obj_type=='CreditSupportList':
                new_obj     = checkArray ( json.loads(obj) )
                return Utils.CreditSupportList ( OrderedDict([(rating, val) for val, rating in new_obj]) ) if new_obj is not None else None
            elif obj_type=='ResetArray':
                resets      = json.loads(obj)
                field_types = ['DatePicker','DatePicker','DatePicker','Float','Text','Percent']
                return [ [setRepr(data_obj,data_type) for data_obj,data_type in zip(reset,field_types)] for reset in resets ]
            elif isinstance(obj_type, list):
                if field_name=='Description':
                    return json.loads(obj)
                elif field_name=='Sampling_Data':
                    new_obj  = checkArray ( json.loads(obj) )
                    return [ [ setRepr(data_obj,data_type) for data_obj, data_type, field in zip ( data_row, obj_type, field_meta['col_names'] ) ] for data_row in new_obj ] if new_obj else []
                elif field_name in ['Items','Properties','Cash_Collateral','Equity_Collateral']:
                    new_obj  = checkArray ( json.loads(obj) )
                    return [ OrderedDict( [ ( field,setRepr(data_obj,data_type) ) for data_obj, data_type, field in zip ( data_row, obj_type, field_meta['col_names'] ) ] ) for data_row in new_obj ] if new_obj else []
                else:
                    raise Exception('Unknown list field type {0}'.format(field_name))
            elif obj_type=='DatePicker':
                return pd.Timestamp(obj) if obj else None
            elif obj_type=='Float':
                try:
                    new_obj = obj if obj=='<undefined>' else float(obj) 
                except ValueError:
                    print 'FLOAT',obj,field_meta
                    new_obj = 0.0
                return new_obj
            elif obj_type=='Container':
                return json.loads(obj)
            else:
                return obj
            
        field_name  = field_meta['description'].replace(' ','_')
        obj_type    = field_meta.get('obj', field_meta['widget'])
        
        #try:
        instrument[field_name] = setRepr(new_val, obj_type)
        #except:
        #    logging.debug('Field {0} could not be set to {1} - obj type {2}'.format(field_name, new_val, obj_type))
            
    def GenerateHandler(self, field_name, widget_elements, label):
        def handleEvent(trait_name, new_value):
            #update the json representation
            widget_elements[field_name]['value'] = new_value
            
        return handleEvent
        
    def ParseConfig(self):
        
        def load_items(config, field_name, field_data, storage):
            #make sure we get the right set of properties
            properties = field_data[field_name]
            for property_name, property_data in sorted(properties.items()):
                value = copy.deepcopy ( property_data )
                value['value'] = self.GetValueforWidget ( config, field_name, value )
                storage[property_name] = value
                
        def BuildPrices ( prices, instrument_fields, market_prices, parent, parent_cache ):
            
            for rate, node in prices.items():
                #get the instrument
                instrument = node['instrument']
                node_data  = {}
                rate_name  = Utils.CheckRateName(rate)[0]
                load_items ( {rate_name:node['instrument']}, rate_name, market_prices, node_data )

                json_data = {  "text" : rate,
                               "type" : "group" if node.get('Children') else "default",
                               "data" : {},
                               "children" : [] }

                parent.append(json_data)
                parent_cache[(rate,)] = [{'price':node_data}, node]
                
                for child in node.get('Children',[]):
                    quote_data          = {}
                    load_items ( child, 'quote', market_prices, quote_data )
                    quote_name          = '{0}.{1}'.format(child['quote']['DealType'], quote_data['Descriptor']['value'])
                    
                    json_quote_data = { "text" : quote_name,
                                        "type" : "default",
                                        "data" : {},
                                        "children" : [] }
                    
                    json_data["children"].append(json_quote_data)
                    
                    instrument_data     = {}
                    load_items ( {child['quote']['DealType']:child['instrument']}, child['quote']['DealType'], instrument_fields, instrument_data )
                    
                    parent_cache[ (rate,quote_name) ] = [{'quote':quote_data, 'instrument':instrument_data}, child]

        deals_to_append = []
        self.data       = {}
        
        #get all benchmark instruments
        all_benchmark_instruments = reduce(set.union, [set(x) for x in FieldMappings['MarketPrices']['sections'].values()], set())
        #all rates in a flat hierarchy
        self.benchmark_fields  = self.load_fields(FieldMappings['MarketPrices']['types'], FieldMappings['MarketPrices']['fields'])
        #map all the fields to one flat hierarchy
        instrument_types = { key: reduce ( operator.concat, [ FieldMappings['Instrument']['sections'][group] for group in FieldMappings['Instrument']['types'][key] if group!='Admin'] ) for key in all_benchmark_instruments }
        #get the fields from the master list
        self.instrument_fields  = self.load_fields(instrument_types, FieldMappings['Instrument']['fields'])
        #get the fields from the instrument prices list
        self.market_prices = self.load_fields(FieldMappings['MarketPrices']['types'], FieldMappings['MarketPrices']['fields'])
        
        #fill it with data
        BuildPrices ( self.config.params['Market Prices'], self.instrument_fields, self.market_prices, deals_to_append, self.data )
        
        self.tree_data = [{    "text" : "Market Data",  
                               "type" : "root", 
                               "id"   : "ROOT", 
                               "state" : { "opened" : True, "selected" : True }, 
                               "children" : deals_to_append} ]
        
        type_data = { "root"  : { "icon" : "/custom/glyphicons/png/glyphicons_041_charts.png", "valid_children" : ["group"] },
                      "group" : { "icon" : "/custom/glyphicons/png/glyphicons_051_eye_open.png", "valid_children" : ["group", "default"] },
                      "default" : {  "icon" : "/custom/glyphicons/png/glyphicons_050_link.png", "valid_children" : [] } }

        market_prices = OrderedDict()
        for k,v in sorted(FieldMappings['MarketPrices']['groups'].items()):
            group_type = v[0]
            for pricetype in sorted(v[1]):
                market_prices.setdefault(k,OrderedDict()).setdefault ( pricetype, group_type )
                
        #tree widget data
        self.tree = MarketTree()
        self.tree.type_data = json.dumps(type_data)
        self.tree.market_prices = json.dumps(market_prices)
        
        #have a placeholder for the selected price/point
        self.current_point    = None
       
    def CalcFrames(self, selection):
        key                        = tuple(json.loads(selection))
        frame, self.current_point  = self.data.get( key, [{}, None] )
        
        #factor_fields
        if frame:
            frames          = []
            if 'instrument' in frame:
                #get the quote
                frames.append ( self.DefineInput ( ( 'Quote', key[-1] ), frame['quote'] ) )
                #get the benchmark instrument
                obj_type        = frame['quote']['DealType']['value']
                for frame_name in FieldMappings['Instrument']['types'][ obj_type ]:
                    #skip the admin section - this is a generic benchmark instrument
                    if frame_name == 'Admin':
                        continue
                    #load the values:
                    instrument_fields = FieldMappings['Instrument']['sections'][frame_name]
                    frame_fields      = {k: v for k, v in frame['instrument'].iteritems() if k in instrument_fields}
                    frames.append ( self.DefineInput ( ( frame_name, key[-1] ), frame_fields ) )
            else:
                frames.append ( self.DefineInput ( ( 'Price', key[-1] ), frame['price'] ) )
                    
            #only store the container (first component)
            self.right_container.children = [x[0] for x in frames]
            return frames
        else:
            #load up an empty frame
            self.right_container = widgets.VBox()
            return []
        
    def Create(self, val):
        key             = tuple(json.loads(val))
        instrument_type = key[-1][:key[-1].find('.')]
        reference       = key[-1][key[-1].find('.')+1:]
        
        #load defaults for the new object
        fields  = self.instrument_fields.get(instrument_type)
        ins     = OrderedDict()
        
        for value in fields.values():
            self.SetValueFromWidget(ins, value, value['value'])

        #set it up        
        ins['Object']       = instrument_type
        ins['Reference']    = reference
        
        #Now check if this is a group or a regular deal
        if instrument_type in FieldMappings['Instrument']['groups']['STR'][1]:
            deal      = {'instrument':ConstructInstrument( ins ), 'Children':[]}
        else:
            deal      = {'instrument':ConstructInstrument( ins )}

        #add it to the xml
        parent = self.data[key[:-1]] [DealCache.Deal]
        #add this to parent
        parent['Children'].append(deal)
        
        #store it away
        view_data = copy.deepcopy( self.instrument_fields.get(instrument_type) )
        
        #make sure we record the instrument type
        for field in ['Object','Reference']:
            view_data[field]['value'] = ins[field]
            
        #update the cache
        count = self.data[key[:-1]][DealCache.Count].setdefault(reference, 0)
        #increment it
        self.data[key[:-1]][DealCache.Count][reference]+=1
        #store it away
        self.data[key]  = [view_data, deal, parent, {} if 'Children' in deal else None]
        
    def Delete(self, val):
        key         = tuple(json.loads(val))
        reference   = key[-1][key[-1].find('.')+1:]
        parent      = self.data[key] [DealCache.Parent]
        
        print key, parent
        #delete the deal
        parent['Children'].remove( self.data[key] [DealCache.Deal] )
        #decrement the count
        self.data[key[:-1]][DealCache.Count][reference]-=1
        #delte the view data
        del self.data[key]

class PortfolioPage(TreePanel):
    def __init__(self, config):
        super(PortfolioPage, self).__init__(config)        
        
    def GetLabel(self, label):
        return '<h4>{0}:</h4><i>{1}</i>'.format( label[0], label[1] ) if type(label)==types.TupleType else '<h4>{0}</h4>'.format( '.'.join(label) )
    
    def SetValueFromWidget(self, instrument, field_meta, new_val):
        
        def checkArray(new_obj):
            return [x for x in new_obj if x[0] is not None] if new_obj is not None else None

        def setRepr(obj, obj_type):
            if obj_type=='Percent':
                return Utils.Percent(100.0*obj)
            elif obj_type=='Basis':
                return Utils.Basis(10000.0*obj)
            elif obj_type=='Period':
                return self.config.periodparser.parseString(obj)[0]
            elif obj_type=='DateList':
                new_obj     = checkArray ( json.loads(obj) )
                return Utils.DateList ( OrderedDict([(pd.Timestamp(date),val) for date,val in new_obj]) ) if new_obj is not None else None
            elif obj_type=='DateEqualList':
                new_obj     = checkArray ( json.loads(obj) )
                return Utils.DateEqualList ( [ [ pd.Timestamp(item[0])] + item[1:] for item in new_obj ] ) if new_obj is not None else None
            elif obj_type=='CreditSupportList':
                new_obj     = checkArray ( json.loads(obj) )
                return Utils.CreditSupportList ( OrderedDict([(rating, val) for val, rating in new_obj]) ) if new_obj is not None else None
            elif obj_type=='ResetArray':
                resets      = json.loads(obj)
                field_types = ['DatePicker','DatePicker','DatePicker','Float','Text','Percent']
                return [ [setRepr(data_obj,data_type) for data_obj,data_type in zip(reset,field_types)] for reset in resets ]
            elif isinstance(obj_type, list):
                if field_name=='Description':
                    return json.loads(obj)
                elif field_name=='Sampling_Data':
                    new_obj  = checkArray ( json.loads(obj) )
                    return [ [ setRepr(data_obj,data_type) for data_obj, data_type, field in zip ( data_row, obj_type, field_meta['col_names'] ) ] for data_row in new_obj ] if new_obj else []
                elif field_name in ['Items','Properties','Cash_Collateral','Equity_Collateral']:
                    new_obj  = checkArray ( json.loads(obj) )
                    return [ OrderedDict( [ ( field,setRepr(data_obj,data_type) ) for data_obj, data_type, field in zip ( data_row, obj_type, field_meta['col_names'] ) ] ) for data_row in new_obj ] if new_obj else []
                else:
                    raise Exception('Unknown list field type {0}'.format(field_name))
            elif obj_type=='DatePicker':
                return pd.Timestamp(obj) if obj else None
            elif obj_type=='Float':
                try:
                    new_obj = obj if obj=='<undefined>' else float(obj) 
                except ValueError:
                    print 'FLOAT',obj,field_meta
                    new_obj = 0.0
                return new_obj
            elif obj_type=='Container':
                return json.loads(obj)
            else:
                return obj
            
        field_name  = field_meta['description'].replace(' ','_')
        obj_type    = field_meta.get('obj', field_meta['widget'])
        
        #try:
        instrument[field_name] = setRepr(new_val, obj_type)
        #except:
        #    logging.debug('Field {0} could not be set to {1} - obj type {2}'.format(field_name, new_val, obj_type))
            
    def GenerateHandler(self, field_name, widget_elements, label):
        def handleEvent(trait_name, new_value):
            #update the json representation
            widget_elements[field_name]['value'] = new_value
            #find the instrument obj
            instrument_obj  = self.current_deal['instrument']            
            #get the fields
            instrument = instrument_obj.field
            #update the value in the model
            if isinstance(label, list):
                for key in label:
                    if key not in instrument:
                        #should only happen if a container attribute was not set (should be a dict by default)
                        instrument[key]={}
                    instrument = instrument[key]

            self.SetValueFromWidget ( instrument, widget_elements[field_name], new_value )
            
        return handleEvent
        
    def ParseConfig(self):
        
        def load_items(config, field_name, field_data, storage):
            #make sure we get the right set of properties
            properties = field_data[field_name] if field_name in field_data else field_data[ config[field_name]['Object'] ]
            for property_name, property_data in sorted(properties.items()):
                value = copy.deepcopy ( property_data )
                if 'sub_fields' in value:
                    new_field_name = value['description'].replace(' ','_')
                    load_items(config[field_name], new_field_name, {new_field_name:value['sub_fields']}, value['sub_fields'])
                value['value'] = self.GetValueforWidget ( config, field_name, value )
                
                storage[property_name] = value
                
        def walkPortfolio ( deals, path, instrument_fields, parent, parent_cache ):
            
            for node in deals:
                #get the instrument
                instrument = node['instrument']
                #get its name
                reference = instrument.field.get('Reference')
                #update the parent cache
                count = parent_cache[path][DealCache.Count].setdefault(reference, 0) 
                #get the deal_id (unique key)
                deal_id = '{0}{1}'.format( reference, ':{0}'.format(count) if count else '' ) if reference else '{0}'.format( count )
                #establish the name
                name = "{0}.{1}".format( instrument.field['Object'], deal_id )
                #increment the counter
                parent_cache[path][DealCache.Count][reference]+=1
                #full path name
                path_name = path+(name,)
                
                #if node.attrib.get('Ignore')=='True':
                #    continue

                node_data = {}
                load_items ( instrument.__dict__, 'field', instrument_fields, node_data )

                json_data = {  "text" : name,
                               "type" : "group" if node.get('Children') else "default",
                               "data" : {},
                               "children" : [] }

                parent.append(json_data)
                parent_cache[path_name] = [node_data, node, deals, {} if 'Children' in node else None ]
                
                if node.get('Children'):
                    walkPortfolio ( node['Children'], path_name, instrument_fields, json_data['children'], parent_cache )

        deals_to_append = []
        self.data       = {():[{}, self.config.deals['Deals'], None, {}]}
        
        #map all the fields to one flat hierarchy
        instrument_types = { key: reduce ( operator.concat, [ FieldMappings['Instrument']['sections'][group] for group in groups ] ) for key, groups in FieldMappings['Instrument']['types'].iteritems() }
        #get the fields from the master list
        self.instrument_fields  = self.load_fields(instrument_types, FieldMappings['Instrument']['fields'])
        #fill it with data
        walkPortfolio ( self.config.deals['Deals']['Children'], (), self.instrument_fields, deals_to_append, self.data )
        
        self.tree_data = [{    "text" : "Postions",  
                               "type" : "root", 
                               "id"   : "ROOT", 
                               "state" : { "opened" : True, "selected" : True }, 
                               "children" : deals_to_append} ]
        
        type_data = { "root"  : { "icon" : "/custom/glyphicons/png/glyphicons_022_fire.png", "valid_children" : ["group"] },
                      "group" : { "icon" : "/custom/glyphicons/png/glyphicons_144_folder_open.png", "valid_children" : ["group", "default"] },
                      "default" : {  "icon" : "/custom/glyphicons/png/glyphicons_037_coins.png", "valid_children" : [] } }

        instrument_def = OrderedDict()
        for k,v in sorted(FieldMappings['Instrument']['groups'].items()):
            group_type = v[0]
            for instype in sorted(v[1]):
                instrument_def.setdefault(k,OrderedDict()).setdefault ( instype, group_type )
                
        #tree widget data
        self.tree = PortfolioTree()
        self.tree.type_data = json.dumps(type_data)
        self.tree.instrument_def = json.dumps(instrument_def)
        
        #have a placeholder for the selected model (deal)
        self.current_deal           = None
        self.current_deal_parent    = None
       
    def CalcFrames(self, selection):
        key                                                         = tuple(json.loads(selection))
        frame, self.current_deal, self.current_deal_parent, count   = self.data.get( key, [{}, None, None, 0] )
        
        #factor_fields
        if frame:
            frames          = []
            #get the object type - this should always be defined
            obj_type        = frame['Object']['value']
            #get the instrument obj
            instrument_obj  = self.current_deal['instrument']
            
            for frame_name in FieldMappings['Instrument']['types'][ obj_type ]:
                #load the values:
                instrument_fields = FieldMappings['Instrument']['sections'][frame_name]
                frame_fields = {k: v for k, v in frame.iteritems() if k in instrument_fields}
                frames.append( self.DefineInput ( ( frame_name, key[-1] ), frame_fields ) )
                
            #only store the container (first component)
            self.right_container.children = [x[0] for x in frames]            
            return frames
        else:
            #load up a set of defaults
            self.right_container = widgets.Box()
            return []
        
    def Create(self, val):
        key             = tuple(json.loads(val))
        instrument_type = key[-1][:key[-1].find('.')]
        reference       = key[-1][key[-1].find('.')+1:]
        
        #load defaults for the new object
        fields  = self.instrument_fields.get(instrument_type)
        ins     = OrderedDict()
        
        for value in fields.values():
            self.SetValueFromWidget(ins, value, value['value'])

        #set it up        
        ins['Object']       = instrument_type
        ins['Reference']    = reference
        
        #Now check if this is a group or a regular deal
        if instrument_type in FieldMappings['Instrument']['groups']['STR'][1]:
            deal      = {'instrument':ConstructInstrument( ins ), 'Children':[]}
        else:
            deal      = {'instrument':ConstructInstrument( ins )}

        #add it to the xml
        parent = self.data[key[:-1]] [DealCache.Deal]
        #add this to parent
        parent['Children'].append(deal)
        
        #store it away
        view_data = copy.deepcopy( self.instrument_fields.get(instrument_type) )
        
        #make sure we record the instrument type
        for field in ['Object','Reference']:
            view_data[field]['value'] = ins[field]
            
        #update the cache
        count = self.data[key[:-1]][DealCache.Count].setdefault(reference, 0)
        #increment it
        self.data[key[:-1]][DealCache.Count][reference]+=1
        #store it away
        self.data[key]  = [view_data, deal, parent, {} if 'Children' in deal else None]
        
    def Delete(self, val):
        key     = tuple(json.loads(val))
        reference   = key[-1][key[-1].find('.')+1:]
        parent      = self.data[key] [DealCache.Parent]
        
        print key, parent
        #delete the deal
        parent['Children'].remove( self.data[key] [DealCache.Deal] )
        #decrement the count
        self.data[key[:-1]][DealCache.Count][reference]-=1
        #delte the view data
        del self.data[key]

class RiskFactorsPage(TreePanel):
    def __init__(self, config):
        super(RiskFactorsPage, self).__init__(config)
        
    def ParseConfig(self):
        
        def load_items(config, factor_data, field_data, storage):
            for property_name, property_data in sorted(field_data[factor_data.type].items()):
                value = copy.deepcopy( property_data )
                value['value'] = self.GetValueforWidget ( config, Utils.CheckTupleName(factor_data), value )
                storage[property_name] = value
                
        risk_factor_fields  = self.load_fields(FieldMappings['Factor']['types'], FieldMappings['Factor']['fields'])
        risk_process_fields = self.load_fields(FieldMappings['Process']['types'], FieldMappings['Process']['fields'])
        
        #only 1 level of config paramters here - unlike the other 2
        self.sys_config     = self.load_fields(FieldMappings['System']['types'], FieldMappings['System']['fields']).values()[0]
        for value in self.sys_config.values():
            value['value'] = self.GetValueforWidget ( self.config.params, 'System Parameters', value )
        
        possible_risk_process = {}
        for k,v in FieldMappings['Process_factor_map'].items():
            fields_to_add = {}
            for process in v:
                fields_to_add[process]=risk_process_fields[process]
            possible_risk_process[k] = fields_to_add
        
        loaded_data         = []
        self.data           = {}
        factor_process_map  = {}
        
        for price_factor, price_data in self.config.params['Price Factors'].iteritems():
            raw_factor        = Utils.CheckRateName(price_factor)
            factor            = Utils.Factor(raw_factor[0], raw_factor[1:])
            factor_to_append  = {}
            process_to_append = {}
            process_data      = possible_risk_process[factor.type].copy()
            
            factor_process_map[price_factor] = ''
            
            load_items(self.config.params['Price Factors'], factor, risk_factor_fields, factor_to_append)
            stoch_proc = self.config.params['Model Configuration'].Search ( factor, price_data )
            if stoch_proc:                
                factor_model = Utils.Factor (stoch_proc, factor.name) 
                load_items(self.config.params['Price Models'], factor_model, risk_process_fields, process_to_append)
                process_data[stoch_proc]=process_to_append
                factor_process_map[price_factor]=stoch_proc
                
            factor_node = {'text':price_factor,
                           'type':'default',
                           'data':{},
                           'children':[]}
            
            loaded_data.append(factor_node)
            self.data[price_factor] = {'Factor':factor_to_append, 'Process':process_data }

        self.tree_data = [{ "text" : "Price Factors",  
                           "type" : "root",  
                           "state" : { "opened" : True, "selected" : False }, 
                           "children" : loaded_data} ]
        
        type_data = { "root" : { "valid_children" : ["default"] },
                      "default" : { "valid_children" : [] } }
        
        #simple lookup to match the params in the config file to the json in the UI
        self.config_map              = {'Factor':'Price Factors', 'Process':'Price Models', 'Config':'System Parameters'}
        #fields to store new objects
        self.new_factor 			 = {'Factor':risk_factor_fields, 'Process':possible_risk_process}
        
        #tree widget data
        self.tree               = FactorTree()
        self.tree.type_data     = json.dumps(type_data)
        self.tree.risk_factors  = json.dumps(sorted(risk_factor_fields.keys()))

    def GetLabel(self, label):
        return '<h4>{0}:</h4><i>{1}</i>'.format( label[0], label[1] + ( '' if label[1] in self.config.params[ self.config_map[label[0]] ] else ' - Unsaved') )
    
    def SetValueFromWidget(self, frame_name, section_name, field_meta, new_val):

        def checkArray(new_obj):
            return np.array( new_obj[:-1] if (new_obj[-1][0] is None) else new_obj, dtype=np.float64 )
            
        def SetTablefromVol(vols, tenor=None):
            #skip the header and the nulls at the end
            curve_array = []
            null_filter = slice(1,-1) if (vols[0][-1] is None) else slice(1,None)
            moneyness = vols[0][null_filter]
            for exp_vol in vols[null_filter]:
                exp = exp_vol[0]
                curve_array.extend([([money, exp, tenor, vol] if (tenor is not None) else [money, exp, vol] ) for money,vol in zip(moneyness, exp_vol[null_filter])])
            return curve_array        
        
        def setRepr(obj, obj_type):
            if obj_type=='Flot':
                new_obj = json.loads(obj)
                if field_name=='Eigenvectors':
                    obj     = []
                    for new_eigen_data in new_obj:
                        eigen_data  = OrderedDict()
                        eigen_data['Eigenvector'] = Utils.Curve([], checkArray (new_eigen_data['data']) )
                        eigen_data['Eigenvalue']  = float(new_eigen_data['label'])
                        obj.append(eigen_data)
                    return obj
                else:
                    return Utils.Curve([], checkArray(new_obj[0]['data']) )
            elif obj_type=='Three':
                new_obj = json.loads(obj)
                if rate_type in Utils.TwoDimensionalFactors:
                    #2d Surfaces
                    return Utils.Curve([2,'Linear'], np.array ( SetTablefromVol(new_obj), dtype=np.float64 ) )
                else:
                    #3d Spaces
                    vol_cube=[]
                    for tenor, vol_surface in new_obj.items():
                        vol_cube.extend( SetTablefromVol ( vol_surface, float(tenor) ) )
                    return Utils.Curve([3,'Linear'], np.array( vol_cube, dtype=np.float64 ) )
            elif obj_type=='Percent':
                return Utils.Percent(100.0*obj)
            elif obj_type=='DatePicker':
                #might want to make this none by default
                return pd.Timestamp(obj) if obj else None
            else:
                return obj
            
        rate_type  = Utils.CheckRateName(section_name)[0]
        field_name = field_meta['description'].replace(' ','_')
        obj_type   = field_meta.get('obj', field_meta['widget'])
        config     = self.config.params [ self.config_map [ frame_name ] ]

        if section_name == "":
            config[field_name] = setRepr(new_val, obj_type)
        elif section_name in config and field_name in config[section_name]:
            config[section_name][field_name] = setRepr(new_val, obj_type)
        elif new_val!=field_meta['default']:
            config.setdefault ( section_name, OrderedDict() ).setdefault(field_name, setRepr(new_val, obj_type) )
            #store the new object with all the usual defaults
            if frame_name=='Factor':
                frame_defaults = self.data[self.tree.selected][frame_name]
            elif frame_name=='Config':
                frame_defaults = self.sys_config
            else:
                frame_defaults = self.data[self.tree.selected][frame_name][rate_type]
            
            for new_field_meta in frame_defaults.values():
                new_field_name = new_field_meta['description'].replace(' ','_')
                new_obj_type   = new_field_meta.get('obj', new_field_meta['widget'])
                if new_field_name not in config[section_name]:
                    config[section_name][new_field_name] = setRepr(new_field_meta['value'], new_obj_type)
            
    def GenerateHandler(self, field_name, widget_elements, label):
        def handleEvent(trait_name, new_value):
            #update the json representation
            widget_elements[field_name]['value']=new_value
            #update the value in the config object
            self.SetValueFromWidget ( label[0], label[1], widget_elements[field_name], new_value )
        return handleEvent
    
    def Create(self, val):
        factor_type = val[:val.find('.')]
        #load defaults for the new riskfactor
        self.data[val] = {'Factor':copy.deepcopy(self.new_factor['Factor'].get(factor_type)), 'Process':copy.deepcopy(self.new_factor['Process'].get(factor_type))}

    def Delete(self, val):
        factor = Utils.CheckRateName(val)
        #delete the factor
        if val in self.config.params ['Price Factors']:
            del self.config.params ['Price Factors'][val]
        for model in self.config.params ['Price Models']:
            if Utils.CheckRateName(model)[1:]==factor[1:]:
                del self.config.params ['Price Models'][model]
                #also delete any correlations this model had
                #TODO!
                
        #delte the view data
        del self.data[val]
        
    def CorrelationFrame(self, stoch_proc_list):
        
        def GenerateHandler(correlation_matrix, process_names, process_lookup):
            def handleEvent(trait_name, new_value):
                #update the value in the config object
                correlations = json.loads ( new_value )
                for proc1,proc2 in itertools.combinations(process_names,2):
                    i, j = process_lookup[proc1], process_lookup[proc2]
                    key = (proc1, proc2) if (proc1,proc2) in correlation_matrix else (proc2, proc1)
                    correlation_matrix[key] = correlations[ j ] [ i ] if correlations[ j ] [ i ]!=0.0 else correlations[ i ] [ j ]
            return handleEvent
        
        #generate dummy processes
        stoch_factors = OrderedDict( (stoch_proc, ConstructProcess(stoch_proc.type, None, self.config.params['Price Models'][Utils.CheckTupleName(stoch_proc)])) for stoch_proc in stoch_proc_list if Utils.CheckTupleName(stoch_proc) in self.config.params['Price Models'] )
        num_factors   = sum ( [x.NumFactors() for x in stoch_factors.values()] )
        
        #prepare the correlation matrix (and the offsets of each stochastic process)
        correlation_factors = []
        correlation_matrix  = np.eye(num_factors, dtype=np.float32)

        for key, value in stoch_factors.items():
            proc_corr_type, proc_corr_factors = value.CorrelationName()
            for sub_factors in proc_corr_factors:
                correlation_factors.append ( Utils.CheckTupleName ( Utils.Factor(proc_corr_type, key.name+sub_factors) ) )
                
        for index1 in range(num_factors):
            for index2 in range(index1+1,num_factors):
                factor1,factor2 = correlation_factors[index1], correlation_factors[index2]
                key = (factor1, factor2) if (factor1, factor2) in self.config.params['Correlations'] else (factor2, factor1)
                rho = self.config.params['Correlations'].get( key, 0.0 ) if factor1!=factor2 else 1.0
                correlation_matrix[index1,index2] = rho
                correlation_matrix[index2,index1] = rho

        container = widgets.Box()
        
        #label this container
        wig = [widgets.HTML()]
        vals = ['<h4>Correlation:</h4>']

        col_types   = '['+','.join(['{ "type": "numeric", "format": "0.0000" }']*len(correlation_factors))+']'
        col_headers = json.dumps( correlation_factors )

        w = Table ( description="Matrix", colTypes=col_types, colHeaders=col_headers )
        vals.append ( json.dumps( correlation_matrix.tolist() ) )
        w.on_trait_change (GenerateHandler(self.config.params['Correlations'], correlation_factors, {x:i for i,x in enumerate(correlation_factors)} ), 'value')
        wig.append(w)
        
        #print correlation_factors, correlation_matrix, col_headers
        
        container.children=wig
        return container, vals
    
    def ModelConfigFrame(self):
        
        def GenerateHandler (model_config):
            def handleEvent (trait_name, new_value):
                #update the value in the config object
                model_config.modeldefaults=OrderedDict()
                model_config.modelfilters=OrderedDict()
                for config in json.loads ( new_value ):
                    if config and config[0] is not None:
                        factor, process = config[0].split('.')
                        if ( config[1] is not None ) and ( config[2] is not None ) :
                            filter_on, value = config[1], config[2]
                            model_config.modelfilters.setdefault(factor, []).append( ( (filter_on, value), process ) )
                            continue
                        model_config.modeldefaults.setdefault(factor, process)
            return handleEvent
        
        container = widgets.Box()
        
        #label this container
        wig = [widgets.HTML()]
        vals = ['<h4>Stochastic Process Mapping:</h4>']
        
        optionmap = {k:v for k,v in FieldMappings['Process_factor_map'].items() if v}
        col_types = json.dumps( [ {"type": "dropdown", "source": sorted ( reduce(operator.concat, [['{0}.{1}'.format(key,val) for val in values] for key,values in optionmap.items()], []) ), "strict":True}, {}, {} ] )
        col_headers = json.dumps( ["Risk_Factor.Stochastic_Process","Where","Equals"] )

        model_config = [['{0}.{1}'.format(k,v), None, None] for k,v in sorted(self.config.params['Model Configuration'].modeldefaults.items())]
        model_config.extend ( reduce(operator.concat, [ [ ['{0}.{1}'.format(k,rule[1]), rule[0][0], rule[0][1]] for rule in v ] for k,v in sorted(self.config.params['Model Configuration'].modelfilters.items()) ], []) )

        w = Table ( description="Model Configuration", colTypes=col_types, colHeaders=col_headers )        
        vals.append ( json.dumps( sorted( model_config ) if model_config else [[None,None,None]]) )
        w.on_trait_change ( GenerateHandler( self.config.params['Model Configuration'] ), 'value' )
        wig.append(w)
        
        container.children=wig
        return container, vals
        
    def CalcFrames(self, selection):

        frame  = self.data.get(selection, {})
        frames = []        
        sps    = set()
        
        if frame:
            #get the name 
            factor_name  = Utils.CheckRateName (selection)
            factor       = Utils.Factor(factor_name[0], factor_name[1:])
            #load the values:
            for frame_name, frame_value in sorted(frame.items()):
                if frame_name=='Process':
                    stoch_proc = self.config.params['Model Configuration'].Search ( factor, self.config.params['Price Factors'].get(selection,{}) )
                    if stoch_proc:
                        full_name = Utils.Factor(stoch_proc, factor.name)
                        sps.add ( full_name )
                        frames.append ( self.DefineInput ( ( frame_name, Utils.CheckTupleName (full_name) ), frame_value[stoch_proc] ) )
                    else:
                        frames.append ( self.DefineInput( (frame_name, '' ), {} ) )
                elif frame_name=='Factor':
                    frames.append ( self.DefineInput( ( frame_name, Utils.CheckTupleName (factor) ), frame_value ) )
                    
            #need to add a frame for correlations
            if self.tree.allselected:
                for selected in json.loads(self.tree.allselected):
                    factor_name  = Utils.CheckRateName (selected)
                    factor       = Utils.Factor(factor_name[0], factor_name[1:])
                    stoch_proc   = self.config.params['Model Configuration'].Search ( factor, self.config.params['Price Factors'].get(selected,{}) )
                    if stoch_proc:
                        full_name =  Utils.Factor(stoch_proc, factor.name)
                        sps.add ( full_name )
                if len(sps)>1:
                    frames.append ( self.CorrelationFrame( sorted(sps) ) )
                    
        #show the system config screen if there are no factors selected
        if not frame:
            frames.append ( self.DefineInput( ('Config', ''), self.sys_config ) )
            frames.append ( self.ModelConfigFrame() )
            
        self.right_container.children = [x[0] for x in frames]

        return frames
    
class CalculationPage(TreePanel):
    def __init__(self, config):
        super(CalculationPage, self).__init__(config)
        
    def GetLabel(self, label):
        return '<h4>{0}:</h4><i>{1}</i>'.format( label[0], label[1] )
    
    def ParseConfig(self):
        self.data                = {}
        self.calculation_fields  = self.load_fields(FieldMappings['Calculation']['types'], FieldMappings['Calculation']['fields'])
        
        self.tree_data = [{ "text" : "Calculations",  
                            "type" : "root",  
                            "state" : { "opened" : True, "selected" : False }, 
                            "children" : [] } ]
        
        type_data = { "root" : { "valid_children" : ["default"] },
                      "default" : { "valid_children" : [] } }
        
        #tree widget data
        self.tree = CalculationTree()
        self.tree.type_data = json.dumps(type_data)
        self.tree.calculation_types = json.dumps( dict.fromkeys( FieldMappings['Calculation']['types'].keys(), 'default' ) )

    def GenerateSlides(self, calc, filename, output):
        nb      = nbf.new_notebook()
        cells   = []
        
        for cell in calc.slides:
            nb_cell             = nbf.new_text_cell('markdown', cell['text'] )
            nb_cell['metadata'] = cell['metadata']
            cells.append(nb_cell)
            
        nb['metadata']= {'celltoolbar': 'Slideshow'}
        nb['worksheets'].append(nbf.new_worksheet(cells=cells))
        
        with open(filename+'.ipynb', 'w') as f:
            nbf.write(nb, f, 'ipynb')
            
        #now generate the slideshow
        call([r'C:\python27\Scripts\ipython','nbconvert', filename+'.ipynb', '--to', 'slides', '--reveal-prefix', "http://cdn.jsdelivr.net/reveal.js/2.6.2"])
        
        #let the gui know about the slides - need to create a custom widget for the label - this is hacky
        output['value'] = '<div class="widget-hbox-single"><div class="widget-hlabel" style="display: block;">Slides</div><a href=\'{0}\' target=''_blank''>{0}</a></div>'.format(filename+'.slides.html')
        
    def GetResults(self, selection, frame, key):
        
        def Define_Click(widget):
            calc   = frame['calc']
            input  = frame['frames']['input']
            output = frame['frames']['output']            
            
            param  = {k:v['value'] for k,v in input.items()}
            #send the name of the calc to the calculation engine
            param['calc_name'] = key
            result = calc.Execute(param)
            
            #Disable unneeded fields
            for k,v in input.items():
                if v.get('Output'):
                    output[ v['Output'] ]['isvisible'] = 'True' if v['value']=='Yes' else 'False'
                    
            #Format the results            
            calc.FormatOutput(result, output)
                        
            #generate slides        
            if calc.slides:
                self.GenerateSlides ( calc, key[0], output['Slideshow'] )

            #flag the gui that we have output
            frame['output']=True
            #trigger redraw
            self._on_selected_changed({'new':selection})
            
        return Define_Click
    
    def CalcFrames(self, selection):
        key             = tuple(json.loads(selection))
        frame           = self.data.get(key, {})
        
        if frame:
            frames   = []
            input    = self.DefineInput( key[-1].split('.'), frame['frames']['input'] )
            
            #add the calc button
            execute_button    = widgets.Button(description='Execute')            
            execute_button.on_click( self.GetResults( selection, frame, key ) )
            
            input[0].children = input[0].children + ( execute_button, )
            input[1].append('')
           
            frames.append ( input )
            
            #now the output
            if frame['output']:
                output = self.DefineInput( [key[-1],'Results'], frame['frames']['output'] )
                frames.append ( output )

            self.right_container.children = [x[0] for x in frames]
            return frames
        else:
            #load up a set of defaults
            self.right_container = widgets.Box()
            return []
        
    def GenerateHandler(self, field_name, widget_elements, label):
        def handleEvent(trait_name, new_value):
            #update the json representation
            widget_elements[field_name]['value']=new_value
            #update the value in the config object
            #self.SetValueFromWidget ( label[0], label[1], widget_elements[field_name], new_value )
        return handleEvent
    
    def Create(self, val):
        key             = tuple(json.loads(val))
        calc_type       = key[-1][:key[-1].find('.')]
        reference       = key[-1][key[-1].find('.')+1:]
        #print key, calc_type, reference
        #load defaults for the new riskfactor
        self.data[key] = {'frames':copy.deepcopy(self.calculation_fields.get(calc_type)), 'calc':ConstructCalculation(calc_type, self.config), 'output':False}

    def Delete(self, val):
        del self.data[tuple(json.loads(val))]
        
class CalibrationPage(TreePanel):
    def __init__(self, config):
        self.current_calib = None
        self.calib_root    = None
        super(CalibrationPage, self).__init__(config)
        
    def GetLabel(self, label):
        return '<h4>{0}:</h4><i>{1}</i>'.format( label[0], label[1] ) if type(label)==types.TupleType else '<h4>{0}</h4>'.format( '.'.join(label) )
    
    def SetValueFromWidget(self, calibration, field_meta_data, new_val):
        
        def checkArray(new_obj):
            return [x for x in new_obj if x[0] is not None] if new_obj is not None else None        

        def setRepr(obj, obj_type, field_meta):
            
            if obj_type=='Percent':
                return Utils.Percent(100.0*obj)
            elif obj_type=='Basis':
                return Utils.Basis(10000.0*obj)
            elif obj_type=='Period':
                return self.config.periodparser.parseString(obj)[0]
            elif obj_type=='DatePicker':
                return pd.Timestamp(obj) if obj else None
            elif obj_type=='Float':
                try:
                    new_obj = obj if obj=='<undefined>' else float(obj) 
                except ValueError:
                    print 'FLOAT',obj,field_meta
                    #raise Exception("DICK")
                    new_obj = 0.0
                return new_obj
            elif obj_type=='Container':
                new_obj = OrderedDict()
                for key, val in json.loads(obj).items():
                    new_field_meta = field_meta['sub_fields'][key]
                    new_objtype    = new_field_meta.get('obj', new_field_meta['widget'])
                    new_obj[key]   = setRepr(val, new_objtype, new_field_meta)
                return new_obj
            else:
                return obj
            
        field_name  = field_meta_data['description'].replace(' ','_')
        obj_type    = field_meta_data.get('obj', field_meta_data['widget'])
        
        #this call should always succeed
        calibration[field_name] = setRepr(new_val, obj_type, field_meta_data)
            
    def GenerateHandler(self, field_name, widget_elements, label):
        def handleEvent(trait_name, new_value):
            #update the json representation
            widget_elements[field_name]['value'] = new_value
            #find the calibration obj
            param_block = self.current_calib.calibration.param
            #update the value in the model
            if isinstance(label, list):
                for key in label:
                    param_block = param_block[key]
            #print label, field_name
            self.SetValueFromWidget ( param_block, widget_elements[field_name], new_value )
        return handleEvent
    
    def ParseConfig(self):
        def load_items(config, field_name, field_data, storage):
            #make sure we get the right set of properties
            properties = field_data[field_name] if field_name in field_data else field_data[ config['model'] ]
            for property_name, property_data in sorted(properties.items()):
                value = copy.deepcopy ( property_data )
                if 'sub_fields' in value:
                    new_field_name = value['description'].replace(' ','_')
                    load_items(config[field_name], new_field_name, {new_field_name:value['sub_fields']}, value['sub_fields'])
                value['value'] = self.GetValueforWidget ( config, field_name, value )
                
                storage[property_name] = value
                
        self.data                   = {():[{}, self.config.calibrations.getroot()]}
        loaded_data                 = []
        self.calibration_fields     = self.load_fields(FieldMappings['Calibration']['types'], FieldMappings['Calibration']['fields'])
        
        for elem in self.config.calibrations.getroot():
            if elem.tag=='MarketDataArchiveFile':
                pass
            elif elem.tag=='Calibrations':
                self.calib_root = elem
                for node_num,node in enumerate(elem):
                    calib_data = {}
                    load_items ( node.calibration.__dict__, 'param', self.calibration_fields, calib_data )
                    name = "{0}.{1}".format( node.calibration.model, node.attrib['ID'] if node.attrib['ID'] else node_num )

                    calibration = {"text" : name,
                                   "type" : "default",  
                                   "data" : {},
                                   "children" : [] }
                    
                    loaded_data.append(calibration)
                    self.data[name] = [ calib_data, node ]
        
        self.tree_data = [{ "text" : "Calibration",  
                            "type" : "root",  
                            "state" : { "opened" : True, "selected" : False }, 
                            "children" : loaded_data } ]
        
        type_data = { "root" : { "valid_children" : ["default"] },
                      "default" : { "valid_children" : [] } }
        
        #tree widget data
        self.tree = CalibrationTree()
        self.tree.type_data = json.dumps(type_data)
        self.tree.calibration_types = json.dumps( dict.fromkeys( FieldMappings['Calibration']['types'].keys(), 'default' ) )

    def CalcFrames(self, selection):
        key                                 = (json.loads(selection) or [None])[0]
        frame, self.current_calib           = self.data.get(key, ({}, None))
        
        if frame:
            frames                          = [ self.DefineInput ( tuple ( key.split('.') ), frame ) ]
            self.right_container.children   = [ x[0] for x in frames ]
            return frames
        else:
            #load up a set of defaults
            self.right_container            = widgets.VBox()
            return []
        
    def Create(self, selection):
        key                 = (json.loads(selection) or [None])[0]
        calibration_type    = key[:key.find('.')]
        reference           = key[key.find('.')+1:]
        
        #load defaults for this object
        fields              = copy.deepcopy ( self.calibration_fields.get(calibration_type) )
        calib               = OrderedDict()

        for value in fields.values():
            self.SetValueFromWidget(calib, value, value['value'])

        
        calibration_config= {'Method': calibration_type[:-5]+'Calibration', #This is a fairly ugly way of linking calibration functions to Price Models 
                             'PriceModel':calibration_type,
                             'ID': reference }
        
        calibration_elem                    = Element ( tag='Calibration', attrib=calibration_config )
        calibration_elem.calibration        = ConstructCalibrationConfig ( calibration_config, calib )
        
        #link it to the root calibration tag
        self.calib_root.append ( calibration_elem )
        
        self.data[key]                      = ( fields, calibration_elem )

    def Delete(self, selection):
        key = (json.loads(selection) or [None])[0]
        #delete the calibration
        self.calib_root.remove ( self.data[key] [1] )
        #delte the view data
        del self.data[key]

class MyStream(object):
    def __init__(self, log_widget):
        self.widget         = log_widget
        self.widget.value   = u''
        
    def flush(self):
        pass

    def write(self, record):
        self.widget.value+=record
        
class Workbench(object):
    '''
    Main class for the crstal WorkBench
    Needs to be able to load and save config files/trade files
    '''
    def __init__(self, market_data='', trade_data='', calib_data='', calendar='', rundate=''):
        self.config           = Parser(None)
        if market_data:
            self.config.ParseMarketfile(market_data) if market_data.endswith('.dat') else self.config.ParseJson(market_data)
        if trade_data:
            if isinstance(trade_data, list):
                #load up a list of hedges and partition
                self.config.ProcessHedgeCounterparties(trade_data)
                #clear the filenames
                trade_data = ''
            else:
                #load up a single counterparty
                self.config.ParseTradefile(trade_data) if trade_data.endswith('.aap') else self.config.ParseJson(trade_data)
                #default a pfe recon file
                FieldMappings ['Calculation']['fields']['PFE_Recon_File']['value']=os.path.splitext(trade_data)[0]+'.csv'
                
        if calib_data:
            self.config.ParseCalibrationfile(calib_data)
            
        if calendar:
            self.config.ParseCalendarfile(calendar)
            
        #Setup some useful defaults
        FieldMappings ['Calculation']['fields']['Run_Date']['value'] = rundate
        
        #load the file selectors
        self.filenames          = widgets.VBox()
        self.trade_file         = FileWidget(description='Load from file', filename=trade_data)
        self.market_file        = FileWidget(description='Load from file', filename=market_data)
        self.calib_file         = FileWidget(description='Load from file', filename=calib_data)
        
        self.trade_area, self.save_trade_button, self.save_trade_name  = self.buildFilewidget(self.trade_file, 'Trade data')
        self.market_area, self.save_market_button, self.save_market_name  = self.buildFilewidget(self.market_file, 'Market data')
        self.calib_area, self.save_calib_button, self.save_calib_name  = self.buildFilewidget(self.calib_file, 'Calibration data')

        self.filenames.children = [self.market_area, self.trade_area, self.calib_area]

        #the main containers/widgets
        self.main       = None
        self.log_area   = widgets.Textarea()

        formatter           = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logstream      = MyStream(self.log_area)
        self.loghandler     = logging.StreamHandler(self.logstream)        
        self.loghandler.setFormatter(formatter)
        
        self.log            = logging.getLogger()
        self.log.handlers   = []
        self.log.setLevel(logging.DEBUG)
        self.log.addHandler(self.loghandler)
        
        #connect the file handlers
        self.link_file_traits( self.market_file, lambda filename: self.config.ParseJson if filename.endswith('.json') else self.config.ParseMarketfile, self.save_market_name )
        self.link_file_traits( self.trade_file, lambda filename: self.config.ParseJson if filename.endswith('.json') else self.config.ParseTradefile, self.save_trade_name )
        self.link_file_traits( self.calib_file, lambda filename: self.config.ParseCalibrationfile )
        #now the save file handlers
        self.link_save_traits ( self.save_trade_name, self.save_trade_button, lambda filename: self.config.WriteTradedataJson if filename.endswith('.json') else self.config.WriteTradefile )
        self.link_save_traits ( self.save_market_name, self.save_market_button, lambda filename: self.config.WriteMarketdataJson if filename.endswith('.json') else self.config.WriteMarketfile )
        #display it
        self.refresh()
        
    def buildFilewidget(self, file_widget, label):
        header       = widgets.HTML ( value='<h4>{0}:</h4>'.format( label ) )
        save_button  = widgets.Button ( description='Save' )
        filename     = widgets.Text ( description='Save to filename', value=file_widget.filename )
        container    = widgets.FlexBox ( padding='4px', border_width='2px', border_style='outset', background_color='#DCE6FA', children = [header, file_widget, filename, save_button ] )
        
        return container, save_button, filename
    
    def refresh(self):
        #Load the pages            
        self.market       = MarketPricePage ( self.config )
        self.factors      = RiskFactorsPage ( self.config )
        self.portfolio    = PortfolioPage ( self.config )
        self.calculations = CalculationPage ( self.config )
        self.calibrations = CalibrationPage ( self.config )
        
        pages           = [self.market, self.portfolio, self.factors, self.calibrations, self.calculations]
        self.tabs       = widgets.Tab(children=[self.filenames]+[x.main_container for x in pages])

        #disable the current widget
        if self.main:
            self.main.visible = False
            
        #draw a new one
        self.main = widgets.VBox()
        self.main.children = [self.tabs, self.log_area]
        
        display(self.main)
        
        self.tabs.set_title(0, 'Files')
        self.tabs.set_title(1, 'Market')
        self.tabs.set_title(2, 'Portfolio')
        self.tabs.set_title(3, 'Price Factors')
        self.tabs.set_title(4, 'Calibrations')
        self.tabs.set_title(5, 'Calculations')

    def link_save_traits(self, save_filename, save_button, call_back ):
        
        def file_save(widget):
            try:
                #callback is a function that returns a function depending on the filename
                call_back ( save_filename.value )( save_filename.value )
            except:
                logging.error('Could not save file {0}'.format(save_filename.value))
            else:
                logging.info('Saved file {0}'.format(save_filename.value))
                
        save_button.on_click( file_save )
            
    def link_file_traits(self, file_widget, call_back, label=None):
        
        def file_loading():
            logging.info('Loading {0}'.format(file_widget.filename))

        def file_loaded():
            with open(file_widget.filename, 'wb') as f:
                f.write( file_widget.value )
            try:
                #callback is a function that returns a function depending on the filename
                call_back(file_widget.filename)(file_widget.filename)
            except:
                logging.error('Could not parse file {0}'.format(file_widget.filename))
                logging.error('File {0} NOT loaded'.format(file_widget.filename))
            else:
                logging.info('Loaded file {0}'.format(file_widget.filename))
                if label:
                    label.value=file_widget.filename
                self.refresh()
            
        def file_failed():
            logging.error('Could not load file contents of {0}'.format(file_widget.filename))
            
        file_widget.on_trait_change(file_loading, 'filename')
        file_widget.on_trait_change(file_loaded, 'value')
        
        # The registration of the handler is different than the two traits above.
        # Instead the API provided by the CallbackDispatcher must be used.
        file_widget.errors.register_callback(file_failed)

        #TODO - make files generated on the server available for download
        #trick to do it is via a HTML widget
        #widgets.HTML(value="<a href='mlp.py' target='_blank'>mlp.py</a><br>")

    def __del__(self):
        self.log.removeHandler(self.loghandler)
        self.loghandler.close()