require('flot');
require('flottime');
require('flotnavigate');

define(["./Tree","jupyter-js-widgets","handsontable","underscore"], function(treebase, widgets, Handsontable, _) {
    
    //setup Flot for drawing stuff
    //useful definition to format the date
    function yyyymmdd(date) {
       var yyyy = date.getFullYear().toString();
       var mm = (date.getMonth()+1).toString(); // getMonth() is zero-based
       var dd  = date.getDate().toString();
       return yyyy + "-" + (mm[1]?mm:"0"+mm[0]) + "-" + (dd[1]?dd:"0"+dd[0]); // padding
    };
    
    function settabledata(data, format_time) {
        console.log('set');
        var columns = [];
        if (data.length>1) {
            columns.push( ['term'].concat(data.map(function(x) {return x['label'];})) );
            for (var i=0; i< data[0]['data'].length; i++) {
                var row = data[0]['data'][i].slice(0);
                if (format_time)
                    row[0] = yyyymmdd( new Date(row[0]) );
                for (var j=1; j<data.length; j++ ) {
                    row.push(data[j]['data'][i][1]);
                }
                columns.push( row );
            }
        } else {
            columns = data[0]['data'];
        }
        return columns;        
    }

    function gettabledata(ht) {
        console.log('get');
        var columns = [];

        if (ht.countCols ()>2) {
            var terms = ht.getDataAtCol(0).slice(1,-1);
            var labels = ht.getDataAtRow(0).slice(1);
            for (var i=0; i< labels.length; i++) {
                columns.push ({'label':labels[i], 'data':_.zip(terms, ht.getDataAtCol(i+1).slice(1,-1))});
            }
        } else {
            columns= [ {'label':'None', 'data':ht.getData()} ];
        }
        return columns;        
    }
    
    var FlotView = widgets.DOMWidgetView.extend({
				
			update: function() {
				//var flot = this.$flot;            
				var data = $.parseJSON(this.model.get('value'));
				this.renderGraph(data);
				return FlotView.__super__.update.apply(this);
			},
			
			render: function(){
				//console.log("render");
				
				this.$el.addClass('widget-hbox');
				
				this.$label = $('<div />')
								.addClass('widget-label')
								.appendTo(this.$el);
				//always initialize with the label        
				this.$label.text(this.model.get('description'));
				
				this.$flot = $('<div/>')
							 .addClass('flotplot')
							 .appendTo(this.$el);
				
				this.$table = $('<div/>')
							 //.addClass('handsonplot')
							 .appendTo(this.$el);				

				this.displayed.then(_.bind(this.update, this));
			},

			renderGraph: function(data) {
				var view = this;
				
				$.plot( this.$flot, data,   {
												legend: { show: true, container: '.flotlegend' },
												series: {
													lines: {
														show: true
													},
													points: {
														show: true
													},
													shadowSize: 0
												},                                                
												zoom: {
													interactive: true
												},
												pan: {
													interactive: true
												}
											} );
											
				// Create the Handsontable table.
                this.hot = new Handsontable(view.$table[0], 
                    {
					  data: settabledata(data, false),
					  // when working in HoT, don't listen for command mode keys
					  afterSelection: function(){ IPython.keyboard_manager.disable(); },
					  afterDeselect: function(){ IPython.keyboard_manager.enable(); },
					  startRows  : 3,
					  startCols  : 3,
					  contextMenu: true,
					  minSpareRows: 1,
					 // the data changed. `this` is the HoT instance
					  afterChange: function(changes, source){
						// don't update if we did the changing!
						if(source === "loadData"){ return; }
							view.handle_table_change(this.getData());
					  },
					  width : 400,
					  height : 300
					});
			},
			
			//events: {"change": "handle_table_change"},
			
			handle_table_change: function(event) {
				// Get the data, and serialize it in JSON.
				var data = gettabledata(this.hot);
				// Update the model with the JSON string.
				this.model.set('value', JSON.stringify(data));
				// Don't touch this...
				this.touch();
			}
			
	});
        
    var FlotModel = widgets.DOMWidgetModel.extend({
        defaults: _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
            _model_name: 'FlotModel',
            _view_name: 'FlotView'
        })
    });
    
    var FlotTreeView = treebase.TreeView.extend({
        
            update: function() {
				var selected = this.model.get('selected');
				var curve_data = $.parseJSON(this.model.get('profiles'));
				
				//only draw the profile if its in the curve_data
				if (selected in curve_data)
				{
					this.renderGraph(curve_data[selected]);
				}
                
				return FlotTreeView.__super__.update.apply(this);
			},
            
            render: function() {
				//console.log("render");
				
				this.$el.addClass('widget-hbox');
				
				this.$label = $('<div />')
								.addClass('widget-label')
								.appendTo(this.$el);
								
				//always initialize with the label        
				this.$label.text(this.model.get('description'));				
				
				this.$layout = $('<div/>')
								 .addClass('widget-vbox')
								 .appendTo(this.$el);
				
				this.$search  = $('<input type="text" />')
					.addClass('input')
					.addClass('widget-text')
					.appendTo(this.$layout);
					
				// Add a <div> in the widget area.
				this.$tree = $('<div />')
					.appendTo(this.$layout);
					
				this.to = false;	
				
				this.$flot = $('<div/>')
							 .addClass('flotplot')
							 .appendTo(this.$layout);
				
				this.$table = $('<div/>')
							 .appendTo(this.$layout);
				
				//Only show the flot if we are not drawing a MTM
				if (this.model.get('description')=="MTM" || this.model.get('description')=="Cashflows")
				{
					this.only_table = true;
					this.$flot.hide();
				} else {
					this.only_table = false;
				}				

				this.displayed.then(_.bind(this.update, this));
			},
            
            renderGraph: function(data) {
				//name the view			 
				var view = this;		
				var format_time = false;

				if (!this.only_table) {
					$.plot( this.$flot, data,   {
												legend: { show: true, container: '.flotlegend' },
												xaxis: { mode: "time" },
												series: {
													lines: {
														show: true
													},
													points: {
														show: true
													},
													shadowSize: 0
												},                                                
												zoom: {
													interactive: true
												},
												pan: {
													interactive: true
												}
											} );
					format_time = true;						
				}
				// Create the Handsontable table.
                this.hot = new Handsontable(view.$table[0], 
                                    {
									  data: settabledata(data, format_time),
									  // when working in HoT, don't listen for command mode keys
									  afterSelection: function(){ IPython.keyboard_manager.disable(); },
									  afterDeselect: function(){ IPython.keyboard_manager.enable(); },
									  //startRows  : 3,
									  //startCols  : 3,
									  copyColsLimit : 5000,
									  copyRowsLimit : 5000,
									 // the data changed. `this` is the HoT instance
									  afterChange: function(changes, source){
										if(source === "loadData"){ return; }
										view.handle_table_change(this.getData());
									  },
									  width:  500,
									  height: 300
									});
			},
            
            handle_table_change: function(event) {
				//don't do anything
				
				// Don't touch this...
				this.touch();
			}

    });
        
    var FlotTreeModel = treebase.TreeModel.extend({
        defaults: _.extend({}, treebase.TreeModel.prototype.defaults, {
            _model_name: 'FlotTreeModel',
            _view_name: 'FlotTreeView',

            profiles: ''
        })
    });

    return {
        FlotView:FlotView,
        FlotModel:FlotModel,
        FlotTreeView: FlotTreeView,
        FlotTreeModel: FlotTreeModel
    };
});
