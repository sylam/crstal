define(["jupyter-js-widgets","handsontable","underscore"], function(widgets, Handsontable, _) {
	
     // Define the TableView
		var TableView = widgets.DOMWidgetView.extend({
            
            initialize: function(options) {
                TableView.__super__.initialize.apply(this, arguments);
                this.loaded = false;
            },

			update: function() {
				//var data = $.parseJSON(this.model.get('value'));
                var val   = this.model.get('value');
                var data = (val==="") ? null : $.parseJSON(val);
                    
                if (!this.loaded)
                {
                    var view = this;
                    
                    switch (this.label) {
                        case "Matrix":
                            this.hot = new Handsontable(view.$table[0], {
                                  // when working in HoT, don't listen for command mode keys
                                  data : data,
                                  afterSelection: function(){ IPython.keyboard_manager.disable(); },
                                  afterDeselect: function(){ IPython.keyboard_manager.enable(); },
                                  colHeaders: view.colHeaders,
                                  rowHeaders: view.colHeaders,
                                  columns: view.colTypes,
                                  startCols : view.colHeaders.length,
                                  startRows : view.colHeaders.length,
                                 // the data changed. `this` is the HoT instance
                                  afterChange: function(changes, source) {
                                    // don't update if we did the changing!
                                    if(source === "loadData"){ return; }
                                    view.handle_table_change(this.getData());
                                  },
                                  width:  700,
                                  height: 300
                                });
                            break;
                        case "Model Configuration":
                            this.hot = new Handsontable(view.$table[0], {
                                  data : data,
                                  // when working in HoT, don't listen for command mode keys
                                  afterSelection: function(){ IPython.keyboard_manager.disable(); },
                                  afterDeselect: function(){ IPython.keyboard_manager.enable(); },
                                  //colHeaders is now an object with the headers and the row autocomplete metadata
                                  colHeaders: view.colHeaders,
                                  columns: view.colTypes,
                                  startCols : view.colHeaders.length,
                                 // the data changed. `this` is the HoT instance
                                  afterChange: function(changes, source) {
                                    // don't update if we did the changing!
                                    if (source === "loadData") { return; }
                                    view.handle_table_change(this.getData());
                                  },
                                  contextMenu: true,
                                  minSpareRows  : 1,
                                  width:  700,
                                  height: 300
                                });
                                this.hasContext = true;
                            break;
                        default:
                            // Create the Handsontable table.
                            this.hot = new Handsontable(view.$table[0], {
                                  data : data,
                                  // when working in HoT, don't listen for command mode keys
                                  afterSelection: function(){ IPython.keyboard_manager.disable(); },
                                  afterDeselect: function(){ IPython.keyboard_manager.enable(); },
                                  colHeaders: view.colHeaders,
                                  columns: view.colTypes,
                                  manualColumnMove: true,
                                  minSpareRows: 1,
                                  startRows  : 1,
                                  startCols : view.colHeaders.length,
                                  // the data changed. `this` is the HoT instance
                                  afterChange: function(changes, source){
                                    // don't update if we did the changing!
                                    if(source === "loadData"){ return; }
                                    view.handle_table_change(this.getData());
                                  },
                                  width:  400,
                                  height: 200
                                });
                    }
                    this.loaded = true;
				}
                else
                {
                    this.hot.loadData(data);
                }
                
				return TableView.__super__.update.apply(this);
			},

			render: function() {
				this.$el.addClass('widget-hbox');

				this.$label = $('<div />')
								.addClass('widget-label')
								.appendTo(this.$el);
								
				//always initialize with the label        
				var label = this.model.get('description');
				this.$label.text(label);
				
				this.$table = $('<div/>')
							 .appendTo(this.$el);
				
				//load up the column type info			 
				this.colTypes = $.parseJSON(this.model.get('colTypes'));
				//load up the column header info			 
				this.colHeaders = $.parseJSON(this.model.get('colHeaders'));
				this.label = label;
                
                //update the table
                this.displayed.then(_.bind(this.update, this));
			},
            
			handle_table_change: function(data) {
				// JS --> PYTHON UPDATE.
				// Update the model with the JSON string.
				if ( (data.length>1 && data[0].length==this.colHeaders.length) || this.model.get('description')=="Model Configuration")
				{
					this.model.set('value', JSON.stringify(data));
				}
				// Don't touch this...
				this.touch();
			}
		});
	
	var TableModel = widgets.DOMWidgetModel.extend({

        defaults: _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
            _model_name: "TableModel",
			_view_name: "TableView",
            _model_module : 'jupyter-crstal',
            _view_module : 'jupyter-crstal',
			
			value : '',
			colTypes : '',
			colHeaders : '',
			colHeaders : ''
        })

    });

    return {
		TableModel: TableModel,
        TableView: TableView
    };
});
