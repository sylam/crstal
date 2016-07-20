define(["jupyter-js-widgets"], function(widgets) {
	
     // Define the TreeView
	var FilePickerView = widgets.DOMWidgetView.extend({
			render: function(){
				
				this.$el.addClass('widget-hbox');
				
				this.$label = $('<div />')
						.addClass('widget-label')
						.appendTo(this.$el)
						.hide();
				
				this.$file = $('<input />')
						.attr('type', 'file')
						.appendTo(this.$el);
				
				this.update(); // Set defaults.
			},
			
			update: function() {
				// Set the value of the file control and then call base.
				var model_filename = this.model.get('filename');
				if (model_filename != '') {
					model_filename = ' ('+model_filename+')'
				}

				// Hide or show the label depending on the existance of a description.
				var description = this.model.get('description');
				if (description == undefined || description == '') {
					this.$label.hide();
				} else {
					this.$label.show();
					this.$label.text( description );
				}

				return FilePickerView.__super__.update.apply(this);
			},
			
			events: {
				// List of events and their handlers.
				'change': 'handle_file_change',
			},
		   
			handle_file_change: function(evt) { 
				// Handle when the user has changed the file.
				
				// Retrieve the first (and only!) File from the FileList object
				var file = evt.target.files[0];
				if (file) {

					// Read the file's textual content and set value to those contents.
					var that = this;
					var file_reader = new FileReader();
					file_reader.onload = function(e) {
						that.model.set('value', e.target.result);
						that.touch();
					}
					file_reader.readAsText(file);
				} else {

					// The file couldn't be opened.  Send an error msg to the
					// back-end.
					this.send({ 'event': 'error' });
				}

				// Set the filename of the file.
				this.model.set('filename', file.name);
				this.touch();
			},
		});
	
	var FilePickerModel = widgets.DOMWidgetModel.extend({

        defaults: _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
            _model_name: "FilePickerModel",
			_view_name: "FilePickerView",
            _model_module : 'jupyter-crstal',
            _view_module : 'jupyter-crstal',
			
			value : '',
			filename : '',
			description : ''
        })

    });

    return {
		FilePickerModel: FilePickerModel,
        FilePickerView: FilePickerView
    };
});
