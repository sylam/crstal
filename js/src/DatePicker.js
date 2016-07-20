define(["jupyter-js-widgets"], function(widgets) {
	
     // Define the TreeView
	var DatePickerView = widgets.DOMWidgetView.extend({
        
            render: function() {
                this.$el.addClass('widget-hbox');
                /* Apply this class to the widget container to make  fit with the other built in widgets.*/
                // Create a label.
                this.$label = $('<div />')
                    .addClass('widget-label')
                    .appendTo(this.$el)
					.hide();

                // Create the date picker control.
                this.$date = $('<input />')
                    .attr('type', 'date')
                    .appendTo(this.$el);

				this.update(); // Set defaults.
            },

            update: function() {

                // Set the value of the date control and then call base.
                this.$date.val(this.model.get('value')); // ISO format "YYYY-MM-DDTHH:mm:ss.sssZ" is required

                // Hide or show the label depending on the existance of a description.
                var description = this.model.get('description');
                if (description == undefined || description == '') {
                    this.$label.hide();
                } else {
                    this.$label.show();
                    this.$label.text(description);
                }

                return DatePickerView.__super__.update.apply(this);
            },

            // Tell Backbone to listen to the change event of input controls (which the HTML date picker is)
            events: {
                "change": "handle_date_change"
            },

            // Callback for when the date is changed.
            handle_date_change: function(event) {
                this.model.set('value', this.$date.val());
                this.touch();
            },
        });		
	
	var DatePickerModel = widgets.DOMWidgetModel.extend({

        defaults: _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
            _model_name: "DatePickerModel",
			_view_name: "DatePickerView",
            _model_module : 'jupyter-crstal',
            _view_module : 'jupyter-crstal',
			
			value : '',
            description : ''
        })

    });

    return {
		DatePickerModel: DatePickerModel,
        DatePickerView: DatePickerView
    };
});
