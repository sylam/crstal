require('jstree');

define(["jupyter-js-widgets", "underscore"], function(widgets, _) {
	
     // Define the TreeView
	var TreeView = widgets.DOMWidgetView.extend({
		
		initialize: function(options) {
            TreeView.__super__.initialize.apply(this, arguments);
			this.loaded = false;
		},
		
		generateActionMethod: function (subtree, $node, node_data, obj_type) 
		{
			return function(data) {
				$node = subtree.create_node($node, {"type":node_data, "data":obj_type});
				subtree.edit($node, obj_type+".");
			};
		},

		generateContextMethod: function (subtree, $node, obj_types)
		{
			var submenus = {};
			for (var key in obj_types) {
			  if (obj_types.hasOwnProperty(key)) {
				submenus[key] = {
									"separator_before": false,
									"separator_after": false,
									"label"				: key,
									"action"			: this.generateActionMethod(subtree, $node, obj_types[key], key)
								};
			  }
			}
			return submenus;
		},

		perform_update: function ( tree, data, type_data ) {				
			tree.jstree({
				"core"  : {"data":data},
				"types" : type_data,
				"plugins" : [ "sort", "unique", "types", "search" ]
			}).on('select_node.jstree', this.handle_select.bind(this));
		},
		
		update: function() {
			if (!this.loaded) {
				var tree = this.$tree;            
				var data = $.parseJSON(this.model.get('value'));
				var type_data = $.parseJSON(this.model.get('type_data'));
				
				//var self = this;               
				
				this.perform_update(tree, data, type_data);
				  
				this.$search
					.on("keyup", this.handle_search.bind(this))
					.on("paste", this.handle_search.bind(this));
					
				//tree is now loaded
				this.loaded=true;
			}
			
			//We can only create the tree once via an update  
			return TreeView.__super__.update.apply(this);
		},

		render: function() {
			// Create the control.
			
			this.$search  = $('<input type="text" />')
				.addClass('input')
				.addClass('widget-text')
				.appendTo(this.$el);
				
			// Add a <div> in the widget area.
			this.$tree = $('<div />')
				.appendTo(this.$el);
				
			this.to = false;	
            
			//call update
            this.displayed.then(_.bind(this.update, this));
		},

		get_json_path: function(obj) {
			var path = this.$tree.jstree("get_path",obj.node);
			return JSON.stringify( path.slice(1) );
		},
		
		handle_search: function(event, obj) {
			var self = this;
			
			if (this.to) { clearTimeout(this.to); }
			this.to = setTimeout ( function () {
				var v = self.$search.val();
				self.$tree.jstree(true).search(v);
			}, 250);
		},
		
		// Callback for when the user makes a new selection
		handle_select: function(event, obj) {
			if (obj!=null) {
				//console.log(obj.node);
				this.model.set('selected', this.get_json_path(obj) );
			}
			this.touch();
		},
		
		// Callback for when the user creates a new node
		handle_tree_change: function(event, obj) {
			if (obj!=null) {					
				if ( obj.node.text.indexOf(obj.node.data)!=0 )
				{
					//console.log('Create',obj.node.text, obj.node.data);
					this.$tree.jstree("rename_node", obj.node, obj.node.data+"."+obj.node.text);
				}
				this.model.set('created', this.get_json_path(obj) );
			}
			this.touch();
		},
		
		// Callback for when the user deletes a node
		handle_delete: function(event, obj) {
			if (obj!=null) {
				//console.log('Delete',obj.node);
				this.model.set('deleted', this.get_json_path(obj));
			}
			this.touch();
		}
	});
	
	var TreeModel = widgets.DOMWidgetModel.extend({

        defaults: _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
            _model_name: "TreeModel",
			_view_name: "TreeView",
            _model_module : 'jupyter-crstal',
            _view_module : 'jupyter-crstal',
			
			value : '',
			type_data : '',
			selected : '',
			created : '',
			deleted : ''
        })

    });

    return {
		TreeModel: TreeModel,
        TreeView: TreeView
    };
});
