define(["./Tree"], function(treebase) {
    
    var CalibrationTreeView = treebase.TreeView.extend({

            perform_update: function ( tree, data, type_data ) {
            
					var calibration_types = $.parseJSON(this.model.get('calibration_types'));
					var self = this;

					tree.jstree({
						"core"  : {"data":data, "check_callback" : true},
						"types" : type_data,
						"plugins" : [ "contextmenu", "sort", "unique", "types", "search" ],
						"contextmenu": {
							"items": function ($node) {
								var subtree = tree.jstree(true);
								return {
									"Create" : {
											"separator_before": false,
											"separator_after": true,
											"label"				: "Create New Calibration",
											"action"			: false,
											"submenu" : self.generateContextMethod(subtree, $node, calibration_types)
										},
									"Delete": {
										"separator_before"	: true,
										"label": "Delete Calibration",
										"action": function (data) {
											if ($node.type!="root")
												subtree.delete_node($node);
										}
									}
								}
							}
						}
					}) .on("rename_node.jstree",  this.handle_tree_change.bind(this))
						.on('select_node.jstree', this.handle_select.bind(this))
						.on('delete_node.jstree', this.handle_delete.bind(this));
			}

        });
        
    var CalibrationTreeModel = treebase.TreeModel.extend({
        defaults: _.extend({}, treebase.TreeModel.prototype.defaults, {
            _model_name: 'CalibrationTreeModel',
            _view_name: 'CalibrationTreeView',

            calibration_types: ''
        })
    });

    return {
        CalibrationTreeView: CalibrationTreeView,
        CalibrationTreeModel: CalibrationTreeModel
    };
});
