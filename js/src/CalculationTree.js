define(["./Tree"], function(treebase) {
    
    var CalculationTreeView = treebase.TreeView.extend({

            perform_update: function ( tree, data, type_data ) {
            
                    var calculation_types = $.parseJSON(this.model.get('calculation_types'));
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
                                            "label"				: "Create New Calculation",
                                            "action"			: false,
                                            "submenu" : self.generateContextMethod(subtree, $node, calculation_types)
                                        },
                                    "Delete": {
                                        "separator_before"	: true,
                                        "label": "Delete Calculation",
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
        
    var CalculationTreeModel = treebase.TreeModel.extend({
        defaults: _.extend({}, treebase.TreeModel.prototype.defaults, {
            _model_name: 'CalculationTreeModel',
            _view_name: 'CalculationTreeView',

            calculation_types: ''
        })
    });

    return {
        CalculationTreeView: CalculationTreeView,
        CalculationTreeModel: CalculationTreeModel
    };
});
