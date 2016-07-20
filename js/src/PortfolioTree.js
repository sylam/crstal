define(["./Tree"], function(treebase) {
    
    var PortfolioTreeView = treebase.TreeView.extend({

            perform_update: function ( tree, data, type_data ) {

                var instruments = $.parseJSON(this.model.get('instrument_def'));
                var self = this;               
                
                tree.jstree({
                    "core"  : {"data":data, "check_callback" : true},
                    "types" : type_data,
                    "plugins" : [ "contextmenu", "sort", "types", "search", "unique" ],
                    "contextmenu": {
                        "items": function ($node) {
                            var subtree = tree.jstree(true);
                            //console.log("Context");
                            var create = {
                                "CreateGroup" : {
                                        "separator_before": false,
                                        "separator_after": false,
                                        "label"				: "Create Structure",
                                        "action"			: false,
                                        "submenu" : self.generateContextMethod(subtree, $node, instruments['STR'])
                                    },
                                "CreateIRDeriv" : {
                                        "separator_before": false,
                                        "separator_after": false,
                                        "label"				: "Create Interest Rate Derivative",
                                        "action"			: false,
                                        "submenu" : self.generateContextMethod(subtree, $node, instruments['IR'])
                                    },
                                "CreateFXDeriv" : {
                                        "separator_before": false,
                                        "separator_after": false,
                                        "label"				: "Create FOREX Derivative",
                                        "action"			: false,
                                        "submenu" : self.generateContextMethod(subtree, $node, instruments['FX'])
                                    },
                                "CreateEQDeriv" : {
                                        "separator_before": false,
                                        "separator_after": false,
                                        "label"				: "Create Equity Derivative",
                                        "action"			: false,
                                        "submenu" : self.generateContextMethod(subtree, $node, instruments['ED'])
                                    },
								"CreateENDeriv" : {
                                        "separator_before": false,
                                        "separator_after": false,
                                        "label"				: "Create Energy Derivative",
                                        "action"			: false,
                                        "submenu" : self.generateContextMethod(subtree, $node, instruments['EN'])
                                    },	
                                "CreateCreditDeriv" : {
                                        "separator_before": false,
                                        "separator_after": false,
                                        "label"				: "Create Credit Derivative",
                                        "action"			: false,
                                        "submenu" : self.generateContextMethod(subtree, $node, instruments['CR'])
                                    }
                            };
                            var del = {
                                "Delete": {
                                    "separator_before"	: true,
                                    "label": "Delete Node",
                                    "action": function (data) {
                                        subtree.delete_node($node);
                                    }
                                }
                            };
                            var items;
                            switch ($node.type) {
                                case "root":
                                    items = create;
                                    break;
                                case "group":
                                    items = $.extend({}, create, del);
                                    break;
                                default:
                                    items = del;
                            }
                            
                            return items;
                        }
                    }
                }).on('select_node.jstree', this.handle_select.bind(this))
                  .on('rename_node.jstree', this.handle_tree_change.bind(this))
                  .on('delete_node.jstree', this.handle_delete.bind(this));
			}

        });
        
    var PortfolioTreeModel = treebase.TreeModel.extend({
        defaults: _.extend({}, treebase.TreeModel.prototype.defaults, {
            _model_name: 'PortfolioTreeModel',
            _view_name: 'PortfolioTreeView',

            instrument_def: ''
        })
    });

    return {
        PortfolioTreeView: PortfolioTreeView,
        PortfolioTreeModel: PortfolioTreeModel
    };
});
