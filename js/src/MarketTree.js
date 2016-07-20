define(["./Tree"], function(treebase) {
    
    var MarketTreeView = treebase.TreeView.extend({

            perform_update: function ( tree, data, type_data ) {
                var market_prices = $.parseJSON(this.model.get('market_prices'));
                var self = this;

                tree.jstree({
                    "core"  : {"data":data, "check_callback" : true},
                    "types" : type_data,
                    "plugins" : [ "contextmenu", "sort", "unique", "types", "search" ],
                    "contextmenu":  {
                        "items": function ($node) {
                            var subtree = tree.jstree(true);
                            //console.log("Context");
                            var create = {
                                "CreateGroup" : {
                                        "separator_before": false,
                                        "separator_after": false,
                                        "label"				: "Create Market Price",
                                        "action"			: false,
                                        "submenu" : self.generateContextMethod(subtree, $node, market_prices['MarketPrices'])
                                    }
                            };
							var create_point = {
                                "CreatePoint" : {
                                        "separator_before": false,
                                        "separator_after": false,
                                        "label"				: "Create Point",
                                        "action"			: false,
                                        "submenu" : self.generateContextMethod(subtree, $node, market_prices['PointFields'])
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
                                    items = $.extend({}, create_point, del);
                                    break;
                                default:
                                    items = del;
                            }
                            
                            return items;
                        }
                    }
                }) .on("rename_node.jstree",  this.handle_tree_change.bind(this)) 
                    .on('select_node.jstree', this.handle_select.bind(this))
                    .on('delete_node.jstree', this.handle_delete.bind(this)); 
			}

        });
        
    var MarketTreeModel = treebase.TreeModel.extend({
        defaults: _.extend({}, treebase.TreeModel.prototype.defaults, {
            _model_name: 'MarketTreeModel',
            _view_name: 'MarketTreeView',

            market_prices: ''
        })
    });

    return {
        MarketTreeView: MarketTreeView,
        MarketTreeModel: MarketTreeModel
    };
});
