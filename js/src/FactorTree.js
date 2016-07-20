define(["./Tree"], function(treebase) {
    
    var FactorTreeView = treebase.TreeView.extend({

            generateActionMethod: function (subtree, $node, riskfactortype) 
			{
				return function(data) {
					$node = subtree.create_node($node, {"type":"default", "data":riskfactortype});
					subtree.edit($node, riskfactortype+".");
				};
			},

			generateContextMethod: function (subtree, $node, RiskFactors)
			{
				var submenus = {};
				if ($node.type!="root") { $node = $node.parent; }
				for (var i in RiskFactors) {
					var key = RiskFactors[i];
					submenus [ key ] = {
										"separator_before": false,
										"separator_after": false,
										"label"				: key,
										"action"			: this.generateActionMethod(subtree, $node, key)
									};
				}
				return submenus;
			},

			perform_update: function ( tree, data, type_data ) {
                var risk_factors = $.parseJSON(this.model.get('risk_factors'));
                var self = this;

                tree.jstree({
                    "core"  : {"data":data, "check_callback" : true},
                    "types" : type_data,
                    "plugins" : [ "contextmenu", "sort", "unique", "types", "search", "checkbox" ],
                    "contextmenu": {
                        "items": function ($node) {
                            var subtree = tree.jstree(true);
                            return {
                                "Create" : {
                                        "separator_before": false,
                                        "separator_after": true,
                                        "label"				: "Create Risk Factor",
                                        "action"			: false,
                                        "submenu" : self.generateContextMethod(subtree, $node, risk_factors)
                                    },
                                "Delete": {
                                    "separator_before"	: true,
                                    "label": "Delete Risk Factor",
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
                    .on('deselect_node.jstree', this.handle_deselect.bind(this))
                    .on('delete_node.jstree', this.handle_delete.bind(this)); 
			},

			handle_select: function(event, obj) {
				if (obj!=null) {
					this.model.set('selected', obj.node.text );
					var list_selected = this.$tree.jstree("get_selected", true).map (function (x) {return x["text"];} ); 
					this.model.set ( 'allselected', JSON.stringify( list_selected ) );
					//$("#"+obj.node.id).css("color","white");
				}
				this.touch();
			},
			
			handle_deselect: function(event, obj) {
				if (obj!=null) {
					var list_selected = this.$tree.jstree("get_selected", true).map (function (x) {return x["text"];} ); 
					this.model.set ( 'allselected', JSON.stringify( list_selected ) );
					//$("#"+obj.node.id).css("color","white");
				}
				this.touch();
			},

			// Callback for when the user deletes a node
			handle_delete: function(event, obj) {
				if (obj!=null) {
					this.model.set('deleted', obj.node.text );
				}
				this.touch();
			},

			handle_tree_change: function(event, obj) {
				if (obj!=null) {
					if ( obj.node.text.indexOf(obj.node.data)!=0 )
					{
						//console.log('Create',obj.node.text, obj.node.data);
						this.$tree.jstree("rename_node", obj.node, obj.node.data+"."+obj.node.text);
					}
					this.model.set('created', obj.node.text );
				}
				//this.model.set('value', JSON.stringify(this.$tree.jstree("get_json")));
				this.touch();
			}

        });
        
    var FactorTreeModel = treebase.TreeModel.extend({
        defaults: _.extend({}, treebase.TreeModel.prototype.defaults, {
            _model_name: 'FactorTreeModel',
            _view_name: 'FactorTreeView',

            risk_factors: '',
            all_selected: []
        })
    });

    return {
        FactorTreeView: FactorTreeView,
        FactorTreeModel: FactorTreeModel
    };
});
