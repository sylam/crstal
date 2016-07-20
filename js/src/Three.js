//load up the trackball controls
require("imports?THREE=three!three/examples/js/controls/TrackballControls.js");

//load up 3gl - at some stage will need to replace this entirely
define(["jupyter-js-widgets","three","handsontable","underscore"], function(widgets, THREE, Handsontable, _) {

		// setup three.js
		var WIDTH  = 400,
			HEIGHT = 300,
			CUBE   = 10;

		// set some camera attributes
		var VIEW_ANGLE = 45,
		  ASPECT = WIDTH / HEIGHT,
		  NEAR = 0.1,
		  FAR = 10000;

		function getColor(max,min,val){
			var MIN_L=50,MAX_L=120;
			var color = new THREE.Color();
			var h = 60/240;
			var s = 160/240;
			var l = (((MAX_L-MIN_L)/(max-min))*val)/240;
			color.setHSL(h,s,l);
			return color;
		}

		function initGrid(cols, rows) {
			var geometry = new THREE.Geometry();
			var color       = new THREE.Color();

			color.setRGB(155, 150, 250);
			var  textMaterial  	  = new THREE.MeshLambertMaterial({ vertexColors: color });
			textMaterial.shading = THREE.SmoothShading;

			for(var i=0;i<cols.length;i++) {
				geometry.vertices.push(new THREE.Vector3( (cols[i] - cols.ofs)*cols.scale, (rows[0]-rows.ofs)*rows.scale , 0));
				geometry.vertices.push(new THREE.Vector3( (cols[i] - cols.ofs)*cols.scale, (rows[rows.length-1]-rows.ofs)*rows.scale, 0));
			}

			for(var i=0;i<rows.length;i++){
				geometry.vertices.push(new THREE.Vector3((cols[0] - cols.ofs)*cols.scale, (rows[i]-rows.ofs)*rows.scale,0));
				geometry.vertices.push(new THREE.Vector3((cols[cols.length-1] - cols.ofs)*cols.scale,(rows[i]-rows.ofs)*rows.scale,0));
			}

			var material = new THREE.LineBasicMaterial( { color: 0x999999, opacity: 0.2 } );

			var line = new THREE.Line(geometry, material);
			line.type = THREE.LinePieces;
			return line;
		}

		function initGraph(cols, rows, surface)
		{
			var geometry = new THREE.Geometry();
			var colors = [];

			var width = cols.length, height =rows.length, concat = [].concat.apply([],surface);
			var min_val = Math.min.apply(null, concat), max_val = Math.max.apply(null, concat);
			var val_scale = CUBE/max_val;

			for (var x=0; x<width; x++){
				for (var y=0; y<height; y++) {
					var z = surface[x][y];
					if ( z == null ) { z = 0.0 } else { z = parseFloat(z) };
					geometry.vertices.push( new THREE.Vector3( (parseFloat(cols[x])-cols.ofs)*cols.scale, (parseFloat(rows[y])-rows.ofs )*rows.scale, z*val_scale) );
					colors.push(getColor(max_val,min_val,z));
				}
			}

			var offset = function(x,y){
				return x*height+y;
			}

			for(var x=0;x<width-1;x++){
				for(var y=0;y<height-1;y++){
					var vec0 = new THREE.Vector3(), vec1 = new THREE.Vector3(), n_vec = new THREE.Vector3();
                    
					// one of two triangle polygons in one rectangle
					vec0.subVectors(geometry.vertices[offset(x,y)],geometry.vertices[offset(x+1,y)]);
					vec1.subVectors(geometry.vertices[offset(x,y)],geometry.vertices[offset(x,y+1)]); 
					n_vec.crossVectors(vec0,vec1).normalize();
					geometry.faces.push(new THREE.Face3(offset(x,y),offset(x+1,y),offset(x,y+1), n_vec, [colors[offset(x,y)],colors[offset(x+1,y)],colors[offset(x,y+1)]]));
					geometry.faces.push(new THREE.Face3(offset(x,y),offset(x,y+1),offset(x+1,y), n_vec.negate(), [colors[offset(x,y)],colors[offset(x,y+1)],colors[offset(x+1,y)]]));
                    
					// the other one
					vec0.subVectors(geometry.vertices[offset(x+1,y)],geometry.vertices[offset(x+1,y+1)]);
					vec1.subVectors(geometry.vertices[offset(x,y+1)],geometry.vertices[offset(x+1,y+1)]); 
					n_vec.crossVectors(vec0,vec1).normalize();
					geometry.faces.push(new THREE.Face3(offset(x+1,y),offset(x+1,y+1),offset(x,y+1), n_vec, [colors[offset(x+1,y)],colors[offset(x+1,y+1)],colors[offset(x,y+1)]]));
					geometry.faces.push(new THREE.Face3(offset(x+1,y),offset(x,y+1),offset(x+1,y+1), n_vec.negate(), [colors[offset(x+1,y)],colors[offset(x,y+1)],colors[offset(x+1,y+1)]]));
				}
			}

			var material = new THREE.MeshLambertMaterial({ vertexColors: THREE.VertexColors});
			material.shading = THREE.SmoothShading;
			var mesh = new THREE.Mesh(geometry, material);
			return mesh;
		}

		var ThreeView = widgets.DOMWidgetView.extend({

			initialize: function(options) {

				this.renderer = new THREE.WebGLRenderer();

				this.renderer.setClearColor( 0xd0e0f0, 1);
				this.camera =
				  new THREE.PerspectiveCamera(
					VIEW_ANGLE,
					ASPECT,
					NEAR,
					FAR);

				this.scene = new THREE.Scene();

				// add the camera to the scene
				this.scene.add(this.camera);

				// the camera starts at 0,0,0
				// so pull it back
				this.camera.position.z = 2*CUBE;

				// start the renderer
				this.renderer.setSize(WIDTH, HEIGHT);

				var positions = [[10,10,10],[-10,-10,10],[-10,10,10],[10,-10,10],[10,10,-10],[-10,-10,-10],[-10,10,-10],[10,-10,-10]];

				for(var i=0;i<positions.length;i++){
					var light=new THREE.DirectionalLight(0xffffff);
					light.position.set(positions[i][0],positions[i][1],positions[i][2]);
					this.scene.add(light);
				}

				this.controls = new THREE.TrackballControls ( this.camera, this.renderer.domElement );

				this.controls.rotateSpeed = 1.0;
				this.controls.zoomSpeed = 1.2;
				this.controls.panSpeed = 0.8;

				this.controls.noZoom = false;
				this.controls.noPan = false;

				this.controls.staticMoving = true;
				this.controls.dynamicDampingFactor = 0.3;

				this.controls.addEventListener( 'change', this.draw.bind(this) );

				this.clock     = new THREE.Clock();
				this.selection = null;
				return ThreeView.__super__.initialize.apply(this, [options]);
			},

			refresh_grid: function() {
				var ht         = this.ht;
				this.rows    = ht.getDataAtRow(0).slice(1,-1);
				this.cols     = ht.getDataAtCol(0).slice(1,-1);            
				this.surface = ht.getData (1,1 , ht.countRows()-2, ht.countCols()-2);

				var redraw = true;
				for(var x = 0; x<this.cols.length;x++) {
					if ( this.cols[x] == null ) { redraw=false; break; }
				}

				for(var y = 0; y<this.rows.length;y++) {
					if ( this.rows[y] == null ) { redraw=false; break; }
				}

				if (redraw && this.rows.length>1 && this.cols.length>1) {
					var min_row  = Math.min.apply(null, this.rows),  max_row = Math.max.apply(null, this.rows);
					var min_col  = Math.min.apply(null, this.cols),  max_col = Math.max.apply(null, this.cols);
					this.rows.scale = CUBE/(max_row-min_row);
					this.cols.scale = CUBE/(max_col-min_col);
					this.rows.ofs   = (max_row+min_row)/2.0;
					this.cols.ofs   = (max_col+min_col)/2.0;

					//delete previous meshes
					var selectedObject = this.scene.getObjectByName("mesh");
					this.scene.remove( selectedObject );
					var selectedlines = this.scene.getObjectByName("lines");
					this.scene.remove( selectedlines );

					var mesh = initGraph ( this.cols, this.rows, this.surface );
					mesh.name = "mesh";
					this.scene.add(mesh);
					
					var grid = initGrid(this.cols, this.rows);
					grid.name = "lines";
					this.scene.add(grid);
				}
			},

			update: function() {
				//console.log("UPDATE");
				var all_data = $.parseJSON(this.model.get('value'));
				var self = this;
				
				if ( Object.prototype.toString.call( all_data ) === '[object Array]' ) {
					this.$modify.hide();
					this.all_data = {null:all_data};
					this.selection = null;
				} else {
					this.$modify.show();                
					this.$all_tenor.children().remove();
					
					var keys = Object.keys(all_data).sort( function (a,b) {return a - b;} );                                
					
					for (var key in keys)
					{
						 $('<option />')
							.text(keys[key])
							.attr('value_name', keys[key])
							.appendTo(self.$all_tenor);
					}
					
					this.$all_tenor.on('change', function (e, data) { self.handle_select_change(e);} );
					this.all_data = all_data;
					if (this.selection == null) { this.selection = keys[key]; }
					this.$all_tenor.val(this.selection);
				}
				
				//console.log( this.$modify.is(":visible") );
				var view = this;
				
				// Create the Handsontable table.
                this.ht = new Handsontable(view.$table[0], 
                    {
					  data: self.all_data[self.selection],
					  // when working in HoT, don't listen for command mode keys
					  afterSelection: function(){ IPython.keyboard_manager.disable(); },
					  afterDeselect: function(){ IPython.keyboard_manager.enable(); },

					  // the data changed. `this` is the HoT instance
					  afterChange: function(changes, source) {
						// don't update if we did the changing!
						  if (source === "loadData") { return; }
						  view.handle_data_change(this.getData());
					  },
					  minSpareRows: 1,
					  minSpareCols : 1,
					  contextMenu: true,
                      width:  500,
                      height: 300
					});

				this.refresh_grid();
				this.draw();
					
				return ThreeView.__super__.update.apply(this);
			},

			render: function(){ 
				var self = this;
				
				this.$el.addClass('widget-hbox');
				this.$label = $('<div />')
								.addClass('widget-label')
								.appendTo(this.$el);

				//always initialize with the label        
				this.$label.text(this.model.get('description'));
				
				//Now prepare to choose different surfaces
				this.$modify = $('<div/>')
								 .addClass('widget-label')
				
				this.$tenor = $('<input type="text" />')
								.addClass('input')
								.addClass('widget-tenor-text')
								.appendTo(this.$modify);
				
				this.$add_tenor = $('<button/>')
								.addClass('btn')
								.appendTo(this.$modify)
								.on('click', function (e) { self.handle_add(e);} );
				this.$add_tenor.text('Add');
				
				this.$rem_tenor = $('<button/>')
								.addClass('btn')
								.appendTo(this.$modify)
								.on('click', function (e) { self.handle_delete(e);} );
				
				this.$rem_tenor.text('Del');
				
				this.$all_tenor = $('<select/>')
								.addClass('widget-tenorlistbox')
								.appendTo(this.$modify);
				
				//hide this by default.            
				this.$modify.hide();
				
				this.$three = $('<div/>')
							 .appendTo(this.$el);

				this.$three.html(this.renderer.domElement);
				
				this.$modify.appendTo(this.$three);

				this.$table = $('<div/>')
							 //.addClass('handsonplot')
							 .appendTo(this.$el);
				
				this.anim();
				this.draw();
                
				this.displayed.then(_.bind(this.update, this));
			},

			handle_data_change: function(data) {
				// JS --> PYTHON UPDATE.
				var json;
				
				if (this.selection != null) {
					if (data!=null) {
						this.all_data[this.selection] = data;
					}
					json = JSON.stringify(this.all_data);
				} else {
					json = JSON.stringify(data);
				}
				
				// Update the model with the JSON string.
				this.model.set('value', json);            
				// Don't touch this...
				this.touch();
			},
			
			handle_delete: function (e) {
				// Handle when a tenor is to be deleted
				if ( this.$all_tenor.children().length>1 ) {
					var self      = this;
					var to_delete = this.$all_tenor.val();
					
					//find the next value that is in the selection 
					this.$all_tenor.children().each(function(){
						if (this.value != to_delete) {
							self.selection = this.value;
							return false;
						}
					});
					
					delete this.all_data [ to_delete ];
					this.handle_data_change(null);
				}
				// Don't touch this...
				this.touch();
			},    
			
			handle_add: function (e) {
				// Handle when a tenor is to be added
				var new_val = this.$tenor.val()
				if ( new_val ) {
					var exists = false;
					this.$all_tenor.children().each(function(){
						if (this.value == new_val) {
							exists = true;
							return false;
						}
					});
					
					if (!exists)
					{
                        var ht = this.ht;
						this.selection = new_val;
						this.handle_data_change( ht.getData() );
					}
				}
				this.touch();
			},    
			
			handle_select_change: function (e) {
				// Handle when a selection is changed.
				var  self = this;
				this.selection = this.$all_tenor.val();                        
				this.$tenor.val(this.selection);
                this.ht = new Handsontable(self.$table[0], 
                                    {
										data: self.all_data[self.selection],
										minSpareRows: 1,
										minSpareCols : 1,
										contextMenu: true
									 });
				
				this.refresh_grid();
				this.draw();
				this.touch();
			},    

			anim : function () {
				var animate = this.anim;
				requestAnimationFrame(animate.bind(this));
				var delta = this.clock.getDelta();
				this.controls.handleResize();
				this.controls.update(delta * 100);
			},

			draw : function() {
				this.renderer.render(this.scene, this.camera);
			}

		});
	
	var ThreeModel = widgets.DOMWidgetModel.extend({

        defaults: _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
            _model_name: "ThreeModel",
			_view_name: "ThreeView",
            _model_module : 'jupyter-crstal',
            _view_module : 'jupyter-crstal',
			
			description : ''
        })

    });

    return {
		ThreeModel: ThreeModel,
        ThreeView: ThreeView
    };
});
