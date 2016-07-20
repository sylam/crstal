// Entry point for the npmcdn bundle containing custom model definitions.
//
// It differs from the notebook bundle in that it does not need to define a
// dynamic baseURL for the static assets and may load some css that would
// already be loaded by the notebook otherwise.
// Load css
/*
require('handsontable/dist/pikaday/pikaday.css');
require('handsontable/dist/handsontable.css');
require('jstree/dist/themes/default/style.css');*/
require("./crstal.less");

// Export widget models and views, and the npm package version number.
module.exports = require('./jupyter-crstal.js');
module.exports['version'] = require('../package.json').version;
