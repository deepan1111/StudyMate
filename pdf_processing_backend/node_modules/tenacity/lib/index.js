#!/usr/bin/env node --harmony

// Depends
const _ = require('lodash');
const FS = require('fs');
const PATH = require('path');
const getFiles = require('./get-files');
const ROOT_PATH = './scss';
const camelCase = require('camelcase');
const jsonfile = require('jsonfile');
const isTenacity = PATH.basename(process.cwd()) === 'tenacity';

const defaults = {
  scssDir: './sandbox/scss',
  jsonDir: './sandbox',
  excludes: ['_windup.scss']
};

class Tenacity {
  constructor(options) {
    this.options = Object.assign({}, defaults, options);
    if (isTenacity) {
      this.init();
    }
  }
  removeWhiteSpace(str) {
    return str.replace(/ /g, '');
  }
  removeNewLine(str) {
    return str.replace(/\r?\n|\r/g, '');
  }
  removeComment(str) {
    return this.removeWhiteSpace(
      this.removeNewLine(
        str.replace(/\/\*[\s\S]*?\*\/|([^\\:]|^)\/\/.*$/gm, '')
      )
    );
  }
  objectifySass(globalContent) {
    const data = {};
    const contentArray = globalContent.split(';');
    contentArray.forEach(styleProperties => {
      const styleProperty = styleProperties.replace(':', '\x01').split('\x01');
      if (styleProperty[1]) {
        const styleName = camelCase(styleProperty[0].replace('$', ''));
        data[styleName] = styleProperty[1].replace('!default', '');
      }
    });
    return data;
  }
  getSassData(dirData, filename) {
    const [{ content }] = dirData.children.filter(
      child => child.name === filename
    );
    const cleanContent = this.removeComment(content);
    return this.objectifySass(cleanContent);
  }
  writeJsonFile(data, filename) {
    const file = PATH.join(this.options.jsonDir, filename);
    jsonfile.writeFile(file, data, { spaces: 2 }, err => {
      if (err) {
        console.error(err);
      } else {
        console.log('\x1b[1m%s\x1b[0m', 'TENACITY');
        console.log(
          '\x1b[32m%s\x1b[0m',
          `${PATH.basename(this.options.scssDir)} converted to ${PATH.basename(
            file
          )}`
        );
      }
    });
  }
  init() {
    const dirData = getFiles(this.options.scssDir, this.options);
    //console.log(JSON.stringify(dirData, null, 4));
    const globalData = this.getSassData(dirData, '_global.scss');
    const overridesData = this.getSassData(dirData, '_overrides.scss');
    const mergedData = _.merge(globalData, overridesData);
    const file = PATH.join(this.options.jsonDir, 'styleguide-data.json');
    // console.log(JSON.stringify(mergedData, null, 4));
    jsonfile.readFile(file, (err, syleguideJsonData) => {
      if (!_.isEqual(syleguideJsonData, dirData)) {
        this.writeJsonFile(mergedData, 'global-data.json');
        this.writeJsonFile(dirData, 'styleguide-data.json');
      } else {
        console.log('\x1b[1m%s\x1b[0m', 'TENACITY');
        console.log('\x1b[32m%s\x1b[0m', 'Files are updated');
      }
    });
  }
  apply(compiler) {
    compiler.plugin('done', () => {
      this.init();
    });
  }
}
if (isTenacity) {
  new Tenacity();
}

module.exports = Tenacity;
module.exports.Options = Tenacity;
