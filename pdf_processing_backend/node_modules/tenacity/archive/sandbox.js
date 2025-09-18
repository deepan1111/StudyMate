#!/usr/bin/env node --harmony
const defaults = {
  excludes: ['_overrides.scss', '_windup.scss']
};

// Depends
const fs = require('fs');
const path = require('path');
const ROOT_PATH = './scss';

class Tenacity {
  constructor(options) {
    this.options = Object.assign({}, defaults, options);
    this.init();
  }

  getFiles(dir, newFiles) {
    const allFiles = newFiles || [];
    const files = fs.readdirSync(dir);
    files.forEach(file => {
      const isDotFile = file.startsWith('.');
      const isExcluded = this.options.excludes.includes(file);
      if (!isDotFile && !isExcluded) {
        const subDir = path.join(dir, file);
        const isDirectory = fs.statSync(subDir).isDirectory();
        if (isDirectory) {
          this.getFiles(subDir, allFiles);
        } else {
          allFiles.push(subDir);
        }
      }
    });
    return allFiles;
  }
  getFilesObj(dir, newFiles) {
    //array.some((e)=>str.includes(e)
    const allFiles = newFiles || {};
    const files = fs.readdirSync(dir);
    files.forEach(file => {
      const isDotFile = file.startsWith('.');
      const isExcluded = this.options.excludes.includes(file);
      if (!isDotFile && !isExcluded) {
        const noRootDir = dir.replace(path.normalize(ROOT_PATH) + path.sep, '');
        const subDir = path.join(dir, file);
        const stats = fs.statSync(subDir);
        if (stats.isDirectory()) {
          this.getFilesObj(subDir, allFiles);
        } else {
          const isProp = allFiles[noRootDir];
          if (isProp) {
            const isString = typeof isProp === 'string';
            if (isString) {
              allFiles[noRootDir] = [allFiles[noRootDir], file];
            } else {
              allFiles[noRootDir].push(file);
            }
          } else {
            const isSubDir = noRootDir.includes(path.sep);
            if (isSubDir) {
              const subDirArray = noRootDir.split(path.sep);
              allFiles[subDirArray[0]] = subDirArray[1];
            } else {
              allFiles[noRootDir] = file;
            }
          }
        }
      }
    });
    return allFiles;
  }
  getObject(fileList) {
    let obj = {};

    const noRootList = fileList.map(file =>
      file.replace(path.normalize(ROOT_PATH) + path.sep, '')
    );

    noRootList.forEach(dir => {
      const dirArray = dir.split(path.sep);
      console.log(dirArray);
      dirArray.forEach(file => {
        const { ext } = path.parse(file);
        const isDir = !path.extname(file);
        if (isDir) {
          obj;
        } else {
          if (ext === '.scss') {
            //console.log(file);
          }
        }
      });
    });
    return obj;
  }
  init() {
    // const fileListArray = this.getFiles(ROOT_PATH);
    // const fileListObject = this.getObject(fileListArray);
    // console.log(fileListObject);
    const fileList = this.getFilesObj(ROOT_PATH);
    console.log(fileList);
  }
}

new Tenacity();
// module.exports = Tenacity;
// module.exports.Options = Tenacity;
