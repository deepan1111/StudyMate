const FS = require('fs');
const PATH = require('path');
const safeReadDirSync = require('./safe-read-dir-sync');

function getFiles(path, options, onEachFile) {
  const name = PATH.basename(path);
  const item = { name, path };
  let stats;

  try {
    stats = FS.statSync(path);
  } catch (e) {
    return null;
  }
  if (options.excludes.includes(name) || name.startsWith('.')) return null;

  if (stats.isFile()) {
    const content = FS.readFileSync(path, 'utf8');
    //console.log(content);
    item.content = content;
    if (onEachFile) {
      onEachFile(item, PATH);
    }
  } else if (stats.isDirectory()) {
    let dirData = safeReadDirSync(path);
    if (dirData === null) return null;

    item.children = dirData
      .map(child => getFiles(PATH.join(path, child), options, onEachFile))
      .filter(e => !!e);
  } else {
    return null; // Or set item.size = 0 for devices, FIFO and sockets ?
  }
  return item;
}

module.exports = getFiles;
