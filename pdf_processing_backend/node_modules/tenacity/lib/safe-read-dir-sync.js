const FS = require('fs');

function safeReadDirSync(path) {
  let dirData = {};
  try {
    dirData = FS.readdirSync(path);
  } catch (ex) {
    if (ex.code == 'EACCES')
      //User does not have permissions, ignore directory
      return null;
    else throw ex;
  }
  return dirData;
}

module.exports = safeReadDirSync;
