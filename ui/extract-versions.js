// extract-versions.js
const fs = require('fs');

try {
  const lockfile = JSON.parse(fs.readFileSync('package-lock.json', 'utf8'));

  if (lockfile.packages) { // For lockfileVersion 2 or 3
    for (const key in lockfile.packages) {
      if (key.startsWith('node_modules/')) {
        const packageName = key.substring('node_modules/'.length);
        const version = lockfile.packages[key].version;
        if (packageName && version) {
          console.log(`${packageName} ${version}`);
        }
      }
    }
  } else if (lockfile.dependencies) { // For lockfileVersion 1
    for (const packageName in lockfile.dependencies) {
      const version = lockfile.dependencies[packageName].version;
      if (packageName && version) {
        console.log(`${packageName} ${version}`);
      }
    }
  } else {
    console.error("Could not find 'packages' or 'dependencies' section in package-lock.json");
  }
} catch (error) {
  console.error("Error reading or parsing package-lock.json:", error);
}
