'use strict';
// Run existing legacy tests with the new flag OFF, without editing tests/files.
const fs=require('fs'),path=require('path'),original=fs.readFileSync;
fs.readFileSync=function(file,options){
 if(typeof file==='string'&&path.resolve(file)===path.resolve('cardcount_profile.json')){
  const profile=JSON.parse(original.call(fs,file,'utf8'));profile.physics_priority_v1.enabled=false;const text=JSON.stringify(profile);
  return typeof options==='string'||options?.encoding?text:Buffer.from(text);
 }
 return original.apply(fs,arguments);
};
require(path.resolve(process.argv[2]));
