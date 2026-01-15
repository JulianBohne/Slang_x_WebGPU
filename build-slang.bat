@echo off

echo Compiling slang to wgsl
call slangc slang\test-compute.slang -target wgsl -o wgsl/compiled-test-compute.wgsl || exit /b 1
echo Success!
