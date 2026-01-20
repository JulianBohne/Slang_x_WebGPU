@echo off

echo Compiling slang to wgsl
@REM call slangc slang/test-compute.slang -target wgsl -o wgsl/compiled-test-compute.wgsl || exit /b 1
@REM call slangc slang/matmul.slang -target wgsl -O3 -o wgsl/compiled-matmul.wgsl || exit /b 1
call slangc slang/splat.slang -entry vert -target wgsl -o wgsl/compiled-splat-vertex.wgsl || exit /b 1
call slangc slang/splat.slang -entry frag -target wgsl -o wgsl/compiled-splat-fragment.wgsl || exit /b 1
echo Success!
