#/usr/bin/env bash

set -xe

echo Compiling slang to wgsl
# slangc slang/test-compute.slang -target wgsl -o wgsl/compiled-test-compute.wgsl
slangc slang/matmul.slang -target wgsl -O3 -o wgsl/compiled-matmul.wgsl
echo Success!
