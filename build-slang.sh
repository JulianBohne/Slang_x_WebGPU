#/usr/bin/env bash

set -xe

echo Compiling slang to wgsl
# slangc slang/test-compute.slang -target wgsl -o wgsl/compiled-test-compute.wgsl
slangc slang/matrix.slang -target wgsl -o wgsl/compiled-matrix.wgsl
echo Success!
