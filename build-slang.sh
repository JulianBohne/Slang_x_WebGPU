#/usr/bin/env bash

set -e

echo Compiling slang to wgsl
# slangc slang/test-compute.slang -target wgsl -o wgsl/compiled-test-compute.wgsl
# slangc slang/matrix.slang -target wgsl -o wgsl/compiled-matrix.wgsl
slangc slang/splat.slang -entry vert -target wgsl -o wgsl/compiled-splat-vertex.wgsl
slangc slang/splat.slang -entry frag -target wgsl -o wgsl/compiled-splat-fragment.wgsl
echo Success!
