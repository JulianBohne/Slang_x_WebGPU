#/usr/bin/env bash

set -e

echo Compiling slang to wgsl
# slangc slang/test-compute.slang -target wgsl -o wgsl/compiled-test-compute.wgsl
# slangc slang/matrix.slang -target wgsl -o wgsl/compiled-matrix.wgsl
slangc slang/present.slang -entry vert -target wgsl -o wgsl/compiled-present-vertex.wgsl
slangc slang/present.slang -entry frag -target wgsl -o wgsl/compiled-present-fragment.wgsl
slangc slang/splat.slang -target wgsl -o wgsl/compiled-splat.wgsl
echo Success!
