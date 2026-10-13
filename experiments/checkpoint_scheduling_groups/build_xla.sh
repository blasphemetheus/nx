#!/usr/bin/env bash
set -o pipefail
P=/home/blewf/.cache/xla_build/xla-bb760b047bdbfeff962f0366ad5cc782c98657e0
cd $P || exit 1
echo "start $(date)"
bazel build --jobs=12 --local_ram_resources=20480 \
  --define framework_shared_object=false -c opt \
  --config=cuda --action_env=TF_NVCC_CLANG=1 --@local_config_cuda//:cuda_compiler=nvcc \
  --repo_env=HERMETIC_CUDA_VERSION=12.9.1 --repo_env=HERMETIC_CUDNN_VERSION=9.8.0 \
  --repo_env=HERMETIC_NVSHMEM_VERSION=3.3.9 --repo_env=HERMETIC_NCCL_VERSION=2.27.7 \
  --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=sm_50,sm_60,sm_70,sm_80,sm_90,sm_100,compute_120 \
  --action_env=HOME=/home/blewf --repo_env=CC=clang --repo_env=CXX=clang++ \
  --copt=-Wno-error=unused-command-line-argument --copt=-Wno-gnu-offsetof-extensions \
  --copt=-Qunused-arguments --copt=-Wno-error=c23-extensions \
  //xla/extension:xla_extension
rc=$?
echo "bazel exit $rc at $(date)"
ls -la bazel-bin/xla/extension/xla_extension.tar.gz
exit $rc
