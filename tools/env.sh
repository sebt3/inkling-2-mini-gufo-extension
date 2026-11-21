# toolchain of the gufo-runtime:0.33.0 image (nix store paths, pinned to that image)
export CLR=/nix/store/al4qwi7jfyrdkyb25xsri8b00b1agj1r-clr-10.1
export PATH=/nix/store/92bpbb2rmrkszhw83x63x7qzj9jsz324-rocm-toolchain/bin:/nix/store/r5ri78axx1qggkjq3vawywlzglg70b5w-gcc-wrapper-16.2.0/bin:/nix/store/rlpcs1h48m058z811xxkn5672v77p5qj-cmake-4.4.4/bin:/nix/store/nyjr5b3ib747yls4n9jq5ijr6wqg6wdc-ninja-1.13.2/bin:$PATH
export DEVLIB=/nix/store/gyzwfffwdwj8dpkrasiqd5hkr2adx844-rocm-device-libs-24.0.0-rocm/amdgcn/bitcode
export HIPFLAGS="-x hip --offload-arch=gfx1151 --rocm-path=$CLR --rocm-device-lib-path=$DEVLIB -isystem $CLR/include -L$CLR/lib -lamdhip64 -Wl,-rpath,$CLR/lib"
