#!/usr/bin/env bash
# Extract target LLVM and runtime dependencies; never install foreign packages
# into the build host. Host clang still produces the BPF objects.
set -euo pipefail
root=$(realpath -m "${1:?usage: setup-riscv64-sysroot.sh OUTPUT_DIRECTORY}")
mkdir -p "$root"
work=$(mktemp -d)
mkdir -p "$work/lists/partial" "$work/cache/archives/partial" "$work/debs"
cat > "$work/sources.list" <<'EOF'
deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports noble main universe
deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports noble-updates main universe
deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports noble-security main universe
EOF
apt_options=(
  -o APT::Architecture=riscv64
  -o APT::Architectures::=riscv64
  -o "Dir::Etc::sourcelist=$work/sources.list"
  -o Dir::Etc::sourceparts=-
  -o "Dir::State::lists=$work/lists"
  -o "Dir::Cache=$work/cache"
  -o Debug::NoLocking=1
)
apt-get "${apt_options[@]}" update
cd "$work/debs"
apt-get "${apt_options[@]}" download \
  llvm-18-dev:riscv64 libllvm18:riscv64 llvm-18:riscv64 \
  llvm-18-runtime:riscv64 llvm-18-tools:riscv64 llvm-18-linker-tools:riscv64 \
  libc6:riscv64 libc6-dev:riscv64 linux-libc-dev:riscv64 \
  libgcc-s1:riscv64 libstdc++6:riscv64 \
  libelf-dev:riscv64 libelf1t64:riscv64 \
  zlib1g-dev:riscv64 zlib1g:riscv64 libzstd-dev:riscv64 libzstd1:riscv64 \
  libffi-dev:riscv64 libffi8:riscv64 libedit-dev:riscv64 libedit2:riscv64 \
  libncurses-dev:riscv64 libtinfo6:riscv64 libncurses6:riscv64 libncursesw6:riscv64 \
  libxml2-dev:riscv64 libxml2:riscv64 libicu74:riscv64 \
  liblzma-dev:riscv64 liblzma5:riscv64 libz3-dev:riscv64 libz3-4:riscv64 \
  libbsd-dev:riscv64 libbsd0:riscv64 libmd-dev:riscv64 libmd0:riscv64
for package in ./*.deb; do
  dpkg-deb -x "$package" "$root"
done
# Ubuntu packages assume a merged-/usr filesystem; dpkg-deb does not create
# the base-files package's directory aliases for us.
for directory in lib bin sbin; do
  if [[ ! -e "$root/$directory" ]]; then
    ln -s "usr/$directory" "$root/$directory"
  fi
done
# bpftime uses Boost headers only. Keep these architecture-neutral headers
# inside the sysroot so CMake does not discard /usr/include as a host implicit
# include directory when generating cross-compiler flags.
cp -a /usr/include/boost "$root/usr/include/"
echo "RISC-V sysroot extracted to $root (package cache: $work/debs)"
