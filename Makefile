# data-beans: Data Backend for Extraction And Neighbourhood Search.
#
# One package, two binaries: `data-beans` and `data-beans-sim` (the latter
# behind the `sim` feature). A single `cargo install --path .` with `sim` on
# installs both, so there is no per-binary loop here. The HDF5 detection
# below is kept in step with the legume-rs Makefile.
#
# data-beans does no GPU work, so there is no backend to choose.
#
# HDF5 (.h5/.h5ad I/O) is opt-in, as libhdf5 isn't on every host:
#   make install HDF5=on
BINARIES := data-beans data-beans-sim

# libhdf5 discovery (called only when HDF5=on or auto-detection runs).
#
# Why we can't just trust h5cc: on some HPC module systems h5cc is on PATH
# but its -showconfig output points at unreachable paths, so hdf5-metno-sys's
# own discovery fails with "Unable to locate HDF5 root directory and/or
# headers". We bypass that by setting HDF5_DIR ourselves to a prefix where
# we've verified both `include/hdf5.h` and a `libhdf5` library exist.
#
# Probes, in order of preference:
#   1. $HDF5_DIR if the caller already set it
#   2. $CONDA_PREFIX (caller is in an active conda env shipping HDF5)
#   3. h5cc on PATH → derive prefix via `dirname $(dirname $(which h5cc))`
#      (h5cc lives at <prefix>/bin/h5cc, so two dirnames gives the prefix)
#   4. Common system prefixes: /opt/homebrew, /usr/local, /usr
#
# Override with `HDF5_DIR=<prefix>` if your install lives somewhere unusual.
H5CC_PATH := $(shell command -v h5cc 2>/dev/null)
# Linux + macOS: avoid `readlink -f` (BSD readlink on macOS doesn't have it).
# `dirname $(dirname …)` is fine here because h5cc always lives at
# `<prefix>/bin/h5cc` whether the entry on PATH is the real file or a symlink.
H5CC_PREFIX := $(if $(H5CC_PATH),$(shell dirname $$(dirname $(H5CC_PATH))))

HDF5_CANDIDATES := $(HDF5_DIR) $(CONDA_PREFIX) $(H5CC_PREFIX) /opt/homebrew /usr/local /usr

# Stage 1: pick the first candidate that has BOTH `<prefix>/include/hdf5.h`
# and `<prefix>/lib*/libhdf5.{so,a,dylib}`. This handles conda, homebrew,
# `/usr/local`, and the HPC-Anaconda case where h5cc -showconfig is unreliable
# but the install is intact under a clean prefix.
HDF5_DETECTED_DIR := $(firstword $(foreach d,$(HDF5_CANDIDATES), \
    $(if $(strip $(d)), \
    $(if $(wildcard $(d)/include/hdf5.h), \
    $(if $(or $(wildcard $(d)/lib/libhdf5.so), \
              $(wildcard $(d)/lib/libhdf5.a), \
              $(wildcard $(d)/lib/libhdf5.dylib), \
              $(wildcard $(d)/lib64/libhdf5.so), \
              $(wildcard $(d)/lib64/libhdf5.a)), \
    $(d))))))

# Stage 2 (Debian/Ubuntu split layout): header lives at
# `/usr/include/hdf5/serial/hdf5.h` and library at
# `/usr/lib/<arch>-linux-gnu/hdf5/serial/libhdf5.so` — no single prefix has
# both, so we can't set HDF5_DIR. But `libhdf5-dev` ships pkg-config files
# (`hdf5` and/or `hdf5-serial`) and hdf5-metno-sys's own discovery succeeds
# there. Detecting this just flips HDF5 on; HDF5_DIR stays unset.
ifeq ($(HDF5_DETECTED_DIR),)
HDF5_PKGCFG_OK := $(shell if pkg-config --exists hdf5 2>/dev/null || pkg-config --exists hdf5-serial 2>/dev/null; then echo yes; fi)
else
HDF5_PKGCFG_OK := yes
endif

ifeq ($(HDF5_PKGCFG_OK),yes)
DEFAULT_HDF5 := on
else
DEFAULT_HDF5 := off
endif

HDF5 ?= $(DEFAULT_HDF5)

# Only export HDF5_DIR when stage 1 found a clean prefix. In the stage-2
# (Debian split) case we deliberately leave HDF5_DIR unset and let
# hdf5-metno-sys's internal probe handle the discovery.
ifeq ($(HDF5),on)
ifneq ($(HDF5_DETECTED_DIR),)
export HDF5_DIR := $(HDF5_DETECTED_DIR)
endif
endif

# Validate so a typo (e.g. `HDF5=yes`) fails fast instead of silently
# producing an unknown --features flag.
ifeq (,$(filter $(HDF5),on off))
$(error Unknown HDF5='$(HDF5)'; valid values are: on, off)
endif

# The `sim` feature is always on so that `data-beans-sim` is built too.
comma := ,
CARGO_FEATURES := --features sim$(if $(filter on,$(HDF5)),$(comma)hdf5)

.PHONY: all install uninstall build test clean help

help:
	@echo "data-beans Makefile"
	@echo ""
	@echo "Auto-detected HDF5 support: $(DEFAULT_HDF5)$(if $(HDF5_DETECTED_DIR), (HDF5_DIR=$(HDF5_DETECTED_DIR)))"
	@echo ""
	@echo "Targets:"
	@echo "  install    - Install $(BINARIES)"
	@echo "  uninstall  - Uninstall both binaries"
	@echo "  build      - Release build"
	@echo "  test       - Run all tests"
	@echo "  clean      - Clean build artifacts"
	@echo ""
	@echo "Overrides:"
	@echo "  make <target> HDF5={on|off}   # default = auto-detected above"
	@echo "  HDF5_DIR=<prefix> make ...    # override the detected prefix"

all: install

install:
	@echo "Installing $(BINARIES) (HDF5: $(HDF5))..."
	cargo install --locked --force --path . $(CARGO_FEATURES)

uninstall:
	cargo uninstall data-beans

build:
	cargo build --release $(CARGO_FEATURES)

test:
	cargo test --features sim

clean:
	cargo clean
