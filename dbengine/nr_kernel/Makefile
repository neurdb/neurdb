# List of subdirectories to process
SUBDIRS := nr_ext nr_pipeline nr_am nr_store nr_molqo

# PG_CONFIG can be overridden by the caller (e.g. make PG_CONFIG=/opt/neurdb/bin/pg_config)
PG_CONFIG ?= pg_config

# Build directory (for out-of-source build when source dir is read-only)
NEURDBPATH ?= /code/neurdb-dev
BUILD_DIR ?= $(NEURDBPATH)/build/nr_kernel
SRC_DIR := $(CURDIR)

# Default target: build all subprojects
.PHONY: all $(SUBDIRS) install debug clean help sync distclean build-local install-local debug-local

# Sync source to build directory
sync:
	@echo "Syncing source to build directory..."
	@mkdir -p $(BUILD_DIR)
	@rsync -a --delete --exclude='*.o' --exclude='*.so' --exclude='build/' $(SRC_DIR)/ $(BUILD_DIR)/

# Default 'all' target: build directly in source dir (no sync required)
all: build-local

# Local build (called from build dir, no sync)
build-local: $(SUBDIRS)

# Rule to handle each subdirectory
$(SUBDIRS):
	@echo "Building $@..."
	$(MAKE) -C $@ PG_CONFIG=$(PG_CONFIG)

# Install target: install directly from source dir (no sync required)
install: install-local

# Local install (called from build dir)
install-local:
	@for dir in $(SUBDIRS); do \
		echo "Running 'make install' in $$dir"; \
		$(MAKE) -C $$dir install PG_CONFIG=$(PG_CONFIG) || exit 1; \
	done

# Debug target: build directly in source dir
debug: debug-local

# Local debug (called from build dir)
debug-local:
	@for dir in $(SUBDIRS); do \
		echo "Running 'make debug' in $$dir"; \
		$(MAKE) -C $$dir debug PG_CONFIG=$(PG_CONFIG); \
	done

# Clean target: clean build directories in source dir
clean:
	@for dir in $(SUBDIRS); do \
		echo "Cleaning in $$dir"; \
		$(MAKE) -C $(SRC_DIR)/$$dir clean || true; \
	done

distclean:
	@echo "Cleaning CMake caches..."
	@find $(SRC_DIR) -name "CMakeCache.txt" -delete
	@find $(SRC_DIR) -name "cmake_install.cmake" -delete
	@find $(SRC_DIR) -type d -name "CMakeFiles" -exec rm -rf {} +

# Help target to list available commands
help:
	@echo "Available targets:"
	@echo "  all       - Build all subprojects"
	@echo "  install   - Install all subprojects"
	@echo "  debug     - Debug-build all subprojects"
	@echo "  clean     - Clean all subprojects"
	@echo "  sync      - Sync source to build directory (legacy dev workflow)"
	@echo "  distclean - Clean CMake caches"
	@echo ""
	@echo "Build directory: $(BUILD_DIR)"
