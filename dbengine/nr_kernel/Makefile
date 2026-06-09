# List of subdirectories to process
SUBDIRS := nr_ext nr_pipeline nr_am nr_store nr_molqo

# Build directory (for out-of-source build when source dir is read-only)
NEURDBPATH ?= /code/neurdb-dev
BUILD_DIR ?= $(NEURDBPATH)/build/nr_kernel
SRC_DIR := $(CURDIR)

# Default target: build all subprojects
.PHONY: all $(SUBDIRS) install debug clean help sync

# Sync source to build directory
sync:
	@echo "Syncing source to build directory..."
	@mkdir -p $(BUILD_DIR)
	@rsync -a --delete --exclude='*.o' --exclude='*.so' --exclude='build/' $(SRC_DIR)/ $(BUILD_DIR)/
	@echo "Cleaning CMake caches..."
	@find $(BUILD_DIR) -name "CMakeCache.txt" -delete 2>/dev/null || true
	@find $(BUILD_DIR) -name "cmake_install.cmake" -delete 2>/dev/null || true
	@find $(BUILD_DIR) -type d -name "CMakeFiles" -exec rm -rf {} + 2>/dev/null || true

# Default 'all' target: sync then build in build dir
all: sync
	@$(MAKE) -C $(BUILD_DIR) build-local

# Local build (called from build dir, no sync)
build-local: $(SUBDIRS)

# Rule to handle each subdirectory
$(SUBDIRS):
	@echo "Building $@..."
	$(MAKE) -C $@

# Install target: sync then install from build dir
install: sync
	@$(MAKE) -C $(BUILD_DIR) install-local

# Local install (called from build dir)
install-local:
	@for dir in $(SUBDIRS); do \
		echo "Running 'make install' in $$dir"; \
		$(MAKE) -C $$dir install; \
	done

# Debug target: sync then debug from build dir
debug: sync
	@$(MAKE) -C $(BUILD_DIR) debug-local

# Local debug (called from build dir)
debug-local:
	@for dir in $(SUBDIRS); do \
		echo "Running 'make debug' in $$dir"; \
		$(MAKE) -C $$dir debug; \
	done

# Clean target: clean build directory
clean:
	@if [ -d "$(BUILD_DIR)" ]; then \
		for dir in $(SUBDIRS); do \
			echo "Cleaning in $$dir"; \
			$(MAKE) -C $(BUILD_DIR)/$$dir clean || true; \
		done \
	fi

# Help target to list available commands
help:
	@echo "Available targets:"
	@echo "  all      - Sync and build all subprojects"
	@echo "  install  - Sync and install all subprojects"
	@echo "  debug    - Sync and debug all subprojects"
	@echo "  clean    - Clean all subprojects"
	@echo "  sync     - Sync source to build directory"
	@echo ""
	@echo "Build directory: $(BUILD_DIR)"
