# List of subdirectories to process
SUBDIRS := nr_ext nr_pipeline

# Default target: build all subprojects
.PHONY: all $(SUBDIRS) install debug clean help

# Default 'all' target builds all subdirectories
all: $(SUBDIRS)

# Rule to handle each subdirectory
$(SUBDIRS):
	@echo "Building $@..."
	$(MAKE) -C $@

# Install target: run 'make install' in each subdirectory
install:
	@for dir in $(SUBDIRS); do \
		echo "Running 'make install' in $$dir"; \
		$(MAKE) -C $$dir install; \
	done

# Debug target: run 'make debug' in each subdirectory
debug:
	@for dir in $(SUBDIRS); do \
		echo "Running 'make debug' in $$dir"; \
		$(MAKE) -C $$dir debug; \
	done

# Clean target: clean all subdirectories
clean:
	@for dir in $(SUBDIRS); do \
		echo "Cleaning in $$dir"; \
		$(MAKE) -C $$dir clean; \
	done

# Help target to list available commands
help:
	@echo "Available targets:"
	@echo "  all      - Build all subprojects"
	@echo "  install  - Install all subprojects"
	@echo "  debug    - Debug all subprojects"
	@echo "  clean    - Clean all subprojects"
