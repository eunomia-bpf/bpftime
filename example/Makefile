# Find direct subdirectories with Makefiles
SUBDIRS := $(shell find . -maxdepth 2 -name "Makefile" -not -path "./Makefile" -exec dirname {} \; | sed 's|^\./||' | sort)

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	@$(MAKE) -C $@

clean:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done