host_arch := x86_64

CC := gcc
CFLAGS := -Wall -fPIC -fPIE -g
LDFLAGS := -lpthread -lm -ldl

frida_version := 16.1.2
frida_os_arch := linux-$(host_arch)
frida_core_devkit_url := https://github.com/frida/frida/releases/download/$(frida_version)/frida-core-devkit-$(frida_version)-$(frida_os_arch).tar.xz
frida_gum_devkit_url := https://github.com/frida/frida/releases/download/$(frida_version)/frida-gum-devkit-$(frida_version)-$(frida_os_arch).tar.xz

all: bin/inject bin/agent.so bin/victim

bin/inject: inject.c ext/frida-core/.stamp
	@mkdir -p $(@D)
	$(CC) -Wl,-pie $(CFLAGS) -O2  -I./ext/frida-core inject.c -o $@ -L./ext/frida-core -lfrida-core $(LDFLAGS)

bin/agent.so: agent.c ext/frida-gum/.stamp
	@mkdir -p $(@D)
	$(CC) -shared $(CFLAGS) -O2  -I./ext/frida-gum agent.c -o $@ -L./ext/frida-gum -lfrida-gum $(LDFLAGS)

bin/victim: victim.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LDFLAGS) -O0 victim.c -o $@

ext/frida-core/.stamp:
	@mkdir -p $(@D)
	@rm -f $(@D)/*
	curl -Ls $(frida_core_devkit_url) | xz -d | tar -C $(@D) -xf -
	@touch $@

ext/frida-gum/.stamp:
	@mkdir -p $(@D)
	@rm -f $(@D)/*
	curl -Ls $(frida_gum_devkit_url) | xz -d | tar -C $(@D) -xf -
	@touch $@

clean:
	rm -rf bin

.PHONY: all deploy clean
