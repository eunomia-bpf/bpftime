// SPDX-License-Identifier: MIT
package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/rlimit"
)

const (
	eventMalloc uint32 = 1
	eventFree   uint32 = 2
)

type eventKey struct {
	PID   uint32
	Event uint32
}

func main() {
	var (
		objectPath = flag.String("object", "malloc_free.bpf.o", "compiled BPF object")
		libcPath   = flag.String("libc", "", "path to libc.so.6")
		interval   = flag.Duration("interval", time.Second, "print interval")
		duration   = flag.Duration("duration", 0, "optional runtime before exiting")
	)
	flag.Parse()

	if *libcPath == "" {
		path, err := findLibc()
		if err != nil {
			exitf("find libc: %v", err)
		}
		*libcPath = path
	}

	if err := rlimit.RemoveMemlock(); err != nil {
		exitf("remove memlock limit: %v", err)
	}

	spec, err := ebpf.LoadCollectionSpec(*objectPath)
	if err != nil {
		exitf("load BPF spec: %v", err)
	}

	coll, err := ebpf.NewCollection(spec)
	if err != nil {
		exitf("create BPF collection: %v", err)
	}
	defer coll.Close()

	prog := coll.Programs["kprobe__generic"]
	if prog == nil {
		exitf("BPF program kprobe__generic not found")
	}
	counts := coll.Maps["allocation_events"]
	if counts == nil {
		exitf("BPF map allocation_events not found")
	}

	exe, err := link.OpenExecutable(*libcPath)
	if err != nil {
		exitf("open executable %q: %v", *libcPath, err)
	}

	mallocLink, err := exe.Uprobe("malloc", prog, &link.UprobeOptions{
		Cookie: uint64(eventMalloc),
	})
	if err != nil {
		exitf("attach malloc uprobe: %v", err)
	}
	defer mallocLink.Close()

	freeLink, err := exe.Uprobe("free", prog, &link.UprobeOptions{
		Cookie: uint64(eventFree),
	})
	if err != nil {
		exitf("attach free uprobe: %v", err)
	}
	defer freeLink.Close()

	fmt.Printf("attached malloc/free uprobes to %s\n", *libcPath)

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	ticker := time.NewTicker(*interval)
	defer ticker.Stop()

	var timer <-chan time.Time
	if *duration > 0 {
		timer = time.After(*duration)
	}

	for {
		select {
		case <-ticker.C:
			if err := printCounts(counts); err != nil {
				exitf("read counts: %v", err)
			}
		case <-timer:
			return
		case <-stop:
			return
		}
	}
}

func printCounts(counts *ebpf.Map) error {
	var (
		key   eventKey
		value uint64
	)

	fmt.Printf("%s\n", time.Now().Format("15:04:05"))
	iter := counts.Iterate()
	for iter.Next(&key, &value) {
		fmt.Printf("  pid=%-6d %-6s calls=%d\n", key.PID,
			eventName(key.Event), value)
	}
	return iter.Err()
}

func eventName(event uint32) string {
	switch event {
	case eventMalloc:
		return "malloc"
	case eventFree:
		return "free"
	default:
		return fmt.Sprintf("event-%d", event)
	}
}

func findLibc() (string, error) {
	candidates := []string{
		"/lib/x86_64-linux-gnu/libc.so.6",
		"/usr/lib/x86_64-linux-gnu/libc.so.6",
		"/lib64/libc.so.6",
		"/usr/lib64/libc.so.6",
		"/lib/aarch64-linux-gnu/libc.so.6",
		"/usr/lib/aarch64-linux-gnu/libc.so.6",
		"/lib/riscv64-linux-gnu/libc.so.6",
		"/usr/lib/riscv64-linux-gnu/libc.so.6",
	}
	for _, path := range candidates {
		if fileExists(path) {
			return path, nil
		}
	}

	path, err := findLibcWithLdconfig()
	if err == nil {
		return path, nil
	}

	return "", errors.New("libc.so.6 not found; pass -libc")
}

func findLibcWithLdconfig() (string, error) {
	out, err := exec.Command("ldconfig", "-p").Output()
	if err != nil {
		return "", err
	}
	scanner := bufio.NewScanner(strings.NewReader(string(out)))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.Contains(line, "libc.so.6") {
			continue
		}
		parts := strings.Split(line, "=>")
		if len(parts) != 2 {
			continue
		}
		path := strings.TrimSpace(parts[1])
		if fileExists(path) {
			return path, nil
		}
	}
	return "", scanner.Err()
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func exitf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
