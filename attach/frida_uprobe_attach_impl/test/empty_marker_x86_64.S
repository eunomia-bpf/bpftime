#if defined(__linux__) && defined(__x86_64__)
/*
 * Empty-marker uprobe target, hand-written so its first byte is always
 * `ret` (0xc3). A compiler-generated naked function can receive an
 * `endbr64` prefix under CET (-fcf-protection) on some GCC point releases;
 * Frida-Gum 16.1.2 accepts that prefixed form instead of rejecting it,
 * defeating the empty-marker rejection test. The GNU assembler emits these
 * bytes verbatim and never inserts endbr64 at an existing label, so the
 * marker stays ret-first regardless of the compiler's CET default.
 */
	.text
	.globl	__bpftime_test_empty_marker_target
	.type	__bpftime_test_empty_marker_target, @function
	.balign 8
__bpftime_test_empty_marker_target:
	ret
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	.size	__bpftime_test_empty_marker_target, . - __bpftime_test_empty_marker_target

/* Mark the stack non-executable (silences the GNU-stack linker note). */
	.section .note.GNU-stack, "", @progbits
#endif
