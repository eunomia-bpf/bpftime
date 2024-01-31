import os
import platform
import tempfile
import struct
import re
from subprocess import Popen, PIPE
from nose.plugins.skip import Skip, SkipTest
import ubpf.assembler
import testdata
import pytest
import os
_test_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../test-cases")
VM = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", "build/test/test_Tests")
try:
    xrange
except NameError:
    xrange = range

def jit_supported_platform():
    """Is the JIT supported on the current platform."""
    return platform.machine() in ['amd64', 'x86_64', 'arm64', 'aarch64']
    
def check_datafile(filename):
    """
    Given assembly source code and an expected result, run the eBPF program and
    verify that the result matches. Uses the JIT compiler.
    """
    if not jit_supported_platform():
        raise SkipTest("JIT is not supported on the current platform")

    data = testdata.read(_test_data_dir, filename)
    if 'asm' not in data and 'raw' not in data:
        raise SkipTest("no asm or raw section in datafile")
    if 'result' not in data and 'error' not in data and 'error pattern' not in data:
        raise SkipTest("no result or error section in datafile")
    if not os.path.exists(VM):
        raise SkipTest("VM not found")
    if 'no jit' in data:
        raise SkipTest("JIT disabled for this testcase (%s)" % data['no jit'])

    if 'raw' in data:
        code = b''.join(struct.pack("=Q", x) for x in data['raw'])
    else:
        code = ubpf.assembler.assemble(data['asm'])

    memfile = None

    if 'mem' in data:
        memfile = tempfile.NamedTemporaryFile()
        memfile.write(data['mem'])
        memfile.flush()

    num_register_offsets = 20
    if 'no register offset' in data:
        # The JIT relies on a fixed register mapping for the call instruction
        num_register_offsets = 1

    try:
        for register_offset in xrange(0, num_register_offsets):
            cmd = [VM]
            if memfile:
                cmd.extend(['-m', memfile.name])
            if 'reload' in data:
                cmd.extend(['-R'])
            if 'unload' in data:
                cmd.extend(['-U'])
            cmd.extend(['-j', '-r', str(register_offset), '-'])

            vm = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

            stdout, stderr = vm.communicate(code)
            stdout = stdout.decode("utf-8")
            stderr = stderr.decode("utf-8")
            stderr = stderr.strip()

            if 'error' in data:
                if data['error'] != stderr:
                    raise AssertionError("Expected error %r, got %r" % (data['error'], stderr))
            elif 'error pattern' in data:
                if not re.search(data['error pattern'], stderr):
                    raise AssertionError("Expected error matching %r, got %r" % (data['error pattern'], stderr))
            else:
                if stderr:
                    raise AssertionError("Unexpected error %r" % stderr)

            if 'result' in data:
                if vm.returncode != 0:
                    raise AssertionError("VM exited with status %d, stderr=%r" % (vm.returncode, stderr))
                expected = int(data['result'], 0)
                result = int(stdout, 0)
                if expected != result:
                    raise AssertionError("Expected result 0x%x, got 0x%x, stderr=%r" % (expected, result, stderr))
            else:
                if vm.returncode == 0:
                    raise AssertionError("Expected VM to exit with an error code")
    finally:
        if memfile:
            memfile.close()

@pytest.mark.parametrize("filename", testdata.list_files(_test_data_dir))
def test_datafiles(filename):
    # This is now a regular test function that will be called once for each filename
    check_datafile(filename)

