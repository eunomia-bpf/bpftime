import struct
import difflib
from nose.plugins.skip import Skip, SkipTest
import ubpf.disassembler
import testdata
import pytest
import os
_test_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../test-cases")

def check_datafile(filename):
    """
    Verify that the result of disassembling the 'raw' section matches the
    'asm' section.
    """
    data = testdata.read(_test_data_dir, filename)
    if 'asm' not in data:
        raise SkipTest("no asm section in datafile")
    if 'raw' not in data:
        raise SkipTest("no raw section in datafile")

    binary = b''.join(struct.pack("=Q", x) for x in data['raw'])
    result = ubpf.disassembler.disassemble(binary)

    # TODO strip whitespace and comments from asm
    if result.strip() != data['asm'].strip():
        diff = difflib.unified_diff(data['asm'].splitlines(), result.splitlines(), lineterm="")
        formatted = ''.join('  %s\n' % x for x in diff)
        raise AssertionError("Assembly differs:\n%s" % formatted)

@pytest.mark.parametrize("filename", testdata.list_files(_test_data_dir))
def test_datafiles(filename):
    # This is now a regular test function that will be called once for each filename
    check_datafile(filename)
