import struct
import difflib
from nose.plugins.skip import Skip, SkipTest
import ubpf.assembler
import ubpf.disassembler
import testdata
import pytest
import os
_test_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../test-cases")

# Just for assertion messages
def try_disassemble(inst):
    data = struct.pack("=Q", inst)
    try:
        return ubpf.disassembler.disassemble(data).strip()
    except ValueError:
        return "<error>"

def check_datafile(filename):
    """
    Verify that the reassembling the output of the disassembler produces
    the same binary, and that disassembling the output of the assembler
    produces the same text.
    """
    data = testdata.read(_test_data_dir, filename)

    if 'asm' not in data:
        raise SkipTest("no asm section in datafile")

    assembled = ubpf.assembler.assemble(data['asm'])
    disassembled = ubpf.disassembler.disassemble(assembled)
    reassembled = ubpf.assembler.assemble(disassembled)
    disassembled2 = ubpf.disassembler.disassemble(reassembled)

    if disassembled != disassembled2:
        diff = difflib.unified_diff(disassembled.splitlines(), disassembled2.splitlines(), lineterm="")
        formatted = ''.join('  %s\n' % x for x in diff)
        raise AssertionError("Assembly differs:\n%s" % formatted)

    if assembled != reassembled:
        raise AssertionError("binary differs")

@pytest.mark.parametrize("filename", testdata.list_files(_test_data_dir))
def test_datafiles(filename):
    # This is now a regular test function that will be called once for each filename
    check_datafile(filename)
