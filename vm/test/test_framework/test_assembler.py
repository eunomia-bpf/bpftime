import struct
from nose.plugins.skip import Skip, SkipTest
import ubpf.assembler
import ubpf.disassembler
import testdata
import pytest
import os
_test_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../test-cases")

try:
    xrange
except NameError:
    xrange = range

# Just for assertion messages
def try_disassemble(inst):
    data = struct.pack("=Q", inst)
    try:
        return ubpf.disassembler.disassemble(data).strip()
    except ValueError:
        return "<error>"

def check_datafile(filename):
    """
    Verify that the result of assembling the 'asm' section matches the
    'raw' section.
    """
    data = testdata.read(_test_data_dir, filename)
    if 'asm' not in data:
        raise SkipTest("no asm section in datafile")
    if 'raw' not in data:
        raise SkipTest("no raw section in datafile")

    bin_result = ubpf.assembler.assemble(data['asm'])
    assert len(bin_result) % 8 == 0
    assert len(bin_result) / 8 == len(data['raw'])

    for i in xrange(0, len(bin_result), 8):
        j = int(i/8)
        inst, = struct.unpack_from("=Q", bin_result[i:i+8])
        exp = data['raw'][j]
        if exp != inst:
            raise AssertionError("Expected instruction %d to be %#x (%s), but was %#x (%s)" %
                (j, exp, try_disassemble(exp), inst, try_disassemble(inst)))

@pytest.mark.parametrize("filename", testdata.list_files(_test_data_dir))
def test_datafiles(filename):
    # This is now a regular test function that will be called once for each filename
    check_datafile(filename)

