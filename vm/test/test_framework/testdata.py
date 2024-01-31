import os
import re

def list_files(test_data_dir):
    """
    Return a list of data files under tests/data

    These strings are suitable to be passed to read().
    """

    result = []
    for dirname, dirnames, filenames in os.walk(test_data_dir):
        dirname = (os.path.relpath(dirname, test_data_dir) + '/').replace('./', '')
        for filename in filenames:
            if filename.endswith('.data') and not filename.startswith('.'):
                result.append(dirname + filename)
    return sorted(result)

def read(_test_data_dir, name):
    """
    Read, parse, and return a test data file

    @param name Filename relative to the tests/data directory
    @returns A hash from section to the string contents
    """

    section_lines = {}
    cur_section = None

    with open(os.path.join(_test_data_dir, name)) as f:
        for line in f:
            line = line.rstrip().partition('#')[0].rstrip()
            if line == '':
                continue
            elif line.startswith('--'):
                cur_section = line[2:].strip()
                if cur_section in section_lines:
                    raise Exception("section %s already exists in the test data file")
                section_lines[cur_section] = []
            elif cur_section:
                section_lines[cur_section].append(line)
    data = { section: '\n'.join(lines) for (section, lines) in list(section_lines.items()) }


    # Resolve links
    for k in list(data):
        if '@' in k:
            del data[k]
            section, path = k.split('@')
            section = section.strip()
            path = path.strip()
            fullpath = os.path.join(_test_data_dir, os.path.dirname(name), path)
            with open(fullpath) as f:
                data[section] = f.read()

    # Special case: convert 'raw' section into binary
    # Each line is parsed as an integer representing an instruction.
    if 'raw' in data:
        insts = []
        for line in data['raw'].splitlines():
            num, _, _ = line.partition("#")
            insts.append(int(num, 0))
        data['raw'] = insts
    #
    # Special case: convert 'mem' section into binary
    # The string '00 11\n22 33' results in "\x00\x11\x22\x33"
    # Ignores hexdump prefix ending with a colon.
    if 'mem' in data:
        hex_strs = []
        for line in data['mem'].splitlines():
            if ':' in line:
                line = line[(line.rindex(':')+1):]
            hex_strs.extend(re.findall(r"[0-9A-Fa-f]{2}", line))
        data['mem'] = bytes(bytearray([(int(x, 16)) for x in hex_strs]))

    return data
