#!/usr/bin/env python

import re
import subprocess
import sys
from tempfile import NamedTemporaryFile

TEST_STRING = """This is a test sentence.
This is another test sentence.
"""

SURPRISAL_RE = re.compile(r"sentence_id\ttoken_id\ttoken\tsurprisal\n"
                           "(\d+\s+\d+\s+[\w.<>]+\s+[-\d.]+\n)+(\d+\s+\d+\s+[\w.<>]+\s+[-\d.]+)",
                          flags=re.MULTILINE)

with NamedTemporaryFile("w") as text_f:
    text_f.write(TEST_STRING)
    text_f.flush()

    output = subprocess.check_output(["get_surprisals", text_f.name]).decode("utf-8")
    print(output)
    assert SURPRISAL_RE.match(output)

lines = [line.strip().split("\t") for line in output.strip().split("\n")[1:]]
assert set(int(line[0]) for line in lines) == {1, 2}, "Both sentences retained"
for line in lines:
    # attempt to parse surprisal
    surp = float(line[3])
    assert surp >= 0
