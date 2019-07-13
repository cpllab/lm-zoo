#!/usr/bin/env python

from collections import defaultdict
import re
import subprocess
import sys
from tempfile import NamedTemporaryFile
import unittest

from nose.tools import *

TEST_STRING = """This is a test sentence.
This is another test sentence.
"""

SURPRISAL_RE = re.compile(r"sentence_id\ttoken_id\ttoken\tsurprisal\n"
                           "(\d+\s+\d+\s+[\w.<>]+\s+[-\d.]+\n)+(\d+\s+\d+\s+[\w.<>]+\s+[-\d.]+)",
                          flags=re.MULTILINE)

class LMTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # super(LMTest, cls).setUpClass()

        with NamedTemporaryFile("w") as text_f:
            text_f.write(TEST_STRING)
            text_f.flush()

            print("== tokenize %s" % text_f.name)
            cls.tokenized_output = subprocess.check_output(["tokenize", text_f.name]).decode("utf-8")
            print(cls.tokenized_output)
            print("\n")

            print("== get_surprisals %s" % text_f.name)
            cls.surprisals_output = subprocess.check_output(["get_surprisals", text_f.name]).decode("utf-8")
            print(cls.surprisals_output)

        cls.tokenized_lines = [line.strip() for line in cls.tokenized_output.strip().split("\n")]
        cls.surprisal_lines = [line.strip().split("\t") for line in cls.surprisals_output.strip().split("\n")]

    @property
    def _parsed_surprisals(self):
        surprisals = defaultdict(dict)
        for line in self.surprisal_lines[1:]:
            sentence_id, token_id, token, surprisal = line
            surprisals[int(sentence_id)][int(token_id)] = (token, float(surprisal))

        return surprisals

    def test_tokenization(self):
        # token sequence output from `get_surprisals` should match token
        # sequences from `tokenize`
        surprisals = self._parsed_surprisals
        for i, tokenized_line in enumerate(self.tokenized_lines):
            tokens = tokenized_line.split(" ")
            eq_(tokens, [surprisals[i + 1][j + 1][0] for j in range(len(tokens))], "Token sequences should match exactly")

    def test_surprisal_output_format(self):
        ok_(SURPRISAL_RE.match(self.surprisals_output))

    def test_surprisal_parse(self):
        eq_(set(int(line[0]) for line in self.surprisal_lines[1:]), {1, 2}, "Both sentences retained")
        for line in self.surprisal_lines[1:]:
            # attempt to parse surprisal
            surp = float(line[3])
            ok_(surp >= 0, "valid surprisal")


if __name__ == "__main__":
    import nose
    nose.runmodule()
