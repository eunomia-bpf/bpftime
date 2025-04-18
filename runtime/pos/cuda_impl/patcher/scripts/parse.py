# Copyright 2024 The PhoenixOS Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pyparsing as pp

VISIBLE, EXTERN, ENTRY, FUNC, VERSION, TARGET, ADDR_SIZE, GLOBAL = map(
    pp.Keyword,
    [
        ".visible",
        ".extern",
        ".entry",
        ".func",
        ".version",
        ".target",
        ".address_size",
        ".global",
    ],
)
ident = pp.Word(pp.identchars, pp.identbodychars)
comment = "//" + pp.restOfLine()
directives = pp.MatchFirst(
    map(lambda x: x + pp.restOfLine(), [VERSION, TARGET, ADDR_SIZE, GLOBAL])
)

kernel_decl = EXTERN + FUNC + ident + pp.nestedExpr() + ";"
kernel_def = (
    VISIBLE
    + ENTRY
    + ident
    + pp.nestedExpr()
    + pp.Suppress(pp.SkipTo("{"))
    + pp.Suppress(pp.nestedExpr("{", "}"))
)
# make sure we're matching the entire string
ptx_grammer = (
    pp.ZeroOrMore(kernel_decl | kernel_def | comment | directives) + pp.stringEnd()
)

with open("ptx/torch.ptx") as f:
    s = f.read()
ptx_grammer.parse_string(s)
