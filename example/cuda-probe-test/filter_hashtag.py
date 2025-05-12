with open("gemm.cpp","r") as f:
    t=f.readlines()

q=[x for x in t if not x.startswith("#")]

with open("gemm-new.cpp","w") as f:
    f.writelines(q)

import os
os.system("clang-format -i ./gemm-new.cpp")
