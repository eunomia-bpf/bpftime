with open("vec_add.cpp","r") as f:
    t=f.readlines()

q=[x for x in t if not x.startswith("#")]

with open("vec_add-new.cpp","w") as f:
    f.writelines(q)

import os
os.system("clang-format -i ./vec_add.cpp")
