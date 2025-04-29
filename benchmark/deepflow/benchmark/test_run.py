for c in ["--http", "--https"]:
    for a in ["no-probe", "kernel-uprobe", "no-uprobe", "user-uprobe"]:
        for b in range(1, 11):
            import os
            file = f"test-{a}-{b}_{c}.txt"
            print(file)
            if os.path.exists(file):
                print("skipped")            
            else:
                print("running")
                os.system(f"python3 run.py -t {a} {c} > {file}")
