import os
import json
from collections import defaultdict
import math

def parse_file(file_name: str) -> list:
    result = []
    with open(file_name, "r") as f:
        lines = f.readlines()
    while lines[-1].strip() == "":
        lines.pop()
    for sec in lines[-6:]:
        a, b, c = (x.strip() for x in sec[1:-2].split("|"))
        size = int(a[:-2].strip())
        req = float(b)
        trans = float(c[:-2])
        result.append({"size": size, "request": req, "transfer": trans})
    return result

def main():
    ENTRY = json.dumps(
        {
            "details": [],
            "statistics": {"transfer": [], "request": []},
        }
    )
    _EXAMPLE = {
        "https": {
            "kernel-uprobe": {
                "details": [{"size": 1, "transfer": 11111, "request": 1111}],
                "statistics": {
                    "transfer": [
                        {
                            "size": 1,
                            "avg": 0,
                            "min": math.inf,
                            "max": -math.inf,
                            "std_dev": 0,
                            "sqr_dev": 0,
                            "count": 0,
                        }
                    ],
                    "request": [],
                },
            },
            "no-probe": json.loads(ENTRY),
            "no-uprobe": json.loads(ENTRY),
            "user-uprobe": json.loads(ENTRY),
        },
        "http": {
            "kernel-uprobe": json.loads(ENTRY),
            "no-probe": json.loads(ENTRY),
            "no-uprobe": json.loads(ENTRY),
            "user-uprobe": json.loads(ENTRY),
        },
    }
    data = {
        "https": {
            "kernel-uprobe": json.loads(ENTRY),
            "no-probe": json.loads(ENTRY),
            "no-uprobe": json.loads(ENTRY),
            "user-uprobe": json.loads(ENTRY),
        },
        "http": {
            "kernel-uprobe": json.loads(ENTRY),
            "no-probe": json.loads(ENTRY),
            "no-uprobe": json.loads(ENTRY),
            "user-uprobe": json.loads(ENTRY),
        },
    }
    for file in os.listdir("."):
        if file.startswith("test-") and file.endswith(".txt"):
            stripped = file[5:-4]
            http = "http" if stripped.endswith("--http") else "https"
            idx = int(stripped.split("_")[0].split("-")[-1])
            print(file)
            if stripped.startswith("kernel-uprobe"):
                data[http]["kernel-uprobe"]["details"].append(parse_file(file))
            elif stripped.startswith("no-uprobe"):
                data[http]["no-uprobe"]["details"].append(parse_file(file))
            elif stripped.startswith("no-probe"):
                data[http]["no-probe"]["details"].append(parse_file(file))
            elif stripped.startswith("user-uprobe"):
                data[http]["user-uprobe"]["details"].append(parse_file(file))
    for http in data.values():
        for ty in http.values():
            for key in {"request", "transfer"}:
                size_diff = defaultdict(
                    lambda: {
                        "avg": 0,
                        "min": math.inf,
                        "max": -math.inf,
                        "std_dev": 0,
                        "sqr_dev": 0,
                        "count": 0,
                    }
                )
                for entry in ty["details"]:
                    for entry_2 in entry:
                        t = size_diff[entry_2["size"]]
                        t["count"] += 1
                        t["avg"] += entry_2[key]
                        t["min"] = min(t["min"], entry_2[key])
                        t["max"] = max(t["max"], entry_2[key])
                        t["sqr_dev"] += entry_2[key] ** 2
                for val in size_diff.values():
                    val["avg"] /= val["count"]
                    val["sqr_dev"] /= val["count"]
                    val["sqr_dev"] -= val["avg"] ** 2
                    val["std_dev"] = val["sqr_dev"] ** 0.5
                ty["statistics"][key] = list(
                    sorted(
                        ({"size": x, **y} for x, y in size_diff.items()),
                        key=lambda t: t["size"],
                    )
                )
    with open("test-data-multi-without-smatrt-ptr.json", "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    main()
