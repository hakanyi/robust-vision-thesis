import os
import json
import argparse

def get_arguments():
    arg_parser = argparse.ArgumentParser(
        description="Make a condition list as JSON from a stimulus directory."
    )
    arg_parser.add_argument(
        "--stimdir",
        required=True,
        help="Where to find the stimuli.",
    )
    arg_parser.add_argument(
        "--savedir",
        required=True,
        help="Where to store the JSON file.",
    )

    args = arg_parser.parse_args()
    return args

CONDITION = 0

if __name__ == "__main__":
    args = get_arguments()

    os.makedirs(args.savedir, exist_ok=True)

    names = sorted(os.listdir(args.stimdir))

    # we'll show the same image condition twice
    new_names = []
    [new_names.extend([name, name]) if "same-image" in name \
     else new_names.append(name) for name in names]
    print(f"There are {len(new_names)} trials in total.")

    condlist = {str(CONDITION) : new_names}

    pth = os.path.join(args.savedir, "condlist.json")
    with open(pth, "w") as f:
        json.dump(condlist, f)
