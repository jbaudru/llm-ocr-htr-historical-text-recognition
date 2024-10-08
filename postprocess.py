# for each text file in a specific folder remove the "|" and the tabulation

import os

def postprocess(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(folder_path + filename, 'r') as f:
                lines = f.readlines()
            with open(folder_path + filename, 'w') as f:
                for line in lines:
                    f.write(line.replace("|", "").replace("\t", ""))
    print("Postprocessing done")
    return

if __name__ == "__main__":
    postprocess(