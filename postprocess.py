# for each text file in a specific folder remove the "|" and the tabulation

import os

def postprocess(folder_path):
    folder_path = os.getcwd() + folder_path
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(folder_path + filename, 'r', encoding="utf-8") as f:
                lines = f.readlines()
            with open(folder_path + filename, 'w', encoding="utf-8") as f:
                for line in lines:
                    f.write(line.replace("|", "").replace("\t", ""))
    print("Postprocessing done")
    return

if __name__ == "__main__":
    postprocess("/results/predictions/two-example_prompt/claude-3-5-sonnet-20240620/")