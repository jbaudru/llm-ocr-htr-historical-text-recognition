# for each text file in a specific folder remove the "|" and the tabulation

import os
import re

def postprocess(folder_path):
    folder_path = os.getcwd() + folder_path
    pattern = re.compile(r"^(-\s+)+-*$")  # Regular expression to match lines with repeated hyphens separated by spaces

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(folder_path + filename, 'r', encoding="utf-8") as f:
                lines = f.readlines()
            with open(folder_path + filename, 'w', encoding="utf-8") as f:
                for line in lines:
                    # Remove lines containing "```plaintext", "```", blank lines, or lines matching the pattern
                    if "```plaintext" in line or "```" in line or line.strip() == "" or pattern.match(line.strip()):
                        continue
                    f.write(line.replace("|", "").replace("\t", ""))
    print("Postprocessing done")
    return

if __name__ == "__main__":
    postprocess("/results/postprocessed/one-example_prompt/gpt-4o/")
    postprocess("/results/postprocessed/one-example_prompt/claude-3-5-sonnet-20240620/")
    postprocess("/results/postprocessed/two-example_prompt/gpt-4o/")
    postprocess("/results/postprocessed/two-example_prompt/claude-3-5-sonnet-20240620/")
    #postprocess("/results/postprocessed/refine_complex-prompt/gpt-4o/")
    #postprocess("/results/postprocessed/refine_complex-prompt/claude-3-5-sonnet-20240620/")