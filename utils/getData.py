import requests
from bs4 import BeautifulSoup

def main():
    # Open the text file in append mode
    with open("names.txt", "a") as file:
        # Send a HTTP request to the webpage
        for i in range(1, 50): # the 50 pages
            response = requests.get("https://nl.geneanet.org/genealogie/?page=" + str(i))
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                li_elements = soup.select("div.panel li")
                for li in li_elements:
                    name = li.get_text(strip=True).split("(")[0]
                    # Write the name to the text file
                    file.write(name + "\n")
            else:
                print(f"Failed to get the webpage: {response.status_code}")

if __name__ == '__main__':
    main()