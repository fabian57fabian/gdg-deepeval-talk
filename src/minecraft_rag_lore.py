import requests
from requests.auth import HTTPBasicAuth
import json


def ask_lore(message):
    print("Asking Minecraft question...")
    payload = json.dumps({"message": message})
    response = requests.get("http://217.160.188.167:10000//api/chat",
                            auth=HTTPBasicAuth("gdgfirenze!challenge","gdgfirenze!challenge"),
                            data=payload)
    return response.text


def read_lore_book(filename):
    with open(filename, 'r') as file:
        return "\n".join(file.readlines())


if __name__ == '__main__':
    print(ask_lore("come si chiama il pescatore?"))