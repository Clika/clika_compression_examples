"""Download a file from Google Drive"""
import sys

import requests


def download_file_from_google_drive(id_, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id_}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {"id": id_, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def _save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        sys.stderr.write(f"usage: {args[0]} <gdrive ID> <destination>?")

    download_file_from_google_drive(id_=args[1], destination=args[2])
