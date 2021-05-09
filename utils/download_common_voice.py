from patoolib import extract_archive
import os
import wget
import re


def dl_commonvoice_data(download_path, args, unpack=True):
    """Download common voice file from mozilla commonvoice repo

    Args:
        download_path (str): common voice full url link. Get link from commonvoice website
        args (dict)
        unpack (bool, optional): whether to unzip the file after download. Defaults to True.
    """

    file_name = re.search('\w+-*\w+\.tar\.gz', download_path)[0]

    file_path = download_path+os.sep+file_name
    if not os.path.isfile(file_path):
        filename = wget.download(args.get('url'), download_path)
        print(f"File sucessfully downloaded in dir {download_path}")
    else:
        print(f'File {file_name} already exists in dir {download_path}')

    if unpack:
        extract_archive(file_name, outdir=download_path, verbose=0)
        print(f"file sucessfully unpacked in dir {download_path}")
    
    print("Finished downloading files")
        