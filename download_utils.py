import requests
from tqdm import tqdm
import time
from pathlib import Path
from magic import Magic
import tarfile
import gzip
from shutil import copyfileobj


class Unpack:
    step = "unpack"

    def __init__(self):
        self.magic = Magic(mime=True, uncompress=True)
        self.magic_formatted = Magic(mime=False, uncompress=True)

    def __call__(self, source, dest):
        source = Path(source)
        dest = Path(dest)
        mime = self.magic.from_file(str(source))
        if mime == 'application/x-tar':
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(source, "r:*") as tar:
                tar.extractall(dest)
        elif mime == 'text/x-tex':
            dest.mkdir(parents=True, exist_ok=True)
            with gzip.open(source, "rb") as src, open(dest / "main.tex", "wb") as dst:
                copyfileobj(src, dst)
        elif mime == 'application/pdf':
            raise RuntimeError(f"No LaTeX source code available for this paper, PDF only")
        elif mime == 'text/plain' and 'withdrawn' in self.magic_formatted.from_file(str(source)):
            raise RuntimeError(f"The paper has been withdrawn and there is"
                              f" no LaTeX source code available")
        else:
            raise RuntimeError(f"Cannot unpack file of type {mime}")




def get_eprint_link(paper):
    return f'http://export.arxiv.org/e-print/{paper}'


def download_eprints(papers, root=Path('sota/eprints')):
    links = [get_eprint_link(paper) for paper in papers]

    for i, link in tqdm(enumerate(links), "Downloading eprints...", total=len(links)):
        # play nice by arxiv - https://info.arxiv.org/help/bulk_data.html#harvest 
        if i % 4 == 0:
            time.sleep(1)
            
        response = requests.get(link)

        filename = link.split('/')[-1]
        path = root / filename
            
        with open(path, 'wb') as file:
            file.write(response.content)
        
    print(f"Downloaded {len(links)} eprints")

