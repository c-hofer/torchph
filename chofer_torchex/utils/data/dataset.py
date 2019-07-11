import requests
import torch


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def labels(self):
        raise NotImplementedError

    @property
    def sample_labels(self):
        raise NotImplementedError


class SimpleDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]
    
    def __iter__(self):
        return zip(self.X, self.Y) 


def _download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        content_iter = response.iter_content(CHUNK_SIZE)
        with open(destination, "wb") as f:

            for i, chunk in enumerate(content_iter):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    print(i, end='\r')

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)

    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
