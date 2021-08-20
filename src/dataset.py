import glob
import torch
import torch.nn.functional as F
import torchvision.io as Tvio
import numpy as np
import multiprocessing as mp
class Dataset(torch.utils.data.Dataset):
    def __init__(self, directory='../data/*', get_subdirs=True, size=(16,16), max_ctx_length=4096, dry_run=False):
        print("Loading dataset...")
        self.data = glob.glob(directory)
        if get_subdirs:
            self.data_temp = []
            with mp.Pool(mp.cpu_count()) as data_procs:
                for i in data_procs.imap_unordered(self.extract_from_data, list(range(len(self.data)))):
                    self.data_temp.extend(i)
            data_procs.join()
            self.data = self.data_temp
            self.max_ctx_length = max_ctx_length
        self.size = size
    def extract_from_data(self, i):
        i_data = self.data[i]
        file_data = glob.glob(i_data+"/*")
        file_data.sort(key=lambda r: int(''.join(x for x in r if (x.isdigit()))))
        return file_data

    def __len__(self):
        return len(self.data)*self.size[0]-self.max_ctx_length-1
    def __getitem__(self, key):
        frame_start = int(np.floor(key / self.size[0]))
        patch_start = int(np.mod(key, self.size[0]))
        
        patches = []
        i_frame = frame_start

        while len(patches) <= self.max_ctx_length+1:
            frame = (Tvio.read_image(self.data[i_frame], mode=Tvio.ImageReadMode.RGB).float() / 255).unsqueeze(0)
            if len(patches) == 0:
                patches.extend(F.unfold(frame, self.size, stride=self.size).transpose(1,2).split(1,1)[patch_start:])
            else:
                patches.extend(F.unfold(frame, self.size, stride=self.size).transpose(1,2).split(1, 1))
            i_frame += 1
        patches = patches[:self.max_ctx_length+1]

        data_x = patches[0:-1]
        data_y = patches[1:]

        return torch.cat(data_x, dim=1).squeeze(0), torch.cat(data_y, dim=1).squeeze(0)

if __name__ == "__main__":
    dataset = Dataset(max_ctx_length=4096, size=(16,16), dry_run=True)

    print(dataset.__getitem__(8)[0].shape)