import os
from pathlib import Path

path = Path(os.path.dirname(os.path.realpath(__file__)))
model_path = path / '..' / 'model'

openpose_url = 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/'
pose_mpi_url = openpose_url + 'pose/mpi/'
mpi_filename = 'pose_iter_160000.caffemodel'

command = f'curl {pose_mpi_url}{mpi_filename} -o {model_path}/{mpi_filename}'
print('Run command: ', command)

os.system(command)
