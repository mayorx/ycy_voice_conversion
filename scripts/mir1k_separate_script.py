import os

source_root = './MIR-1K/UndividedWavfile' #place your
target_root = './tmp/mir1k'

for filename in os.listdir(source_root):
    source_path = os.path.join(source_root, filename)
    target_path = os.path.join(target_root, filename)
    os.system('ffmpeg -i {} -map_channel 0.0.1 {}'.format(source_path, target_path))


