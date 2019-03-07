import os
import librosa

raw_dataset_dir = './raw_data'
target_dataset_dir = '/home/cly/datacenter/songs/'

print(os.listdir(raw_dataset_dir))
sr = 20000
length = 1000

for singer in os.listdir(raw_dataset_dir):
    singer_dir = os.path.join(raw_dataset_dir, singer)
    target_singer_dir = os.path.join(target_dataset_dir, singer)
    try:
        os.mkdir(target_singer_dir)
    except Exception:
        pass
    part_id = 0
    for song in os.listdir(singer_dir):
        song_path = os.path.join(singer_dir, song)
        sample, _ = librosa.load(song_path, sr)

        st = 0
        ed = sample.shape[0]

        while st + length < ed:
            part_path = os.path.join(target_singer_dir, '{:06}.wav'.format(part_id))
            librosa.output.write_wav(part_path, sample[st:st + length], sr)

            st += length
            part_id += 1

        print('working on singer: {}, song: {}, part_id: {}'.format(singer, song, part_id))


