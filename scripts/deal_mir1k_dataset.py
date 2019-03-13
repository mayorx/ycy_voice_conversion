#将人声提取， 将其处理为我们需要的数据
import os
import shutil

DATA_ROOT = './tmp/mir1k'
RAWDATA_ROOT = './raw_data'

gender_names = {
    'boys': ['abjones', 'bobon', 'bug', 'davidson', 'fdps', 'geniusturtle', 'jmzen', 'Kenshin', 'khair', 'leon', 'stool'],
    'girls': ['amy', 'Ani', 'annar', 'ariel', 'heycat', 'tammy', 'titon', 'yifen']
}

total_count = 0

for gender in gender_names:
    names = gender_names[gender]
    for name in names:
        cnt = 0
        print('working on {}'.format(name))
        while True:
            cnt += 1
            filename = '{}_{}.wav'.format(name, cnt)
            scr_path = os.path.join(DATA_ROOT, filename)
            dst_path = os.path.join(RAWDATA_ROOT, gender, filename)
            if not os.path.isfile(scr_path):
                break
            shutil.copyfile(scr_path, dst_path)
        total_count += cnt - 1
        print('....  cnt {}'.format(cnt - 1))
    target_dir = os.path.join(RAWDATA_ROOT, gender)

assert total_count == 110
print('finished.')



