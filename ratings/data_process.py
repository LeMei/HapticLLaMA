import json as js

def load_json(path):
    with open(path, 'rb') as f:
        data = js.load(f)
    return data

def to_json(data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        js.dump(data, f, ensure_ascii=False, indent=4)

def cal_avg(captions, rate_dict):
    sum_rate, count = 0, 0
    rate_list = []
    for cap in captions:
        rate = rate_dict[cap]
        rate_list.append(rate)
        sum_rate = sum_rate + rate
        count = count + 1

    return sum_rate/count,rate_list

def process(part1, part2):
    signal_p1 = part1['haptic_signal']
    signal_p2 = part2['haptic_signal']
    signal_dict = {}

    for sig in signal_p1:
        if sig in ['1228.wav','F291_loop.wav','F521_loop.wav']:
            continue
        signal_dict[sig] = {}
        rate_dict1 = signal_p1[sig]
        caption_1 = list(rate_dict1.keys())
        rate_dict2 = signal_p2[sig]
        caption_2 = list(rate_dict2.keys())
        assert len(caption_1) == len(caption_2)

        avg = [(rate_dict2[cap] + rate_dict1[cap])/2 for cap in caption_1]

        for i,j in zip(caption_1, avg):
            signal_dict[sig][i] = j
        
    
    return signal_dict
#1,3,5,7,9,11
total_num = 41
rating_sets = range(1, total_num, 2)  
data = {}
dict_list = []
for i in rating_sets:
    part1_rating = r'./{}/captionrating.json'.format(i)
    part2_rating = r'./{}/captionrating.json'.format(i+1)

    print(i)
    data1 = load_json(part1_rating)
    data2 = load_json(part2_rating)

    signal_dict = process(data1, data2)
    dict_list.append(signal_dict)

for d in dict_list:
    data.update(d)

print('len sample:{}'.format(len(data.keys())))

json_path = r'./combined_rating.json'
to_json(data, json_path=json_path)






