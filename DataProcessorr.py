import csv
import numpy as np

global key_dict
global key_iter
global pq_dict
global pq_iter
global class_dict
global class_iter
global type_iter
global type_dict
global rarity_dict
global rarity_iter
global race_iter
global race_dict
import pickle

def redo_race(data_vec):
    global race_iter
    global race_dict

def pp_race(data_vec):
    global race_iter
    global race_dict
    if data_vec[-3] in race_dict:
        data_vec[-3] = race_dict[data_vec[-3]]/ 7.0
    else:
        race_iter += 1
        race_dict[data_vec[-3]] = float(rarity_iter)
        data_vec[-3] = float(race_dict[data_vec[-3]])/ 7.0

    return

def pp_rarity(data_vec):
    global rarity_iter
    global rarity_dict
    #print(data_vec[6])
    if data_vec[6] in rarity_dict:
        data_vec[6] = rarity_dict[data_vec[6]]/ 5.0
    else:
        rarity_iter += 1
        rarity_dict[data_vec[6]] = rarity_iter
        data_vec[6] = float(rarity_dict[data_vec[6]])/ 5.0

    return

def pp_type(data_vec):
    global type_iter
    global type_dict
    if data_vec[2] in type_dict:
        data_vec[2] = type_dict[data_vec[2]]/ 6.0
    else:
        type_dict[data_vec[2]] = type_iter
        data_vec[2] = float(type_dict[data_vec[2]])/ 6.0
        type_iter += 1
    return

def pp_classes(data_vec):
    global class_dict
    global class_iter
    #print(data_vec[1])
    if data_vec[1] in class_dict:
        data_vec[1] = class_dict[data_vec[1]]/ 12.0
    else:
        class_dict[data_vec[1]] = class_iter
        data_vec[1] = float(class_dict[data_vec[1]])/ 12.0
        class_iter += 1
    return

def extract_mechanics(data_vec):
    mechanics = ['','','','']
    try:
        tmp = eval(data_vec[7])
        if type(tmp) == list:
            for i in range(len(tmp)):
                mechanics[i] = tmp[i]
    except:
        pass
    global key_dict
    global key_iter
    for i in range(len(mechanics)):
        if mechanics[i] in key_dict:
            mechanics[i] = float(key_dict[mechanics[i]])/34.0
        else:
            key_dict[mechanics[i]] = key_iter
            key_iter += 1
            mechanics[i] = float(key_dict[mechanics[i]])/34.0

    data_vec = data_vec[:7]+mechanics+data_vec[8:]
    return data_vec

def extract_play_requirements(data_vec):
    prq = ['','','','','']
    prq_v = ['','','','','']
    try:
        tmp = eval(data_vec[8+3])
        i = 0
        for key in tmp:
            prq[i] = key
            prq_v[i] = tmp[key]
            i+=1

        global pq_dict
        global pq_iter
        for i in range(len(prq)):
            if prq[i] in pq_dict:
                prq[i] = float(pq_dict[prq[i]])/33.0
            else:
                pq_dict[prq[i]] = pq_iter
                pq_iter += 1
                prq[i] = float(pq_dict[prq[i]])/33.0
    except:
        pass
    total = prq + prq_v
    data_vec = data_vec[:8+3]+total+data_vec[9+3:]
    return data_vec


def read_data():
    data_vec = list()
    wrtr = csv.writer(open('gan_card.csv','w+'),delimiter=',', lineterminator='\n')
    wrtr.writerow(['id', 'player_class', 'type', 'cost', 'attack', 'health', 'rarity', 'mechanics1', 'mechanics2',\
                   'mechanics3', 'mechanics4', 'play_requirements1', 'play_requirements2', 'play_requirements3',\
                   'play_requirements4', 'play_requirements5', 'play_requirements_val1', 'play_requirements_val2',\
                   'play_requirements_val3', 'play_requirements_val4', 'play_requirements_val5', 'race', 'durability', 'entourage'])
    with open(r'.\Data\cards_flat.csv','r') as fp:
        max = -1
        min = 9999
        rdr = csv.reader(fp,delimiter=',')
        c = 0
        effects = dict()
        first = True

        for line in rdr:
            skip = False
            #print("hi")
            if first:
                first = False
                continue
            data_vec += line[:3]
            data_vec += line[6:10]
            data_vec += [line[12]]
            data_vec += line[14:16]
            data_vec += line[-2:]

            #print(c)
            data_vec = extract_mechanics(data_vec)
            data_vec = extract_play_requirements(data_vec)
            for i in range(len(data_vec)):
                if data_vec[i] == '':
                    data_vec[i] = 1.0
            if data_vec[1] != 1:
                pp_classes(data_vec)
            if data_vec[2] != 1:
                if data_vec[2] != 'MINION':
                    skip = True
                pp_type(data_vec)
            if data_vec[3] != 1:
                data_vec[3] = float(data_vec[3])/10.0
            if data_vec[4] != 1:
                data_vec[4] = float(data_vec[4])/30.0
            if data_vec[5] != 1:
                data_vec[5] = float(data_vec[5])/200.0
            if data_vec[6] != 1:
                if data_vec[6] == 'LEGENDARY':
                    skip = True
                pp_rarity(data_vec)
            if data_vec[-3] != 1:
                pp_race(data_vec)
                #print(data_vec[-3])

            if data_vec[5] != -1:
                if int(data_vec[5]) > max:
                    max = int(data_vec[5])
                if int(data_vec[5]) < min:
                    min = int(data_vec[5])
            if not skip:
                c += 1
                wrtr.writerow(data_vec)
           # print(len(data_vec))
            data_vec = list()

def read_data_one_hot():
    data_vec = list()
    wrtr = csv.writer(open('gan_card.csv', 'w+'), delimiter=',', lineterminator='\n')

    #wrtr.writerow(['id', 'player_class', 'type', 'cost', 'attack', 'health', 'rarity', 'mechanics1', 'mechanics2', \
     #              'mechanics3', 'mechanics4', 'play_requirements1', 'play_requirements2', 'play_requirements3', \
      #             'play_requirements4', 'play_requirements5', 'play_requirements_val1', 'play_requirements_val2', \
       #            'play_requirements_val3', 'play_requirements_val4', 'play_requirements_val5', 'race', 'durability',
        #           'entourage'])
    header = ['id']

    with open('dicts.pkl', 'rb') as fp:
        class_dict = pickle.load(fp)
        class_iter = pickle.load(fp)
        key_dict = pickle.load(fp)
        key_iter = pickle.load(fp)
        pq_dict = pickle.load(fp)
        pq_iter = pickle.load(fp)
        race_dict = pickle.load(fp)
        race_iter = pickle.load(fp)
        rarity_dict = pickle.load(fp)
        rarity_iter = pickle.load(fp)
        type_dict = pickle.load(fp)
        type_iter = pickle.load(fp)
    header = ['id'] + (class_iter*['class']) + (type_iter*['type']) + ['cost', 'attack', 'health'] + (rarity_iter*['rarity'])\
            + (key_iter*['mechanic']) + (pq_iter*['play_req'])+ (pq_iter*['play_req_val'])+(rarity_iter*['race']+['durability','entourage'])

    #print(len(header))
    wrtr.writerow(header)
    with open(r'.\Data\cards_flat.csv', 'r') as fp:
        max = -1
        min = 9999
        rdr = csv.reader(fp, delimiter=',')
        c = 0
        effects = dict()
        first = True

        for line in rdr:
            skip = False
            # print("hi")
            if first:
                first = False
                continue
            data_vec += line[:3]
            data_vec += line[6:10]
            data_vec += [line[12]]
            data_vec += line[14:16]
            data_vec += line[-2:]

            solution = []
           # print(c)
            #data_vec = extract_mechanics(data_vec)
            #data_vec = extract_play_requirements(data_vec)
            for i in range(len(data_vec)):
                if data_vec[i] == '':
                    data_vec[i] = 1.0
            solution = [data_vec[0]]
            arr = np.zeros(class_iter)
            if data_vec[1] != 1:
                #p_classes(data_vec)

                arr[class_dict[data_vec[1]]] = 1.0
                data_vec[1] = arr
            solution += arr.tolist()
            arr = np.zeros(type_iter)
            if data_vec[2] != 1:
                if data_vec[2] != 'MINION':
                    skip = True
                arr[type_dict[data_vec[2]]] = 1.0
                data_vec[2] = arr
            solution += arr.tolist()
                #pp_type(data_vec)
            if data_vec[3] != 1:
                data_vec[3] = float(data_vec[3]) / 10.0
            if data_vec[4] != 1:
                data_vec[4] = float(data_vec[4]) / 30.0
            if data_vec[5] != 1:
                data_vec[5] = float(data_vec[5]) / 200.0
            solution += data_vec[3:6]
            arr = np.zeros(rarity_iter)
            if data_vec[6] != 1:
                if data_vec[6] == 'LEGENDARY':
                    skip = True
                #pp_rarity(data_vec)

                arr[rarity_dict[data_vec[6]]-1] = 1.0
                data_vec[6] = arr
            solution += arr.tolist()
            arr = np.zeros(len(key_dict))
            if data_vec[7] != 1:
                #pp_rarity(data_vec)

                tmp = eval(data_vec[7])
                for x in tmp:
                    arr[key_dict[x]] = 1.0
                data_vec[7] = arr
            solution += arr.tolist()
            arr = np.zeros(pq_iter * 2)
            if data_vec[8] != 1:
                #pp_rarity(data_vec)
                tmp = eval(data_vec[8])
                arr = np.zeros(pq_iter*2)
                for key in tmp:
                    arr[pq_dict[key]] = 1.0
                    arr[pq_dict[key]+pq_iter-1] = tmp[key]
                data_vec[8] = arr
            solution += arr.tolist()
            solution += data_vec[9:-3]
            arr = np.zeros(race_iter)
            if data_vec[-3] != 1:
                arr[race_dict[data_vec[-3]]] = 1.0
                data_vec[2] = arr
            solution += arr.tolist()
            solution += data_vec[-2:]

            if data_vec[5] != -1:
                if int(data_vec[5]) > max:
                    max = int(data_vec[5])
                if int(data_vec[5]) < min:
                    min = int(data_vec[5])


            if not skip:
                c += 1
               # print('lel',len(solution))
                wrtr.writerow(solution)
                # print(len(data_vec))
            data_vec = list()
            solution = list()



def preprocess_data(mbatch_size = 128):
    result = list()
    with open('gan_card.csv','r') as fp:
        rdr = csv.reader(fp,delimiter=',')
        c = 0
        first = True
        for line in rdr:
            if first:
                first = False
                continue
            c += 1
            if c > mbatch_size:
                break
            line = (line[1:-1])
            for i in range(len(line)):
                line[i] = float(line[i])
            result.append(line)


    return result

type_dict = dict()
type_iter = 0
key_iter = 0
pq_iter = 0
class_iter = 0
rarity_iter = 0
race_iter = 0
class_dict = dict()
key_dict = dict()
pq_dict = dict()
rarity_dict = dict()
race_dict = dict()
#read_data()
read_data_one_hot()
#preprocess_data()



