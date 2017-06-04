import csv

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

def redo_race(data_vec):
    global race_iter
    global race_dict

def pp_race(data_vec):
    global race_iter
    global race_dict
    if data_vec[-3] in race_dict:
        data_vec[-3] = race_dict[data_vec[-3]]
    else:
        race_iter += 1
        race_dict[data_vec[-3]] = float(rarity_iter) / 7.0
        data_vec[-3] = float(race_dict[data_vec[-3]])

    return

def pp_rarity(data_vec):
    global rarity_iter
    global rarity_dict
    #print(data_vec[6])
    if data_vec[6] in rarity_dict:
        data_vec[6] = rarity_dict[data_vec[6]]
    else:
        rarity_iter += 1
        rarity_dict[data_vec[6]] = rarity_iter/ 5.0
        data_vec[6] = float(rarity_dict[data_vec[6]])

    return

def pp_type(data_vec):
    global type_iter
    global type_dict
    if data_vec[2] in type_dict:
        data_vec[2] = type_dict[data_vec[2]]
    else:
        type_dict[data_vec[2]] = type_iter/ 6.0
        data_vec[2] = float(type_dict[data_vec[2]])
        type_iter += 1
    return

def pp_classes(data_vec):
    global class_dict
    global class_iter
    #print(data_vec[1])
    if data_vec[1] in class_dict:
        data_vec[1] = class_dict[data_vec[1]]
    else:
        class_dict[data_vec[1]] = class_iter/ 12.0
        data_vec[1] = float(class_dict[data_vec[1]])
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

            print(c)
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
   #print(min,max)
        #print("['id', 'player_class', 'type', 'cost', 'attack', 'health', 'rarity', 'mechanics', 'play_requirements', 'race', 'durability', 'entourage']")
        #for i,key in enumerate(effects.keys()):
        #    print(i,key)
    #wrtr.close()

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
            #print(line)
        #print(len(result))


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
read_data()
preprocess_data()
