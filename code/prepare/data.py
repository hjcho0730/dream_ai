from func.utils import *

def logClear():
    with open(log_path, "w") as f:
            f.write("")

if __name__ == "__main__":
      logClear()

#data1
def getMovieReviewData(usingMethod="csv", *args, **kwargs):
    if usingMethod == False:
        return ([], [])
    if usingMethod == "csv":
        return getMovieReviewDataByCsv(*args, **kwargs)
    elif usingMethod == "json":
        return getMovieReviewDataByJson(*args, **kwargs)
    #else
    return getMovieReviewDataByJson(*args, **kwargs)
def getMovieReviewDataByCsv(limits = 7000, max_length=64, min_length=1): # 최대 199176개
    import csv 
    texts = []
    labels = []

    path = os.path.join(real_path, "datas", "nsmc_raw") # 원래 파일 이름은 raw였음
    file_list = os.listdir(path)

    txtFiles = tuple(file for file in file_list if file.endswith('.txt'))
    
    for i in txtFiles:
        p = os.path.join(path, i)
        
        with open(p, "r") as f:
            data = csv.reader(f, delimiter="\t")
            for row in list(data)[1:]: 
                line = row[0]
                text = '\t'.join(row[1:len(row)-1])
                label = row[-1]

                #print(', '.join(row)) 

                if max_length != -1 and len(text) > max_length:
                    continue
                if min_length != -1 and len(text) < min_length:
                    continue
                score = label

                texts.append(text)
                labels.append(score)
                if limits == -1:
                    continue
                if len(texts) >= limits:
                    break
        if limits == -1:
            continue
        if len(texts) >= limits:
            break
    return (texts[:limits], labels[:limits])

def getMovieReviewDataByJson(limits = 7000, max_length=64, min_length=1):# 최대 709812개
    import json
    texts = []
    labels = []

    path = os.path.join(real_path, "datas", "nsmc_raw") # 원래 파일 이름은 raw였음
    file_list = os.listdir(path)

    jsonFiles = tuple(file for file in file_list if file.endswith('.json'))

    n = -1
    l = len(jsonFiles)
    print(l, "개의 파일")

    maxLen= len(str(l)) + 1
    orbit= min(l // 30, (limits // 30) if limits != -1 else 1000)
    for i in jsonFiles:
        n+=1
        if n % orbit  == 0:
            print(f"[{max(((len(texts)*100 / limits) if limits != -1 else 0), n*100/l):10f}%] {n:{maxLen}d} / {l:{maxLen}d}{f"  | ({len(texts)} / {limits})" if limits != -1 else ""}")

        p = os.path.join(path, i)
        with open(p, 'r') as f:
            json_data = json.load(f)    
            for i in json_data:
                if max_length != -1 and len(i["review"]) > max_length:
                    continue
                if min_length != -1 and len(i["review"]) < min_length:
                    continue
                score = float(i["rating"])
                if score >= 9:
                    score = 1
                else: #if score <= 4:
                    score = 0
                texts.append(i["review"])
                labels.append(score)
                if limits == -1:
                    continue
                if len(texts) >= limits:
                    break
        if limits == -1:
            continue
        if len(texts) >= limits:
            break
    
    if n % orbit  != 0:
        print(f"[{100:10f}%]")
    return (texts[:limits], labels[:limits])

####

def merge_tuples(*tuples):
    if len(tuples) == 0:
        return [], []
    
    return tuple(
        list(chain.from_iterable(group)) for group in zip(*tuples)
    )
