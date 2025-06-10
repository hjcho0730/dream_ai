from utils import *
def logClear():
    with open(log_path, "w") as f:
            f.write("")

if __name__ == "__main__":
      logClear()

#data1
def getMovieReviewData(limits = 7000, max_length=64, min_length=1):
    texts = []
    labels = []

    path = os.path.join(real_path, "datas", "nsmc_raw") # 원래 파일 이름은 raw였음
    file_list = os.listdir(path)

    jsonFiles = [file for file in file_list if file.endswith('.json')]
    
    for i in jsonFiles:
        p = os.path.join(path, i)
        with open(p, 'r') as f:
            json_data = json.load(f)    
            for i in json_data:
                if len(i["review"]) >= max_length or len(i["review"]) < min_length:
                    continue
                score = float(i["rating"])
                if score >= 8:
                    score = 2
                elif score <= 4:
                    score = 0
                else:
                    score = 1
                texts.append(i["review"])
                labels.append(score)
        if len(texts) >= limits:
            break
    return (texts[:limits], labels[:limits])

#

def merge_tuples(*tuples):
    return tuple(
        list(chain.from_iterable(group)) for group in zip(*tuples)
    )
