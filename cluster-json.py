import json
import time
from kiwipiepy import Kiwi
kiwi = Kiwi()

def jaccard_from_string_set(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    return (len(set1 & set2) / len(set1 | set2))

start = time.time()
filename = 'esg2022-09-05-2022-09-06'
# filename= '한국타이어 사회공헌'
threadhold = 0.2
newslist = []

with open(f'esg-news-list-json/{filename}.json', encoding='utf-8') as file:
    newslist = json.load(file)

#calculate jaccard coefficient and make array for every news in newslist
jaccard_calculate = []
for index, news in enumerate(newslist):
    clustertmp = [index]
    jndex = index + 1
    while jndex < len(newslist):
        if (jaccard_from_string_set(newslist[index]['token_set'], newslist[jndex]['token_set']) > threadhold):
            clustertmp.append(jndex)
        jndex = jndex + 1
    jaccard_calculate.append(clustertmp)

#make clusters by greedy algorithm
clusters = []
ignoreIdx = []
count = 0 

while count < len(jaccard_calculate):
    # print(count)
    if count in ignoreIdx:
        count = count + 1
        continue
    cluster = jaccard_calculate[count]
    # print(cluster)
    for index in cluster:
        if len(set(jaccard_calculate[index]) - set(cluster)) != 0:
            # print(cluster)
            cluster = cluster + list(set(jaccard_calculate[index]) - set(cluster)) #greedy part 
            # print(cluster)
    print(cluster)
    clusters.append(cluster)
    ignoreIdx = ignoreIdx + cluster
    # print(len(ignoreIdx))
    count = count + 1

# Compare two clusters if Same News Clustered both in different clusters
i = 0
while i < len(clusters):
    j = i + 1
    while j < len(clusters):
        if(len(set(clusters[i]) & set(clusters[j]))) > 0:
            duplicateItems = list(set(clusters[i]) & set(clusters[j])) #make list of duplicate items

            #make cluster without duplicate items
            clusterI = list(set(clusters[i]) - (set(clusters[i]) & set(clusters[j])))
            clusterJ = list(set(clusters[i]) - (set(clusters[i]) & set(clusters[j])))

            #make tokens to compare
            clusterIToken = []
            for idx in clusterI:
                clusterIToken = clusterIToken + newslist[idx]['token_set']
            clusterITokenSet = set(clusterIToken)

            clusterJToken = []
            for idx in clusterJ:
                clusterJToken = clusterJToken + newslist[idx]['token_set']
            clusterJTokenSet = set(clusterJToken)

            #compare jaccard
            for item in duplicateItems:
                jaccardI = len(clusterITokenSet & set(newslist[item]['token_set'])) / len(clusterITokenSet | set(newslist[item]['token_set']))
                jaccardJ = len(clusterJTokenSet & set(newslist[item]['token_set'])) / len(clusterJTokenSet | set(newslist[item]['token_set']))
                if jaccardI > jaccardJ:
                    clusters[j] = list(set(clusters[j])-set([item]))
                else:
                    clusters[i] = list(set(clusters[i])-set([item]))

            #reset count
            i=0
            j=0
        j = j + 1
    i = i + 1

#sort cluster by published date
print(f"length: {len(clusters)}")
for cluster in clusters:
    cluster.sort()

clusters = sorted(clusters,key=lambda x:newslist[x[0]]['published_at'],reverse=False)





#make txt file
file = open(f'cluster/{filename}.txt', 'w+')
for cluster in clusters:
    for idx, newsIndex in enumerate(cluster):
        if idx == 0:
            file.write(newslist[newsIndex]['title'] +' | '+ newslist[newsIndex]['provider'] + ' | ' + newslist[newsIndex]['provider_link_page'] + '\n')
        else:
            file.write('\t' + newslist[newsIndex]['title'] + ' | '+ newslist[newsIndex]['provider'] + ' | ' + newslist[newsIndex]['provider_link_page'] + '\n')
file.close()

#make json file
json_file = open(f'cluster-json/{filename}.json','w+',encoding='UTF-8')

#make obj
json_cluster = {'date':f'{filename[0:10]}', 'news_cluster_list':[]}

for cluster in clusters:
    json_entitiy = {'id':f"cl:{newslist[cluster[0]]['news_id']}",'news': []}
    for newsIndex in cluster:
        json_entitiy['news'].append(newslist[newsIndex])
    json_cluster['news_cluster_list'].append(json_entitiy)

json.dump(json_cluster, json_file, indent=2, ensure_ascii=False)

json_file.close()

print(f"finished clustering in {time.time() - start}sec")



# cluster_list = []
# filename='포스코'
# with open(f'cluster-json/{filename}.json', encoding='utf-8') as file:
#     cluster_list = json.load(file)['news_cluster_list']
#     print(len(cluster_list))

# weight_list = []
# for i, cluster in enumerate(cluster_list): 
#     # print(cluster)
#     #cluster 별로 키워드와의 연관도 계산 (키워드 포함 문장 개수)/(전체문장 개수)
#     weight_of_cluster_news = []
#     cluster_news_length = len(cluster)

#     for news in cluster['news']:
#         count = 0
#         sents = kiwi.split_into_sents(news['content'])
#         length = len(sents)
#         for sent in sents:
#             if(sent.text.upper().count(filename.upper()) > 0):
#                 count = count + 1
#         weight_of_cluster_news.append(count/length)
    
#     # print('count',count)
#     weight_list.append({'index':i,'weight':sum(weight_of_cluster_news)/len(weight_of_cluster_news)})

# weight_list = sorted(weight_list, key=lambda x:x['weight'], reverse=True)

# for weight in weight_list:
#     if(weight['weight']>0.1):
#         print(cluster_list[weight['index']]['news'][0]['title'])