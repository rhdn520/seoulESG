#-*- coding: utf-8 -*-
from kiwipiepy import Kiwi
kiwi = Kiwi()
kiwi.load_user_dictionary('user_dict.txt')
import json
from collections import Counter

def count_tokens(text):
    tokens =kiwi.tokenize(text)
    nn_tokens = [token.form for token in tokens if token.tag == 'NNG' or token.tag =='NNP']
    # print(tokens)
    tokens_count = Counter(nn_tokens)
    return tokens_count

def update_keyword_relations(tokens_count):
    #기존 json 파일 로드
    # jsonfile = {}
    with open(f'dictionary/keyword_relations.json','r', encoding='UTF-8') as jsonfile:
        data = json.load(jsonfile)
    
    for token in tokens_count:
        if token in data: 
            # 기존 단어 정보 update
            for tokenn in tokens_count:
                if token == tokenn: 
                    continue
                elif(tokenn in data[token]): #tokens count 정보가 이미 있는 경우
                    data[token][tokenn] = data[token][tokenn] + tokens_count[tokenn]
                else: #tokens count 정보가 없는 경우 새로운 tokens count 를 넣어줌
                    data[token][tokenn] = tokens_count[tokenn]
        else:
            # 새로운 단어에 대한 정보 추가
            data[token] = {}
            for tokenn in tokens_count:
                if token == tokenn:
                    continue
                else:
                    # print(tokens_count[tokenn])
                    data[token][tokenn] = tokens_count[tokenn]
            # jsonfile[token] = tokens_count[token]
    
    with open(f'dictionary/keyword_relations.json','w', encoding='UTF-8') as jsonfile:
        json.dump(data, jsonfile, indent=2, ensure_ascii=False)

def main():
    test_text = "현대건설이 차세대 원전사업 핵심역량 강화를 위해 국내 최고 원자력 종합연구개발기관과 손잡았다. \n \n\n 현대건설은 10일 서울 종로구 현대건설 본사에서 한국원자력연구원과 ‘소형모듈원전, 원자력 수소생산 및 원전해체 기술 개발 협력을 위한 업무협약(MOU)’을 체결했다. 협약식에는 윤영준 현대건설 사장과 박원석 한국원자력연구원장 등 양사 주요 관계자들이 참석한 가운데 진행됐다. \n \n\n 이번 협약을 바탕으로 양측은 △비경수로형 SMR 개발 △경수로형 SMR 시공 기술 △연구용 원자로 관련 기술협력 △원자력을 이용한 수소 생산 △원전해체 기술개발 등 핵심 분야에서 상호 협력한다. 또한 해당분야의 기술 및 정보 교류, 해외 시장 진출 등에 관한 협력을 적극 추진할 예정이다. \n \n\n 현대건설은 한국원자력연구원과 협력체계를 구축하는데 성공하면서, 기존의 경수로형 뿐만 아니라 4세대 소형모듈원전 기술 개발에 속도가 붙을 것이라고 기대했다. 또한 원자력 산업의 신시장인 원전해체와 원자력을 이용한 수소 생산 분야에서 다양한 시너지 효과도 나타날 수 있다고 전망하고 있다. \n \n\n 윤영준 현대건설 사장은 “K원전 기술을 선도하고 있는 현대건설과 한국원자력연구원이 전략적 제휴를 맺음으로써 탄소제로 신형 원전기술 개발과 차세대 원전사업 추진 속도가 더욱 빨라질 것으로 기대한다”며 “원천 기술 확보와 사업 다각화를 통해 글로벌 원전산업의 게임 체인저로서 현대건설의 입지를 확고히 하는 한편, K원전기술 강국의 위상을 제고할 것”이라고 말했다."
    tokens_count = count_tokens(test_text)
    update_keyword_relations(tokens_count)




if __name__ == "__main__":
    main()