# -*- coding: utf-8 -*-
import re
from kiwipiepy import Kiwi
from collections import Counter
kiwi = Kiwi()

query_keyword = "폐배터리 재활용"


def clean_text(input):
    output = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', input)
    return output


def jaccard_from_string_set(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    return (len(set1 & set2) / len(set1 | set2))


title_list = ["포스코홀딩스, 폴란드에 폐배터리 재활용 공장 준공",
              "\"전기차 배터리 다쓰고 나면 어디로?'…폐배터리 재활용주 '훨훨'",
              "[제20회 서울국제A&D컨퍼런스]\"전자폐기물·플라스틱 재활용 성장 두드러질 것\"",
              "[특징주]유일에너테크, 세계 최고 폐배터리 리튬 회수율…대중국 의존도 낮출 '해결사' 재영텍",
              "LG에너지솔루션, 난징 1·2공장서 전기차 130만대 분량 배터리 생산"]

# for title in title_list:
#     print(title)
#     query_keyword_token = kiwi.tokenize(clean_text(query_keyword))
#     keyword_morph = [item.form for item in query_keyword_token if item.tag == 'NNG' or item.tag == 'NNP' or item.tag =='NNB' or item.tag == 'NR' or item.tag == 'NP' or item.tag == 'SN']
#     keyword_token_list = list(set(keyword_morph))

#     title_token = kiwi.tokenize(clean_text(title))
#     title_morph = [item.form for item in title_token if item.tag == 'NNG' or item.tag == 'NNP' or item.tag =='NNB' or item.tag == 'NR' or item.tag == 'NP' or item.tag == 'SN']
#     title_token_list = list(set(title_morph))

#     print(jaccard_from_string_set(keyword_token_list, title_token_list))

content = "임성주 SK에코플랜트 가치혁신 담당임원 \n \n 제20회 서울국제A&D컨퍼런스가 본사 주최로 24일 서울 영등포구 페어몬트 앰배서더 서울 호텔에서 열렸다. 임성주 SK에코플랜트 가치혁신 담당임원이 강연을 하고 있다. 사진=서동일 기자 \n[파이낸셜뉴스] 전자폐기물(E-waste), 플라스틱 재활용 등 업스트림(후방산업) 사업의 성장이 두드러질 것이라는 시각이 나왔다. 폐기물 시장은 그동안 매립이나 소각과 같이 생활 환경 유지에 필수적인 다운스트림(전방산업) 중심으로 성장해왔다. \n \n임성주 SK에코플랜트 가치혁신 담당임원은 24일 파이낸셜뉴스 주최로 서울 여의도 페어몬트 앰배서더 서울 그랜드볼룸에서 열린 제20회 서울국제A&D컨퍼런스에서 \"글로벌 폐기물 시장은 전자폐기물, EV배터리, 폐플라스틱 등 업스트림 중심으로 시장이 성장 할 것으로 예상된다\"며 \"국가 간 폐기물 수출입 금지 조치와 순환경제의 중요성이 높아져서다\"고 밝혔다. \n \n미국 글로벌 시장조사업체인 얼라이드마켓리서치의 2021년 조사결과에 따르면 글로벌 폐기물 시장은 2022년 3870억달러(한화 약 519조원), 2026년 4870억달러(약 653조원), 2030년 6180억달러(약 829조원)로 급성장이 예상된다. \n \n업스트림에 해당하는 폐플라스틱의 CAGR(연평균 성장률)은 8%로, 전자폐기물은 13%로 예측된다. 기존 생활, 건설 폐기물 등 다운스트림의 CAGR 예상치 4%를 훨씬 뛰어넘는 성장이다. \n \n업스트림 공략을 위한 SK에코플랜트의 선택은 싱가포르 소재 글로벌 전기·전자 폐기물(E-waste) 분야 선도기업인 테스(TES) 인수다. 약 10억달러(지난 2월 당시 약 1조2000억원)를 베팅했다. \n \nITAD(IT자산처분서비스)를 통해 IT자산의 정보 제거 및 재활용 등이 핵심이다. 21개국에서 서비스하고 있는 '캐시카우'(수익창출원) 역할을 하고 있다. 유럽을 넘어 미국에서 본격적인 사업을 한다는 계획이다. \n \nEV(전기차) 폐배터리 재활용에도 도전한다. 싱가포르에 있는 IT기기용 배터리 재활용 공장이 기반이다. SK온을 포함해 SK그룹 내에서 배터리 재활용 밸류체인을 만든다. 전기차를 제조하는 국내 대형 OEM(주문생산) 메이커들과도 협업하고 있다. \n \n플라스틱 재활용도 키 포인트다. 노르웨이의 RVM(폐기물회수자동화기기) 제조업체인 '톰라(Tomra)'와도 수거 사업에 협력한다. SK그룹 내 석유사업 자회사와도 협업하고 있다. \n \n투명 폐페트병을 식품용기로 재활용하는 등 B2B(Bottle to Bottle) 재활용은 물론 폐플라스틱 밸류체인 전 단계에서 선순환 구조를 마련한다. 관련 시장을 고도화하고 규모의 경제를 실현한다는 계획이다. \n \n에너지 사업은 연료전지와 함께 국내외 태양광, 해상풍력 개발사업 중심으로 확대한다. \n \nSK에코플랜트는 고체산화물 연료전지 기술을 보유한 미국 블룸에너지에 지난해 10월 3000억원을 투자 지분 5.4%를 확보했다. 이달에도 4047억원을 들여 블룸에너지 지분을 늘렸다. \n \n국내 태양광 모듈 제조 및 발전소 사업을 운영하는 탑선에는 1300억원을 투자, 인수했다. 국내 태양광 공급을 대폭 늘린다는 계획이다. \n \n삼강엠앤티 지분 31.8%를 3426억원에 매입, 해상풍력 밸류체인에 진입한다. 부유식 해상풍력에 사용되는 고유 부유체 기술을 포스코와 함께 개발한다. 동해 앞바다에 디벨로퍼들이 개발하는 수십 GW 규모 프로젝트에도 진입 중이다. \n \n고체 산화물 전해조(SOEC) 기술을 통해 그린수소도 만든다. 원전과 연계하면 재생에너지원을 공급받을 때보다 안정적으로 그린수소를 생산할 수 있는 것으로 평가되는 기술이다. 미국, 인도네시아에선 그린수소 에너지를 개발 중이다. \n \n이를 통해 그린 솔루션 기반 순환경제를 완성하는 것이 SK에코플랜트의 목표다. 향후 5년 내 그린 플랫폼을 완성, 글로벌 ESG 대표기업으로 성장하겠다는 포부다. \n \n임 담당임원은 \"환경, 에너지 연계해 순환경제의 설계자 및 시행자가 되는 것이 사업 모델\"이라며 \"기업가치(EV)는 2020년 7000억원에서 올해 프리IPO투자(상장전지분투자)를 받으면서 약 3조8000억원으로 5배 넘게 증가했다. 지난 2년 간 사업모델 혁신으로 글로벌 종합환경기업, 글로벌 연료전지 사업자, 국내 톱 10 종합건설업체이자 대표 그린 디벨로퍼로 도약 할 것\"이라고 말했다. \n \n특별취재팀 김경아 팀장 서혜진 김현정 강구귀 차장 김민기 최두선 한영준 김태일 이주미 이승연 김동찬 기자"
content1 = "[머니투데이 김사무엘 기자, 방진주 PD] \n\"지금 안 사면 10년 뒤에 후회할 수 있습니다.\"\n\n폐배터리 재활용 산업이 주목받고 있다. 배터리 산업이 매년 고성장을 이어 온 것처럼 폐배터리 재활용 역시 고성장이 예약된 신성장 산업이라는 인식 때문이다.\n\n배터리 산업 전문가로 통하는 윤혁진 SK증권 신성장산업분석팀장은 머니투데이 증권 전문 유튜브 채널 '부꾸미-부자를 꿈꾸는 개미'와의 인터뷰에서 \"자동차 폐배터리가 쏟아져 나오는 2025년쯤부터 폐배터리 재활용 산업의 본격적인 성장이 시작될 것\"이라며 \"배터리 기업이 10년 전에 비해 지금 굉장히 성장한 것처럼 폐배터리 산업도 10년 뒤에는 많이 커져 있을 것\"이라고 밝혔다.\n\n \nQ. 최근 폐배터리 재활용 혹은 재사용이 주목받고 있는데요. 어떤 사업인가요?\n▶윤혁진 연구원 : 배터리를 오래 쓰다보면 성능이 떨어지게 됩니다. 완충을 해도 원래 성능의 70% 수준 이하로 떨어지면 교체 혹은 폐기를 해야 하죠. 전기차에 들어가는 배터리는 대용량인데요. 성능이 떨어졌다고 대용량 배터리를 그냥 버리기 아까우니 이를 ESS(에너지 저장 시스템)나 UPS(무정전 전원 장치)에 다시 사용합니다. 이를 재사용(reuse)이라고 하고요.\n\n재사용하기에도 성능이 많이 떨어지는 폐배터리는 전부 분해해서 이 안에 있는 니켈, 코발트, 망간, 알루미늄 등 값비싼 소재들을 뽑아내는데 이를 재활용(recycle)이라고 합니다.\n\nQ. 폐배터리 재활용 산업이 고성장 할 것으로 보는 이유는 무엇인가요?\n▶일반적으로 전기차를 사면 수명은 7~10년 정도로 봅니다. 그러면 오늘 팔린 전기차는 7~10년 뒤에 폐배터리가 나온다고 보면 되겠죠. 전 세계적으로 전기차가 100만대 이상 팔린 첫 해가 2017년이었습니다. 중국에서는 2018년 처음으로 연간 판매량 100만대를 넘었죠. 그러면 2024년이나 2025년쯤부터 폐배터리가 쏟아져 나올 겁니다. \n\n유럽에서 전기차 판매가 100만대를 넘은 건 2020년이고요. 미국은 올해 100만대 이상 팔릴 것 같습니다. 2027년쯤 이후부터는 이 시장에서 폐배터리가 엄청 나오겠죠.\n\n아직은 폐배터리가 본격적으로 나오기 전인데도 재활용 업체들은 셀 스크랩(배터리 생산 과정에서 나온 불량품)만으로도 충분히 고성장하고 있습니다. 지금이 1차 성장이라면 2025년 이후로 예상되는 2차 성장은 더 가파르게 성장할 수 있죠.\n\n세계적으로 친환경 정책을 적극 추진하고 있다는 점도 긍정적입니다. 유럽은 2024년부터 배터리 생산과정에서 발생하는 탄소 발자국을 모두 공개하도록 했습니다. 배터리 하나를 만드는 데 발생하는 탄소의 양을 공개해야 하는 거죠. 2030년부터는 유럽에서 팔리는 배터리를 생산할 때 재활용 소재를 일정 비율 이상 사용하도록 의무화 했습니다. 코발트 12%, 니켈 4%, 리튬 4%는 재활용 소재를 써야하죠.\n\n2035년부터는 규제가 더 강해집니다. 코발트 20%, 리튬 10%, 니켈 12% 이상은 재활용 소재를 써야합니다. 이런 정책들이 폐배터리 재활용 시장을 장기적으로 크게 만들수 있는 거죠.\n\nQ. 폐배터리 재활용 산업에서 대표적인 업체는 어디인가요?\n▶미국에 라이사이클이란 기업이 있고요. 우리나라에는 성일하이텍과 에코프로 등이 있습니다.\n\n라이사이클은 2016년에 만들어진 신생회사입니다. 공장도 별로 없는데 시가총액은 13억달러(1조7000억원)나 돼요. 이건 라이사이클이 미국이라는 지역적 특성을 갖고 있기 때문입니다. 폐배터리는 환경오염과 운송비 등의 문제로 국가간 이동이 어렵습니다. 철저하게 로컬 비즈니스죠. 앞으로 폐배터리가 쏟아져 나올 시장이 어딜지 생각해보면 당연히 한국보다 미국에 있는 기업이 더 높은 가치를 받을 수밖에 없습니다.\n\n성일하이텍은 최근 상장해서 현재 시가총액이 약 1조원입니다. 폐배터리 재활용은 크게 전처리와 후처리 과정으로 나뉘고요. 후처리는 다시 습식제련과 건식제련으로 나뉩니다. 성일하이텍은 전처리와 후공정 중 습식제련을 하는 회사입니다. 전처리 생산능력은 지난해 연 6만1000톤에서 올해는 10만6000톤, 내년에는 13만톤으로 두 배씩 커질 예정입니다. 습식제련은 올해 기준 4320톤을 처리할 수 있는 규모고요. 현재 군산에 1만톤 규모의 새로운 공장을 짓고 있습니다.\n\n미국과 유럽에 진출할 계획도 있습니다. 내년에 군산공장이 잘 가동되면 시장에서 자금을 조달하든 자체 조달하든 자금을 모아 미국과 유럽에 1만 톤짜리 습식제련 공장을 짓겠다는 계획이고요.\n\n에코프로는 자회사 에코프로CNG가 폐배터리 재활용 사업을 합니다. 전처리 규모가 연 2만톤 정도고요. 습식제련은 연 1만2000톤 규모입니다. \n\n재활용 업체들은 지금 성장 단계이기 때문에 당장 매출은 큰 의미가 없고요. 장기적인 산업의 성장성에 더 주목을 하는 것이 좋습니다. ☞ 폐배터리 재활용 산업에 관한 보다 자세한 내용은 유튜브 채널 '부꾸미-부자를 꿈꾸는 개미'에서 확인하실 수 있습니다."
content2 = "포스코홀딩스가 25일 폴란드 브젝돌니시에 이차전지 리사이클링 공장 'PLSC(Poland Legnica Sourcing Center)'를 준공했다. \n \n\n\n \n\n 이날 준공식 행사는 유병옥 포스코홀딩스 친환경미래소재팀장, 임훈민 주폴란드 대사, 야누스 지아르스키 군수, 파베우 피렉 시장과  LG에너지솔루션, GS에너지, 성일하이텍 등 고객사 및 협력사 관계자 등이 참석했다. \n \n\n\n \n\n유병옥 포스코홀딩스 친환경미래소재팀장은 \"리사이클링 사업은 친환경 배터리 선순환 경제의 중심 축이자 포스코그룹이 추진하는 이차전지소재 사업의 핵심\"이라며  \"리사이클링 사업을 지속적으로 확대하여 기업의 사회적 책임에 앞장서고 동시에 이차전지소재 원료 경쟁력을 확보해 나가겠다\"라고 말했다. \n \n\n\n \n\n포스코홀딩스는 전기차 시장 확대에 따른 이차전지 재활용 시장의 성장과 세계 각국 정부 및 주요 고객사의 이차전지 재활용 원료 사용 의무화 요구에 대응하기 위해 2021년 3월 이차전지 재활용 자회사 PLSC를 설립했다. 이차전지 재활용 고유 기술을 보유한 국내 업체 성일하이텍과 협업해 공장을 운영하고 있다. \n \n\n\n \n\n2021년 10월 공장 착공 후 10개월여 만에 준공한 PLSC는 연산 7000톤의 생산능력을 갖춘 이차전지 재활용 공장이다. 이곳에선 유럽의 배터리 제조과정에서 발생하는 폐기물인 스크랩과 폐배터리를 수거, 분쇄해 가루형태의 중간가공품(블랙 매스, Black Mass)을 만들고, 이 중간가공품에서 리튬 니켈 코발트 망간 등을 추출하는 포스코HY클린메탈에 공급할 예정이다. \n \n\n\n \n\n블랙매스는 리튬이온 배터리 스크랩을 파쇄하고 선별 채취한 검은색의 분말로. 니켈 리튬 코발트, 망간 등을 함유하고 있다. \n \n\n\n \n\n한편 포스코그룹은 2010년 이차전지소재사업에 진출한 이래 핵심 원료인 리튬 니켈 분야에서 지속적인 투자와 기술개발을 진행하고 있다. 2030년까지 안정적인 이차전지소재 밸류체인을 구축해 리튬 30만톤, 니켈 22만톤, 양극재 61만톤, 음극재 32만톤을 생산해 매출액 41조원 달성한다는 계획이다."
content3 = "SK이노베이션이 미래 신성장동력으로 차세대 소형모듈원자로(SMR)와 전기차용 배터리, 재활용 사업에 주목하고 있다고 밝혔다. \n \n 김준(사진) SK이노베이션 부회장은 11일(현지시간) 미국 캘리포니아주 새너제이에서 열린 ‘SK이노베이션 글로벌 포럼’ 기조연설을 통해 “차별적 기술 기반의 무탄소·저탄소 에너지, 순환경제 중심 친환경 포트폴리오 개발을 통해 미래 성장을 추진하겠다”며 “전기가 에너지의 핵심이 되는 전동화, 폐기물·소재의 재활용 등에 초점을 맞춰 포트폴리오를 구축해 나갈 것”이라고 밝혔다. \n \n 김 부회장은 특히 전동화 영역과 관련해 “원자력(SMR), 전기차 배터리 및 소재 등 다양한 차세대 성장 분야에 주목하고 있다”고 강조했다. 앞서 지난달 SK이노베이션은 SK㈜와 함께 차세대 SMR 기업인 미국 ‘테라파워’와 포괄적 사업 협력을 위한 양해각서를 체결했다. 테라파워는 2008년 마이크로소프트 창업자인 빌 게이츠가 설립한 회사로, 대형 원전보다 발전 용량과 크기를 줄이고 안정성을 높인 SMR의 핵심 기술력을 보유하고 있다. \n \n 김 부회장은 이와 함께 순환경제 영역에서도 폐배터리 재활용과 폐자원 활용 등 신규 성장동력 발굴을 지속해서 이어 갈 것이라고 밝혔다. \n \n SK이노베이션은 향후 전동화, 폐기물·소재 재활용 분야에서 자체 보유 기술에 더해 각 분야의 글로벌 선도 기업 지분투자와 기술·사업 협력으로 관련 프로젝트를 추진할 계획이다. 이를 바탕으로 2050년 이전까지 온실가스 순배출량을 제로(0)로 하는 ‘넷제로’ 목표를 달성할 방침이다. \n \n 글로벌 포럼은 SK그룹이 미래 신성장동력 발굴을 위해 2012년부터 개최해 온 행사다. SK이노베이션은 11∼12일 현지의 산학 전문가들을 초청해 무탄소·저탄소 에너지 및 자원순환, 배터리 등 사업 분야와 관련한 포럼을 개최했다. 이번 포럼에는 김 부회장을 비롯해 지동섭 SK온 사장 등 SK이노베이션 계열의 주요 경영진이 참석했다. \n\n \n우상규 기자 skwoo@segye.com"

token = kiwi.tokenize(clean_text(content))
morph = [item.form for item in token if item.tag == 'NNG' or item.tag == 'NNP' or item.tag =='NNB' or item.tag == 'NR' or item.tag == 'NP' or item.tag == 'SN']
a = Counter(morph)
# print(a.get('석유'))


def count_token_occurrence(keyword, content):
    keyword_token = kiwi.tokenize(clean_text(keyword))
    keyword_morph = [item.form for item in keyword_token if item.tag == 'NNG' or item.tag == 'NNP' or item.tag =='NNB' or item.tag == 'NR' or item.tag == 'NP' or item.tag == 'SN']

    content_token = kiwi.tokenize(clean_text(content))
    content_morph = [item.form for item in content_token if item.tag == 'NNG' or item.tag == 'NNP' or item.tag =='NNB' or item.tag == 'NR' or item.tag == 'NP' or item.tag == 'SN']
    content_counter = Counter(content_morph)

    count = 0
    for morph in keyword_morph:
        if(content_counter.get(morph) == None):
            continue
        count = count + content_counter.get(morph)

    return count/len(content_morph)

print(count_token_occurrence(query_keyword, content3))

