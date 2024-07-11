from logging import basicConfig
from urllib import response
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.callbacks import get_openai_callback

import os
import re
import urllib.parse
import random
import time
import datetime
import pandas as pd
import ast

import requests

from datasets.download import DownloadManager
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import tiktoken

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MIRO_API_KEY = os.environ.get("MIRO_API_KEY")
if not MIRO_API_KEY:
    MIRO_API_KEY = st.secrets["keys"]["MIRO_API_KEY"]

theme = ""
prompt_ptrn = ""
sytle_prompt = ""

# 現在時刻の取得
dt_now = datetime.datetime.now()
# 現在時刻の書式変更
dt_now_formatted = dt_now.strftime("%Y%m%d_%H%M%S")

def theme_translate(user_theme,openai_api_key):
    theme = "私が入力するのは、" + user_theme + "についてのデータです。"
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name=st.session_state["model_name"])
    translating_prompt = f"""
Please translate the following Japanese sentence into English.
{theme}
"""
    st.session_state.messages.append(SystemMessage(content=translating_prompt))
    with st.spinner("KJ-GPTがテーマを分析しています ..."):
        answer, cost = get_answer(llm, st.session_state.messages[-1:])
    translated_theme = answer
    st.session_state.costs.append(cost)
    st.session_state["translated_theme"] = translated_theme
    return translated_theme

def eng_translates(text,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name=st.session_state["model_name"])
    translating_prompt = f"""
Please translate the following Japanese sentence into English.
{text}
"""
    st.session_state.messages.append(SystemMessage(content=translating_prompt))
    with st.spinner("KJ-GPTが内容を分析しています ..."):
        answer, cost = get_answer(llm, st.session_state.messages[-1:])
    translated_text = answer
    st.session_state.costs.append(cost)
    return translated_text

def summarize(text,openai_api_key,style):
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7, model_name=st.session_state["model_name"])

    if style=="formal":
        translating_prompt = f"""
### Instruction:
Act as an introspective person who excels at looking deep into his or her own mind.
Briefly summarise the following statement in about 50 words.

### Conditions:
Please write in Japanese.

### Input:
（その中でも一番気になっているのが、）宿題が多すぎて課題をこなすだけになっているのが嫌ということ。（答えのある問題をただ強制的に解答させられるのは無駄だと思う。インターネットやChatGPTなどが急速に発展しているので、そういった単なる暗記や論理計算は、そのうち人間がやる必要はなくなると思う。それなのに、このまま偏差値至上主義の詰め込み教育で今後もやっていくならば、何の役にも立たない大人を育てることになるだろう。）
そうではなくて、もっと未来の社会を発展させるような意味ある勉強がしたい。（答えのない問いに試行錯誤しながら立ち向かったり、自分だけの特別な興味関心を育てて専門性を高めたりする勉強の方が今後求められるのは明らかだ。）つまり、目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたいということ。そしてそのためには、子どもだけじゃなく大人の教育も必要だと思う。
（そもそも今の教師が昔ながらの詰め込み式の教育で育ったので、その意識改革が必要だ。教師自身が答えのない自分の心の底から出てきた問いを設定し、生徒と一緒にそれに取り組む姿勢を見せないと、子供達はついていかない。それだけではなく、子供の親たちも新しい学びを人生に取り入れなければならない。答えのない探究活動は従来の学習に比べて、より日常生活に深く関わるものだ。普段過ごしている中で感じる疑問や違和感などを起点にした、実体験に即した問いであるほど、今後の長い人生で取り組むに値する深いものになりやすい。なので、これまでのように親が教育を学校や塾に任せっぱなしにして、家庭で子供に無関心でいては子供の探究心が育ちにくくなる。教師と同じように、親たちも自分の問いを立ててそれを追求する営みを実際にやるべきだ。そして、その行動が子供たちを感化させ、家庭を活気づかせて、さらには職場のパフォーマンスも上げることになるのが理想だ。）これらをまとめると、未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要なのだと言える。

### Output:
考える力や新しいことに挑戦する力を育むような本当に意味のある学びは、年齢に関係なく誰にでも大切だ。

### Input:
{text}
"""
    else:
        translating_prompt = f"""
### Instruction:
Act as an introspective person who excels at looking deep into his or her own mind.
Briefly summarise the following statement in about 50 words.

### Conditions:
Please write in Japanese.

### Input:
（その中でも一番気になっているのが、）宿題が多すぎて課題をこなすだけになっているのが嫌ということ。（答えのある問題をただ強制的に解答させられるのは無駄だと思う。インターネットやChatGPTなどが急速に発展しているので、そういった単なる暗記や論理計算は、そのうち人間がやる必要はなくなると思う。それなのに、このまま偏差値至上主義の詰め込み教育で今後もやっていくならば、何の役にも立たない大人を育てることになるだろう。）
そうではなくて、もっと未来の社会を発展させるような意味ある勉強がしたい。（答えのない問いに試行錯誤しながら立ち向かったり、自分だけの特別な興味関心を育てて専門性を高めたりする勉強の方が今後求められるのは明らかだ。）つまり、目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたいということ。そしてそのためには、子どもだけじゃなく大人の教育も必要だと思う。
（そもそも今の教師が昔ながらの詰め込み式の教育で育ったので、その意識改革が必要だ。教師自身が答えのない自分の心の底から出てきた問いを設定し、生徒と一緒にそれに取り組む姿勢を見せないと、子供達はついていかない。それだけではなく、子供の親たちも新しい学びを人生に取り入れなければならない。答えのない探究活動は従来の学習に比べて、より日常生活に深く関わるものだ。普段過ごしている中で感じる疑問や違和感などを起点にした、実体験に即した問いであるほど、今後の長い人生で取り組むに値する深いものになりやすい。なので、これまでのように親が教育を学校や塾に任せっぱなしにして、家庭で子供に無関心でいては子供の探究心が育ちにくくなる。教師と同じように、親たちも自分の問いを立ててそれを追求する営みを実際にやるべきだ。そして、その行動が子供たちを感化させ、家庭を活気づかせて、さらには職場のパフォーマンスも上げることになるのが理想だ。）これらをまとめると、未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要なのだと言える。

### Output:
ガチで意味ある勉強をすれば、考える力とか挑戦する力が伸びるし、それは子どもだけじゃなくて大人になっても大切。

### Input:
{text}
"""
    st.session_state.messages.append(SystemMessage(content=translating_prompt))
    with st.spinner("KJ-GPTが内容を要約しています ..."):
        answer, cost = get_answer(llm, st.session_state.messages[-1:])
    summarized_text = answer
    st.session_state.costs.append(cost)
    return summarized_text

def data_generating(user_theme,openai_api_key):
    theme = user_theme
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name=st.session_state["model_name"])
    translating_prompt = f"""
Please translate the following Japanese sentence into English.
{theme}
"""
    st.session_state.messages.append(SystemMessage(content=translating_prompt))
    with st.spinner("KJ-GPTがテーマについて考えています ..."):
        answer, cost = get_answer(llm, st.session_state.messages[-1:])
    translated_theme = answer
    st.session_state.costs.append(cost)

    generating_prompt = f"""
### Instruction:
Please list about 15 bulleted assertions or opinions from a variety of perspectives, including those that may be somehow related to the following topic.
Items should begin with the letter "・".
Items should be generated with more subjective items that are full of emotions and feelings.
Keep sentences as concise as possible, avoiding duplication.
Please write in Japanese.

### Topic:
{translated_theme}
"""
    return generating_prompt

def select_style():
    # 文章スタイルの選択
    style_choice = st.sidebar.radio("文章化のスタイル:", ("カジュアル", "フォーマル"))
    if style_choice == "カジュアル":
        style = "casual"
        sytle_prompt = "Write in a casual manner."
    else:
        style = "formal"
        sytle_prompt = "Please write in a formal manner."
    return style

def prompt_grouping(lines, translated_theme):
    # 例示データの出典
    # 「R1 ネパール山地に技術協力するには？」
    # 四注記 (1) 1985.1.11 (2)自宅 (3)川喜田二郎 (4)川喜田二郎 (5)『KJ法』執筆用 (6)川喜田二郎1986『KJ法 渾沌をして語らしめる』中央公論社、pp.374-375
    theme = translated_theme
    grouping1 = f"""
### Instruction:
{theme}
Please group all of these items with those that are closer in deeper underlying meaning.

### Condition:
Please write in Japanese.
Each group should have at most 3 items. Items that cannot be grouped should be listed as is.
Items with conflicting content should be grouped separately.

### Input:
#### Theme:
ネパール山地に技術協力するには？
#### Data:
ヘマをするとネパールは戦争の引き金となったバルカン・ベトナムと化してしまう。
山岳地の村々には、素朴な活力がありそうだ。
伝統的な暮らし方を、生態系として捉え直そう。
近代文明にはネパール山民にとって有害な面も多くある。
山岳地域の社会は病気にかかっているので、その原因を突き止めよう。
近代化のもと、過疎過密が重大な問題となっている。
外国人の私にもできることは、技術協力である。
私は外国人だから、内政には干渉したくない。
近外文明にはネパール山村の活性化に役立つものも多くある。
現地人より私の方が近代文明の遺産には詳しい。
過疎は伝統的な文化と環境を破壊しつつある。
私にできることは、地理学・民族学での経験を活かすことだ。
住民の力で維持できるような技術導入でありたい。
ネパールに車道ばかり先行して発達させれば、山中は過疎化するばかりではないか。
住民にとって最も効果的な事業から重点的に始めたい。
山中の村々が自力更生で活性化するのを激励したい。
住民と一緒になって活性化を図ることが成功の鍵。
ヒマラヤ山中の土地と人とが好きだ。
地すべりが頻発しているようだ。
成功した実例は広域の人々を励ます効果があろう。
僻地でも活性化の道はあるという実例を、小地域で実現したい。
山民の福祉向上が国際紛争の未然の防止策である。
新技術の導入は、住民にとって黒字の事業であるべきだ。
現地の独自な地域性を捉え、それを活用したい。
技術協力の結果、住民の喜ぶ顔を見たい。
近代文明のやり方をそのまま押し付けるのは疑問。
個々の村が活性化しないと、人体に例えれば筋肉が発達しないようなもの。
住民は生活苦に喘ぎ、見るに忍びない。
経済的向上は福祉向上の必要条件だが十分条件ではない。
森林がどんどん失われつつあるらしい。
まず現地の自然・文化・人間からできるだけ学ぶことだ。
最初から大きな資本投下をするのは、私の力では及ばない。
ネパールの人口は急増しているが、どうやって人々は食っていけば良いのか。

### Output:
グループ1:
成功した実例は広域の人々を励ます効果があろう。
僻地でも活性化の道はあるという実例を、小地域で実現したい。
グループ2:
ネパールに車道ばかり先行して発達させれば、山中は過疎化するばかりではないか。
個々の村が活性化しないと、人体に例えれば筋肉が発達しないようなもの。
グループ3:
現地の独自な地域性を捉え、それを活用したい。
近代文明のやり方をそのまま押し付けるのは疑問。
グループ4:
近代化のもと、過疎過密が重大な問題となっている。
過疎は伝統的な文化と環境を破壊しつつある。
グループ5:
森林がどんどん失われつつあるらしい。
地すべりが頻発しているようだ。
グループ6:
山民の福祉向上が国際紛争の未然の防止策である。
ヘマをするとネパールは戦争の引き金となったバルカン・ベトナムと化してしまう。
グループ7:
山岳地域の社会は病気にかかっているので、その原因を突き止めよう。
伝統的な暮らし方を、生態系として捉え直そう。
まず現地の自然・文化・人間からできるだけ学ぶことだ。
グループ8:
近外文明にはネパール山村の活性化に役立つものも多くある。
現地人より私の方が近代文明の遺産には詳しい。
グループ9:
外国人の私にもできることは、技術協力である。
私は外国人だから、内政には干渉したくない。
グループ10:
ヒマラヤ山中の土地と人とが好きだ。
住民は生活苦に喘ぎ、見るに忍びない。
グループ11:
山中の村々が自力更生で活性化するのを激励したい。
技術協力の結果、住民の喜ぶ顔を見たい。
グループ12:
住民の力で維持できるような技術導入でありたい。
新技術の導入は、住民にとって黒字の事業であるべきだ。
住民にとって最も効果的な事業から重点的に始めたい。
グループ13:
近代文明にはネパール山民にとって有害な面も多くある。
グループ14:
経済的向上は福祉向上の必要条件だが十分条件ではない。
グループ15:
ネパールの人口は急増しているが、どうやって人々は食っていけば良いのか。
グループ16:
最初から大きな資本投下をするのは、私の力では及ばない。
グループ17:
私にできることは、地理学・民族学での経験を活かすことだ。
グループ18:
住民と一緒になって活性化を図ることが成功の鍵。
グループ19:
山岳地の村々には、素朴な活力がありそうだ。

### Input:
"""
    grouping2 = f"""
### Instruction:
{theme}
Please group all of these items with those that are closer in deeper underlying meaning.

### Condition:
Please write in Japanese.
Each group should have at most 3 items. Items that cannot be grouped should be listed as is.
Items with conflicting content should be grouped separately.

### Input:
#### Theme:
ネパール山地に技術協力するには？
#### Data:
僻地でも活性化の道はあることを小地域で成功させ、同じ僻地の広域の人々を励ましたい。
ネパール山中に車道のみ発達させ個々の村の活性化がないと、筋肉を忘れ血管のみ太らせたのと同じく、過疎化の弊害をもたらすのみ。
近代文明のやり方の型通りの押しつけでなく、現地の独自な地域性を活用すべきだ。
近代化のもとで、過疎化は伝統的な文化と環境の解体までもたらし、過疎過密化は重大問題化している。
自然環境がどんどん破壊されつつあるらしい。
山岳地域を国際紛争の温床にせぬためには、住民の福祉向上こそ未然の防止策である。
まず現地から学び、山岳地帯での暮らし方を生態系として捉え直し、その生態系の病因をつきとめよう。
ネパール山村の活性化に役立つ近代文明の遺産については、住民より私の方が詳しい。
私は外国人だから内政干渉は好まず、ネパールに対してできることは技術協力である。
ヒマラヤ山中の土地と人とが好きなので、住民の生活苦を見るに忍びない。
山中の村々の自力更生による活性化を励ますような技術協力を行い、住民の喜ぶ顔を見たい。
住民の力で黒字の事業として維持できる、最も効果的な技術協力から始めたい。
近代文明にはネパール山民にとって有害な面も多くある。
経済的向上は福祉向上の必要条件だが十分条件ではない。
ネパールの人口は急増しているが、どうやって人々は食っていけば良いのか。
最初から大きな資本投下をするのは、私の力では及ばない。
私にできることは、地理学・民族学での経験を活かすことだ。
住民と一緒になって活性化を図ることが成功の鍵。
山岳地の村々には、素朴な活力がありそうだ。

### Output:
グループ1:
ネパール山中に車道のみ発達させ個々の村の活性化がないと、筋肉を忘れ血管のみ太らせたのと同じく、過疎化の弊害をもたらすのみ。
近代文明のやり方の型通りの押しつけでなく、現地の独自な地域性を活用すべきだ。
グループ2:
近代化のもとで、過疎化は伝統的な文化と環境の解体までもたらし、過疎過密化は重大問題化している。
自然環境がどんどん破壊されつつあるらしい。
ネパールの人口は急増しているが、どうやって人々は食っていけば良いのか。
グループ3:
住民と一緒になって活性化を図ることが成功の鍵。
山岳地の村々には、素朴な活力がありそうだ。
グループ4:
ネパール山村の活性化に役立つ近代文明の遺産については、住民より私の方が詳しい。
私は外国人だから内政干渉は好まず、ネパールに対してできることは技術協力である。
私にできることは、地理学・民族学での経験を活かすことだ。
グループ5:
山中の村々の自力更生による活性化を励ますような技術協力を行い、住民の喜ぶ顔を見たい。
住民の力で黒字の事業として維持できる、最も効果的な技術協力から始めたい。
グループ6:
僻地でも活性化の道はあることを小地域で成功させ、同じ僻地の広域の人々を励ましたい。
グループ7:
近代文明にはネパール山民にとって有害な面も多くある。
グループ8:
経済的向上は福祉向上の必要条件だが十分条件ではない。
グループ9:
山岳地域を国際紛争の温床にせぬためには、住民の福祉向上こそ未然の防止策である。
グループ10:
まず現地から学び、山岳地帯での暮らし方を生態系として捉え直し、その生態系の病因をつきとめよう。
グループ11:
最初から大きな資本投下をするのは、私の力では及ばない。
グループ12:
ヒマラヤ山中の土地と人とが好きなので、住民の生活苦を見るに忍びない。

### Input:
"""
    grouping3 = f"""
### Instruction:
{theme}
Please group all of these items with those that are closer in deeper underlying meaning.

### Condition:
Please write in Japanese.
Each group should have at most 3 items. Items that cannot be grouped should be listed as is.
Items with conflicting content should be grouped separately.

### Input:
#### Theme:
ネパール山地に技術協力するには？
#### Data:
僻地でも活性化の道はあることを小地域で成功させ、同じ僻地の広域の人々を励ましたい。
近代文明にはネパール山民にとって有害な面も多くある。
経済的向上は福祉向上の必要条件だが十分条件ではない。
山岳地域を国際紛争の温床にせぬためには、住民の福祉向上こそ未然の防止策である。
まず現地から学び、山岳地帯での暮らし方を生態系として捉え直し、その生態系の病因をつきとめよう。
最初から大きな資本投下をするのは、私の力では及ばない。
ヒマラヤ山中の土地と人とが好きなので、住民の生活苦を見るに忍びない。
近代文明の開発力の型通りな押しつけでなく、ネパール山地の独自な地域性を活用するのでなければ、過疎化の弊害をもたらすのみ。
人口急増を伴う近代化の直接間接の影響下に、生活問題をめぐる全体状況の変化で過疎過密化が進み、ひいては伝統的な文化・環境が解体され、重大問題化している。
山地の村々が持つらしい素朴な活力を信頼し、住民と一体になり地域の活性化を図ることが成功の鍵。
私にできることは近代文明の遺産を活用し、そこで培われた地域理解についての専門家的経験をもとに、内政干渉にならぬ技術協力を行うこと。
山中の村々が黒字で自力更生でき活性化することを励ますような技術協力をやり、住民の喜ぶ顔を見たい。

### Output:
グループ1:
近代文明にはネパール山民にとって有害な面も多くある。
経済的向上は福祉向上の必要条件だが十分条件ではない。
近代文明の開発力の型通りな押しつけでなく、ネパール山地の独自な地域性を活用するのでなければ、過疎化の弊害をもたらすのみ。
グループ2:
山地の村々が持つらしい素朴な活力を信頼し、住民と一体になり地域の活性化を図ることが成功の鍵。
まず現地から学び、山岳地帯での暮らし方を生態系として捉え直し、その生態系の病因をつきとめよう。
グループ3:
私にできることは近代文明の遺産を活用し、そこで培われた地域理解についての専門家的経験をもとに、内政干渉にならぬ技術協力を行うこと。
最初から大きな資本投下をするのは、私の力では及ばない。
グループ4:
ヒマラヤ山中の土地と人とが好きなので、住民の生活苦を見るに忍びない。
山中の村々が黒字で自力更生でき活性化することを励ますような技術協力をやり、住民の喜ぶ顔を見たい。
グループ5:
人口急増を伴う近代化の直接間接の影響下に、生活問題をめぐる全体状況の変化で過疎過密化が進み、ひいては伝統的な文化・環境が解体され、重大問題化している。
グループ6:
僻地でも活性化の道はあることを小地域で成功させ、同じ僻地の広域の人々を励ましたい。
グループ7:
山岳地域を国際紛争の温床にせぬためには、住民の福祉向上こそ未然の防止策である。

### Input:
"""

    if lines >= 20:
        prompt_ptrn = grouping1
    elif 13 <= lines < 20:
        prompt_ptrn = grouping2
    else:
        prompt_ptrn = grouping3
    return prompt_ptrn

def prompt_labeling(number_of_items, break_point, style):

    if style == "formal":
        labeling1 = f"""
### Instructions:
Please summarize the items of groups in one concise sentence with a deeper meaning.

### Conditions:
Please write in Japanese.
{sytle_prompt}

###Input:
成功した実例は広域の人々を励ます効果があろう。
僻地でも活性化の道はあるという実例を、小地域で実現したい。
###Output:
僻地でも活性化の道はあることを小地域で成功させ、同じ僻地の広域の人々を励ましたい。

###Input:
現地の独自な地域性を捉え、それを活用したい。
近代文明のやり方をそのまま押し付けるのは疑問。
###Output:
近代文明のやり方の型通りの押しつけでなく、現地の独自な地域性を活用すべきだ。

###Input:
森林がどんどん失われつつあるらしい。
地すべりが頻発しているようだ。
###Output:
自然環境がどんどん破壊されつつあるらしい。

###Input:
山岳地域の社会は病気にかかっているので、その原因を突き止めよう。
伝統的な暮らし方を、生態系として捉え直そう。
まず現地の自然・文化・人間からできるだけ学ぶことだ。
###Output:
まず現地から学び、山岳地帯での暮らし方を生態系として捉え直し、その生態系の病因をつきとめよう。

###Input:
外国人の私にもできることは、技術協力である。
私は外国人だから、内政には干渉したくない。
###Output:
私は外国人だから内政干渉は好まず、ネパールに対してできることは技術協力である。

###Input:
山中の村々が自力更生で活性化するのを激励したい。
技術協力の結果、住民の喜ぶ顔を見たい。
###Output:
山中の村々の自力更生による活性化を励ますような技術協力を行い、住民の喜ぶ顔を見たい。

### Input:
"""

        labeling2 = f"""
### Instructions:
Please summarize the items of groups in one concise sentence with a deeper meaning.

### Conditions:
Please write in Japanese.
{sytle_prompt}

###Input:
ネパール山中に車道のみ発達させ個々の村の活性化がないと、筋肉を忘れ血管のみ太らせたのと同じく、過疎化の弊害をもたらすのみ。
近代文明のやり方の型通りの押しつけでなく、現地の独自な地域性を活用すべきだ。
###Output:
近代文明の開発力の型通りな押しつけでなく、ネパール山地の独自な地域性を活用するのでなければ、過疎化の弊害をもたらすのみ。

###Input:
近代化のもとで、過疎化は伝統的な文化と環境の解体までもたらし、過疎過密化は重大問題化している。
自然環境がどんどん破壊されつつあるらしい。
ネパールの人口は急増しているが、どうやって人々は食っていけば良いのか。
###Output:
人口急増を伴う近代化の直接間接の影響下に、生活問題をめぐる全体状況の変化で過疎過密化が進み、ひいては伝統的な文化・環境が解体され、重大問題化している。

###Input:
住民と一緒になって活性化を図ることが成功の鍵。
山岳地の村々には、素朴な活力がありそうだ。
###Output:
山地の村々が持つらしい素朴な活力を信頼し、住民と一体になり地域の活性化を図ることが成功の鍵。

###Input:
ネパール山村の活性化に役立つ近代文明の遺産については、住民より私の方が詳しい。
私は外国人だから内政干渉は好まず、ネパールに対してできることは技術協力である。
私にできることは、地理学・民族学での経験を活かすことだ。
###Output:
私にできることは近代文明の遺産を活用し、そこで培われた地域理解についての専門家的経験をもとに、内政干渉にならぬ技術協力を行うこと。

###Input:
山中の村々の自力更生による活性化を励ますような技術協力を行い、住民の喜ぶ顔を見たい。
住民の力で黒字の事業として維持できる、最も効果的な技術協力から始めたい。
###Output:
山中の村々が黒字で自力更生でき活性化することを励ますような技術協力をやり、住民の喜ぶ顔を見たい。

### Input:
"""

        labeling3 = f"""
### Instructions:
Please summarize the items of groups in one concise sentence with a deeper meaning.

### Conditions:
Please write in Japanese.
{sytle_prompt}

###Input:
近代文明にはネパール山民にとって有害な面も多くある。
経済的向上は福祉向上の必要条件だが十分条件ではない。
近代文明の開発力の型通りな押しつけでなく、ネパール山地の独自な地域性を活用するのでなければ、過疎化の弊害をもたらすのみ。
###Output:
ネパール山地の独自な地域性を無視し、経済至上主義その他の弊害を持った近代文明の開発力を型通り押しつけるのでは、過疎化の弊害を除いた本物の技術協力はできない。

###Input:
山地の村々が持つらしい素朴な活力を信頼し、住民と一体になり地域の活性化を図ることが成功の鍵。
まず現地から学び、山岳地帯での暮らし方を生態系として捉え直し、その生態系の病因をつきとめよう。
###Output:
実態把握による山岳地生態系の診断と、住民の素朴な活力を信頼した相互参画とが、互いに強めあってこそ、地域の活性化に通ずる技術協力に成功しよう。

###Input:
私にできることは近代文明の遺産を活用し、そこで培われた地域理解についての専門家的経験をもとに、内政干渉にならぬ技術協力を行うこと。
最初から大きな資本投下をするのは、私の力では及ばない。
###Output:
私にできることは、小資本で足りる対象を選び、近代文明の遺産を生かし、身につけた地域理解についての専門家的経験をもとに、内政干渉にならぬ技術協力を行うこと。

###Input:
ヒマラヤ山中の土地と人とが好きなので、住民の生活苦を見るに忍びない。
山中の村々が黒字で自力更生でき活性化することを励ますような技術協力をやり、住民の喜ぶ顔を見たい。
###Output:
ヒマラヤ山中の土地と人とが好きなので、その村々が黒字の自力更生で生活苦を除き、活性化するのを励ますような技術協力を行い、住民の喜ぶ顔を見たい。

### Input:
"""
        
        symbol = """
### Instructions:
Paraphrase each group of sentences in a couple of words that can be understood instantaneously.

### Conditions:
Please write in Japanese.
Rephrase it with an adjective, verb or metaphor with creativity.

###Input:
ネパール山地の独自な地域性を無視し、経済至上主義その他の弊害を持った近代文明の開発力を型通り押しつけるのでは、過疎化の弊害を除いた本物の技術協力はできない。
###Output:
アテハメは失敗

###Input:
僻地でも活性化の道はあることを小地域で成功させ、同じ僻地の広域の人々を励ましたい。
###Output:
インパクト

###Input:
人口急増を伴う近代化の直接間接の影響下に、生活問題をめぐる全体状況の変化で過疎過密化が進み、ひいては伝統的な文化・環境が解体され、重大問題化している。
###Output:
過疎過密

###Input:
山岳地域を国際紛争の温床にせぬためには、住民の福祉向上こそ未然の防止策である。
###Output:
国際紛争の防止

###Input:
実態把握による山岳地生態系の診断と、住民の素朴な活力を信頼した相互参画とが、互いに強めあってこそ、地域の活性化に通ずる技術協力に成功しよう。
###Output:
実態把握と参画

###Input:
私にできることは、小資本で足りる対象を選び、近代文明の遺産を生かし、身につけた地域理解についての専門家的経験をもとに、内政干渉にならぬ技術協力を行うこと。
###Output:
小さなモデルを！

###Input:
ヒマラヤ山中の土地と人とが好きなので、その村々が黒字の自力更生で生活苦を除き、活性化するのを励ますような技術協力を行い、住民の喜ぶ顔を見たい。
###Output:
自立を励ます

### Input:
"""

    else:
        labeling1 = f"""
### Instructions:
Please summarize the items of groups in one concise sentence with a deeper meaning.

### Conditions:
Please write in Japanese.
{sytle_prompt}

###Input:
成功した実例は広域の人々を励ます効果があろう。
僻地でも活性化の道はあるという実例を、小地域で実現したい。
###Output:
人里離れた場所でも良くなる方法があるということを、小さい地域で成功させて、同じような場所に住むたくさんの人を元気づけたい。

###Input:
現地の独自な地域性を捉え、それを活用したい。
近代文明のやり方をそのまま押し付けるのは疑問。
###Output:
新しい時代のやり方をそのまま押し付けるのではなく、その場所ならではの特徴を生かすべき。

###Input:
森林がどんどん失われつつあるらしい。
地すべりが頻発しているようだ。
###Output:
自然環境がどんどん破壊されているらしい。

###Input:
山岳地域の社会は病気にかかっているので、その原因を突き止めよう。
伝統的な暮らし方を、生態系として捉え直そう。
まず現地の自然・文化・人間からできるだけ学ぶことだ。
###Output:
まずは現地から学び、山での生活を生態系のなかで理解し、その仕組みがうまくいかない原因を見つけ出す。

###Input:
外国人の私にもできることは、技術協力である。
私は外国人だから、内政には干渉したくない。
###Output:
私は外国人なので、他の国の政治に口出しするのは控えて、技術支援でネパールを助ける。

###Input:
山中の村々が自力更生で活性化するのを激励したい。
技術協力の結果、住民の喜ぶ顔を見たい。
###Output:
山中の村が自分たちの力で元気になれるような技術支援を行い、住民たちの喜ぶ顔を見たい。

### Input:
"""

        labeling2 = f"""
### Instructions:
Please summarize the items of groups in one concise sentence with a deeper meaning.

### Conditions:
Please write in Japanese.
{sytle_prompt}

###Input:
ネパール山中に車道のみ発達させ個々の村の活性化がないと、筋肉を忘れ血管のみ太らせたのと同じく、過疎化の弊害をもたらすのみ。
近代文明のやり方の型通りの押しつけでなく、現地の独自な地域性を活用すべきだ。
###Output:
新しい時代の開発のやり方をそのまま強要するのではなく、ネパールの山岳地域の特別な特徴を生かさないと、過疎の問題を悪化させるだけ。

###Input:
近代化のもとで、過疎化は伝統的な文化と環境の解体までもたらし、過疎過密化は重大問題化している。
自然環境がどんどん破壊されつつあるらしい。
ネパールの人口は急増しているが、どうやって人々は食っていけば良いのか。
###Output:
人口急増と近代化の影響で、生活を巡る状況が変わり、人口が減る場所と増える場所ができ、結果として、昔からの文化や自然が壊れ大きな問題になっている。

###Input:
住民と一緒になって活性化を図ることが成功の鍵。
山岳地の村々には、素朴な活力がありそうだ。
###Output:
山の村々が自然に持っている活力を大切に、住民たちと一緒になって地域を元気にすることが成功の大事なポイント。

###Input:
ネパール山村の活性化に役立つ近代文明の遺産については、住民より私の方が詳しい。
私は外国人だから内政干渉は好まず、ネパールに対してできることは技術協力である。
私にできることは、地理学・民族学での経験を活かすことだ。
###Output:
私ができるのは、近代の良い面を使って、地域をよく理解する専門の経験を活かし、政治に口を出さないようにしながら技術支援をすること。

###Input:
山中の村々の自力更生による活性化を励ますような技術協力を行い、住民の喜ぶ顔を見たい。
住民の力で黒字の事業として維持できる、最も効果的な技術協力から始めたい。
###Output:
山の中の村々が自分たちの力でお金を稼ぎ、元気になれるように手伝い、その結果、住民たちが嬉しそうにする顔を見たい。

### Input:
"""

        labeling3 = f"""
### Instructions:
Please summarize the items of groups in one concise sentence with a deeper meaning.

### Conditions:
Please write in Japanese.
{sytle_prompt}

###Input:
近代文明にはネパール山民にとって有害な面も多くある。
経済的向上は福祉向上の必要条件だが十分条件ではない。
近代文明の開発力の型通りな押しつけでなく、ネパール山地の独自な地域性を活用するのでなければ、過疎化の弊害をもたらすのみ。
###Output:
ネパールの山岳地域の特別な性質を無視して、ただお金が一番大事といった悪影響を持つ考え方など、新しい時代の開発方法をそのまま押し付けるのでは、過疎化問題に本当に役立つ技術支援はできない。

###Input:
山地の村々が持つらしい素朴な活力を信頼し、住民と一体になり地域の活性化を図ることが成功の鍵。
まず現地から学び、山岳地帯での暮らし方を生態系として捉え直し、その生態系の病因をつきとめよう。
###Output:
山の自然状態をしっかり理解し、住んでいる人たちの自然な活力を大切にして、互いに助け合いながら一緒に取り組むことが、技術支援でその地域を盛り上げるキーポイントだ。

###Input:
私にできることは近代文明の遺産を活用し、そこで培われた地域理解についての専門家的経験をもとに、内政干渉にならぬ技術協力を行うこと。
最初から大きな資本投下をするのは、私の力では及ばない。
###Output:
私ができることは、少しの資本でできることを選んで近代の良いところを活用しつつ、地域のことをよく理解する専門の知識を使って、政治に口を出さないようにしながら技術支援をすることだ。

###Input:
ヒマラヤ山中の土地と人とが好きなので、住民の生活苦を見るに忍びない。
山中の村々が黒字で自力更生でき活性化することを励ますような技術協力をやり、住民の喜ぶ顔を見たい。
###Output:
ヒマラヤの土地とそこに住む人たちが好きなので、その村々が自分たちの力でお金を稼ぎ、生活の苦しさを無くすよう手伝って、住民たちが嬉しそうな顔をするのを見たい。

### Input:
"""

        symbol = f"""
### Instructions:
Paraphrase each group of sentences in a couple of words that can be understood instantaneously.

### Conditions:
Please write in Japanese.
Rephrase it with an adjective, verb or metaphor with creativity.
{sytle_prompt}

###Input:
ネパール山地の独自な地域性を無視し、経済至上主義その他の弊害を持った近代文明の開発力を型通り押しつけるのでは、過疎化の弊害を除いた本物の技術協力はできない。
###Output:
アテハメはうまくいかない

###Input:
僻地でも活性化の道はあることを小地域で成功させ、同じ僻地の広域の人々を励ましたい。
###Output:
インパクト

###Input:
人口急増を伴う近代化の直接間接の影響下に、生活問題をめぐる全体状況の変化で過疎過密化が進み、ひいては伝統的な文化・環境が解体され、重大問題化している。
###Output:
人口の増減

###Input:
山岳地域を国際紛争の温床にせぬためには、住民の福祉向上こそ未然の防止策である。
###Output:
世界のけんかを止める

###Input:
実態把握による山岳地生態系の診断と、住民の素朴な活力を信頼した相互参画とが、互いに強めあってこそ、地域の活性化に通ずる技術協力に成功しよう。
###Output:
しっかり理解して、参加

###Input:
私にできることは、小資本で足りる対象を選び、近代文明の遺産を生かし、身につけた地域理解についての専門家的経験をもとに、内政干渉にならぬ技術協力を行うこと。
###Output:
小さなお手本を！

###Input:
ヒマラヤ山中の土地と人とが好きなので、その村々が黒字の自力更生で生活苦を除き、活性化するのを励ますような技術協力を行い、住民の喜ぶ顔を見たい。
###Output:
自立を応援する

### Input:
"""

    if number_of_items >= 20:
        prompt_ptrn = labeling1
    elif 13 <= number_of_items < 20:
        prompt_ptrn = labeling2
    elif 0 < number_of_items < 13:
        prompt_ptrn = labeling3
    else:
        prompt_ptrn = symbol
    return prompt_ptrn

# embeddings へ変換
def to_emb(model, text, prefix="query: "):
    return model.encode([prefix + text], normalize_embeddings=True)

def set_state(i):
    st.session_state.stage = i

def init_page():
    st.set_page_config(
        page_title="KJ-GPT",
        page_icon="🕵️‍♂️"
    )
    st.header("KJ-GPT 🕵️‍♂️")
    st.sidebar.title("オプション")

def init_messages():
    clear_button = st.sidebar.button("履歴をリセット", key="clear", on_click=set_state, args=[0])
    if clear_button or "messages" not in st.session_state:

        st.session_state.messages = [
            SystemMessage(content="")
        ]

        # st.session_state.messages = [
        #     SystemMessage(content=labeling4)
        # ]

        # st.session_state.messages = [
        #     SystemMessage(content=symbol)
        # ]

        st.session_state.costs = []

        # st.session_state["openai_api_key"] = ""

        st.session_state["miro_api_key"] = ""

        st.session_state["model_name"] = ""

        st.session_state["user_input_random"] = ""

        st.session_state["labeling_pair"] = ""

        st.session_state["edited_labeling_pair"] = []

        st.session_state["saved_content"] = []

        st.session_state["grouping_table"] = []

        st.session_state["grouped_tables"] = []

        st.session_state["no_grouped_table"] = []

        st.session_state["number_of_items"] = ""

        st.session_state["labeling_table"] = []

        st.session_state["working_markdown_text"] = ""

        st.session_state["markdown_text"] = ""

        st.session_state["user_theme"] = ""

        st.session_state["translated_theme"] = ""

        st.session_state["summarized_data"] = ""

def select_model(openai_api_key):
    model = st.sidebar.radio("モデル:", ("GPT-3.5-16k", "GPT-4-Turbo", "GPT-4o"),index=2)
    if model == "GPT-3.5-16k":
        model_name = "gpt-3.5-turbo-16k-0613"
    elif model == "GPT-4-Turbo":
        model_name = "gpt-4-1106-preview"
    else:
        model_name = "gpt-4o-2024-05-13"
    st.session_state["model_name"] = model_name
    
    # サイドバーにスライダーを追加し、temperatureを0から1までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.1とする
    # temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    temperature = 0.7
    
    return ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature, model_name=model_name)

def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost

def count_newlines(text):
    # 改行文字を含む正規表現パターンを定義
    pattern = r"\r\n|\n|\r"
    # パターンにマッチする部分を抽出し、その数を返す
    return len(re.findall(pattern, text))

def split_lines_to_list(data):
    lines = data.strip().split("\n")
    return lines

def get_list(data):
    # データを行ごとに分割
    lines = data.strip().split("\n")
    # 各要素から "- " を削除
    lines = [item.lstrip('- ') for item in lines]
    # 各要素から 冒頭の数字 を削除
    lines = [re.sub(r'^\d+\.\s*', '', item) for item in lines]

    result = []
    temp = []

    for line in lines:
        # ”グループ”もしくは"単独"または"単品"の行の場合、tempをリセット
        if "グループ" in line or "単独" in line or "単品" in line:
            if temp:
                if len(temp) == 1:
                    result.append(temp[0])
                else:
                    result.append(temp)
                temp = []
        # 行にデータがある場合のみ追加
        elif line:
            temp.append(line)
    # 最後のグループのデータを追加
    if temp:
        if len(temp) == 1:
            result.append(temp[0])
        else:
            result.append(temp)
    # ""の空白要素を削除
    result = [item for item in result if item != ""]
    
    return result

def parse_dict_from_string(string):
    """文字列から辞書オブジェクトに変換する関数"""
    try:
        # 文字列が辞書形式なら辞書に変換
        if string.startswith("{") and string.endswith("}"):
            return ast.literal_eval(string)
        else:
            # 辞書形式でない場合はそのまま返す
            return string
    except ValueError as e:
        # print(f"Error converting string to dictionary: {e}")
        return string  # 変換に失敗した場合も元の文字列を返す

def add_markdown_entry(level, text):
    """与えられたレベルに基づいてマークダウン形式のエントリを追加する関数"""
    return f"{'#' * level} {text}\n"

def find_sub_items(key, level, dict_items):
    """サブアイテムを探し、その見出しをマークダウン形式で返す関数"""
    markdown = ""
    for dict_item in dict_items:
        if isinstance(dict_item, dict) and key in dict_item:
            for sub_item in dict_item[key]:
                if sub_item:  # 空の文字列でない場合にサブ見出しを追加
                    markdown += add_markdown_entry(level, sub_item)
                    # 再帰的に下のレベルの項目を探す
                    if level < 6:  # 6次見出しまで続ける
                        markdown += find_sub_items(sub_item, level + 1, dict_items)
    return markdown

def headline_to_list(markdown_text):
    """マークダウンされた見出しをリスト箇条書き形式に変換して返す関数"""

    # マークダウンテキストを行に分割します。
    lines = markdown_text.strip().split("\n")
    
    # 変換されたリストを保持するためのリストを初期化します。
    converted_list = []

    # 各行をループして処理します。
    for line in lines:

        if "**" in line:
            converted_list.append("\n" + line + "\n")

        else:
            # 見出しのレベルを決定します（#の数で決まります）。
            level = line.count("#")
            
            # 見出しのテキストを抽出します。
            text = line.replace("#", "").strip()
            
            # リストアイテムのインデントを決定します。
            # インデントは見出しレベルに応じて増やします。
            indent = '    ' * (level - 1)
            
            # 変換されたリストアイテムを追加します。
            converted_list.append(f"{indent}- {text}")

    # リスト形式に変換されたマークダウンを結合して出力します。
    converted_markdown = "\n".join(converted_list)

    return converted_markdown

def segmented_by_three(markdown_input):
    """マークダウンされた見出しを3行ごとに分割して返す関数"""
    # マークダウンテキストを行に分割します。
    lines = markdown_input.strip().split("\n")

    # 3行ごとに区切ってリスト化します。
    segmented_markdown = []
    for i in range(0, len(lines), 3):
        segment = lines[i:i + 3]
        segmented_markdown.append("\n".join(segment))

    return segmented_markdown

def split_sections(text):
    """段落ごとに分割して返す関数"""
    # 各セクションを格納するためのリスト
    sections = []
    # 現在のセクションの内容を格納するための変数
    current_section = []
    
    # テキストを行ごとに分割して処理
    for line in text.strip().split("\n"):
        # セクションのヘッダーを検出した場合、
        # 現在のセクションをリストに追加して新しいセクションを開始する
        if line.startswith("**"):
            if current_section:
                sections.append("\n".join(current_section))
                current_section = []
        current_section.append(line)
    
    # 最後のセクションをリストに追加
    if current_section:
        sections.append("\n".join(current_section))
    
    return sections

def split_by_hashes(text):
    """#の上位下位ごとにセットにして辞書型リストを返す関数（Basic Data for Abduction）"""
    # "#" で始まるテキストを抽出する正規表現パターン
    pattern_hashes = [r"(^" + "#"*i + " [^\n]*(?:\n(?!#).*)*)" for i in range(1, 7)]

    # マッチング結果をリストに格納
    matches_hashes = [re.findall(p, text, re.MULTILINE) for p in pattern_hashes]

    # '**' で挟まれたテキストから "(数字)" を除外して抽出する正規表現パターン
    pattern_bold_text_excluding_numbers = r"\*\*\(\d+\) (.+?)\*\*"

    # マッチング結果をリストに格納
    matches_bold_text_excluding_numbers = re.findall(pattern_bold_text_excluding_numbers, text)

    # セクションを辞書として関連付ける
    related_sections = {}

    # もし一番上位の項目しかない一匹狼の場合は、その項目のみ辞書に入れる。
    if matches_hashes[1] == []:
        # セクションのシンボルマークをkey、一匹狼のラベルをvalueとして辞書に登録
        symbol_mark = matches_bold_text_excluding_numbers[0]
        related_sections[symbol_mark] = matches_hashes[0]

    # 各レベルのセクションごとに関連付けを行う
    for level in range(len(matches_hashes) - 1, -1, -1):
        # matches_hashes[level]の要素を後ろからループ
        for section in reversed(matches_hashes[level]):
            # セクションの開始インデックスを見つける
            section_start_index = text.index(section)
            # このインデックスより前のすべての上位レベルのセクションを見つける
            for upper_level in range(level - 1, -1, -1):
                previous_sections = [s for s in matches_hashes[upper_level] if text.index(s) < section_start_index]
                # もっとも近い上位レベルのセクションを選択する
                if previous_sections:
                    nearest_section = previous_sections[-1]
                    # このセクションをキーとして辞書に追加
                    if nearest_section in related_sections:
                        related_sections[nearest_section].append(section.strip())
                    else:
                        related_sections[nearest_section] = [section.strip()]
                    break

    return related_sections

def split_by_hashes_chart(text):
    # テキストを行ごとに分割
    lines = text.split('\n')
    lines = [item for item in lines if item != ""]
    numbers_of_lines = len(lines)

    # 整理された構造を保持するための辞書
    organized_structure = {}

    # numbers_of_lines が 2 以下の場合の特別な処理
    if numbers_of_lines <= 2:
        if lines:
            bold_section = lines[0].strip('*')
            header = lines[1] if len(lines) > 1 else None
            organized_structure[bold_section] = header if header else bold_section
        return organized_structure

    # 現在のボールドセクションとヘッダーを保持するための変数
    current_bold_section = None
    current_headers = [None] * 7  # 7レベルのヘッダーに対応

    # 各行を繰り返し処理
    for line in lines:
        # ボールドセクションのチェック
        if line.startswith('**') and line.endswith('**'):
            current_bold_section = line.strip('*')
            organized_structure[current_bold_section] = {}
            current_headers = [None] * 7  # ヘッダーをリセット
        elif current_bold_section:
            # ヘッダーレベルを判定
            header_level = line.count('#')
            if header_level > 0:
                current_headers[header_level - 1] = line
                for i in range(header_level, 7):
                    current_headers[i] = None

                for i in range(header_level - 2, -1, -1):
                    if current_headers[i]:
                        if current_headers[i] not in organized_structure[current_bold_section]:
                            organized_structure[current_bold_section][current_headers[i]] = []
                        organized_structure[current_bold_section][current_headers[i]].append(line)
                        break

    return organized_structure

def parse_markdown_with_bold(md_text):
    """マークダウンテキストを解析し、レベル数を追加する関数"""
    lines = md_text.strip().split("\n")
    parsed_content = []

    for line in lines:
        # 太字で書かれた行を識別（レベル0）
        if line.startswith("**") and line.endswith("**"):
            content = line.strip("* ").strip()
            parsed_content.append((0, content))
        # 見出しレベルを判定
        elif line.startswith("#"):
            level = line.count("#")
            content = line.strip("# ").strip()
            parsed_content.append((level, content))
    
    return parsed_content

def add_shape(api_key, board_id, level, text, x, y, shape_width, shape_height):
    url = f"https://api.miro.com/v2/boards/{board_id}/shapes"

    if level == 0:
        style = {
                "borderOpacity": "0",
                "fontSize": "90",
                "borderColor": "#000000"
            }
    elif level == 1:
        style = {
                "borderOpacity": "0",
                "fontSize": "64",
                "borderColor": "#000000"
            }

    elif level == 2:
        style = {
                "fontSize": "50",
                "fillColor": "#ffffff",
                "fillOpacity": "1",
                "borderColor": "#000000",
                "borderWidth": "14.0"
            }

    elif level == 3:
        style = {
                "fontSize": "40",
                "fillColor": "#ffffff",
                "fillOpacity": "1",
                "borderColor": "#000000",
                "borderWidth": "10.0"
            }
    elif level == 4:
        style = {
                "fontSize": "30",
                "fillColor": "#ffffff",
                "fillOpacity": "1",
                "borderColor": "#000000",
                "borderWidth": "6.0"
            }
    else:
        style = {
                "fontSize": "18",
                "borderColor": "#000000",
                "borderWidth": "2.0"
            }

    payload = {
        "data": {
            "content": text,
            "shape": "rectangle"
        },
        "style": style,
        "position": {
            "x": x,
            "y": y
        },
        "geometry": {
            "width": shape_width,
            "height": shape_height
        }
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, json=payload, headers=headers)

def add_rounded_rectangle(api_key, board_id, x, y, width, height, level):
    url = f"https://api.miro.com/v2/boards/{board_id}/shapes"

    payload = {
        "data": {
            "shape": "round_rectangle"
        },
        "position": {
            "x": x,
            "y": y
        },
        "geometry": {
            "width": width,
            "height": height
        },
        "style": {
            "borderOpacity": "1",
            "borderColor": "#000000",
            "borderWidth": 10 / level,
        }
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, json=payload, headers=headers)

    # print(response.text)
    # print(f"角丸四角形のシェイプが位置({x}, {y})に幅{width}、高さ{height}で作成されました")


def add_shape_to_miro(api_key, board_id, text, x, y):
    url = f"https://api.miro.com/v2/boards/{board_id}/shapes"

    payload = {
        "data": {
            "content": text,
            "shape": "rectangle"
        },
        "position": {
            "x": x,
            "y": y
    }
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, json=payload, headers=headers)

    # print(response.text)

def text_to_lines_list(input_text):
  """#で始まるマークダウンの箇条書きを、行ごとに分割してリスト化する関数"""
  # テキストを行に分割
  lines = input_text.strip().split('\n')

  # 出力を整形
  lines_output = []
  current_text = ""
  for line in lines:
      # 新しいセクションの開始を検出
      if line.startswith('#'):
          # 現在のテキストを追加して新しいテキストを開始
          if current_text:
              lines_output.append(current_text.strip())
              current_text = ""
      current_text += line + "\n"
  # 最後のテキストを追加
  if current_text:
      lines_output.append(current_text.strip())
  return lines_output

def list_to_BDA(list):
  """#で始まるマークダウンの箇条書きリストを、つづけた2つの階層ごとにBDAへ変換する関数"""
  # 指定された処理を実行する
  result = []
  current_section = None
  current_subsection = None
  current_subsubsection = None
  current_subsubsubsection = None
  current_subsubsubsubsection = None
  current_subsubsubsubsubsection = None

  # 処理開始
  for text in list:
      if text.startswith("# "):  # 階層1の要素
          if current_section is not None:
              result.append(current_section)
          current_section = {"title": text, "subsections": []}
          current_subsection = None
          current_subsubsection = None
          current_subsubsubsection = None
          current_subsubsubsubsection = None
          current_subsubsubsubsubsection = None
      elif text.startswith("## ") and current_section is not None:  # 階層2の要素
          current_subsection = {"title": text, "subsubsections": []}
          current_section["subsections"].append(current_subsection)
          current_subsubsection = None
          current_subsubsubsection = None
          current_subsubsubsubsection = None
          current_subsubsubsubsubsection = None
      elif text.startswith("### ") and current_subsection is not None:  # 階層3の要素
          current_subsubsection = {"title": text, "subsubsubsections": []}
          current_subsection["subsubsections"].append(current_subsubsection)
          current_subsubsubsection = None
          current_subsubsubsubsection = None
          current_subsubsubsubsubsection = None
      elif text.startswith("#### ") and current_subsubsection is not None:  # 階層4の要素
          current_subsubsubsection = {"title": text, "subsubsubsubsections": []}
          current_subsubsection["subsubsubsections"].append(current_subsubsubsection)
          current_subsubsubsubsection = None
          current_subsubsubsubsubsection = None
      elif text.startswith("##### ") and current_subsubsubsection is not None:  # 階層5の要素
          current_subsubsubsubsection = {"title": text, "subsubsubsubsubsections": []}
          current_subsubsubsection["subsubsubsubsections"].append(current_subsubsubsubsection)
          current_subsubsubsubsubsection = None
      elif text.startswith("###### ") and current_subsubsubsubsection is not None:  # 階層6の要素
          current_subsubsubsubsubsection = {"title": text, "content": []}
          current_subsubsubsubsection["subsubsubsubsubsections"].append(current_subsubsubsubsubsection)

  # 最後のセクションを追加（もし存在する場合）
  if current_section is not None:
      result.append(current_section)

  # 結果を表示（デバッグ用）
  output_list = []

  # 第1階層目
  joined_text_list = []
  joined_text=""

  for section in result:
    joined_text_list.append(section["title"])
    if current_subsection:
      for subsection in section["subsections"]:
          joined_text_list.append(subsection["title"])
    joined_text = "\n".join(joined_text_list)
    output_list.append(joined_text)
    joined_text_list = []

  # 第2階層目
  joined_text_list = []
  joined_text=""

  for subsection in section["subsections"]:
    joined_text_list.append(subsection['title'])
    if current_subsubsection:
      for subsubsection in subsection["subsubsections"]:
          joined_text_list.append(subsubsection["title"])
          joined_text = "\n".join(joined_text_list)
    output_list.append(joined_text)
    joined_text_list = []

  # 第3階層目
  joined_text_list = []
  joined_text=""

  if current_subsection:
    for subsubsection in subsection["subsubsections"]:
      joined_text_list.append(subsubsection['title'])
      if current_subsubsubsection:
        for subsubsubsection in subsubsection["subsubsubsections"]:
          joined_text_list.append(subsubsubsection["title"])
          joined_text = "\n".join(joined_text_list)
      output_list.append(joined_text)
      joined_text_list = []

  # 第4階層目
  joined_text_list = []
  joined_text=""

  if current_subsubsubsection:
    for subsubsubsection in subsubsection["subsubsubsections"]:
      joined_text_list.append(subsubsubsection['title'])
      if current_subsubsubsubsection:
        for subsubsubsubsection in subsubsubsection["subsubsubsubsections"]:
          joined_text_list.append(subsubsubsubsection["title"])
          joined_text = "\n".join(joined_text_list)
      output_list.append(joined_text)
      joined_text_list = []

  # リスト内の空要素を削除
  output_list_cleaned = [a for a in output_list if a != ""]
  return output_list_cleaned

def sentence_generating(llm,group,translated_theme,summarized_list,openai_api_key,style):
    combined_list = []
    # simplified_list = []
    # summarized_list = []

    if summarized_list:
        last_answer = st.session_state.messages[-1].content
    else:
        last_answer = ""
    
    if last_answer:
        last_answer_summarized = summarize(last_answer,openai_api_key,style)
    else:
        last_answer_summarized = ""

    if style == "formal":
        sentence = f"""
### Instructions:
{translated_theme}
Act as an introspective person who excels at looking deep into his or her own mind.
Make bulleted items into creative sentences by supplementing them with conjunctions.

### Condition:
Please write in Japanese.
Any additional explanations should be enclosed in parentheses.
Please unify the writing style in the Japanese standard form, like as "〜だ" or "〜である".

### Input:
（子どもだけじゃなく大人の教育も必要だと思う）
## 目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたい
### 宿題が多すぎて課題をこなすだけになっているのが嫌
### 未来の社会を発展させるような意味ある勉強がしたい

### Output:
とにかく宿題が多すぎて、考える余裕がなくて課題をこなすだけになっているのが嫌だ。（答えのある問題をただ強制的に解答させられるのは無駄だと思う。インターネットやChatGPTなどが急速に発展しているので、そういった単なる暗記や論理計算は、そのうち人間がやる必要はなくなると思う。それなのに、このまま偏差値至上主義の詰め込み教育で今後もやっていくならば、何の役にも立たない大人を育てることになるだろう。）そうではなくて、もっと未来の社会を発展させるような意味ある勉強に集中できたらどんなに良いだろうか。（答えのない問いに試行錯誤しながら立ち向かったり、自分だけの特別な興味関心を育てて専門性を高めたりする勉強の方が今後求められるのは明らかだ。大人も自分の興味関心とスキルを結びつけることが仕事で求められるのだ。）つまり、目先の宿題を消化するだけの作業は懲り懲りで、将来の社会のことを考えてもっと有意義な学びをしたいということ。

### Input:
（{last_answer_summarized}）
"""
    else:
        sentence = f"""
### Instructions:
{translated_theme}
Act as an introspective person who excels at looking deep into his or her own mind.
Make bulleted items into creative sentences by supplementing them with conjunctions.

### Condition:
Please write in Japanese.
Any additional explanations should be enclosed in parentheses.
Please write in a simple, easy-to-understand manner for junior high school students.

### Input:
# 未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要だ
## 子どもだけじゃなく大人の教育も必要だと思う
## 目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたい
### 宿題が多すぎて課題をこなすだけになっているのが嫌
### 未来の社会を発展させるような意味ある勉強がしたい

### Output:
宿題が多すぎて課題をこなすだけになっているのが嫌だ。（答えがわかってる問題をとけって言われるの、マジないと思う。ネットとかChatGPTとか今の時代めっちゃあるし。それ使えば一瞬だから。そんなことわざわざ人間がやる必要なくね？このまま「偏差値」ばっか言って詰め込みまくったら、将来役に立たない大人になるでしょ。）
だから、もっと未来の社会を発展させるような意味ある勉強がしたい。（答えが決まってない問題をめっちゃ考えたり、自分だけ興味がある分野を掘り下げたりした方がめっちゃ楽しいと思うし、それがこれからは大切。）目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたいということ。そしてそのためには、子どもだけじゃなく大人の教育も必要だと思う。
（ていうか、先生自体が古い詰め込みキョーイクされてきてるんだから、その意識を変えないとダメでしょ。先生が自分でやってないのに、生徒に探究学習をしろとか言っても響かないし。うん。うちの親もそうじゃん。大学試験の面接に必要だからって塾のやってる企業インターンのチラシ持ってくるけど、自分は仕事で何かスキルアップしようとしてんのかな。探究っていつもの生活の中で見つけていくものじゃん。家で過ごすときにそういった環境にないと、いくら学校とか塾でやってもおんなじじゃん。先生も親も自分たちで探究学習ってのをやれば、仕事もプライベートもノリに乗っていい感じになるんじゃないの？それが分かってからじゃないと、子どもたちにも教えることができないと思うけど…。）何が言いたかったかっていうと、つまり、未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要なんだってこと。

### Input:
（{last_answer_summarized}）
"""

    st.session_state.messages.append(SystemMessage(content=sentence))
    st.session_state.messages.append(HumanMessage(content=group))

    # #のマークダウンを削除したBDAをリストにして画面上へ表示
    simplyfied_group = re.sub("\#+ ","",group)
    simplyfied_group_list = simplyfied_group.split()
    simplyfied_group_list[0] = "**" + str(simplyfied_group_list[0]) + "**"
    st.table(simplyfied_group_list)

    last_3_messages = st.session_state.messages[-3:]
    ai_messages = []
    for message in last_3_messages:
        if isinstance(message, AIMessage):
            ai_messages.append(message)
    last_messages = ai_messages + st.session_state.messages[-2:]
    with st.spinner("KJ-GPTが文章化しています ..."):
        answer, cost = get_answer(llm, last_messages)
    combined_list.append(answer)
    st.session_state.messages.append(AIMessage(content=answer))
    st.session_state.costs.append(cost)

    # st.session_state.messages.append(SystemMessage(content=simplifying_sentence))
    # st.session_state.messages.append(HumanMessage(content=answer))
    # with st.spinner("KJ-GPTが文章化しています ..."):
    #     simplified_answer, cost = get_answer(llm, st.session_state.messages[-2:])
    # simplified_list.append(simplified_answer)
    # st.session_state.messages.append(AIMessage(content=simplified_answer))

    # st.session_state.messages.append(SystemMessage(content=summarized_sentence))
    # st.session_state.messages.append(HumanMessage(content=simplified_answer))
    # with st.spinner("KJ-GPTが文章化しています ..."):
    #     summarized_answer, cost = get_answer(llm, st.session_state.messages[-2:])
    # summarized_list.append(summarized_answer)
    # st.session_state.messages.append(AIMessage(content=summarized_answer))

    combined_sentences = "".join(combined_list)
    # simplified_sentences = "\n".join(simplified_list)
    # summarized_sentences = "\n".join(summarized_list)
    st.markdown(combined_sentences)
    if last_answer_summarized:
        return last_answer_summarized
    else:
        return answer
    # st.markdown(simplified_sentences)
    # st.markdown(summarized_sentences)
    # return combined_sentences, simplified_sentences

def sumarized_sentence_generating(llm,group,translated_theme,style):
    combined_list = []

    if style == "formal":
        sentence = f"""
### Instructions:
{translated_theme}
Act as an introspective person who excels at looking deep into his or her own mind.
Connect the following bullet points and write in a creative tone.

### Condition:
Please write in Japanese.
Please supplement the conjunctions so that the places separated by line breaks are logically connected.

### Input Example:
沖縄の暖かさや福島の自然への愛情は、自分のアイデンティティと結びついている。地域の特色を大切にすることは、文化や伝統を味わうことにも繋がる。
地域の自然とフルーツを活かしたスイーツで福島県を活性化し、地域愛や環境への配慮、食品ロス削減に貢献したいと思う。
無駄のない、地域に根付いた食文化の重要性を感じており、それを通じて持続可能な社会づくりに貢献したい。また、地域の交通インフラ整備も重要な課題であると考え、若者として積極的に議論に参加したい。
過疎化と交通問題を克服し、福島の食文化を活かした地域活性化のビジョンを持つことが、地域の人々の協力により過疎化を食い止め、より良い未来を創る鍵だと考えている。
彼女の歴史への興味を共有し、京都での座禅体験や古い町並みを通して、より深い絆を育むことが重要だと考えている。
歴史や地域文化に触れることで、私たちの生き方を見つめ直し、より人間らしさに回帰し共に成長していくことが重要だと感じている。
私は生まれ育った地域への深い愛情を持っており、地元の文化や伝統、祭りや行事への参加を通じて共同体の一員としての絆を感じる。地域愛は、私たちのアイデンティティを育み、未来へ繋げていくべき価値である。
iPadを使ったイラストの才能を活かし、原作の魅力を損なわずに感動を与える作品を創造したいと考えている。
原作の感動をアニメでどう表現するかは大きな課題だが、デジタルの利点を活かし創造的な試みにより新しい魅力を加え、原作への敬意を保ちつつ感動を再創造したい。
ウクライナとロシアの戦争解決には、国際社会全体の平和願望を重んじ、実行可能な対話と解決策を模索する必要がある。
未来の社会に貢献する意義のある学びへの渇望は、単なる感情ではなく、内面からの強い欲求だ。現教育システムへの疑問と宿題の多さへの不満は、教育の本質を見直す必要性を示唆している。
子どもも大人も、年齢に関わらず、常に新しいことを学び続けることが必要だ。
障碍者も含めたすべての人々が、社会において尊重され、自らの能力を発揮できる環境を作ることの重要性を認識し、共生社会を目指すべきである。
共生社会においては、障害者を含め全ての人が、自己の能力を発揮し尊重される環境が求められる。それには、年齢に関係なく意義ある学びが必要であり、それが共生社会の構築に不可欠である。
スポーツのプロとアマは、技術力だけでなく、経済的報酬、精神力、専門知識の面でも異なる。ユニフォームのデザインにもその差が現れ、プロは洗練されたものを用い、アマは実用性やコストを重視する傾向にある。

### Output Example:
沖縄の暖かさや福島の自然への愛情は、自分のアイデンティティと深く結びついており、地域の特色を大切にすることは文化や伝統を味わうことへと繋がる。そのため、地域の自然とフルーツを活かしたスイーツで福島県を活性化することや、地域愛や環境への配慮、食品ロス削減などに興味がある。無駄のない、地域に根付いた食文化の重要性を感じ、持続可能な社会を実現したいと考えており、そのためには地域の交通インフラ整備も重要な課題である。若者として、これらの議論に積極的に参加したいと思う。\n
このように、過疎化と交通問題を克服し、福島の食文化を活かした地域活性化のビジョンを持つことは、地域の人々と協力し過疎化を食い止め、より良い未来を創る鍵だと確信している。また、彼女の歴史への興味を共有し、京都での座禅体験や古い町並みを通じて、より深い絆を育むことも重要だと考えている。歴史や地域文化に触れることは、私たちの生き方を見つめ直し、より人間らしさに回帰し共に成長する機会を提供する。私自身、生まれ育った地域への深い愛情を持ち、地元の文化や伝統、祭りや行事への参加を通じて共同体の一員としての絆を感じている。この地域愛は、私たちのアイデンティティを育み、未来へ繋げていくべき価値だと理解している。\n
またiPadを使ったイラストの才能を活かし、原作の魅力を損なわずに感動を与える作品を創造したいとの思いを持っている。原作の感動をアニメでどう表現するかは大きな課題であるが、デジタルの利点を活かし創造的な試みにより新しい魅力を加え、原作への敬意を保ちつつ感動を再創造したい。それが何らかの形で地域愛の育成につながれば嬉しい。\n
国際社会に目を向ければ、ウクライナとロシアの戦争解決は、全体の平和願望を重んじ、実行可能な対話と解決策を模索する必要がある。未来の社会に貢献する意義のある学びへの渇望は、単なる感情ではなく、内面からの強い欲求であり、現教育システムへの疑問と宿題の多さへの不満は、教育の本質を見直す必要性を示唆している。子どもも大人も、年齢に関わらず、常に新しいことを学び続けることが必要であり、障害者も含めたすべての人々が社会において尊重され、能力を発揮できる環境を作ることの重要性を認識している。そうした社会を実現することで、平和な世界を取り戻すことができると考えている。\n
共生社会においては、障害者を含め全ての人が、自己の能力を発揮し尊重される環境が求められる。それには、年齢に関係なく意義ある学びが必要であり、それが共生社会の構築に不可欠である。スポーツの世界でも、プロとアマの違いは技術力だけではなく、経済的報酬、精神力、専門知識の面でも異なり、ユニフォームのデザインにもその差が現れている。プロは洗練されたものを用い、アマは実用性やコストを重視する傾向にある。\n
これらの視点から、私たちは生活の様々な側面において、共生社会を目指し、独自のアイデンティティを育み、地域愛を大事にしながら、持続可能な未来を創造するために努力していく必要がある。

### Input:
"""
    else:
        sentence = f"""
### Instructions:
{translated_theme}
Act as an introspective person who excels at looking deep into his or her own mind.
Connect the following bullet points and write in a creative tone.

### Condition:
Please write in Japanese.
Please supplement the conjunctions so that the places separated by line breaks are logically connected.
Please write in a simple, easy-to-understand manner for junior high school students.

### Input Example:
沖縄と福島のそれぞれの魅力に心惹かれ、地元の特徴を活かしたスイーツにロマンを感じる。
福島の自然とフルーツを活かしたご当地スイーツで地域を盛り上げ、新鮮で独特な味わいを共有し、訪れる人を増やしたい。食品ロス削減の方法も考え、地元の良さを広めたいと思っています。
地域の食材を活用し食品ロスを減らすことで、福島の魅力を伝えたい。また、交通手段が増えれば、地元をもっと楽しむことができる。
過疎化と交通問題を解決し、食を通じて福島の魅力を伝え、地域活性化を図りたい。
彼女が歴史を愛するゆえに、京都での座禅体験や古い町並みを歩くことを楽しみにしている。
沖縄の自然や文化を大切にしながら、歴史への愛と地域活性化への情熱を持って、愛する人との絆を深めたいと考えている。
iPadでイラストを描くことの自由さを愛し、心を動かす漫画に深い思い入れを持つ。原作の魅力を大切にし、アニメ化する際もその精神を維持すべきだと考える。
アニメ化する際には原作の魅力を尊重しつつ、デジタルの利点を活かした新しい表現を取り入れたい。
平和を望む心は共通しているが、それを実現するにはただ願うだけでなく、具体的な行動が必要だ。それが未来のための意味ある学びに繋がる。
宿題に追われる現状に反発し、自分の内面を見つめながら未来に貢献する学びを求めている。
学びは一生涯にわたる大切なもので、子どもだけでなく大人にも必要だ。生きる力や将来に役立つ知識を持つことが、本質的な教育の目的だと感じる。
障碍者も社会で活躍し、平等な報酬を得る権利がある。全員が自身の可能性を最大限に発揮し、教育がその鍵であると考える。

### Output Example:
沖縄と福島が好きで、地元の名産を活かしたスイーツになんとなくロマンを感じるんだよね。だから、福島の自然とフルーツをつかったご当地スイーツで地域をもりあげたい。フレッシュでユニークな味わいで、観光客も地元の人もよろこばせたい。地元で獲れたものを地元で食べて、食品ロスを減らすのが大事なんだ。その時に交通手段が増えれば、住んでいる町の人ももっと楽しむことができる。このように、過疎と交通の問題をなんとかしながら、食べもので福島の魅力を伝えて、地域をにぎやかにできればいいな。\n
それに、歴史が好きな彼女と一緒に、京都で座禅体験をしたり、古い町並みを歩いたりするのが楽しみ。こんなこと言うの恥ずかしいけど、歴史と地域の情熱をいかして、彼女のことをもっと知りたいんだよね。\n
あと、iPadでイラストを描くことが好きで、心を動かす漫画に今ハマっている。アニメ化する時も原作の良さをちゃんと表現した方がいいと思う。その上で、デジタルの利点を活かした新しい表現を取り入れるべきなんじゃないかな。\n
感動する作品はときに平和な世界につながるんじゃないかな。そういった戦争のない世界を望む心はみんなと同じだけど、大切なのはただ願うだけじゃなくて具体的な行動をすること。それが未来のための意味ある学びにつながっていく。でも、今の宿題に追われてる状況には正直うんざりしていて、もっと自分の内面を見つめながら将来につながる勉強がしたい。学びは一生続く大切なもので、子どもだけでなく大人にも必要だもの。生きる力や将来に役立つ知識が、本質的な教育の目的だと思うね。\n
そうすることで、障碍者も活躍できて平等な報酬を得るような、しあわせな社会が実現すると思う。みんなが自分の可能性を最大限に発揮するためには、教育がその土台になると思うんだ。

### Input:
"""

    st.session_state.messages.append(SystemMessage(content=sentence))
    st.session_state.messages.append(HumanMessage(content=group))
    with st.spinner("KJ-GPTが文章化しています ..."):
        answer, cost = get_answer(llm, st.session_state.messages[-2:])
    combined_list.append(answer)
    st.session_state.messages.append(AIMessage(content=answer))
    st.session_state.costs.append(cost)

    combined_sentences = "".join(combined_list)
    return combined_sentences

def related_sentence_generating(llm,contexts,wiki_text,translated_theme,wiki_extract):

    sentence = f"""
### Instructions:
{translated_theme}
You are a poet par excellence.
Write an abstract poem, referring to the Wikipedia content and summary text.

### Condition:
Please write in Japanese.
Explain the poem in a clear and simple way.
Please include plenty of line breaks in your poem to make it easier to read.

### Summary:
{contexts}

### Wikipedia:
{wiki_text}
"""

    # print("prompt:", sentence)

    st.session_state.messages.append(SystemMessage(content=sentence))
    with st.spinner("検索中 ..."):
        answer, cost = get_answer(llm, st.session_state.messages[-1:])
    st.session_state.messages.append(AIMessage(content=answer))
    st.session_state.costs.append(cost)
    return answer

def related_gal_sentence_generating(llm,contexts,wiki_text,translated_theme,wiki_extract):

    sentence = f"""
### Instructions:
{translated_theme}
You are a Japanese gal attending a vocational school.
Read the text summarising the questioner's concerns and advise on the next steps to take, based on information from Wikipedia.

### Condition:
Please write in Japanese.
Use the same style of writing as in the examples.
Please include three or four of the following emojis in the text in any combination of two of the following emojis.
🙇🏻 💦 ❕　🙏🏻 👏🏻 💕 ✨ 🤍 🏹 👼🏻 💗 🌷 👀 💕 🚶🏻 💨 🤦🏻‍♀️ 💞 🥺 🤭 💡 💖 🙈 💦 😽 ✌🏻 🏃🏻 ➰ 😿 🌀 ❤︎ ‼️ 👍🏻 🕺🏻 ✨ 👩🏻‍❤️‍👩🏻 💞

### Example:
相談者さんへ🌷
質問してくれてありがと🥺
めっちゃいいこと考えててリスペクト🫶🏻
返信おくれたけどゆるしてネ
ウィキペディアは初ナースちゃんだよ🖤
はずかしくて家で一瞬着ただけ(＾＾)
相談者さんはハロウィンどっかいった？？

### Summary:
{contexts}

### Wikipedia:
{wiki_extract}
{wiki_text}
"""
    # print("prompt:", sentence)

    st.session_state.messages.append(SystemMessage(content=sentence))
    # st.session_state.messages.append(HumanMessage(content=group))
    with st.spinner("返信を待っています ..."):
        answer, cost = get_answer(llm, st.session_state.messages[-1:])
    st.session_state.messages.append(AIMessage(content=answer))
    st.session_state.costs.append(cost)
    return answer

@st.cache_resource
def load_wiki():
    # wikipedia 日本語データセットのロード
    wikija_dataset = load_dataset(
        path="singletongue/wikipedia-utils",
        name="passages-c400-jawiki-20230403",
        split="train",
    )
    # faiss index のダウンロード
    dm = DownloadManager()
    index_local_path = dm.download(
        f"https://huggingface.co/datasets/hotchpotch/wikipedia-passages-jawiki-embeddings/resolve/main/faiss_indexes/passages-c400-jawiki-20230403/sup-simcse-ja-base/index_IVF2048_PQ192.faiss"
    )
    # faiss index のロード
    faiss_index = faiss.read_index(index_local_path)

    # embeddings へ変換するモデルのロード
    model = SentenceTransformer("cl-nagoya/sup-simcse-ja-base")
    model.max_seq_length = 512

    return wikija_dataset, faiss_index, model

def messages_init():
    st.session_state.messages = [
        SystemMessage(content=""),
        HumanMessage(content=""),
        AIMessage(content="")
    ]

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def button_disable():
    # next_roundボタンが押されたらボタンを無効にする
    st.session_state["disable_button"] = True

def main():
    init_page()
    init_messages()
    llm = ""
    group_list = ""
    answer = ""
    number_of_items = ""
    break_point = ""
    grouped_tables = []
    labeling_pair = []
    labeling_df = pd.DataFrame([{"表札": ""}])
    labeled_data_df = pd.DataFrame([{"中身": ""}]).reset_index()
    edited_labels_list = []
    edited_labeling_pair = [] # 修正後の表札とラベルの辞書型ペアを格納するリスト
    edited_labeling_pairs = []
    edited_labeling_pair_added_solos = [] # 修正後の表札とラベルの辞書型ペア、単独ラベルを格納するリスト

    style = select_style()

    if not OPENAI_API_KEY:
        # OpenAI API Keyの入力
        with st.form("my_api_key", clear_on_submit=True):
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            api_key_button = st.form_submit_button(label="Enter")
        
            if api_key_button and openai_api_key:
                st.session_state["openai_api_key"] = openai_api_key
    else:
        openai_api_key = OPENAI_API_KEY
        st.session_state["openai_api_key"] = openai_api_key

    st.session_state["miro_api_key"] = MIRO_API_KEY
    
    if openai_api_key:
        llm = select_model(st.session_state["openai_api_key"])

    translated_theme = None

    # ユーザーの入力を監視
    theme_container = st.container()
    with theme_container:
        with st.form(key="my_theme", clear_on_submit=False):
            user_theme = st.text_area(label="テーマ: ", key="theme_input", height=50)
            st.text("について")
            theme_button = st.form_submit_button(label="決定")
        if theme_button and user_theme:
            st.session_state["user_theme"] = user_theme
            translated_theme = theme_translate(user_theme,st.session_state["openai_api_key"])

    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    if st.session_state.stage == 0:

        tab1, tab2 = st.tabs(["初めから", "続きから"])

        with tab1:
            user_input = st.text_area(label="項目ラベル: ", key="input", height=300)

        with tab2:
            continueing = st.text_area(label="続きから: ", key="continueing", height=300)

        grouping_button = st.button(label="データをグルーピング", on_click=set_state, args=[1])
        if not continueing:
            st.session_state.user_input = user_input
        else:
            # 現在の最上位の島のデータと、それまでの階層構造のデータを分けてリスト化
            continueing_split = continueing.split("\n\n--------------------------------\n\n")
            # 階層構造が入ったリストの2番目の要素をリスト化
            continueing_list = continueing_split[1].split("\n")
            # 文字列のリストを実際の辞書のリストに変換
            result_list = [parse_dict_from_string(item) for item in continueing_list]

            # 階層構造の辞書リストで、edited_labeling_pairの値を更新
            st.session_state["edited_labeling_pair"] = result_list
            # 最上位の島ラベルをuser_inputへ
            st.session_state.user_input = continueing_split[0]
        
    if st.session_state.stage == 1:

        grouped_tables = []  # grouped_table を格納するリストを初期化

        # session_state内の以前のグループ化データをリセット
        st.session_state["grouping_table"] = []

        first_group_list = split_lines_to_list(st.session_state.user_input)

        user_input_random_order_list = random.sample(first_group_list, len(first_group_list))

        user_input_random = "\n".join(user_input_random_order_list)

        st.session_state["user_input_random"] = user_input_random

        number_of_first_items = len(first_group_list)
        labeling_pair = []
        dict = {}
        count = 1
        break_point = 7
        if number_of_first_items <= 20:
            break_point = 5
        if not number_of_items:
            number_of_items = number_of_first_items

        # ラベル集めのためのllmセッティング
        model_name_grouping = st.session_state["model_name"]
        llm_group = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.3, model_name=model_name_grouping)

        # 項目数がbreak_pointより少なくなるまでラベル集めをループする
        # lines = count_newlines(user_input)
        if translated_theme is None:
            translated_theme = theme_translate(user_theme,st.session_state["openai_api_key"])
        prompt_ptrn = prompt_grouping(number_of_items, translated_theme)
        st.session_state.messages.append(SystemMessage(content=prompt_ptrn))
        st.session_state.messages.append(HumanMessage(content=st.session_state["user_input_random"]))
        with st.spinner(f"KJ-GPTがラベルを集めています（残りラベル数{number_of_items}） ..."):
            answer, cost = get_answer(llm_group, st.session_state.messages[-2:])

            # answerのtoken数を計算
            tiktoken_encoding = tiktoken.encoding_for_model(model_name_grouping)
            encoded = tiktoken_encoding.encode(answer)
            token_count = len(encoded)

            # 直前のanswerをリスト化
            answer_list = split_lines_to_list(answer)

            # 入力と出力のグループ数を比較
            # 入力のグループ数
            input_data_len = len(first_group_list)
            # 出力のグループ数
            # "グループ:"の入った要素と空要素を削除
            filtered_answer_list = [item for item in answer_list if not re.match(r'(^|#+\s*)グループ\d+:', item) and item != '']
            output_data_len = len(filtered_answer_list)
            if input_data_len != output_data_len:
                print("最初のデータ数と出力の数が違います！")
                # 重複を削除
                unique_answer_list = list(dict.fromkeys(answer_list))
                # "- "を削除
                unique_answer_list = [item.lstrip('- ') for item in unique_answer_list]
                # 冒頭の数字を削除
                unique_answer_list = [re.sub(r'^\d+\.\s*', '', item) for item in unique_answer_list]
                # 入力と出力のどちらか一方にだけ含まれるデータを抽出
                generated_data_set = set(first_group_list) ^ set(unique_answer_list)
                generated_data_list = list(generated_data_set)
                # "グループ:"の入った要素と空要素を削除
                generated_data_list = [item for item in generated_data_list if not re.match(r'(^|#+\s*)グループ\d+:', item) and item != '']
                # generated_data_listから重複を削除
                generated_data_list = list(set(generated_data_list))
                # どちらか一方にだけ含まれるデータを処理
                for item in generated_data_list:
                    if item in first_group_list: # 入力にも含まれているデータ（ラベル集めで漏れているデータ）を判断して追加
                        unique_answer_list.append(f"単独：\n{item}")
                    elif item not in first_group_list and item in unique_answer_list: # 入力には無くて出力のみに含まれているデータ（ラベル集めで混入したデータ）を判断して削除
                        unique_answer_list.remove(item)
                
                answer = "\n".join(unique_answer_list)

            # answerのtoken数が上限（4069tokens）を超えた場合の処理
            if token_count >= 4096:

                # 直前のanswerをリスト化
                answer_list = split_lines_to_list(answer)
                # answerから「グループ：」「単独：」や空要素を削除
                cleaned_answer_list = [item for item in answer_list if not re.match(r'(^|#+\s*)グループ\d+:', item) and item != '']
                cleaned_answer_list = [item for item in cleaned_answer_list if not re.match(r'^単独：', item)]
                # 最初の元データリストからanswerでグループ分けされたものを除いたungrouped_listを作成
                ungrouped_list = set(first_group_list) - set(cleaned_answer_list)

                if ungrouped_list:
                    # ungrouped_listをstr型へ変換
                    ungrouped_data = "\n".join(ungrouped_list)

                    # グループ分けされていない残りラベルについて、再びグループ分け
                    st.session_state.messages.append(SystemMessage(content=prompt_ptrn))
                    st.session_state.messages.append(HumanMessage(content=ungrouped_data))
                    # 未グルーピングのデータ、systemプロンプトをapiに渡す
                    conti_answer, cost = get_answer(llm_group, st.session_state.messages[-2:])

                    # 最初のグループ分けとその続きのグループ分け結果を結合
                    answer += conti_answer
        
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

        set_state(2)

    if st.session_state.stage == 2:

        grouped_data_list = get_list(st.session_state.messages[-1].content)
        no_grouped_items = []  # グループ化しない単独データ を格納するリストを初期化
        for i, item in enumerate(grouped_data_list):
            if isinstance(item, str):
                no_grouped_items.append(item)
            else:
                number_of_groups = i + 1
                st.write(f"グループ{number_of_groups}")
                grouped_table = st.data_editor(item, key=f"group{number_of_groups}", num_rows="dynamic", use_container_width=True)
                grouped_tables.append(grouped_table)
                st.session_state["grouping_table"].append(grouped_table)
        st.write("単独")
        if not no_grouped_items:
            no_grouped_items.append(" ")
        no_grouped_table = st.data_editor(no_grouped_items, key="no_grouped_items", num_rows="dynamic", use_container_width=True)
        # grouped_tables.append(no_grouped_table)

        # if st.session_state["no_grouped_items"] != st.session_state["no_grouped_items"]:
        #     st.session_state["grouped_tables"] = grouped_tables
        
        # st.session_state["grouping_table"].append(no_grouped_table)

        for i, group in enumerate(grouped_tables):
            if len(group) < 2: # ラベルが1つしかない場合
                no_grouped_table.append(group[0]) # 単独のラベル群へ追加
                del grouped_tables[i] # グループのリストから削除

        st.session_state["grouped_tables"] = grouped_tables
        grouped_tables_length = len(st.session_state["grouped_tables"])
        st.session_state["no_grouped_tables"] = no_grouped_table
        # 単独のアイテムが少なくとも1つある場合は、その個数を数える
        if len(no_grouped_table) != 0 and no_grouped_table[0] != " ":
            no_grouped_tables_length = len(st.session_state.no_grouped_tables)
        else:
            no_grouped_tables_length = 0
        st.session_state["number_of_items"] = grouped_tables_length + no_grouped_tables_length

        st.button(label="表札をつくる", on_click=set_state, args=[3])

    if st.session_state.stage == 3:

        print("表札を作るボタンが押されました")

        cleaned_group_list = []
        group_list = st.session_state["grouped_tables"]
        cleaned_no_group_list = []
        no_group_list = st.session_state["no_grouped_tables"]
        number_of_items = st.session_state["number_of_items"]
        if not break_point:
            first_group_list = split_lines_to_list(st.session_state.user_input)
            number_of_first_items = len(first_group_list)
            if number_of_first_items <= 20:
                break_point = 5
            else:
                break_point = 7
        
        # 空の要素を削除
        for each_group in group_list:
            cleaned_group_a = list(filter(None, each_group))
            cleaned_group_b = [item for item in cleaned_group_a if item != "" ]
            cleaned_group_list.append(cleaned_group_b)

        for item in no_group_list:
            if item is not None and item != "":
                cleaned_no_group_list.append(item)
        st.session_state["no_grouped_tables"] = cleaned_no_group_list

        for group in cleaned_group_list:    
            if isinstance(group, list) and len(group) >= 2:

                # 各ラベルから、「数字＋.」および「アルファベット＋数字＋.」のパターンを削除
                cleaned_group_list = [re.sub(r'^[a-zA-Z]?\d+\.', '', text) for text in group]

                group_string = "\n".join(cleaned_group_list)

                # ラベル集めのプロンプト
                prompt_ptrn = prompt_labeling(number_of_items,break_point,style)

                # 表札づくりのためのllmセッティング
                llm_label = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7, model_name=st.session_state["model_name"])

                st.session_state.messages.append(SystemMessage(content=prompt_ptrn))
                st.session_state.messages.append(HumanMessage(content=group_string))
                with st.spinner("KJ-GPTが表札を考えています ..."):
                    answer, cost = get_answer(llm_label, st.session_state.messages[-2:])
                group_list.append(answer)
                dict = {answer: group}
                labeling_pair.append(dict)
                st.session_state.messages.append(AIMessage(content=answer))
                st.session_state.costs.append(cost)
            elif isinstance(group, list) and len(group) == 1:
                added_no_grouped_table = st.session_state["no_grouped_tables"]
                added_no_grouped_table.append(group[0])
                st.session_state["no_grouped_tables"] = added_no_grouped_table

        group_list = [item for item in group_list if not isinstance(item, list)]
        # print(labeling_pair)
        group_list_str = "\n".join(group_list)
        user_input = group_list_str
        # result = st.text_area(label=f"{count}回目の結果: ", key=f"result{count}", value=group_list_str, height=300)
        number_of_items = len(group_list)

        # ラベルのペアをsession_stateに格納
        st.session_state["labeling_pair"] = labeling_pair
        
        set_state(4)

    if st.session_state.stage == 4:

        st.markdown("**グループ**")
        for i, item in enumerate(st.session_state["labeling_pair"]):
            new_item = {}
            for key, value in item.items():
                edited_labeling_df = st.data_editor([key], column_config={"value": st.column_config.Column(f"表札{i+1}")}, use_container_width=True)
                edited_labels_list.append(edited_labeling_df[0])
                for value in item.values():
                    # edited_labeled_df = st.data_editor(value, column_config={"value": st.column_config.Column("ラベル",width=None)}, use_container_width=True, key=f"values{i}")
                    st.table(value) #　表示用のテーブル
                st.write("--")
                new_item[edited_labeling_df[0]] = list(value)
                edited_labeling_pair.append(new_item)
        edited_labeling_pair_added_solos = edited_labeling_pair

        st.markdown("**単体**")
        edited_labeled_solo_df = st.data_editor(st.session_state["no_grouped_tables"], column_config={"value": st.column_config.Column("ラベル")}, use_container_width=True)
        for solo_item in list(edited_labeled_solo_df):
            if solo_item is not None and solo_item != "":
                edited_labels_list.append(solo_item)
                edited_labeling_pair_added_solos.append(solo_item)

        # ボタンのdisabled状態を設定
        should_disable_button = st.session_state.get("disable_button", False)

        # 条件に応じたボタンの状態でボタンを表示
        next_round_button = st.button(label="この表札で決定", disabled=should_disable_button, on_click=button_disable)

        if next_round_button:

            st.session_state.user_input = "\n".join(edited_labels_list)
            st.session_state["edited_labeling_pair"].extend(edited_labeling_pair)

            if "saved_content" not in st.session_state:
                st.session_state["saved_content"] = []
            if "wip_saved_content" not in st.session_state:
                st.session_state["wip_saved_content"] = []

            st.session_state["wip_saved_content"].extend(edited_labeling_pair_added_solos)

            # ラベルのペアリストの要素を改行で連結して一つの文字列にする
            if isinstance(st.session_state["edited_labeling_pair"], list):
                edited_labeling_pair_txt = "\n".join(str(item) for item in st.session_state["edited_labeling_pair"])
            else:
                edited_labeling_pair_txt = st.session_state["edited_labeling_pair"]
            st.session_state["saved_content"] = st.session_state.user_input + "\n\n--------------------------------\n\n" + edited_labeling_pair_txt
            wip_user_theme = st.session_state["user_theme"]

            st.button(label="このままラベル集めを続ける👉", on_click=set_state, args=[1])

            st.download_button("保存💾＆ラベル集めを続ける👉", st.session_state["saved_content"], f"{dt_now_formatted}_{wip_user_theme}", on_click=set_state, args=[1])
            
            st.download_button("完了してシンボルを作る👌", st.session_state["saved_content"], f"{dt_now_formatted}_{wip_user_theme}", on_click=set_state, args=[5])

            # next_roundボタンを再度有効にする
            st.session_state['disable_button'] = False

        # symbol_button = st.button(label="シンボルマークを作る")

        # if symbol_button:
        #     st.session_state.user_input = "\n".join(edited_labels_list)
        #     st.session_state["edited_labeling_pair"].extend(edited_labeling_pair)
        #     set_state(5)
        #     st.rerun()

    if st.session_state.stage == 5:

        edited_labeling_pairs = st.session_state["edited_labeling_pair"]
        
        # symbol_input_list = get_list(user_input)
        # print("symbol_input_list:", symbol_input_list)
        top_items = []
        symbol_sets = []
        symbol_dict = {}
        symbol_count = 0

        # 最上位の島の数をカウント
        highest_level_labels = st.session_state["user_input"]
        highest_level_labels_list = highest_level_labels.split("\n")
        symbol_list_length = len(highest_level_labels_list)

        # 最上位の島をループしてシンボルマークを作成
        for item in highest_level_labels_list:
            
            # シンボルマークの総数からループごとに1ずつ減らした数
            # symbol_reversed_count = symbol_list_length - symbol_count

            item_str = str(item)

            prompt_ptrn = prompt_labeling(0,break_point,style)

            st.session_state.messages.append(SystemMessage(content=prompt_ptrn))
            st.session_state.messages.append(HumanMessage(content=item_str))
            with st.spinner("KJ-GPTがシンボルを作成しています ..."):
                symbol_answer, cost = get_answer(llm, st.session_state.messages[-2:])
            symbol_set = symbol_answer + "：" + item_str
            symbol_title = "**" + f"({symbol_count + 1}) " +  symbol_answer + "**"
            symbol_dict["# " + item_str] = symbol_title + "\n" + "# " + item_str
            # print("symbol_set", symbol_set)
            top_items.append(item_str)
            symbol_sets.append(symbol_set)
            st.session_state.messages.append(AIMessage(content=answer))
            st.session_state.costs.append(cost)

            symbol_count += 1

        symbol_str = "\n".join(symbol_sets)

        markdown_text = ""

        # top_itemsのリストを逆順で処理
        # reversed_top_items = list(reversed(top_items))

        # reversed_top_itemsリストをループして最上位の見出しを処理します。
        for top_item in highest_level_labels_list:
            # top_text = top_item.split("：")[1].strip()
            markdown_text += add_markdown_entry(1, top_item)  # 最上位の見出しを逆順で読み込んで追加
            markdown_text += find_sub_items(top_item, 2, edited_labeling_pairs)  # サブアイテムを探して追加
        for key, value in symbol_dict.items():
            markdown_text = markdown_text.replace(key, value)

        st.session_state["markdown_text"] = markdown_text

        converted_markdown = headline_to_list(st.session_state["markdown_text"])
        st.markdown(converted_markdown)

        set_state(6)

    # 作業中の箇条書きを表示
    if st.session_state["saved_content"] and st.session_state.stage != 6:

        wip_container = st.container(border=True)
        with wip_container:
            st.write("作業中の結果")

            st.code(st.session_state["saved_content"], language="python")

            working_markdown_text = ""
            working_highest_level_labels = st.session_state["user_input"]
            working_highest_level_labels_list = working_highest_level_labels.split("\n")

            for top_item in working_highest_level_labels_list:
                # top_text = top_item.split("：")[1].strip()
                working_markdown_text += add_markdown_entry(1, top_item)  # 最上位の見出しを逆順で読み込んで追加
                working_markdown_text += find_sub_items(top_item, 2, st.session_state["wip_saved_content"])  # サブアイテムを探して追加
            # for key, value in st.session_state["saved_content"].items():
            #     working_markdown_text = working_markdown_text.replace(key, value)

            st.session_state["working_markdown_text"] = working_markdown_text

            converted_working_markdown = headline_to_list(working_markdown_text)
            st.markdown(converted_working_markdown)

    if st.session_state.stage == 6:

        # 作業中の箇条書きを削除
        converted_working_markdown = ""

        chart_container = st.container()
        with chart_container:
            with st.form(key="my_chart", clear_on_submit=False):
                if "markdown_text" not in st.session_state:
                    st.session_state["markdown_text"] = ""
                labels_all = st.text_area(label="ラベルのペア: ", key="labels_all", value=st.session_state["markdown_text"], height=300)
                miro_teamId = st.text_input("Miro チームID", type="password")
                boardId = st.text_input("Miro ボードID")
                sentence_button = st.form_submit_button(label="データを図解化")


            if sentence_button and labels_all and miro_teamId and boardId:

                parsed_markdown = parse_markdown_with_bold(labels_all)

                api_token = st.session_state["miro_api_key"]
                
                # 各要素のレベルとそのxy座標を保管するためのデータ構造
                element_positions = []

                # シェイプをボードに追加
                x_position = 0
                y_position = 0
                previous_level = -1
                previous_x_position = 0  # 前の要素のx座標
                previous_width = 0
                highest_level = 0
                level_y_positions = {0: 0, 1: 350, 2: 1100, 3: 1400, 4: 1700, 5: 1900, 6: 2100, 8: 2400}  # 各レベルの最新のy座標
                width_by_level =  {0: 800, 1: 3000, 2: 500, 3: 275, 4: 175, 5: 175, 6: 175, 8: 175}  # 各レベルごとのシェイプの幅
                height_by_level =  {0: 100, 1: 450, 2: 250, 3: 100, 4: 75, 5: 75, 6: 75, 8: 75}  # 各レベルごとのシェイプの高さ

                for level, text in parsed_markdown:

                    # シェイプの幅と高さをレベルによって決める
                    shape_width = width_by_level[level]
                    shape_height = height_by_level[level]

                    if level > highest_level:
                        highest_level = level
                    if level > previous_level:
                        # 階層が深くなった場合、レベルが1以下の時は前の要素のx座標をそのまま引き継ぎ、それ以外の時は前の要素の幅の4分の位置左にずらす
                        if level == 1:
                            x_position = previous_x_position + 1000
                        else:
                            x_position = previous_x_position - ( previous_width / 4 )
                        # y座標を更新        
                        y_position = level_y_positions[level]

                    elif level < previous_level:
                        if level != 0:
                            # より浅い階層に移動したら、前の要素のx座標からシェイプの幅分プラス240動かす
                            x_position = previous_x_position + shape_width + 240
                        else:
                            # レベル0へ移動したら、前の要素のx座標からシェイプの幅分プラス1200動かす
                            x_position = previous_x_position + shape_width + 1200
                        # y軸は垂直に移動する
                        y_position = level_y_positions[level]
                    elif level == previous_level: # 同じレベルの場合
                        # x座標のみ120増やす
                        x_position = previous_x_position + shape_width + 120

                    # シェイプを作成
                    add_shape(api_token, boardId, level, text, x_position, y_position, shape_width, shape_height)

                    # この要素の情報をデータベースに追加
                    element_positions.append({"level": level, "x": x_position, "y": y_position})

                    # x座標とy座標を更新
                    level_y_positions[level] = y_position
                    previous_x_position = x_position
                    previous_level = level
                    previous_width = shape_width


                # print("element_positions >>>>", element_positions)
                # 結果の表示
                # for element in element_positions:
                #     print(f"レベル {element['level']} の要素が位置 ({element['x']}, {element['y']}) に追加されました。")

                # レベル0ごとの四角形を計算するための変数
                start_x = 0
                max_y = 0
                max_x = 0

                # 各レベル0ごとに四角形を計算して追加
                # for element in element_positions:
                #     level = element['level']
                #     x, y = element['x'], element['y']

                #     # 最大のx座標とy座標を更新
                #     max_x = max(max_x, x)
                #     max_y = max(max_y, y)

                #     print("max_xは、", max_x, "start_xは", start_x)

                #     if level == 0 and x != start_x:
                #         # 前のレベル0から現在のレベル0までの範囲の四角形を計算
                #         width = max_x - start_x - 700
                #         height = max_y + 360
                #         frame_x = start_x + ( width - 1700 ) / 2 
                #         frame_y = ( height - 360 ) / 2

                #         # 四角形を追加
                #         add_rounded_rectangle(api_token, boardId, frame_x, frame_y, width, height)

                #         # 新しいレベル0の開始点を設定
                #         start_x = x
                #         max_x = x
                #         max_y = 0

                # # 最後の範囲の四角形を計算して追加
                # width = max_x - start_x + 1200
                # height = max_y + 360
                # frame_x = start_x + ( width - 1700 ) / 2 
                # frame_y = ( height - 360 ) / 2
                # add_rounded_rectangle(api_token, boardId, frame_x, frame_y, width, height)


                # レベル1ごとの四角形を作成
                # 1以上の要素のみ抽出
                level_1_or_higher = [item for item in parsed_markdown if item[0] >= 1]

                # levelの値が1以上の要素を抽出
                elements_level_1_or_higher = [item for item in element_positions if item['level'] >= 1]

                # リストをタプルの形に変換
                tuple_list_positions_1_or_higher = [(item['level'], item['x'], item['y']) for item in elements_level_1_or_higher]

                if highest_level == 2:
                    # 2が出現する最後の項目までのリストを生成
                    last_two_index = None
                    for i, (level, _, _) in enumerate(tuple_list_positions_1_or_higher):
                        if level == 2:
                            last_two_index = i

                    # last_two_indexがNoneでなければ、最後の2までのリストをスライスする
                    if last_two_index is not None:
                        result_list = tuple_list_positions_1_or_higher[:last_two_index + 1]
                    else:
                        result_list = tuple_list_positions_1_or_higher  # 2がない場合は元のリストをそのまま使用

                    tuple_list_positions_1_or_higher = result_list

                # レベル1ごとの四角形を計算するための変数
                level1_start_x = 0
                level1_max_x = 0
                level1_max_y = 0

                # 各レベル1ごとに四角形を計算して追加
                for level, x, y in tuple_list_positions_1_or_higher:
                    # 最大のx座標とy座標を更新
                    if level >= 1:
                        level1_max_x = max(level1_max_x, x)
                        level1_max_y = max(level1_max_y, y)

                    if level == 1 and x != level1_start_x:
                        # レベル1の区間の四角形を計算
                        width = level1_max_x - level1_start_x - 240
                        height = level1_max_y - 650 + 960  # y座標の基準点は650
                        frame_x = level1_start_x + ( width - 3600 ) / 2
                        frame_y = 650 + ( height - 600 ) / 2  # y座標の基準点を加算

                        # 子要素がある場合のみ四角形を追加（四角形の高さが450の場合は追加しない）
                        if height != 450:
                            # 四角形を追加
                            add_rounded_rectangle(api_token, boardId, frame_x, frame_y, width, height, level)

                        # 新しいレベル1の開始点を設定
                        level1_start_x = x
                        level1_max_x = x
                        level1_max_y = 960  # y座標の基準点をリセット

                # 最後の範囲の四角形を計算して追加（レベル1が最後に現れた場合）
                if level1_max_x > level1_start_x:
                    width = level1_max_x - level1_start_x + 2100
                    height = level1_max_y - 650 + 960  # y座標の基準点は650
                    frame_x = level1_start_x + ( width - 3200 ) / 2
                    frame_y = 650 + ( height - 600 ) / 2  # y座標の基準点を加算
                    add_rounded_rectangle(api_token, boardId, frame_x, frame_y, width, height, 1)


                # # レベル2ごとの四角形を作成
                # # 2以上の要素のみ抽出
                # level_2_or_higher = [item for item in parsed_markdown if item[0] >= 2]

                # # levelの値が2以上の要素を抽出
                # elements_level_2_or_higher = [item for item in element_positions if item['level'] >= 2]

                # # リストをタプルの形に変換
                # tuple_list_positions_2_or_higher = [(item['level'], item['x'], item['y']) for item in elements_level_2_or_higher]

                # if highest_level == 3:
                #     # 3が出現する最後の項目までのリストを生成
                #     last_three_index = None
                #     for i, (level, _, _) in enumerate(tuple_list_positions_2_or_higher):
                #         if level == 3:
                #             last_three_index = i

                #     # last_three_indexがNoneでなければ、最後の3までのリストをスライスする
                #     if last_three_index is not None:
                #         result_list = tuple_list_positions_2_or_higher[:last_three_index + 1]
                #     else:
                #         result_list = tuple_list_positions_2_or_higher  # 3がない場合は元のリストをそのまま使用

                #     tuple_list_positions_2_or_higher = result_list

                # # レベル2ごとの四角形を計算するための変数
                # level2_start_x = 0
                # level2_max_x = 0
                # level2_max_y = 0

                # # 各レベル2ごとに四角形を計算して追加
                # for level, x, y in tuple_list_positions_2_or_higher:
                #     # 最大のx座標とy座標を更新
                #     if level >= 2:
                #         level2_max_x = max(level2_max_x, x)
                #         level2_max_y = max(level2_max_y, y)

                #     if level == 2 and x != level2_start_x:
                #         # レベル2の区間の四角形を計算
                #         width = level2_max_x - level2_start_x - 120
                #         height = level2_max_y - 360 + 120  # y座標の基準点は360
                #         frame_x = level2_start_x + ( width - 120 ) / 2
                #         frame_y = 360 + ( height - 120 ) / 2  # y座標の基準点を加算

                #         # 子要素がある場合のみ四角形を追加（四角形の高さが120の場合は追加しない）
                #         if height != 120:
                #             # 四角形を追加
                #             add_rounded_rectangle(api_token, boardId, frame_x, frame_y, width, height)

                #         # 新しいレベル2の開始点を設定
                #         level2_start_x = x
                #         level2_max_x = x
                #         level2_max_y = 360  # y座標の基準点をリセット

                # # 最後の範囲の四角形を計算して追加（レベル2が最後に現れた場合）
                # if level2_max_x > level2_start_x:
                #     width = level2_max_x - level2_start_x + 120
                #     height = level2_max_y - 360 + 120  # y座標の基準点は360
                #     frame_x = level2_start_x + ( width - 120 ) / 2
                #     frame_y = 360 + ( height - 120 ) / 2  # y座標の基準点を加算
                #     add_rounded_rectangle(api_token, boardId, frame_x, frame_y, width, height)

                
                # # レベル3ごとの四角形を作成
                # # 3以上の要素のみ抽出
                # level_3_or_higher = [item for item in parsed_markdown if item[0] >= 3]

                # # levelの値が3以上の要素を抽出
                # elements_level_3_or_higher = [item for item in element_positions if item['level'] >= 3]

                # # リストをタプルの形に変換
                # tuple_list_positions_3_or_higher = [(item['level'], item['x'], item['y']) for item in elements_level_3_or_higher]

                # if highest_level == 4:
                #     # 4が出現する最後の項目までのリストを生成
                #     last_four_index = None
                #     for i, (level, _, _) in enumerate(tuple_list_positions_3_or_higher):
                #         if level == 4:
                #             last_four_index = i

                #     # last_four_indexがNoneでなければ、最後の4までのリストをスライスする
                #     if last_four_index is not None:
                #         result_list = tuple_list_positions_3_or_higher[:last_four_index + 1]
                #     else:
                #         result_list = tuple_list_positions_3_or_higher  # 4がない場合は元のリストをそのまま使用

                #     tuple_list_positions_3_or_higher = result_list

                # # レベル3ごとの四角形を計算するための変数
                # level3_start_x = 0
                # level3_max_x = 0
                # level3_max_y = 0

                # # 各レベル3ごとに四角形を計算して追加
                # for level, x, y in tuple_list_positions_3_or_higher:
                #     # 最大のx座標とy座標を更新
                #     if level >= 3:
                #         level3_max_x = max(level3_max_x, x)
                #         level3_max_y = max(level3_max_y, y)

                #     if level == 3 and x != level3_start_x:
                #         # レベル3の区間の四角形を計算
                #         width = level3_max_x - level3_start_x - 120
                #         height = level3_max_y - 480 + 120  # y座標の基準点は480
                #         frame_x = level3_start_x + ( width - 120 ) / 2
                #         frame_y = 480 + ( height - 120 ) / 2  # y座標の基準点を加算

                #         # 子要素がある場合のみ四角形を追加（四角形の高さが120の場合は追加しない）
                #         if height != 120:
                #             # 四角形を追加
                #             add_rounded_rectangle(api_token, boardId, frame_x, frame_y, width, height)

                #         # 新しいレベル3の開始点を設定
                #         level3_start_x = x
                #         level3_max_x = x
                #         level3_max_y = 480  # y座標の基準点をリセット

                # # 最後の範囲の四角形を計算して追加（レベル3が最後に現れた場合）
                # if level3_max_x > level3_start_x:
                #     width = level3_max_x - level3_start_x + 120
                #     height = level3_max_y - 480 + 120  # y座標の基準点は480
                #     frame_x = level3_start_x + ( width - 120 ) / 2
                #     frame_y = 480 + ( height - 120 ) / 2  # y座標の基準点を加算
                #     add_rounded_rectangle(api_token, boardId, frame_x, frame_y, width, height)


                # # レベル4ごとの四角形を作成
                # # 4以上の要素のみ抽出
                # level_4_or_higher = [item for item in parsed_markdown if item[0] >= 4]

                # # levelの値が4以上の要素を抽出
                # elements_level_4_or_higher = [item for item in element_positions if item['level'] >= 4]

                # # リストをタプルの形に変換
                # tuple_list_positions_4_or_higher = [(item['level'], item['x'], item['y']) for item in elements_level_4_or_higher]

                # if highest_level == 5:
                #     # 5が出現する最後の項目までのリストを生成
                #     last_five_index = None
                #     for i, (level, _, _) in enumerate(tuple_list_positions_4_or_higher):
                #         if level == 5:
                #             last_four_index = i

                #     # last_five_indexがNoneでなければ、最後の5までのリストをスライスする
                #     if last_five_index is not None:
                #         result_list = tuple_list_positions_4_or_higher[:last_five_index + 1]
                #     else:
                #         result_list = tuple_list_positions_4_or_higher  # 5がない場合は元のリストをそのまま使用

                #     tuple_list_positions_4_or_higher = result_list

                # # レベル4ごとの四角形を計算するための変数
                # level4_start_x = 0
                # level4_max_x = 0
                # level4_max_y = 0

                # # 各レベル4ごとに四角形を計算して追加
                # for level, x, y in tuple_list_positions_4_or_higher:
                #     # 最大のx座標とy座標を更新
                #     if level >= 4:
                #         level4_max_x = max(level4_max_x, x)
                #         level4_max_y = max(level4_max_y, y)

                #     if level == 4 and x != level4_start_x:
                #         # レベル4の区間の四角形を計算
                #         width = level4_max_x - level4_start_x - 120
                #         height = level4_max_y - 600 + 120  # y座標の基準点は600
                #         frame_x = level4_start_x + ( width - 120 ) / 2
                #         frame_y = 600 + ( height - 120 ) / 2  # y座標の基準点を加算

                #         # 子要素がある場合のみ四角形を追加（四角形の高さが120の場合は追加しない）
                #         if height != 120:
                #             # 四角形を追加
                #             add_rounded_rectangle(api_token, boardId, frame_x, frame_y, width, height)

                #         # 新しいレベル4の開始点を設定
                #         level4_start_x = x
                #         level4_max_x = x
                #         level4_max_y = 600  # y座標の基準点をリセット

                # # 最後の範囲の四角形を計算して追加（レベル4が最後に現れた場合）
                # if level4_max_x > level4_start_x:
                #     width = level4_max_x - level4_start_x + 120
                #     height = level4_max_y - 600 + 120  # y座標の基準点は600
                #     frame_x = level4_start_x + ( width - 120 ) / 2
                #     frame_y = 600 + ( height - 120 ) / 2  # y座標の基準点を加算
                #     add_rounded_rectangle(api_token, boardId, frame_x, frame_y, width, height)


                # # レベル5ごとの四角形を作成
                # # 5以上の要素のみ抽出
                # level_5_or_higher = [item for item in parsed_markdown if item[0] >= 5]

                # # levelの値が5以上の要素を抽出
                # elements_level_5_or_higher = [item for item in element_positions if item['level'] >= 5]

                # # リストをタプルの形に変換
                # tuple_list_positions_5_or_higher = [(item['level'], item['x'], item['y']) for item in elements_level_5_or_higher]
                # print(">>>> tuple_list_positions_5_or_higherは、", tuple_list_positions_5_or_higher)

                # if highest_level == 6:
                #     # 6が出現する最後の項目までのリストを生成
                #     last_six_index = None
                #     for i, (level, _, _) in enumerate(tuple_list_positions_5_or_higher):
                #         if level == 6:
                #             last_six_index = i

                #     # last_six_indexがNoneでなければ、最後の6までのリストをスライスする
                #     if last_six_index is not None:
                #         result_list = tuple_list_positions_5_or_higher[:last_six_index + 1]
                #     else:
                #         result_list = tuple_list_positions_5_or_higher  # 6がない場合は元のリストをそのまま使用

                #     tuple_list_positions_5_or_higher = result_list

                # # レベル5ごとの四角形を計算するための変数
                # level5_start_x = 0
                # level5_max_x = 0
                # level5_max_y = 0

                # # 各レベル5ごとに四角形を計算して追加
                # for level, x, y in tuple_list_positions_5_or_higher:
                #     # 最大のx座標とy座標を更新
                #     if level >= 5:
                #         level5_max_x = max(level5_max_x, x)
                #         level5_max_y = max(level5_max_y, y)
                #         print(">>>> 現在のlevelは、", level, " 更新されたx座標の最大値は、", level5_max_x, " 更新されたy座標の最大値は、", level5_max_y)

                #     if level == 5 and x != level5_start_x:
                #         # レベル5の区間の四角形を計算
                #         width = level5_max_x - level5_start_x - 120
                #         height = level5_max_y - 720 + 120  # y座標の基準点は720
                #         frame_x = level5_start_x + ( width - 120 ) / 2
                #         frame_y = 720 + ( height - 120 ) / 2  # y座標の基準点を加算

                #         # 子要素がある場合のみ四角形を追加（四角形の高さが120の場合は追加しない）
                #         if height != 120:
                #             # 四角形を追加
                #             add_rounded_rectangle(api_token, boardId, frame_x, frame_y, width, height)

                #         # 新しいレベル5の開始点を設定
                #         level5_start_x = x
                #         print(">>>> 新しいlevel5の開始点は、", level5_start_x)
                #         level5_max_x = x
                #         print(">>>> 現在のlevel5のx軸の最大値は、", level5_max_x)
                #         level5_max_y = 720  # y座標の基準点をリセット

                # # 最後の範囲の四角形を計算して追加（レベル5が最後に現れた場合）
                # if level5_max_x > level5_start_x:
                #     width = level5_max_x - level5_start_x + 120
                #     height = level5_max_y - 720 + 120  # y座標の基準点は720
                #     frame_x = level5_start_x + ( width - 120 ) / 2
                #     frame_y = 720 + ( height - 120 ) / 2  # y座標の基準点を加算
                #     add_rounded_rectangle(api_token, boardId, frame_x, frame_y, width, height)

                markdown_text = st.session_state["markdown_text"]
                # マークダウンテキストをネストした箇条書き形式に変換。
                converted_markdown = headline_to_list(markdown_text)

                st.subheader("分類結果：")

                # リスト形式のマークダウンテキストを出力。
                st.markdown(converted_markdown)


        sentence_container = st.container()
        with sentence_container:
            with st.form(key="my_sentence", clear_on_submit=False):
                if "markdown_text" not in st.session_state:
                    st.session_state["markdown_text"] = ""
                layered_data = st.text_area(label="階層化データ: ", key="layered_data", value=st.session_state["markdown_text"], height=300)
                sentence_button = st.form_submit_button(label="データを文章化")
            if sentence_button and layered_data:

                # テキストエリア"階層化データ"のデータでsession_stateの"markdown_text"を更新
                st.session_state["markdown_text"] = layered_data

                # session_stateの"markdown_text"をリセット
                st.session_state.messages = []

                # 変数answerをリセット
                answer = ""
                # 変数labelをリセット
                label = ""
                sentence = f"""
        ### Instructions:
        {translated_theme}
        You are a philosopher who excels at introspection. Please logically connect each of the following bulleted items into a sentence.
        Please add philosophical criticism along the way.

        ### Condition:
        Please write in Japanese.
        Any additional explanations should be enclosed in parentheses.
        The bullet points to be entered can be summarized as follows; {label}

        ### Input:
        ### 宿題が多すぎて課題をこなすだけになっているのが嫌
        ### 未来の社会を発展させるような意味ある勉強がしたい

        ### Output:
        宿題が多すぎて課題をこなすだけになっているのが嫌。（答えのある問題をただ強制的に解答させられるのは無駄だと思う。インターネットやChatGPTなどが急速に発展しているので、そういった単なる暗記や論理計算は、そのうち人間がやる必要はなくなると思う。それなのに、このまま偏差値至上主義の詰め込み教育で今後もやっていくならば、何の役にも立たない大人を育てることになるだろう。）
        そうではなくて、もっと未来の社会を発展させるような意味ある勉強がしたい。（答えのない問いに試行錯誤しながら立ち向かったり、自分だけの特別な興味関心を育てて専門性を高めたりする勉強の方が今後求められるのは明らかだ。）

        ### Input:
        """
                
                simplified_answer = ""
                simplifying_sentence = f"""
        ### Instruction:
        {translated_theme}
        Please summarise the text in 1~2 lines, including the bracketed parts.

        ### Condition:
        Please write in Japanese.
        Please make sure that the text is chewed up in a way that high school students can understand.
        Lines beginning with bracketed numbers should be transcribed in their original position and with no change in content.
        Please add a logical connection to the text below.
        {simplified_answer}

        ### Input:
        (1) 常識はずれの大切な行動

        宿題が多すぎて課題をこなすだけになっているのが嫌。（答えのある問題をただ強制的に解答させられるのは無駄だと思う。インターネットやChatGPTなどが急速に発展しているので、そういった単なる暗記や論理計算は、そのうち人間がやる必要はなくなると思う。それなのに、このまま偏差値至上主義の詰め込み教育で今後もやっていくならば、何の役にも立たない大人を育てることになるだろう。）
        そうではなくて、もっと未来の社会を発展させるような意味ある勉強がしたい。（答えのない問いに試行錯誤しながら立ち向かったり、自分だけの特別な興味関心を育てて専門性を高めたりする勉強の方が今後求められるのは明らかだ。）つまり、目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたいということ。そしてそのためには、子どもだけじゃなく大人の教育も必要だと思う。
        （そもそも今の教師が昔ながらの詰め込み式の教育で育ったので、その意識改革が必要だ。教師自身が答えのない自分の心の底から出てきた問いを設定し、生徒と一緒にそれに取り組む姿勢を見せないと、子供達はついていかない。それだけではなく、子供の親たちも新しい学びを人生に取り入れなければならない。答えのない探究活動は従来の学習に比べて、より日常生活に深く関わるものだ。普段過ごしている中で感じる疑問や違和感などを起点にした、実体験に即した問いであるほど、今後の長い人生で取り組むに値する深いものになりやすい。なので、これまでのように親が教育を学校や塾に任せっぱなしにして、家庭で子供に無関心でいては子供の探究心が育ちにくくなる。教師と同じように、親たちも自分の問いを立ててそれを追求する営みを実際にやるべきだ。そして、その行動が子供たちを感化させ、家庭を活気づかせて、さらには職場のパフォーマンスも上げることになるのが理想だ。）このように、未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要と言える。

        ### Output:
        (1) 常識はずれの大切な行動

        ただ問題の答えを教え込む古いやり方ではなく、なぜそうなるのかを考えたり、新しいことに挑戦したりする学びが大切だって話だよ。ネットやChatGPTみたいな賢いツールがたくさんあるから、単純な暗記や計算はもう人間がわざわざやることじゃなくなるんじゃないかな。でも、学校が今のまま詰め込みで点数だけ追いかける教育を続けたら、本当に必要なスキルを身につけられない大人になってしまう。
        これからは、答えがすぐには出ないような問題にどう立ち向かうか、自分の好きなことを見つけて深く掘り下げる学びが求められるんだ。先生たちも昔のやり方から変わって、生徒と一緒に考えることが大事だし、それは親も同じ。家での学びもすごく重要で、親が自分で疑問を持って考える姿を子供に見せることが、子供の好奇心を育てるんだ。
        つまり、学校や塾だけじゃなくて、家でも親が子供と一緒に新しいことにチャレンジしたり、考えたりすることが、子供の成長にとってはめちゃくちゃ大事ってわけ。そうすると、家の中ももっと楽しくなって、親の仕事のやる気にもつながるんだ。

        ### Input:
        """
                
                summarized_answer = ""
                summarized_sentence = f"""
        ### Instruction:
        {translated_theme}
        Briefly summarize the following sentences in one sentence.

        ### Condition:
        Please write in Japanese.
        Please make sure that the text is chewed up in a way that high school students can understand.
        Please add a logical connection and a conjunction to the the text below.
        {summarized_answer}

        ### Input:
        ただ問題の答えを教え込む古いやり方ではなく、なぜそうなるのかを考えたり、新しいことに挑戦したりする学びが大切だって話だよ。ネットやChatGPTみたいな賢いツールがたくさんあるから、単純な暗記や計算はもう人間がわざわざやることじゃなくなるんじゃないかな。でも、学校が今のまま詰め込みで点数だけ追いかける教育を続けたら、本当に必要なスキルを身につけられない大人になってしまうよ。
        これからは、答えがすぐには出ないような問題にどう立ち向かうか、自分の好きなことを見つけて深く掘り下げる学びが求められるんだ。先生たちも昔のやり方から変わって、生徒と一緒に考えることが大事だし、それは親も同じ。家での学びもすごく重要で、親が自分で疑問を持って考える姿を子供に見せることが、子供の好奇心を育てるんだ。
        つまり、学校や塾だけじゃなくて、家でも親が子供と一緒に新しいことにチャレンジしたり、考えたりすることが、子供の成長にとってはめちゃくちゃ大事ってわけ。そうすると、家の中ももっと楽しくなって、親の仕事のやる気にもつながるよ。

        ### Output:
        いまの時代は、単純な暗記よりも考える力や新しいことに挑戦する力を育てる学びが大切なんだ。そして、家庭でも親が子供と一緒に学ぶことができれば、子供だけじゃなくて親の成長にもつながるってわけ。

        ### Input:
        """
            
                markdown_text = st.session_state["markdown_text"]
                # マークダウンテキストをネストした箇条書き形式に変換。
                converted_markdown = headline_to_list(markdown_text)

                st.subheader("分類結果：")

                # リスト形式のマークダウンテキストを出力。
                st.markdown(converted_markdown)
                # マークダウンテキストを段落ごとに分割。
                segmented_sections = split_sections(markdown_text)
                basic_data_for_abduction = {}
                summarized_list = []
                summarized_each_list = []
                summarized_list_split = []
                summarized_all_list = []
                last_answer = ""
                last_answer_summarized = ""
                just_before_answer_summarized = ""
                
                st.subheader("文章化：")

                for section in segmented_sections:

                    # "**"で挟まれたシンボルマークを抽出する
                    symbol_pattern = r"(\*\*.*?\*\*)"
                    matches_symbol = re.findall(symbol_pattern, section)
                    matches_symbol = "\n" + matches_symbol[0]

                    # シンボルマークを除いたラベル群をlabels_onlyに入れる
                    labels_only = section.replace(matches_symbol, "")

                    # labels_onlyの項目を逆順に並び替えてlabels_only_reversedに入れる
                    # labels_only_lines = labels_only.strip().split("\n")
                    # labels_only_reversed = "\n".join(list(reversed(labels_only_lines)))

                    # シンボルマークを章立てとして出力
                    st.markdown(matches_symbol)

                    # シンボルマークを除いたラベル群をリストに格納
                    labels_only_list = text_to_lines_list(labels_only)

                    # BDA（Basic Data for Abduction）ごとに分割しリスト化
                    BDA_list = list_to_BDA(labels_only_list)
                    # リスト化したBDAを逆順に並び替え
                    BDA_list_reversed = list(reversed(BDA_list))

                    # BDAごとの文章化に使うlast_messagesをリセット
                    last_messages = []
                    # BDAごとに文章化
                    for group in BDA_list_reversed:
                        # groupが0個（シンボルマークのみ）の場合を除いて文章化を実行
                        if len(group) >= 1:
                            # sentence_generatingで文章化し、返し値の要約文をjust_before_answer_summarizedに格納
                            just_before_answer_summarized = sentence_generating(llm,group,st.session_state["translated_theme"],summarized_list,st.session_state["openai_api_key"],style)
                            summarized_list.append(just_before_answer_summarized)

                    # セクションごとのまとめ
                    summarized_text = "\n".join(summarized_list)
                    just_before_answer_summarized = ""
                    st.markdown("まとめ")
                    summarized_all = sumarized_sentence_generating(llm,summarized_text,st.session_state["translated_theme"],style)
                    st.session_state["summarized_data"] = summarized_all
                    st.markdown(summarized_all)
                    summarized_text = ""
                    # セクションのまとめを全体のまとめに格納して、セクションのまとめはリセット
                    summarized_all_list.append(summarized_all)
                    summarized_list = []

                last_answer = st.session_state.messages[-1].content
                if last_answer:
                    last_answer_summarized = summarize(last_answer,openai_api_key,style)
                else:
                    last_answer_summarized = ""
                summarized_all_list.append(last_answer_summarized)
                
                summarized_text = "\n".join(summarized_all_list)
                just_before_answer_summarized = ""
                st.markdown("**全体のまとめ**")
                summarized_all = sumarized_sentence_generating(llm,summarized_text,st.session_state["translated_theme"],style)
                # summarized_all = summarized_all.replace("\n","")
                st.session_state["summarized_data"] = summarized_all
                st.markdown(summarized_all)

        # related_container = st.container()
        # with related_container:
        #     with st.form(key="my_related", clear_on_submit=False):
        #         summarized_data = st.text_area(label="まとめの文章: ", key="summarized_data", value=st.session_state["summarized_data"], height=300)
        #         ask_button = st.form_submit_button(label="ウィキペディアからヒントを探す")
        #         ask_gal_button = st.form_submit_button(label="ギャルからヒントを得る")

        #     if ask_button and summarized_data:

        #         # まとめ文章の総文字数の3分の1ごとに分割してリスト化
        #         split_text_length = int(len(summarized_data) / 3)
        #         split_sentences = [summarized_data[x:x+split_text_length] for x in range(0,len(summarized_data),split_text_length)]
        #         # 分割結果が20文字以下の文章の場合、削除する
        #         for i, item in enumerate(split_sentences):
        #             if len(item) < 20:
        #                 del split_sentences[i]

        #         with st.spinner("検索中 ..."):
        #             # wikipediaデータ、faiss index、embedding化のためのデータをロード
        #             wikija_dataset, faiss_index, model = load_wiki()

        #         st.subheader(f"テーマ：")
        #         st.markdown(f"""#### 『{st.session_state["user_theme"]}』""")
        #         st.markdown("まとめの文章をもとに、ウィキペディアから関係のありそうなヒントを探します。")

        #         wiki_item_list = []
        #         for i, item in enumerate(split_sentences):
        #             emb = to_emb(model, item)
        #             # faiss で検索して、関連 Top-15 を取り出す
        #             TOP_K = 20
        #             scores, indexes = faiss_index.search(emb, TOP_K)
        #             # インデックス順1~20位から1つをランダムで指定
        #             selected_indexes = random.sample(range(1,20), k=1)
        #             # 残り一つは、8~20位のうちからランダムで指定
        #             # selected_indexes.insert(1,random.randint(8,20))

        #             for sel_i, idx in enumerate(selected_indexes):
        #                 if idx < TOP_K:  # 範囲を超えないようにチェック
        #                     id = indexes[0][idx]
        #                     score = scores[0][idx]
        #                     data = wikija_dataset[int(id)]
        #                     print((score, data["title"], data["text"][:100]))
        #                     wiki_title = data["title"]

        #                     # wikiの項目が重複していない場合のみ処理
        #                     if wiki_title not in wiki_item_list:
        #                         wiki_item_list.append(wiki_title)

        #                         wiki_text = ">..." + data["text"] + "..."

        #                         # 抽出したテキストの最初の1文を取り出す
        #                         first_sentence_wiki = data["text"].partition("。")[0]

        #                         url = "https://ja.wikipedia.org/api/rest_v1/page/summary/" + data["title"]

        #                         response = requests.get(url)
        #                         json_data = response.json()

        #                         if "extract" in json_data:
        #                             wiki_extract = json_data["extract"]
        #                         else:
        #                             wiki_extract = ""
        #                         # ヒントとなる詩のためのllmセッティング
        #                         llm_poet = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name=st.session_state["model_name"])
        #                         poet_answer = related_sentence_generating(llm_poet,item,data["text"],st.session_state["translated_theme"],wiki_extract)
        #                         st.markdown(f"### ・{wiki_title}")
        #                         if "thumbnail" in json_data:
        #                             thumbnail_image = json_data["thumbnail"]['source']
        #                             st.image(thumbnail_image)
        #                         if wiki_extract:
        #                             st.caption(wiki_extract)
        #                         st.markdown(poet_answer)
        #                         st.markdown(wiki_text)
        #                         st.link_button("Wikipedia", "https://ja.wikipedia.org/wiki/" + data["title"] + "#:~:text=" + first_sentence_wiki)
                                
        #                         st.markdown("\n\n")

        #                         bing_query_url = "https://www.bing.com/search?q=" + wiki_title
        #                         st.link_button(f"Webで「{wiki_title}」を検索", bing_query_url)

        #                         st.markdown("\n\n")

        #                         st.markdown("##### 📚")

        #                         url = "https://www.googleapis.com/books/v1/volumes?q=" + wiki_title + "&langRestrict=ja&orderBy=newest"

        #                         response = requests.get(url)
        #                         json_data = response.json()

        #                         col1, col2, col3 = st.columns(3)
        #                         columns = [col1, col2, col3]

        #                         if 'items' in json_data:
        #                             for i, idx in enumerate(json_data['items'][0:2]):
        #                                 book_item = json_data['items'][i]
        #                                 volume_info = book_item['volumeInfo']
        #                                 book_id = book_item['id']
        #                                 book_title = volume_info['title']
        #                                 col = columns[i]  # 各アイテムを異なるカラムに均等に割り当てる
        #                                 with col:
        #                                     st.markdown(f"""<span style="word-wrap:break-word;">{book_title}</span>""", unsafe_allow_html=True)
        #                                     if 'imageLinks' in volume_info:
        #                                         book_thumbnail = volume_info['imageLinks']['thumbnail']
        #                                         st.image(book_thumbnail)
        #                                     else:
        #                                         book_thumbnail = ""
        #                                     if 'description' in volume_info:
        #                                         book_description = volume_info['description']
        #                                         st.caption(book_description[:100])
        #                                     else:
        #                                         book_description = ""
        #                                     book_link = "https://www.google.co.jp/books/edition/_/" + book_id + "?hl=ja"
        #                                     # book_link = volume_info['previewLink']
        #                                     st.link_button("詳細", book_link)

        #                             st.markdown("\n")
        #                         else:
        #                             st.markdown("Google Booksでは見つかりませんでした。\n")

        #                         amazon_link = f"https://www.amazon.co.jp/s?k={wiki_title}&i=stripbooks"
        #                         st.link_button("Amazonで関連書籍を探す", amazon_link)
        #                         calil_link = "https://calil.jp/search?q=" + wiki_title
        #                         st.link_button("図書館（カーリル）で関連資料を探す", calil_link)

        #                         st.markdown("\n\n")

        #     if ask_gal_button and summarized_data:

        #         # まとめ文章の総文字数の3分の1ごとに分割してリスト化
        #         split_text_length = int(len(summarized_data) / 3)
        #         split_sentences = [summarized_data[x:x+split_text_length] for x in range(0,len(summarized_data),split_text_length)]
        #         # 分割結果が20文字以下の文章の場合、削除する
        #         for i, item in enumerate(split_sentences):
        #             if len(item) < 20:
        #                 del split_sentences[i]

        #         with st.spinner("インスタでアカウントを検索中 ..."):
        #             # wikipediaデータ、faiss index、embedding化のためのデータをロード
        #             wikija_dataset, faiss_index, model = load_wiki()

        #         st.subheader(f"依頼テーマ：")
        #         st.markdown(f"""#### 『{st.session_state["user_theme"]}』""")
        #         st.markdown("上のまとめをもとに、次に調査すべき対象を大学生から聞きました🤭🤍")
        #         st.caption("""
        # インターネットや図書館で調べたり、実際に現地を訪れたりするための入り口にしてみてください。
        # そこで出会ったり見聞きしたことから、更に次の調査対象に渡り歩き、最終的にあなたの問題意識を更に深めることができます。\n
        # そして、調べるなかで、感動したり、なるほどと思ったり、またはこれは違うなと感じたりしたら、忘れないうちに箇条書きでメモしてみてください。\
        # それをまたこのKJ-GPTを使って分析してみると、アイデアをますますブラッシュアップすることができます。
        # """)
        #         wiki_item_list = []
        #         for i, item in enumerate(split_sentences):
        #             # container = st.container()
        #             # container.write(item)
        #             emb = to_emb(model, item)
        #             # faiss で検索して、関連 Top-15 を取り出す
        #             TOP_K = 15
        #             scores, indexes = faiss_index.search(emb, TOP_K)
        #             # インデックス順4~15位から1つをランダムで指定
        #             selected_indexes = random.sample(range(4,15), k=1)
        #             # 残り一つは、8~20位のうちからランダムで指定
        #             # selected_indexes.insert(1,random.randint(8,20))

        #             for sel_i, idx in enumerate(selected_indexes):
        #                 if idx < TOP_K:  # 範囲を超えないようにチェック
        #                     # if i  == 1 and sel_i == 1:
        #                     #     id = random.randint(1,5555583)
        #                     # if idx % 3  == 0:
        #                     #     id = random.randint(1,5555583)
        #                     # else:
        #                     #     id = indexes[0][idx]
        #                     id = indexes[0][idx]
        #                     score = scores[0][idx]
        #                     data = wikija_dataset[int(id)]
        #                     print((score, data["title"], data["text"][:100]))
        #                     wiki_title = data["title"]

        #                     # wikiの項目が重複していない場合のみ処理
        #                     if wiki_title not in wiki_item_list:
        #                         wiki_item_list.append(wiki_title)

        #                         wiki_text = ">..." + data["text"] + "..."

        #                         # 抽出したテキストの最初の1文を取り出す
        #                         first_sentence_wiki = data["text"].partition("。")[0]

        #                         url = "https://ja.wikipedia.org/api/rest_v1/page/summary/" + data["title"]

        #                         response = requests.get(url)
        #                         json_data = response.json()

        #                         if "extract" in json_data:
        #                             wiki_extract = json_data["extract"]
        #                         else:
        #                             wiki_extract = ""
        #                         gal_answer = related_gal_sentence_generating(llm,item,data["text"],st.session_state["translated_theme"],wiki_extract)
        #                         st.markdown(f"### ・{wiki_title}")
        #                         if "thumbnail" in json_data:
        #                             thumbnail_image = json_data["thumbnail"]['source']
        #                             st.image(thumbnail_image)
        #                         if wiki_extract:
        #                             st.caption(wiki_extract)
        #                         st.markdown(gal_answer)
        #                         st.markdown(wiki_text)
        #                         st.link_button("Wikipedia", "https://ja.wikipedia.org/wiki/" + data["title"] + "#:~:text=" + first_sentence_wiki)
                                
        #                         st.markdown("\n\n")

        #                         if any(adult_keyword in wiki_title for adult_keyword in adult_keywords):
        #                             print("キーワードに不適切な内容が含まれています。")
        #                         else:
        #                             bing_query_url = "https://www.bing.com/search?q=" + wiki_title
        #                             st.link_button(f"Webで「{wiki_title}」を検索", bing_query_url)

        #                             st.markdown("\n\n")

        #                             st.markdown("##### 📚")

        #                             url = "https://www.googleapis.com/books/v1/volumes?q=" + wiki_title + "&langRestrict=ja&orderBy=newest"

        #                             response = requests.get(url)
        #                             json_data = response.json()

        #                             col1, col2, col3 = st.columns(3)
        #                             columns = [col1, col2, col3]

        #                             if 'items' in json_data:
        #                                 for i, idx in enumerate(json_data['items'][0:2]):
        #                                     book_item = json_data['items'][i]
        #                                     volume_info = book_item['volumeInfo']
        #                                     book_id = book_item['id']
        #                                     book_title = volume_info['title']
        #                                     col = columns[i]  # 各アイテムを異なるカラムに均等に割り当てる
        #                                     with col:
        #                                         st.markdown(f"""<span style="word-wrap:break-word;">{book_title}</span>""", unsafe_allow_html=True)
        #                                         if 'imageLinks' in volume_info:
        #                                             book_thumbnail = volume_info['imageLinks']['thumbnail']
        #                                             st.image(book_thumbnail)
        #                                         else:
        #                                             book_thumbnail = ""
        #                                         if 'description' in volume_info:
        #                                             book_description = volume_info['description']
        #                                             st.caption(book_description[:100])
        #                                         else:
        #                                             book_description = ""
        #                                         book_link = "https://www.google.co.jp/books/edition/_/" + book_id + "?hl=ja"
        #                                         # book_link = volume_info['previewLink']
        #                                         st.link_button("詳細", book_link)

        #                                 st.markdown("\n")
        #                             else:
        #                                 st.markdown("Google Booksでは見つかりませんでした。\n")

        #                             amazon_link = f"https://www.amazon.co.jp/s?k={wiki_title}&i=stripbooks"
        #                             st.link_button("Amazonで関連書籍を探す", amazon_link)
        #                             calil_link = "https://calil.jp/search?q=" + wiki_title
        #                             st.link_button("図書館（カーリル）で関連資料を探す", calil_link)

        #                             st.markdown("\n\n")

    # コストの計算と表示
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## 料金 \n#### *現在GPT-4oは料金計算に非対応")
    st.sidebar.markdown(f"**合計金額: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == "__main__":
    main()
