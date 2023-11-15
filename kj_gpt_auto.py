import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.callbacks import get_openai_callback

import re

# theme = "The items in the input field are records of fieldwork at a rural village in Nepal."
theme = ""
prompt_ptrn = ""

# 例示データの出典
# 「何となく気になる興味関心について」
# 四注記 (1) 2023/10/26-10/31(2)自宅 (3) マンダラチャートをもとにブレストして出たアイデア(4)Toshiki Otsuka


def prompt_grouping(lines, translated_theme):
    theme = translated_theme
    grouping1 = f"""
###Instruction:
{theme}
Please group all of these items with those that are closer in deeper underlying meaning.
Each group should have at most 3 items. Be sure to group ALL items. Items that cannot be grouped should be listed as is.
Please write in Japanese.

### Input:
食品ロスをどうやって減らせばいいか気になる
歴史ある京都の町並みがとても好きだ
地域で作ったものは地元で食べたほうがいい
未来の社会を発展させるような意味ある勉強がしたい
卓球部の部活のユニフォームがダサくていやだ
イラストをiPadで描くのが得意
ウクライナとロシアの戦争はどうすれば解決するだろうか
障碍者もちゃんと給料をもらって生活できる社会にすべきだ
歴史が好きな彼女を喜ばせたい
パラリンピックで見た障碍者の姿に感動した
子どもだけじゃなく大人の教育も必要だと思う
福島県の自然を活かした町おこしに興味がある
宿題が多すぎて課題をこなすだけになっているのが嫌
感動して泣けるような漫画がすきだ
漫画をアニメ化しても、原作から全く違うのになってしまうのが気に食わない
スポーツのプロとアマの違いはなんだろう
地元のご当地スイーツがあれば食べたい
住んでいる福島県のフルーツをもっとアピールしたい
おじいちゃんの住む沖縄の陽気な雰囲気が好きだ
住んでいる地域の人口がどんどん減っていて不安
修学旅行で行く京都の座禅体験がたのしみ
友達と遊ぶときに交通手段が少ないのが悩み

###Output:
グループ1:
地元のご当地スイーツがあれば食べたい
住んでいる福島県のフルーツをもっとアピールしたい
グループ2:
宿題が多すぎて課題をこなすだけになっているのが嫌
未来の社会を発展させるような意味ある勉強がしたい
グループ3:
食品ロスをどうやって減らせばいいか気になる
地域で作ったものは地元で食べたほうがいい
グループ4:
住んでいる地域の人口がどんどん減っていて不安
友達と遊ぶときに交通手段が少ないのが悩み
グループ5:
修学旅行で行く京都の座禅体験がたのしみ
歴史ある京都の町並みがとても好きだ
グループ6:
漫画をアニメ化しても、原作から全く違うのになってしまうのが気に食わない
グループ7:
卓球部の部活のユニフォームがダサくていやだ
グループ8:
障碍者もちゃんと給料をもらって生活できる社会にすべきだ
グループ9:
感動して泣けるような漫画がすきだ
グループ10:
パラリンピックで見た障碍者の姿に感動した
グループ11:
子どもだけじゃなく大人の教育も必要だと思う
グループ12:
おじいちゃんの住む沖縄の陽気な雰囲気が好きだ
グループ13:
スポーツのプロとアマの違いはなんだろう
グループ14:
福島県の自然を活かした町おこしに興味がある
グループ15:
イラストをiPadで描くのが得意
グループ16:
歴史が好きな彼女を喜ばせたい
グループ17:
ウクライナとロシアの戦争はどうすれば解決するだろうか

###Input:
"""
    grouping2 = f"""
###Instruction:
{theme}
Please group all of these items with those that are closer in deeper underlying meaning.
Each group should have at most 2 items. Be sure to group ALL items. Items that cannot be grouped should be listed as is.
Please write in Japanese.

### Input:
福島県のスイーツを活かした、ご当地スイーツが食べたい
目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたい
無駄がなくて地域に根付いた食のつながりが大切だと思っている
人口が減っていき、人との交流がなくなっていくのが心配
街並みや座禅など歴史ある京都でしか味わえない体験が好き
漫画をアニメ化しても、原作から全く違うのになってしまうのが気に食わない
卓球部の部活のユニフォームがダサくていやだ
障碍者もちゃんと給料をもらって生活できる社会にすべきだ
感動して泣けるような漫画がすきだ
パラリンピックで見た障碍者の姿に感動した
子どもだけじゃなく大人の教育も必要だと思う
おじいちゃんの住む沖縄の陽気な雰囲気が好きだ
スポーツのプロとアマの違いはなんだろう
福島県の自然を活かした町おこしに興味がある
イラストをiPadで描くのが得意
歴史が好きな彼女を喜ばせたい
ウクライナとロシアの戦争はどうすれば解決するだろうか

###Output:
グループ1:
福島県のスイーツを活かした、ご当地スイーツが食べたい
福島県の自然を活かした町おこしに興味がある
グループ2:
目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたい
子どもだけじゃなく大人の教育も必要だと思う
グループ3:
スポーツのプロとアマの違いはなんだろう
パラリンピックで見た障碍者の姿に感動した
卓球部の部活のユニフォームがダサくていやだ
グループ4:
無駄がなくて地域に根付いた食のつながりが大切だと思っている
人口が減っていき、人との交流がなくなっていくのが心配
グループ5:
街並みや座禅など歴史ある京都でしか味わえない体験が好き
グループ6:
漫画をアニメ化しても、原作から全く違うのになってしまうのが気に食わない
グループ7:
障碍者もちゃんと給料をもらって生活できる社会にすべきだ
グループ8:
感動して泣けるような漫画がすきだ
グループ9:
おじいちゃんの住む沖縄の陽気な雰囲気が好きだ
グループ10:
イラストをiPadで描くのが得意
グループ11:
歴史が好きな彼女を喜ばせたい
グループ12:
ウクライナとロシアの戦争はどうすれば解決するだろうか

###Input:
"""
    grouping3 = f"""
###Instruction:
{theme}
Please group all of these items with those that are closer in deeper underlying meaning.
Each group should have at most 2 items. Be sure to group ALL items. Items that cannot be grouped should be listed as is.
Please write in Japanese.

### Input:
自然や果物、スイーツなどの強みを生かして、福島県を盛り上げたい
未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要だ
スポーツの魅力は、競技が上手いか下手かという以外にもあるのではないか
コミュニティが小さくなっていくからこそ、効率的で地元に根付いた食のつながりを維持したい
街並みや座禅など歴史ある京都でしか味わえない体験が好き
漫画をアニメ化しても、原作から全く違うのになってしまうのが気に食わない
障碍者もちゃんと給料をもらって生活できる社会にすべきだ
感動して泣けるような漫画がすきだ
おじいちゃんの住む沖縄の陽気な雰囲気が好きだ
イラストをiPadで描くのが得意
歴史が好きな彼女を喜ばせたい
ウクライナとロシアの戦争はどうすれば解決するだろうか

###Output:
グループ1:
自然や果物、スイーツなどの強みを生かして、福島県を盛り上げたい
コミュニティが小さくなっていくからこそ、効率的で地元に根付いた食のつながりを維持したい
グループ2:
街並みや座禅など歴史ある京都でしか味わえない体験が好き
歴史が好きな彼女を喜ばせたい
グループ3:
漫画をアニメ化しても、原作から全く違うのになってしまうのが気に食わない
感動して泣けるような漫画がすきだ
グループ4:
未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要だ
グループ5:
スポーツの魅力は、競技が上手いか下手かという以外にもあるのではないか
グループ6:
障碍者もちゃんと給料をもらって生活できる社会にすべきだ
グループ7:
おじいちゃんの住む沖縄の陽気な雰囲気が好きだ
グループ8:
イラストをiPadで描くのが得意
グループ9:
ウクライナとロシアの戦争はどうすれば解決するだろうか

###Input:
"""
    grouping4 = f"""
###Instruction:
{theme}
Please group all of these items with those that are closer in deeper underlying meaning.
Each group should have at most 2 items. Be sure to group ALL items. Items that cannot be grouped should be listed as is.
Please write in Japanese.

###Input:
福島県の自然や果物を活かした地産地消のつながりを作り、人が減っても活気ある地域にしたい
京都での散策や座禅など、歴史ある土地に行かないとできない体験を彼女と一緒に楽しみたい
アニメでは表せないような、漫画にしかない感動の体験がある
未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要だ
スポーツの魅力は、競技が上手いか下手かという以外にもあるのではないか
障碍者もちゃんと給料をもらって生活できる社会にすべきだ
おじいちゃんの住む沖縄の陽気な雰囲気が好きだ
イラストをiPadで描くのが得意
ウクライナとロシアの戦争はどうすれば解決するだろうか

###Output:
グループ1:
福島県の自然や果物を活かした地産地消のつながりを作り、人が減っても活気ある地域にしたい
京都での散策や座禅など、歴史ある土地に行かないとできない体験を彼女と一緒に楽しみたい
グループ2:
未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要だ
スポーツの魅力は、競技が上手いか下手かという以外にもあるのではないか
障碍者もちゃんと給料をもらって生活できる社会にすべきだ
グループ3:
アニメでは表せないような、漫画にしかない感動の体験がある
グループ4:
おじいちゃんの住む沖縄の陽気な雰囲気が好きだ
グループ5:
イラストをiPadで描くのが得意
グループ6:
ウクライナとロシアの戦争はどうすれば解決するだろうか

###Input:
"""
    grouping5 = f"""
###Instruction:
{theme}
Please group all of these items with those that are closer in deeper underlying meaning.
Each group should have at most 2 items. Be sure to group ALL items. Items that cannot be grouped should be listed as is.
Please write in Japanese.

###Input:
自然や食、歴史など、その土地ならではの結びつきや体験に興味があり、自らもそうした地域づくりがしたい
例えば年齢差や障害の有無、競技のレベルといったような常識の枠に捉われず、本質的な未来を切り開いていく必要がある
アニメでは表せないような、漫画にしかない感動の体験がある
おじいちゃんの住む沖縄の陽気な雰囲気が好きだ
イラストをiPadで描くのが得意
ウクライナとロシアの戦争はどうすれば解決するだろうか

###Output:
グループ1:
自然や食、歴史など、その土地ならではの結びつきや体験に興味があり、自らもそうした地域づくりがしたい
おじいちゃんの住む沖縄の陽気な雰囲気が好きだ
グループ2:
例えば年齢差や障害の有無、競技のレベルといったような常識の枠に捉われず、本質的な未来を切り開いていく必要がある
ウクライナとロシアの戦争はどうすれば解決するだろうか
グループ3:
アニメでは表せないような、漫画にしかない感動の体験がある
グループ4:
イラストをiPadで描くのが得意

###Input:
"""
    if lines >= 20:
        prompt_ptrn = grouping1
    elif 15 <= lines < 20:
        prompt_ptrn = grouping2
    elif 10 <= lines < 15:
        prompt_ptrn = grouping3
    elif 6 <= lines < 10:
        prompt_ptrn = grouping4
    else:
        prompt_ptrn = grouping5
    return prompt_ptrn

def theme_translate(user_theme):
    theme = "インプットフィールドに入力したのは、" + user_theme + "についてのデータです。"
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
    translating_prompt = f"""
Please translate the following Japanese sentence into English.
{theme}
"""
    st.session_state.messages.append(SystemMessage(content=translating_prompt))
    with st.spinner("KJ-GPTがテーマを分析しています ..."):
        answer, cost = get_answer(llm, st.session_state.messages[-1:])
    translated_theme = answer
    st.session_state.costs.append(cost)
    return translated_theme


def data_generating(user_theme):
    theme = user_theme
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
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

labeling1 = """
###Instructions:
Please summarize the items of groups in one concise sentence with a deeper meaning.

###Conditions:
Please write in Japanese.

###Input:
地元のご当地スイーツがあれば食べたい
住んでいる福島県のフルーツをもっとアピールしたい
###Output:
福島県のスイーツを活かした、ご当地スイーツが食べたい

###Input:
宿題が多すぎて課題をこなすだけになっているのが嫌
未来の社会を発展させるような意味ある勉強がしたい
###Output:
目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたい

###Input:
食品ロスをどうやって減らせばいいか気になる
地域で作ったものは地元で食べたほうがいい
###Output:
無駄がなくて地域に根付いた食のつながりが大切だと思っている

###Input:
住んでいる地域の人口がどんどん減っていて不安
友達と遊ぶときに交通手段が少ないのが悩み
###Output:
人口が減っていき、人との交流がなくなっていくのが心配

###Input:
修学旅行で行く京都の座禅体験がたのしみ
歴史ある京都の町並みがとても好きだ
###Output:
街並みや座禅など歴史ある京都でしか味わえない体験が好き

###Input:
"""

labeling2 = """
###Instructions:
Please summarize the items of groups in one concise sentence with a deeper meaning.

###Conditions:
Please write in Japanese.

###Input:
福島県のスイーツを活かした、ご当地スイーツが食べたい
福島県の自然を活かした町おこしに興味がある
###Output:
自然や果物、スイーツなどの強みを生かして、福島県を盛り上げたい

###Input:
目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたい
子どもだけじゃなく大人の教育も必要だと思う
###Output:
未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要だ

###Input:
スポーツのプロとアマの違いはなんだろう
パラリンピックで見た障碍者の姿に感動した
卓球部の部活のユニフォームがダサくていやだ
###Output:
スポーツの魅力は、競技が上手いか下手かという以外にもあるのではないか

###Input:
無駄がなくて地域に根付いた食のつながりが大切だと思っている
人口が減っていき、人との交流がなくなっていくのが心配
###Output:
コミュニティが小さくなっていくからこそ、効率的で地元に根付いた食のつながりを維持したい

###Input:
"""

labeling3 = """
###Instructions:
Please summarize the items of groups in one concise sentence with a deeper meaning.

###Conditions:
Please write in Japanese.

###Input:
自然や果物、スイーツなどの強みを生かして、福島県を盛り上げたい
コミュニティが小さくなっていくからこそ、効率的で地元に根付いた食のつながりを維持したい
###Output:
福島県の自然や果物を活かした地産地消のつながりを作り、人が減っても活気ある地域にしたい

###Input:
街並みや座禅など歴史ある京都でしか味わえない体験が好き
歴史が好きな彼女を喜ばせたい
###Output:
京都での散策や座禅など、歴史ある土地に行かないとできない体験を彼女と一緒に楽しみたい

###Input:
漫画をアニメ化しても、原作から全く違うのになってしまうのが気に食わない
感動して泣けるような漫画がすきだ
###Output:
アニメでは表せないような、漫画にしかない感動の体験がある

###Input:
"""

labeling4 = """
###Instructions:
Please summarize the items of groups in one concise sentence with a deeper meaning.

###Conditions:
Use metaphors and proverbs as needed. Please write in Japanese.

###Input:
福島県の自然や果物を活かした地産地消のつながりを作り、人が減っても活気ある地域にしたい
京都での散策や座禅など、歴史ある土地に行かないとできない体験を彼女と一緒に楽しみたい
###Output:
自然や食、歴史など、その土地ならではの結びつきや体験に興味があり、自らもそうした地域づくりがしたい

###Input:
未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要だ
スポーツの魅力は、競技が上手いか下手かという以外にもあるのではないか
障碍者もちゃんと給料をもらって生活できる社会にすべきだ
###Output:
例えば年齢差や障害の有無、競技のレベルといったような常識の枠に捉われず、本質的な未来を切り開いていく必要がある

###Input:
"""

labeling5 = """
###Instructions:
Please summarize the items of groups in one concise sentence with a deeper meaning.

###Conditions:
Use metaphors and proverbs as needed. Please write in Japanese.

###Input:
自然や食、歴史など、その土地ならではの結びつきや体験に興味があり、自らもそうした地域づくりがしたい
おじいちゃんの住む沖縄の陽気な雰囲気が好きだ
###Output:
その土地土地が持つ歴史や自然環境、食文化や雰囲気などを、自分から楽しみ、そして広めたい

###Input:
例えば年齢差や障害の有無、競技のレベルといったような常識の枠に捉われず、本質的な未来を切り開いていく必要がある
ウクライナとロシアの戦争はどうすれば解決するだろうか
###Output:
既存の枠組みや対立軸をはみ出してでも、本当に大切な行動を起こし、平和な未来を目指したい

###Input:
"""

symbol = """
###Instructions:
Paraphrase each group of sentences in a single word that can be understood instantaneously.
Rephrase it with an adjective, verb or metaphor in Japanese.

###Input:
その土地土地が持つ歴史や自然環境、食文化や雰囲気などを、自分から楽しみ、そして広めたい
###Output:
地域独特の風土

###Input:
既存の枠組みや対立軸をはみ出してでも、本当に大切な行動を起こし、平和な未来を目指したい
###Output:
常識はずれの大切な行動

###Input:
アニメでは表せないような、漫画にしかない感動の体験がある
###Output:
漫画ならではの感動

###Input:
イラストをiPadで描くのが得意
###Output:
デジタルアート

###Input:
"""

def init_page():
    st.set_page_config(
        page_title="KJ-GPT",
        page_icon="🕵️‍♂️"
    )
    st.header("KJ-GPT 🕵️‍♂️")
    st.sidebar.title("Options")

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
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

def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4", "GPT-4-Turbo"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo-0613"
    elif model == "GPT-3.5-16k":
        model_name = "gpt-3.5-turbo-16k-0613"
    elif model == "GPT-4":
        model_name = "gpt-4"
    else:
        model_name = "gpt-4-1106-preview"

    # サイドバーにスライダーを追加し、temperatureを0から1までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.1とする
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    
    return ChatOpenAI(temperature=temperature, model_name=model_name)

def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost

def count_newlines(text):
    # 改行文字を含む正規表現パターンを定義
    pattern = r"\r\n|\n|\r"
    # パターンにマッチする部分を抽出し、その数を返す
    return len(re.findall(pattern, text))

def get_init_list(data):
    lines = data.strip().split("\n")
    return lines

def get_list(data):
    # データを行ごとに分割
    lines = data.strip().split("\n")

    result = []
    temp = []

    for line in lines:
        # ”グループ”もしくは"単独"の行の場合、tempをリセット
        if "グループ" in line or "単独" in line:
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

def headline_to_list(markdown_input):
    """マークダウンされた見出しをリスト箇条書き形式に変換して返す関数"""
    # マークダウンテキストを行に分割します。
    lines = markdown_input.strip().split("\n")

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

def messages_init():
    st.session_state.messages = [
        SystemMessage(content=""),
        HumanMessage(content=""),
        AIMessage(content="")
    ]

def main():
    init_page()

    llm = select_model()
    init_messages()

    translated_theme = None

    # ユーザーの入力を監視
    theme_container = st.container()
    with theme_container:
        with st.form(key="my_theme", clear_on_submit=False):
            user_theme = st.text_area(label="テーマ: ", key="theme_input", height=50)
            st.text("について")
            theme_button = st.form_submit_button(label="決定")
        # if theme_button and user_theme:
        #     translated_theme = theme_translate(user_theme)

    container = st.container()
    with container:
        with st.form(key="my_form", clear_on_submit=False):
            user_input = st.text_area(label="項目ラベル: ", key="input", height=300)
            # generating_button = st.form_submit_button(label="項目自動生成")
            grouping_button = st.form_submit_button(label="データを統合")
            # labeling_button = st.form_submit_button(label="表札づくり")
            # symbol_button = st.form_submit_button(label="シンボル作成")

        # if generating_button and user_theme:
        #     prompt_ptrn = data_generating(user_theme)
        #     st.session_state.messages.append(SystemMessage(content=prompt_ptrn))
        #     st.session_state.messages.append(HumanMessage(content=user_input))
        #     with st.spinner("KJ-GPTが元データを生成しています ..."):
        #         answer, cost = get_answer(llm, st.session_state.messages[-2:])
        #     st.session_state.messages.append(AIMessage(content=answer))
        #     st.session_state.costs.append(cost)

        if grouping_button and user_input:
            group_list = get_init_list(user_input)
            number_of_items = len(group_list)
            print(number_of_items)
            labeling_pair = []
            dict = {}
            count = 0
            while number_of_items > 5:
                # lines = count_newlines(user_input)
                if translated_theme is None:
                    translated_theme = theme_translate(user_theme)
                prompt_ptrn = prompt_grouping(number_of_items, translated_theme)
                st.session_state.messages.append(SystemMessage(content=prompt_ptrn))
                st.session_state.messages.append(HumanMessage(content=user_input))
                with st.spinner("KJ-GPTがラベルを集めています ..."):
                    answer, cost = get_answer(llm, st.session_state.messages[-2:])
                st.session_state.messages.append(AIMessage(content=answer))
                st.session_state.costs.append(cost)
                group_list = get_list(answer)
                for group in group_list:
                    if isinstance(group, list):
                        group_string = "\n".join(group)
                        if number_of_items >= 20:
                            prompt_ptrn = labeling1
                        elif 16 <= number_of_items < 20:
                            prompt_ptrn = labeling2
                        elif 12 <= number_of_items < 16:
                            prompt_ptrn = labeling3
                        elif 8 <= number_of_items < 12:
                            prompt_ptrn = labeling4
                        else:
                            prompt_ptrn = labeling5
                        st.session_state.messages.append(SystemMessage(content=prompt_ptrn))
                        st.session_state.messages.append(HumanMessage(content=group_string))
                        with st.spinner("KJ-GPTが表札を考えています ..."):
                            answer, cost = get_answer(llm, st.session_state.messages[-2:])
                        group_list.append(answer)
                        dict = {answer: group}
                        labeling_pair.append(dict)
                        st.session_state.messages.append(AIMessage(content=answer))
                        st.session_state.costs.append(cost)
                    # else:
                    #     labeling_pair.append(group)
                group_list = [item for item in group_list if not isinstance(item, list)]
                # print(labeling_pair)
                group_list_str = "\n".join(group_list)
                user_input = group_list_str
                count += 1
                result = st.text_area(label=f"{count}回目の結果: ", key=f"result{count}", value=group_list_str, height=300)
                number_of_items = len(group_list)
            # symbol_input_list = get_list(user_input)
            # print("symbol_input_list:", symbol_input_list)
            top_items = []
            symbol_sets = []
            symbol_dict = {}
            symbol_count = 0
            for item in group_list:
                symbol_count += 1
                item_str = str(item)
                st.session_state.messages.append(SystemMessage(content=symbol))
                st.session_state.messages.append(HumanMessage(content=item_str))
                with st.spinner("KJ-GPTがシンボルを作成しています ..."):
                    symbol_answer, cost = get_answer(llm, st.session_state.messages[-2:])
                symbol_set = symbol_answer + "：" + item_str
                symbol_title = "**" + f"({symbol_count}) " +  symbol_answer + "**"
                symbol_dict["# " + item_str] = symbol_title + "\n" + "# " + item_str
                # print("symbol_set", symbol_set)
                top_items.append(item_str)
                symbol_sets.append(symbol_set)
                st.session_state.messages.append(AIMessage(content=answer))
                st.session_state.costs.append(cost)
            symbol_str = "\n".join(symbol_sets)
            st.text_area(label="シンボル: ", value=symbol_str, key="final_result", height=300)
            st.text_area(label="辞書: ", value=labeling_pair, key="final_dict", height=300)

            markdown_text = ""

            # top_itemsリストをループして最上位の見出しを処理します。
            for top_item in top_items:
                # top_text = top_item.split("：")[1].strip()
                markdown_text += add_markdown_entry(1, top_item)  # 最上位の見出しを追加
                markdown_text += find_sub_items(top_item, 2, labeling_pair)  # サブアイテムを探して追加
            for key, value in symbol_dict.items():
                markdown_text = markdown_text.replace(key, value)
            
            st.text_area(label="Mark down: ", value=markdown_text, key="markdown", height=450)


#             sentence = f"""
# ###Instructions:
# {translated_theme}
# You are a philosopher who excels at introspection. Please logically connect each of the following bulleted items into a sentence.
# Please add philosophical criticism along the way.

# ### Condition:
# Please write in Japanese.
# Any additional explanations should be enclosed in parentheses.
# The line beginning with "**" is the title of the chapter, so insert a line break before and after it.
# Please add a logical connection to the last sentence of the text below.
# {answer}

# ###Input:
# # 未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要だ
# ## 子どもだけじゃなく大人の教育も必要だと思う
# ## 目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたい
# ### 宿題が多すぎて課題をこなすだけになっているのが嫌
# ### 未来の社会を発展させるような意味ある勉強がしたい

# ###Output:
# 宿題が多すぎて課題をこなすだけになっているのが嫌。（答えのある問題をただ強制的に解答させられるのは無駄だと思う。インターネットやChatGPTなどが急速に発展しているので、そういった単なる暗記や論理計算は、そのうち人間がやる必要はなくなると思う。それなのに、このまま偏差値至上主義の詰め込み教育で今後もやっていくならば、何の役にも立たない大人を育てることになるだろう。）
# そうではなくて、もっと未来の社会を発展させるような意味ある勉強がしたい。（答えのない問いに試行錯誤しながら立ち向かったり、自分だけの特別な興味関心を育てて専門性を高めたりする勉強の方が今後求められるのは明らかだ。）つまり、目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたいということ。そしてそのためには、子どもだけじゃなく大人の教育も必要だと思う。
# （そもそも今の教師が昔ながらの詰め込み式の教育で育ったので、その意識改革が必要だ。教師自身が答えのない自分の心の底から出てきた問いを設定し、生徒と一緒にそれに取り組む姿勢を見せないと、子供達はついていかない。それだけではなく、子供の親たちも新しい学びを人生に取り入れなければならない。答えのない探究活動は従来の学習に比べて、より日常生活に深く関わるものだ。普段過ごしている中で感じる疑問や違和感などを起点にした、実体験に即した問いであるほど、今後の長い人生で取り組むに値する深いものになりやすい。なので、これまでのように親が教育を学校や塾に任せっぱなしにして、家庭で子供に無関心でいては子供の探究心が育ちにくくなる。教師と同じように、親たちも自分の問いを立ててそれを追求する営みを実際にやるべきだ。そして、その行動が子供たちを感化させ、家庭を活気づかせて、さらには職場のパフォーマンスも上げることになるのが理想だ。）このように、未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要と言える。

# ###Input:
# """

            # 変数answerをリセット
            answer = ""
            sentence = f"""
###Instructions:
{translated_theme}
You are a philosopher who excels at introspection. Please logically connect each of the following bulleted items into a sentence.
Please add philosophical criticism along the way.

### Condition:
Please write in Japanese.
Any additional explanations should be enclosed in parentheses.
The line beginning with "**" is the title of the chapter, so insert a line break before and after it.
Please add a logical connection to the last sentence of the text below.
{answer}

###Input:
**(1) 常識はずれの大切な行動**
# 未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要だ
## 子どもだけじゃなく大人の教育も必要だと思う
## 目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたい
### 宿題が多すぎて課題をこなすだけになっているのが嫌
### 未来の社会を発展させるような意味ある勉強がしたい

###Output:

(1) 常識はずれの大切な行動

宿題が多すぎて課題をこなすだけになっているのが嫌。（答えのある問題をただ強制的に解答させられるのは無駄だと思う。インターネットやChatGPTなどが急速に発展しているので、そういった単なる暗記や論理計算は、そのうち人間がやる必要はなくなると思う。それなのに、このまま偏差値至上主義の詰め込み教育で今後もやっていくならば、何の役にも立たない大人を育てることになるだろう。）
そうではなくて、もっと未来の社会を発展させるような意味ある勉強がしたい。（答えのない問いに試行錯誤しながら立ち向かったり、自分だけの特別な興味関心を育てて専門性を高めたりする勉強の方が今後求められるのは明らかだ。）つまり、目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたいということ。そしてそのためには、子どもだけじゃなく大人の教育も必要だと思う。
（そもそも今の教師が昔ながらの詰め込み式の教育で育ったので、その意識改革が必要だ。教師自身が答えのない自分の心の底から出てきた問いを設定し、生徒と一緒にそれに取り組む姿勢を見せないと、子供達はついていかない。それだけではなく、子供の親たちも新しい学びを人生に取り入れなければならない。答えのない探究活動は従来の学習に比べて、より日常生活に深く関わるものだ。普段過ごしている中で感じる疑問や違和感などを起点にした、実体験に即した問いであるほど、今後の長い人生で取り組むに値する深いものになりやすい。なので、これまでのように親が教育を学校や塾に任せっぱなしにして、家庭で子供に無関心でいては子供の探究心が育ちにくくなる。教師と同じように、親たちも自分の問いを立ててそれを追求する営みを実際にやるべきだ。そして、その行動が子供たちを感化させ、家庭を活気づかせて、さらには職場のパフォーマンスも上げることになるのが理想だ。）このように、未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要と言える。

###Input:
"""
            
            simplified_answer = ""
            simplifying_sentence = f"""
### Instruction:
{translated_theme}
Extract the bracketed sentences from the text and create an integrated text of them.

### Condition:
Please write in Japanese.
Please make sure that the text is chewed up in a way that high school students can understand.
Lines beginning with bracketed numbers should be transcribed in their original position and with no change in content.
Please add a logical connection to the text below.
{simplified_answer}

###Input:
(1) 常識はずれの大切な行動

宿題が多すぎて課題をこなすだけになっているのが嫌。（答えのある問題をただ強制的に解答させられるのは無駄だと思う。インターネットやChatGPTなどが急速に発展しているので、そういった単なる暗記や論理計算は、そのうち人間がやる必要はなくなると思う。それなのに、このまま偏差値至上主義の詰め込み教育で今後もやっていくならば、何の役にも立たない大人を育てることになるだろう。）
そうではなくて、もっと未来の社会を発展させるような意味ある勉強がしたい。（答えのない問いに試行錯誤しながら立ち向かったり、自分だけの特別な興味関心を育てて専門性を高めたりする勉強の方が今後求められるのは明らかだ。）つまり、目先の宿題を消化するだけではなく、将来の社会に意義のある学びをしたいということ。そしてそのためには、子どもだけじゃなく大人の教育も必要だと思う。
（そもそも今の教師が昔ながらの詰め込み式の教育で育ったので、その意識改革が必要だ。教師自身が答えのない自分の心の底から出てきた問いを設定し、生徒と一緒にそれに取り組む姿勢を見せないと、子供達はついていかない。それだけではなく、子供の親たちも新しい学びを人生に取り入れなければならない。答えのない探究活動は従来の学習に比べて、より日常生活に深く関わるものだ。普段過ごしている中で感じる疑問や違和感などを起点にした、実体験に即した問いであるほど、今後の長い人生で取り組むに値する深いものになりやすい。なので、これまでのように親が教育を学校や塾に任せっぱなしにして、家庭で子供に無関心でいては子供の探究心が育ちにくくなる。教師と同じように、親たちも自分の問いを立ててそれを追求する営みを実際にやるべきだ。そして、その行動が子供たちを感化させ、家庭を活気づかせて、さらには職場のパフォーマンスも上げることになるのが理想だ。）このように、未来にとって意味のある本当の学びは、年齢に関係なく誰にでも重要と言える。

###Output:
(1) 常識はずれの大切な行動

ただ問題の答えを教え込む古いやり方ではなく、なぜそうなるのかを考えたり、新しいことに挑戦したりする学びが大切だって話だよ。ネットやChatGPTみたいな賢いツールがたくさんあるから、単純な暗記や計算はもう人間がわざわざやることじゃなくなるんじゃないかな。でも、学校が今のまま詰め込みで点数だけ追いかける教育を続けたら、本当に必要なスキルを身につけられない大人になってしまう。
これからは、答えがすぐには出ないような問題にどう立ち向かうか、自分の好きなことを見つけて深く掘り下げる学びが求められるんだ。先生たちも昔のやり方から変わって、生徒と一緒に考えることが大事だし、それは親も同じ。家での学びもすごく重要で、親が自分で疑問を持って考える姿を子供に見せることが、子供の好奇心を育てるんだ。
つまり、学校や塾だけじゃなくて、家でも親が子供と一緒に新しいことにチャレンジしたり、考えたりすることが、子供の成長にとってはめちゃくちゃ大事ってわけ。そうすると、家の中ももっと楽しくなって、親の仕事のやる気にもつながるんだ。

###Input:
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

###Input:
ただ問題の答えを教え込む古いやり方ではなく、なぜそうなるのかを考えたり、新しいことに挑戦したりする学びが大切だって話だよ。ネットやChatGPTみたいな賢いツールがたくさんあるから、単純な暗記や計算はもう人間がわざわざやることじゃなくなるんじゃないかな。でも、学校が今のまま詰め込みで点数だけ追いかける教育を続けたら、本当に必要なスキルを身につけられない大人になってしまうよ。
これからは、答えがすぐには出ないような問題にどう立ち向かうか、自分の好きなことを見つけて深く掘り下げる学びが求められるんだ。先生たちも昔のやり方から変わって、生徒と一緒に考えることが大事だし、それは親も同じ。家での学びもすごく重要で、親が自分で疑問を持って考える姿を子供に見せることが、子供の好奇心を育てるんだ。
つまり、学校や塾だけじゃなくて、家でも親が子供と一緒に新しいことにチャレンジしたり、考えたりすることが、子供の成長にとってはめちゃくちゃ大事ってわけ。そうすると、家の中ももっと楽しくなって、親の仕事のやる気にもつながるよ。

###Output:
いまの時代は、単純な暗記よりも考える力や新しいことに挑戦する力を育てる学びが大切なんだ。そして、家庭でも親が子供と一緒に学ぶことができれば、子供だけじゃなくて親の成長にもつながるってわけ。

###Input:
"""

            converted_markdown = headline_to_list(markdown_text)

            st.markdown(converted_markdown)

            segmented_markdown = segmented_by_three(markdown_text)
            combined_list = []
            simplified_list = []
            summarized_list = []

            for segment in segmented_markdown:

                st.session_state.messages.append(SystemMessage(content=sentence))
                st.session_state.messages.append(HumanMessage(content=segment))
                with st.spinner("KJ-GPTが文章化しています ..."):
                    answer, cost = get_answer(llm, st.session_state.messages[-2:])
                # もし答えが(1)などで始まっていたら、その前に改行を追加
                if answer.startswith("("):
                    answer = "\n" + answer
                combined_list.append(answer)
                st.session_state.messages.append(AIMessage(content=answer))
                st.session_state.costs.append(cost)

                st.session_state.messages.append(SystemMessage(content=simplifying_sentence))
                st.session_state.messages.append(HumanMessage(content=answer))
                with st.spinner("KJ-GPTが文章化しています ..."):
                    simplified_answer, cost = get_answer(llm, st.session_state.messages[-2:])
                # もし答えが(1)などで始まっていたら、その前に改行を追加
                if simplified_answer.startswith("("):
                    simplified_answer = "\n" + simplified_answer
                simplified_list.append(simplified_answer)
                st.session_state.messages.append(AIMessage(content=simplified_answer))

            #     st.session_state.messages.append(SystemMessage(content=summarized_sentence))
            #     st.session_state.messages.append(HumanMessage(content=simplified_answer))
            #     with st.spinner("KJ-GPTが文章化しています ..."):
            #         summarized_answer, cost = get_answer(llm, st.session_state.messages[-2:])
            #     summarized_list.append(summarized_answer)
            #     st.session_state.messages.append(AIMessage(content=summarized_answer))

            combined_sentences = "\n".join(combined_list)
            simplified_sentences = "\n".join(simplified_list)
            # summarized_sentences = "\n".join(summarized_list)
            st.markdown(combined_sentences)
            st.markdown(simplified_sentences)
            # st.markdown(summarized_sentences)

            

        # if labeling_button and user_input:
        #     lines = count_newlines(user_input)
        #     if lines >= 35:
        #         prompt_ptrn = labeling1
        #     elif 25 <= lines < 35:
        #         prompt_ptrn = labeling2
        #     elif 15 <= lines < 25:
        #         prompt_ptrn = labeling3
        #     elif 11 <= lines < 15:
        #         prompt_ptrn = labeling4
        #     else:
        #         prompt_ptrn = labeling5
        #     st.session_state.messages.append(SystemMessage(content=prompt_ptrn))
        #     st.session_state.messages.append(HumanMessage(content=user_input))
        #     print(st.session_state.messages[-2:])
        #     with st.spinner("KJ-GPTが表札を考えています ..."):
        #         answer, cost = get_answer(llm, st.session_state.messages[-2:])
        #     st.session_state.messages.append(AIMessage(content=answer))
        #     st.session_state.costs.append(cost)

        # if symbol_button and user_input:
        #     st.session_state.messages.append(SystemMessage(content=symbol))
        #     st.session_state.messages.append(HumanMessage(content=user_input))
        #     print(st.session_state.messages[-2:])
        #     with st.spinner("KJ-GPTが入力しています ..."):
        #         answer, cost = get_answer(llm, st.session_state.messages[-2:])
        #     st.session_state.messages.append(AIMessage(content=answer))
        #     st.session_state.costs.append(cost)

    # チャット履歴の表示
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        # else:
        #     st.write(f"System message: {message.content}")

    # コストの計算と表示
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == "__main__":
    main()