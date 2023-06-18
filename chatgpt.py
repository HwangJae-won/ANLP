import pandas as pd 
import numpy as np
from datasets import load_dataset
import openai
import json
import os

##data 
data = load_dataset("klue", 'nli')
val = data['validation'][:300]
val=  pd.DataFrame(val)

## OepnAI
api = '' #my api
openai.organization=" " #my information

## Function of making response
def question(prompt, assi):
    openai.api_key = api
    
    messages = [
    {"role": "system", "content" : "당신은 한국어로 말합니다."}, 
    {"role":'user', "content" : prompt}, 
    {"role":"assistant", "content": assi}
    ]
    response = openai.ChatCompletion.create(model= "gpt-3.5-turbo",messages =messages)
    answer = response['choices'][0]['message']['content']
    
    return answer

prompt1= "나는 첫번째 문장을 전체로 두번쨰 문장을 가설로 설정하였어. 가설을 모순, 모순이 아님, 판단할 수 없음의 3가지로 판단해줘. \n"
prompt2 ="판단 결과, 모순이면 0, 모순이 아니면 1, 판단할 수 없으면 2를 출력해줘"

pre =[]
for i in range(len(val)):
    assi =''
    prompt =prompt1 +"전제:"+val.premise[i]+ "  \n 가설:"+val.hypothesis[i]
    txt =question(prompt, assi)
    print(txt)
    print("->")
    answer =question(prompt2, txt)
    print(answer)
    pre.append(answer)
    
pre_new=[]
for i in pre:
    pre_new.append(re.sub(r'[^0-9]', '', i))


file_path = "chatgpt.json"
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(pre_new.tolist(), file)