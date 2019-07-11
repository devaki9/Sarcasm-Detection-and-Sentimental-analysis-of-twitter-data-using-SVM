import pandas as pd
df=pd.read_json('csvjson.json')
print(df)
for a,b in df.iterrows():
    if(b["label"]=="-1"):
       ironicData.append(b['text'])
    elif(b["label"]=="1"):
        nonIronicData.append(b['text'])

