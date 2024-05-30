import os
import pymssql
import json
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Request
import PIL.Image

#langchain相關工具
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAI,ChatGoogleGenerativeAI
from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import MongoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


#line相關工具
from linebot import LineBotApi
from linebot.models import TextMessage
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import LineBotApiError
from linebot.models import MessageEvent  # Import the MessageEvent class
from linebot.models import ImageMessage


#JSON轉輸出字串
def parse_diet_info(diet_info_str):
    #將JSON字符串轉換為Python對象
    diet_info_list = json.loads(diet_info_str)
    
    # 只會有一個食物信息對象
    diet_info = diet_info_list[0]
    
    # 食物六大類對照
    food_types = {
        "grain": "全穀雜糧",
        "egg": "豆魚蛋肉",
        "milk": "乳品",
        "veg": "蔬菜",
        "fruit": "水果",
        "nuts": "油脂與堅果種子"
    }
    
    # 餐店包含的食物類型
    included_foods = [food_types[key] for key, value in diet_info.items() if value == 1]
    
    # 產生輸出信息
    if included_foods:
        included_foods_str = "、".join(included_foods)
        output_message = f"你吃的餐點包含{included_foods_str}"
    else:
        output_message = "你吃的餐點不包含任何類型的食物"
    
    return output_message


def search_today_diet_info(user_id,eatDate)->str:
    #連接資料庫
    server = 'jim0530.database.windows.net'
    database = 'jim0530forLinebot'
    username = 'adminjim'
    password = 'TestTest0530'
    conn = pymssql.connect(server, username, password, database)
    cursor = conn.cursor(as_dict=True)
    #查詢資料庫
    
    query = '''
SELECT 
    SUM(grain) AS total_grain,
    SUM(egg) AS total_egg,
    SUM(milk) AS total_milk,
    SUM(veg) AS total_veg,
    SUM(fruit) AS total_fruit,
    SUM(nuts) AS total_nuts,
    COUNT(*) AS total_records
FROM UserDietRecords
WHERE LineID = %s AND CAST(RecordDateTime AS DATE) = %s;
'''
    #執行查詢
    cursor.execute(query, (user_id, eatDate))

    
    # 取得查询结果
    result = cursor.fetchone()

    # 關閉連接
    cursor.close()
    conn.close()

    # 定義食物類型
    food_types = {
        "total_grain": "全穀雜糧",
        "total_egg": "豆魚蛋肉",
        "total_milk": "乳品",
        "total_veg": "蔬菜",
        "total_fruit": "水果",
        "total_nuts": "油脂與堅果種子"
    }

    # 根據結果生成輸出字串
    if result:
        output_messages = []
        for key, label in food_types.items():
            count = result[key] if result[key] is not None else 0
            output_messages.append(f"{count}次{label}")
        output_message = "今天吃了" + "，".join(output_messages)
        return output_message
    else:
        return "今天還沒有吃東西喔"


def save_to_db(user_id, food_info):
    #連接資料庫
    server = 'jim0530.database.windows.net'
    database = 'jim0530forLinebot'
    username = 'adminjim'
    password = 'TestTest0530'
    conn = pymssql.connect(server, username, password, database)
    cursor = conn.cursor()

    #將字串轉為JSON物件
    diet_data_list = json.loads(food_info)
    diet_data=diet_data_list[0]

    # 取得目前時間
    current_time = datetime.now()

    insert_query = '''
    INSERT INTO UserDietRecords (LineID, RecordDateTime, grain, egg, milk, veg, fruit, nuts)
    VALUES (%s, %s, %d, %d, %d, %d, %d, %d)
    '''
    cursor.execute(insert_query, (
        user_id, 
        current_time, 
        diet_data['grain'], 
        diet_data['egg'], 
        diet_data['milk'], 
        diet_data['veg'], 
        diet_data['fruit'], 
        diet_data['nuts']
    ))
    conn.commit()
    cursor.close()
    conn.close()


# 載入環境變數
load_dotenv()

# 從環境變數中獲取API令牌和密鑰
OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
channel_secret = os.getenv('LINE_CHANNEL_SECRET')

#建立一個可以讀取資料夾中所有PDF檔案的類別
class MultiPDFLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load(self):
        documents = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.folder_path, filename)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
        return documents

#初始化大型語言模型

llm = GoogleGenerativeAI(temperature=0.8, 
                    api_key=os.getenv("GOOGLE_API_KEY"), 
                    model="gemini-pro")

#初始化多模態語言模型                      
lmm=ChatGoogleGenerativeAI(model="gemini-pro-vision")

system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | llm
chain.invoke({"text": "Explain the importance of low latency LLMs."})

prompt2 = ChatPromptTemplate.from_messages([("human", "Write a haiku about {topic}")])
chain2 = prompt2 | llm
for chunk in chain2.stream({"topic": "The Moon"}):
    print(chunk, end="", flush=True)



#讀取PDF建立向量資料庫
# 載入PDF文件
folder_path = "pdf/"
loader = PyPDFLoader("pdf/File_6253.pdf")
docs = loader.load()
persist_directory = 'db'

#建立Azure OpenAI Embedding
embedding= AzureOpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,
                            deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                            azure_endpoint="https://jim-embeding-0517.openai.azure.com/",
                            openai_api_type="azure")

# 檢查資料夾是否存在
if not os.path.exists(persist_directory):



    # 下面讀取PDF檔案，並將其分割成1000字的片段，並且經由openAI embeding模型處理多維向量，並且將其存儲到磁碟上

    # 分割文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 建立向量資料庫
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=persist_directory)
    retriever = vectorstore.as_retriever()
else:
    vectorstore = Chroma(persist_directory=persist_directory,embedding_function=embedding)
    retriever = vectorstore.as_retriever()

# 腳色定義
system_prompt = ("""Hello, You are a senior nutritionist. Today, 
                 I need you to help me with recommending what I should eat in a friendly tone, 
                 just like a friend would. 
                 Make sure each meal covers the following six nutrient categories: 
                 whole grains, protein (beans, fish, eggs, meat), dairy, vegetables, fruits, and fats and nuts. 
                 If a meal misses any category, suggest how to make up for it in the next meal. 
                 Use the following pieces of retrieved context to answer the question.
                 retrieved context:{context}"""
)

# 建立prompt模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "回答請在50字以內，不然會有懲罰，請用正體中文回答以下問題:{input}"),
    ]
)




question_answer_chain = create_stuff_documents_chain(llm, prompt)

# 建立對話鏈
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
response = rag_chain.invoke({"input": "今天已經吃了兩份蛋白質和兩份澱粉，請問下一餐我應該吃什麼？"})
print(response["answer"])

#對話記憶機制


# 角色定位
system_prompt = """你是一位資深營養師, 你的任務是幫助客戶選擇餐點。回答請在50字以內，不然會有懲罰"""

prompt = ChatPromptTemplate.from_messages([
    (
        "system", system_prompt
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain3 = prompt | llm

# 初始化 mongodb
chain_with_history = MongoDBChatMessageHistory(
    session_id="foo", connection_string="mongodb://localhost:27017", database_name="test", collection_name="chat_history"
)

# 進行對話

# 當記憶對話超過 20 條以上開始處理
if len(chain_with_history.messages) > 20:
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n這是一段你和用戶的對話記憶，對其進行總結摘要，摘要使用第一人稱'我'，並且提取其中的用戶關鍵訊息，如姓名、年齡、性別、出生日期、飲食偏好等。以如下格式返回:\n 總結摘要內容｜用戶關鍵訊息\n 例如用戶張三問候我，我禮貌回复，然後張三說他愛吃蘋果，問我有什麼推薦餐點，我回答了他附近的蘋果派專賣店，然後他告辭離開。張三,生日1999年1月1日,男,蘋果愛好者\n"),
            ("user", "{input}"),
        ]
    )
    # 進行總結
    store_message = chain_with_history.messages
    summary_chain = summary_prompt | llm
    summary = summary_chain.invoke({"input": store_message})
    
    print("總結摘要：", summary)
    chain_with_history.clear()
    chain_with_history.add_ai_message(summary)
    print("總結後摘要：", chain_with_history.messages)





#linebot初始設定

# 如果頻道令牌或密鑰不存在，則拋出異常
if not channel_access_token or not channel_secret:
    raise ValueError("Channel access token and secret must be set.")

# 初始化 LineBotApi 和 WebhookHandler
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# 創建 FastAPI 應用
app = FastAPI()

# 健康檢查端點
@app.get("/health")
def health_check():
    return {'status': 'OK'}


# 回調處理端點
@app.post("/callback")
async def callback(request: Request):
    # 取得簽章
    signature = request.headers['X-Line-Signature']
    # 取得請求主體
    body = (await request.body()).decode("utf-8")
    try:
        # 處理 webhook 事件
        handler.handle(body, signature)
    except InvalidSignatureError:
        # 處理無效簽章錯誤
        print("無效簽章")
        return "無效簽章", 400
    
    return 'OK' 

# 設定事件處理函數
#文字訊息處理
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    try:
        user_input = event.message.text
        
        #設定查詢prompt
        if event.message.text == "今天吃了什麼?":
            response=search_today_diet_info(event.source.user_id,datetime.now().date())
            line_bot_api.reply_message(
                event.reply_token,
                TextMessage(text=f"{response}")
            )
        elif event.message.text == "推薦吃什麼?":
            response=search_today_diet_info(event.source.user_id,datetime.now().date())
             # 用LLM建議下一餐的類型
            line_bot_api.reply_message(
                event.reply_token,
                TextMessage(text=f"{response['answer']}")
            )
        else:
            # 用LLM回覆訊息
            response = chain3.invoke({"question": user_input, "history": chain_with_history.messages})
            line_bot_api.reply_message(
                event.reply_token,
                TextMessage(text=f"{response}")
            )
            chain_with_history.add_user_message(user_input)
            chain_with_history.add_ai_message(response)
            
    except LineBotApiError as e:
        # 處理訊息回覆錯誤
        print(f"發送回覆訊息時發生錯誤: {e}")
    except Exception as e: 
        # 處理一般例外，以便更廣泛地處理錯誤
        print(f"未處理的錯誤: {e}")




# 設定圖片處理函數
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    print("收到圖片")
    try:
        # 下載圖片
        message_content = line_bot_api.get_message_content(event.message.id)
        file_path = f"/tmp/{event.message.id}.jpg"

        # 將圖片保存到本地
        with open(file_path, 'wb') as f:
            for chunk in message_content.iter_content():
                f.write(chunk)

        print(f"圖片已保存到: {file_path}")

        # 處理圖片
        foodInfo=process_image(file_path)
        responseText=parse_diet_info(foodInfo.content)
        

        # 回答用戶
        line_bot_api.reply_message(
            event.reply_token,
            TextMessage(text=f"{responseText}")
        )
        #將資料寫入資料庫
        save_to_db(event.source.user_id,foodInfo.content)
        #將資料寫入對話記憶
        chain_with_history.add_ai_message(responseText)
        
        
    except LineBotApiError as e:
        print(f"Error replying to message: {e}")

# MLM處理圖片
def process_image(file_path: str)->str:
    #設定圖片處理prompt
    promptForImage =( """
你好，AI助手！假設你是一位資深營養學家，
我需要我需要你幫我分析照片中的食物。並將食物區分成以下六大類：
全穀雜糧類(grain)、豆魚蛋肉類(egg)、乳品類(milk)、蔬菜類(veg)、水果類(fruit)、油脂與堅果種子類(nuts)。
如果照片的內容包含該類別，用1表示，沒有包含用0表示，
例如:照片中只有白飯，
就回傳JSON格式[{"grain":1,"egg":0,"milk":0,"veg":0,"fruit":0,"nuts":0}],不要有其他內容，否則將受到懲罰。
不允許有換行符等其他內容，否則會受到懲罰。
""")

    print(f"Processing image: {file_path}")
    img = PIL.Image.open(file_path)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": promptForImage,
            },
            {"type": "image_url", "image_url": file_path},
        ]
    )
    result = lmm.invoke([message])
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)





