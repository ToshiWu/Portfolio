import os, requests, json, configparser
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *

# 設定LineBot為工作目錄
if os.getcwd().split('\\')[-1] != 'LineBot':
    os.chdir('./LineBot')
    
import stock_utils as Stock
import thsr_utils as Thsr

# LINE 聊天機器人的基本資料
config = configparser.ConfigParser()
config.read('config.ini')

line_bot_api = LineBotApi(config.get('line-bot', 'channel_access_token'))
handler = WebhookHandler(config.get('line-bot', 'channel_secret'))

# 高鐵參數
thsr = Thsr.ThsrModule()
chat_record = []

thsr_res = {"starting":"", "ending":"", "date":"", "ampm":""}
station_names = Thsr.station_name
date_keys = Thsr.date_key
ampm_keys = Thsr.ampm_key

# 股票參數
StockSymbol = Stock.stock_symbol

print('可使用對話參數')
for i in ['station_names','date_keys','ampm_keys','StockSymbol']:
    print(f'{i}\t: {globals()[i]}')
    
# 建立LineBot app
app = Flask(__name__)


# 接收 LINE 資訊
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    # app.logger.info("Request body: " + body)

    try:
        print("body:",body)
        print("signature:", signature)
        print("===")
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# 回覆設定 (加入高鐵API多輪對話)
@handler.add(MessageEvent, message=TextMessage)
def get_response(event):
    query = event.message.text       
    
    if len(chat_record) < 5:
        chat_record.append(query)
    else:
        chat_record.pop(0)
        chat_record.append(query)
    print("chat_record:",chat_record)


    # 判斷是否為"高鐵查詢意圖"
    if query == "高鐵":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="哪一天出發?"))
    try:
        if chat_record[-2] == "高鐵" and any(chat_record[-1] == i for i in date_keys):
            date_format = thsr.get_date_string_today(chat_record[-1])
            thsr_res['date'] = date_format
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="上午還是下午的車?"))

        elif any(chat_record[-2] == i for i in date_keys) and any(chat_record[-1] == i for i in ampm_keys):
            thsr_res['ampm'] = chat_record[-1]
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="起站是哪裡呢?"))

        elif any(chat_record[-2] == i for i in ampm_keys) and any(chat_record[-1] == i for i in station_names):
            startind_id = Thsr.station_id[chat_record[-1]]
            thsr_res['starting'] = startind_id
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="終點站是哪裡呢?"))

        elif any(chat_record[-2] == i for i in station_names) and any(chat_record[-1] == i for i in station_names):
            ending_id = Thsr.station_id[chat_record[-1]]
            thsr_res['ending'] = ending_id
            text = thsr.get_runs(thsr_res['starting'],thsr_res['ending'],thsr_res['date'],thsr_res['ampm'])
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text))

    except IndexError:
        pass
    
    
    # 判斷是否為"股價詢問意圖"
    try:
        # 若股票名稱 = query ...
        if any(sk == query for sk in list(StockSymbol.keys())):
            stock_symbol = StockSymbol[query]
            stock_data = Stock.get_stockdata(stock_symbol, "2020-12-01", "2020-12-10")
            stock_info = Stock.get_stockinfo(query, stock_data, "Close")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=stock_info))

    except:
        pass
      
    
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請重新輸入"))

if __name__ == "__main__":
    app.run(port=80)