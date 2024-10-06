import streamlit as st
from transformers import GPT2LMHeadModel, BertTokenizer
import torch
import sqlite3
import random

# 调试信息
print("调试信息：正在运行更新后的 app.py")

# ----------------------------
# 加载分词器和模型
# ----------------------------
tokenizer = BertTokenizer.from_pretrained("thu-coai/CDial-GPT2_LCCC-base", cache_dir='D:/model_cache')
model = GPT2LMHeadModel.from_pretrained("thu-coai/CDial-GPT2_LCCC-base", cache_dir='D:/model_cache')


# 设置特殊的 tokens
tokenizer.eos_token = '[SEP]'
tokenizer.pad_token = '[PAD]'

# ----------------------------
# 初始化会话状态
# ----------------------------
if 'chat_history_ids' not in st.session_state:
    st.session_state['chat_history_ids'] = None

if 'interests' not in st.session_state:
    st.session_state['interests'] = []

# ----------------------------
# 初始化数据库连接
# ----------------------------
conn = sqlite3.connect('user_data.db', check_same_thread=False)
c = conn.cursor()

# 创建对话记录表（如果不存在）
c.execute('''
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT,
    ai_response TEXT,
    sentiment REAL,
    interests TEXT,
    feedback INTEGER
)
''')
conn.commit()

# 在应用启动时，加载被差评的 AI 回复
c.execute("SELECT ai_response FROM conversations WHERE feedback = -1")
negative_responses = [row[0] for row in c.fetchall()]

# ----------------------------
# 定义情感分析函数
# ----------------------------
def analyze_sentiment(text):
    if any(word in text for word in ['开心', '高兴', '愉快', '满意', '幸福']):
        sentiment = 0.5
    elif any(word in text for word in ['难过', '伤心', '沮丧', '生气', '烦恼']):
        sentiment = -0.5
    else:
        sentiment = 0.0
    return sentiment

# ----------------------------
# 定义对话生成函数
# ----------------------------
def generate_response(user_input, sentiment, chat_history_ids=None):
    # 根据情感得分，生成情感标签
    if sentiment > 0.1:
        emotion_tag = "心情愉快"
    elif sentiment < -0.1:
        emotion_tag = "心情低落"
    else:
        emotion_tag = "心情平静"

    # 将情感标签添加到用户输入前
    user_input_with_emotion = f"[{emotion_tag}] {user_input}{tokenizer.eos_token}"

    # 对用户输入进行编码
    new_input_ids = tokenizer.encode(user_input_with_emotion, return_tensors='pt')

    # 将新输入与历史对话连接
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # 生成模型输出书
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        temperature=0.7
    )

    # 解码生成的响应
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # 简单优化：避免生成被差评的回复
    retry_count = 0
    max_retries = 3
    while response in negative_responses and retry_count < max_retries:
        temperature = 0.7 + retry_count * 0.1
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            temperature=temperature
        )
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        retry_count += 1

    # 根据用户兴趣，添加个性化内容
    if st.session_state['interests']:
        interests = '、'.join(st.session_state['interests'])
        response += f"\n另外，记得您对 {interests} 感兴趣，我们可以多聊聊这方面的话题。"

    return response, chat_history_ids

# ----------------------------
# 定义个性化提醒函数
# ----------------------------
def get_personalized_reminder(sentiment, interests):
    reminders = []
    if sentiment < -0.1:
        negative_reminders = [
            "听起来你有点不开心，试着深呼吸放松一下吧。",
            "或许出去散散步能让你感觉好些。",
            "记得照顾好自己，一切都会好起来的。",
            "和朋友聊聊天，可能会让你心情变好。"
        ]
        reminders.append(random.choice(negative_reminders))
    elif sentiment > 0.1:
        positive_reminders = [
            "很高兴听到你心情不错！保持积极的心态。",
            "愿你的好心情一直持续下去！",
            "继续保持，你的笑容很有感染力！"
        ]
        reminders.append(random.choice(positive_reminders))
    else:
        neutral_reminders = [
            "希望你有美好的一天！",
            "保持平和的心态，生活会更美好。",
            "或许尝试一些新事物，会带来惊喜。"
        ]
        reminders.append(random.choice(neutral_reminders))

    # 根据兴趣添加提醒
    if interests:
        for interest in interests:
            if interest == "音乐":
                music_reminders = [
                    "听听你喜欢的音乐，放松一下吧。",
                    "最近有新歌发布，去发现一下吧！"
                ]
                reminders.append(random.choice(music_reminders))
            elif interest == "运动":
                sport_reminders = [
                    "别忘了每天锻炼身体，保持健康！",
                    "尝试新的运动项目，可能会很有趣。"
                ]
                reminders.append(random.choice(sport_reminders))
            # 可以继续为其他兴趣添加提醒

    return reminders

# ----------------------------
# 设置页面标题和描述
# ----------------------------
st.title("个性化成长型对话 AI 助手")

st.write("欢迎使用个性化成长型对话 AI 助手。请在下面的输入框中输入您的消息。")

# ----------------------------
# 用户输入
# ----------------------------
user_input = st.text_input("您：", "")

# 定义关键词列表
keywords = ['电影', '音乐', '运动', '旅行', '阅读', '游戏']

# 当用户点击发送按钮时
if st.button("发送"):
    if user_input:
        # 分析用户情感
        sentiment = analyze_sentiment(user_input)
        if sentiment > 0.1:
            st.write("检测到您心情不错！😊")
        elif sentiment < -0.1:
            st.write("抱歉，感觉您心情不太好，希望我能帮到您。😢")
        else:
            st.write("您的心情看起来很平静。😐")

        # 提取用户兴趣
        for word in keywords:
            if word in user_input and word not in st.session_state['interests']:
                st.session_state['interests'].append(word)
                st.write(f"我注意到您对 **{word}** 感兴趣！")

        # 生成 AI 回复
        response, st.session_state['chat_history_ids'] = generate_response(
            user_input, sentiment, st.session_state['chat_history_ids']
        )
        st.write(f"AI 助手：{response}")

        # 保存对话记录到数据库
        interests_str = '、'.join(st.session_state['interests'])
        c.execute("INSERT INTO conversations (user_input, ai_response, sentiment, interests, feedback) VALUES (?, ?, ?, ?, ?)",
                  (user_input, response, sentiment, interests_str, None))
        conn.commit()

        # 获取刚插入记录的 ID
        last_row_id = c.lastrowid

        # 添加反馈按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍", key=f"like_{last_row_id}"):
                feedback = 1
                c.execute("UPDATE conversations SET feedback = ? WHERE id = ?", (feedback, last_row_id))
                conn.commit()
                st.write("感谢您的反馈！")
        with col2:
            if st.button("👎", key=f"dislike_{last_row_id}"):
                feedback = -1
                c.execute("UPDATE conversations SET feedback = ? WHERE id = ?", (feedback, last_row_id))
                conn.commit()
                st.write("感谢您的反馈！")

        # 个性化提醒
        reminders = get_personalized_reminder(sentiment, st.session_state['interests'])
        if reminders:
            st.write("个性化提醒：")
            for reminder in reminders:
                st.info(reminder)
    else:
        st.write("请输入您的消息。")

# 查看历史对话按钮
if st.button("查看历史对话"):
    st.write("**历史对话记录：**")
    c.execute("SELECT user_input, ai_response FROM conversations")
    data = c.fetchall()
    if data:
        for idx, (user_msg, ai_msg) in enumerate(data):
            st.write(f"**您 {idx + 1}：** {user_msg}")
            st.write(f"**AI 助手 {idx + 1}：** {ai_msg}")
    else:
        st.write("暂无历史对话。")


