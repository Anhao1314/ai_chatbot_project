import streamlit as st

st.title("个性化成长型对话 AI 助手")

st.write("欢迎使用个性化成长型对话 AI 助手。请在下面的输入框中输入您的消息。")

user_input = st.text_input("您：", "")

if st.button("发送"):
    if user_input:
        st.write(f"AI 助手：{user_input}")
    else:
        st.write("请输入您的消息。")


