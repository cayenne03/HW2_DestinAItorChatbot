import gradio as gr
import requests

def format_message(text):
    """Ensures proper formatting with newlines."""
    return text.replace("\\n", "\n")

def chat_with_bot(message, history):
    """Send user message to RASA and retrieve response."""
    rasa_url = "http://localhost:5005/webhooks/rest/webhook"
    response = requests.post(
        rasa_url,
        json={"sender": "user", "message": message}
    )

    bot_messages = [msg.get("text", "") for msg in response.json()] if response.json() else ["Sorry, I didn't understand that."]
    formatted_response = "\n\n".join(map(format_message, bot_messages))

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": formatted_response})
    
    return "", history

def fetch_first_message():
    """Automatically sends 'hi' to RASA when the app loads."""
    rasa_url = "http://localhost:5005/webhooks/rest/webhook"
    response = requests.post(
        rasa_url,
        # send "hi" to get the first response
        json={"sender": "user", "message": "hi"}
    )
    
    bot_messages = [msg.get("text", "") for msg in response.json()] if response.json() else ["Welcome! How can I assist you?"]
    formatted_response = "\n\n".join(map(format_message, bot_messages))
    
    return [{"role": "assistant", "content": formatted_response}]

with gr.Blocks(css="footer {visibility: hidden}") as demo:
    with gr.Column(variant="panel"):
        gr.Markdown(
            """
            # 🏝️ DestinAItor - Your AI Travel Companion ✈️🌍
            **Plan your perfect trip & discover amazing places!**
            """,
            elem_id="title"
        )

        chatbot = gr.Chatbot(
            height=600, 
            type="messages",
            elem_classes="dark-chatbot",
            render_markdown=True
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your message here...", 
                label="Your Message",
                container=False,
                scale=7
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        clear = gr.Button("Clear Chat")

        send_btn.click(chat_with_bot, [msg, chatbot], [msg, chatbot])
        msg.submit(chat_with_bot, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], None, chatbot, queue=False)
        demo.load(fetch_first_message, outputs=chatbot)


    gr.Markdown("""
    <style>
    body {
        background-color: #121212 !important;
        color: white !important;
    }
    #title {
        text-align: center;
        font-size: 22px;
        color: #f4f4f4;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .dark-chatbot {
        background-color: #1a1a1a !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px;
    }
    .message {
        background-color: #2d2d2d !important;
        color: white !important;
        padding: 10px !important;
        border-radius: 8px !important;
        font-family: "Arial", sans-serif;
        line-height: 1.5;
    }
    .gradio-container {
        background-color: #121212 !important;
        color: white !important;
    }
    .message-wrap {
        max-width: none !important;
        width: 100% !important;
    }
    </style>
    """)

if __name__ == "__main__":
    demo.launch(share=False)
