import gradio as gr
import requests
import json
from typing import Union, Dict, List, Any, Tuple

def format_message(content: Union[str, Dict[str, Any]]) -> str:
    """
    Formats message content, handling both text and image carousels.
    """
    if isinstance(content, str):
        return (content.replace("\\n", "\n"))
    
    if isinstance(content, dict) and content.get("type") == "image_carousel":
        images = content.get("images", [])
        if not images:
            return ""
        
        return f"""<div style="display: flex; flex-direction: row; overflow-x: scroll; gap: 10px; padding: 20px 0; max-width: 100%; scroll-behavior: smooth; -ms-overflow-style: none; scrollbar-width: none;">
            {"".join([
                f'<img src="{img}" style="width: 150px; height: 150px; object-fit: cover; border-radius: 4px; flex-shrink: 0;">'
                for img in images
            ])}
        </div>"""
    
    return str(content)


def chat_with_bot(message: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:

    """Send user message to RASA and retrieve response."""
    rasa_url = "http://localhost:5005/webhooks/rest/webhook"
    response = requests.post(
        rasa_url,
        json={"sender": "user", "message": message}
    )
    
    if not response.json():
        formatted_response = "Sorry, I didn't understand that."
    else:
        # Process each message from the bot
        bot_messages = []
        for msg in response.json():
            if "text" in msg:
                bot_messages.append(format_message(msg["text"]))
            if "custom" in msg:
                # Handle custom responses with images
                custom_data = msg["custom"]
                if isinstance(custom_data, str):
                    try:
                        custom_data = json.loads(custom_data)
                    except json.JSONDecodeError:
                        continue
                
                if custom_data.get("type") == "image_carousel":
                    bot_messages.append(format_message(custom_data))
        
        formatted_response = "\n\n".join(bot_messages)
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": formatted_response})
    
    return "", history



# def chat_with_bot(message: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
#     """Send user message to RASA and retrieve response."""
#     test_response = [
#         {"text": "I found some great hotels in Athens for you! Here are some photos of Electra Metropolis:"},
#         {
#             "custom": {
#                 "type": "image_carousel",
#                 "images": [
#                     "https://media-cdn.tripadvisor.com/media/photo-l/2b/51/48/63/electra-metropolis-athens.jpg",
#                     "https://media-cdn.tripadvisor.com/media/photo-l/11/54/8a/24/deluxe-acropolis-view.jpg",
#                     "https://media-cdn.tripadvisor.com/media/photo-l/11/54/8c/50/superior-room.jpg",
#                     "https://media-cdn.tripadvisor.com/media/photo-l/15/33/f9/4c/athens.jpg"
#                 ]
#             }
#         },
#         {"text": "I found thing else for you:"},
#         {
#             "custom": {
#                 "type": "image_carousel",
#                 "images": [
#                     "https://media-cdn.tripadvisor.com/media/photo-l/15/33/e8/fa/athens.jpg",
#                 ]
#             }
#         }
#     ]
#
#     response_data = test_response
#
#     if not response_data:
#         formatted_response = "Sorry, I didn't understand that."
#     else:
#         bot_messages = []
#         for msg in response_data:
#             if "text" in msg:
#                 bot_messages.append(format_message(msg["text"]))
#             if "custom" in msg:
#                 custom_data = msg["custom"]
#                 if isinstance(custom_data, str):
#                     try:
#                         custom_data = json.loads(custom_data)
#                     except json.JSONDecodeError:
#                         continue
#
#                 if custom_data.get("type") == "image_carousel":
#                     # Limit to 3 images
#                     images = custom_data.get("images", [])[:3]
#                     custom_data["images"] = images
#                     bot_messages.append(format_message(custom_data))
#
#         formatted_response = "\n".join(bot_messages)
#
#     history.append({"role": "user", "content": message})
#     history.append({"role": "assistant", "content": formatted_response})
#
#     return "", history


def fetch_first_message() -> List[Dict[str, str]]:
    """Automatically sends 'hi' to RASA when the app loads."""
    rasa_url = "http://localhost:5005/webhooks/rest/webhook"
    response = requests.post(
        rasa_url,
        json={"sender": "user", "message": "hi"}
    )
    
    bot_messages = []
    if response.json():
        for msg in response.json():
            if "text" in msg:
                bot_messages.append(format_message(msg["text"]))
            if "custom" in msg:
                try:
                    custom_data = json.loads(msg["custom"]) if isinstance(msg["custom"], str) else msg["custom"]
                    if custom_data.get("type") == "image_carousel":
                        bot_messages.append(format_message(custom_data))
                except json.JSONDecodeError:
                    continue
    else:
        bot_messages = ["Welcome! How can I assist you?"]
    
    formatted_response = "\n\n".join(bot_messages)
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
            height=750,
            type="messages",
            elem_classes="dark-chatbot",
            render_markdown=True,
            bubble_full_width=True,
            sanitize_html=False  # Important for HTML spaces to work
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
            font-family: "Arial", sans-serif !important;
            line-height: 1.5 !important;
            white-space: pre-wrap !important;  /* Added this line */
        }
        
        .gradio-container {
            background-color: #121212 !important;
            color: white !important;
        }
        </style>
        """)

if __name__ == "__main__":
    demo.launch(share=False)