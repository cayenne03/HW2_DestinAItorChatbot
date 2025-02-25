# (OBSOLETE)! THIS FILE NEEDS TO BE UPDATED (See DestinAItor_VisualPresentation.png)

# 🌍 DestinAItor - Your AI Travel Companion ✈️  

Welcome to **DestinAItor**, an intelligent travel assistant built with **RASA**! 🚀  
Plan trips, discover amazing places, and get real-time travel recommendations effortlessly.  

---

## 📌 1️⃣ Features
✅ AI-powered **trip planning** for any city or country  
✅ Smart **recommendations** for hotels, restaurants, and attractions  
✅ **Real-time data** from APIs (TripAdvisor, etc.)  
✅ Seamless integration with **RASA** (database storage: under construction)  
✅ Interactive chatbot for **engaging travel conversations**  

---

## 🚀 Getting Started  

## 📌 2️⃣ Installation & Setup  

### **Clone the Repository** 
```bash
git clone https://github.com/cayenne03/HW2_DestinAItorChatbot.git
cd HW2_DestinAItorChatbot
```

### **Create and Activate a Virtual Environment (Optional but Recommended)**
```bash
# For Conda users
conda create -n rasa310 python=3.10
conda activate rasa310

# For venv users
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Train the RASA Model**
```bash
rasa train
```

## 📌 3️⃣ Running the Chatbot

### **Start the RASA Chatbot**
```bash
rasa run --enable-api
```

### **Start the Action Server**
```bash
rasa run actions
```


### **Chat with the Travel Bot!**
```bash
rasa shell
```

## 📌 4️⃣ API Integration

To set up API keys, add them to a .env file:

```bash
TRIPADVISOR_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## 📌 5️⃣ Database Setup

Run the docker-compose.yml file

```bash
docker-compose up
```

