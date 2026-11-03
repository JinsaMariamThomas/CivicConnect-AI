# 🌍 CivicConnect AI
### AI-Powered Multilingual Citizen Grievance Chatbot

**Neural Storm 2026 | Problem Statement: HI-02 | Team: Jinsa**

🔗 **Live Demo:** https://civicconnect-ai.streamlit.app/

## 📌 About the Project

CivicConnect AI is an AI-powered citizen grievance management platform that helps people report community problems such as water shortages, damaged roads, electricity failures and waste management issues.

The platform uses Natural Language Processing (NLP) to understand citizen complaints, identify the relevant issue category, suggest the responsible department and assign a priority level.

It aims to make public grievance reporting easier, faster and more accessible through multilingual support.

## 🎯 Problem Statement

Citizens often face difficulties reporting local community problems and reaching the correct government department. Language barriers and a lack of transparent complaint tracking can make the process more challenging.

CivicConnect AI addresses these challenges through an intelligent and user-friendly complaint management system.

## ✨ Key Features

- 🤖 AI-powered grievance classification
- 🌐 Multilingual complaint input
- 🏢 Automatic department suggestions
- 🚨 Complaint priority identification
- 📝 Easy complaint submission
- 🔍 Complaint tracking using unique IDs
- 📊 Community analytics dashboard
- 📋 View and manage all reports
- 🔄 Complaint status updates

## 🛠️ Technologies Used

- **Python** – Main programming language
- **Streamlit** – Web application framework
- **Scikit-learn** – Machine learning and text classification
- **TF-IDF** – Text feature extraction
- **LinearSVC** – Complaint classification model
- **Langdetect** – Language detection
- **Deep Translator** – Translation support
- **Pandas** – Data processing
- **Matplotlib** – Data visualization
- **SQLite** – Complaint database

## ⚙️ How It Works

1. A citizen enters a complaint in the chatbot or submission form.
2. The system detects the language.
3. The NLP model analyses the complaint text.
4. The model predicts the complaint category.
5. The platform suggests the responsible department and priority.
6. A unique tracking ID is generated.
7. Citizens can track the complaint status.
8. Administrators can review and update reports.

## 🏷️ Supported Complaint Categories

- Water
- Roads
- Electricity
- Waste Management
- Public Safety

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/JinsaMariamThomas/CivicConnect-AI.git
```

### 2. Open the project folder

```bash
cd CivicConnect-AI
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 🌐 Live Application

https://civicconnect-ai.streamlit.app/

## 🔮 Future Improvements

- Integration with official government grievance portals
- Persistent cloud database for online complaint storage
- Improved multilingual NLP models
- Real-time complaint notifications
- Location-based complaint mapping
- Administrative dashboard with role-based access

## 👩‍💻 Project Information

**Project:** CivicConnect AI  
**Event:** Neural Storm 2026  
**Problem ID:** HI-02  
**Track:** Human Impact  
**Team:** Jinsa

---

*Making community problem reporting smarter, simpler and more accessible.*
