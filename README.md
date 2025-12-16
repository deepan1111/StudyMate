# 📚 StudyMate

> **Your AI-powered study companion** – upload notes, ask questions, analyze content, and learn smarter.

![StudyMate Banner](https://dummyimage.com/1200x400/0f172a/ffffff\&text=StudyMate+%7C+Learn+Smarter)

---

## 🚀 What is StudyMate?

🔗 GitHub: [https://study-mate-arqq.vercel.app/](https://study-mate-arqq.vercel.app/)

**StudyMate** is a modern web platform designed to help students learn efficiently using AI. Upload PDFs or study materials, ask questions, get summaries, and analyze content in seconds.

Whether you're preparing for **placements, exams, or self-learning**, StudyMate acts like your personal tutor.

---

## ✨ Key Features

* 📄 **PDF Upload & Analysis** – Upload notes and extract insights instantly
* 🤖 **AI Q&A** – Ask questions directly from your study material
* 🧠 **Smart Summaries** – Get concise explanations and key points
* 🔍 **Content Understanding** – Deep analysis, not just keyword search
* ⚡ **Fast & Scalable API** – Built with performance in mind
* 🌐 **Clean UI** – Simple, student-friendly interface

---

## 🛠️ Tech Stack

### Frontend

* ⚛️ React
* 🎨 Tailwind CSS
* 🌐 Axios

### Backend

* 🐍 FastAPI
* 🧠 LLM-powered processing
* 📄 PDF Parsing & Text Extraction

### Deployment

* ☁️ Render

---

## 📂 Project Structure

```bash
studymate/
├── frontend/        # React application
├── backend/         # FastAPI server
│   ├── routes/
│   ├── services/
│   ├── pdf_processor/
│   └── main.py
├── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/studymate.git
cd studymate
```

### 2️⃣ Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3️⃣ Frontend Setup

```bash
cd frontend
npm install
npm start
```

---

## 🧪 API Example

```http
POST /analyze-pdf
Content-Type: multipart/form-data
```

Response:

```json
{
  "summary": "This document explains...",
  "insights": ["Point 1", "Point 2"]
}
```

---

## 📸 Screenshots

<img width="1903" height="900" alt="Screenshot 2025-12-16 153216" src="https://github.com/user-attachments/assets/dcca102e-0c51-4731-a072-fa69da751afb" />
<img width="1866" height="866" alt="Screenshot 2025-12-16 153237" src="https://github.com/user-attachments/assets/ba423494-35b0-46a8-bb74-c7ffc79b3cbc" />
<img width="1750" height="873" alt="Screenshot 2025-12-16 153259" src="https://github.com/user-attachments/assets/923ab8a1-bc86-482c-990b-d10c80c43ae2" />
<img width="1871" height="888" alt="Screenshot 2025-12-16 153313" src="https://github.com/user-attachments/assets/791151db-8e0e-4103-9113-a326ea7ca1ab" />
<img width="1845" height="867" alt="Screenshot 2025-12-16 153328" src="https://github.com/user-attachments/assets/33004d74-f160-401c-b6dc-840b1c8d88ac" />


---

## 🎯 Use Cases

* 📖 Exam & Placement Preparation
* 🧑‍🎓 College Notes Understanding
* 📑 Research Paper Analysis
* 🤯 Last-minute Revision

---

## 🧩 Future Enhancements

* 🔐 User Authentication (JWT)
* 🗂️ Notes History & Dashboard
* 🧠 Multiple AI Models Support
* 📱 Mobile-friendly UI

---

## 👨‍💻 Author

**Deepan B**
🎓 Engineering Student | 💻 Full Stack Developer
🔗 GitHub: [https://github.com/deepan1111](https://github.com/deepan1111)

---

## ⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub — it motivates me to build more!

---

> *“Study smarter, not harder.” – StudyMate*
