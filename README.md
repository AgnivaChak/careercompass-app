# 🚀 CareerCompass - Resume & JD Matchmaking App

[![Live Demo](https://img.shields.io/badge/Live-Demo-green?style=for-the-badge&logo=render)](https://careercompass-app.onrender.com)
[![Built with Python](https://img.shields.io/badge/Built%20with-Python-blue.svg?style=for-the-badge&logo=python)](#tech-stack)
[![License](https://img.shields.io/badge/License-MIT-informational?style=flat-square)](#license)

CareerCompass is a smart, interactive web app that helps candidates compare their **Resume** against any **Job Description (JD)** and discover:
- ✅ Skill Match %
- 📉 Missing skills
- 💡 Recommended projects to improve alignment

This project was designed as a **personal job-seeking assistant** to impress ATS systems and interviewers — and now it's open-sourced for others to benefit too!

---

## 🌟 Features

| Capability                      | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| 📄 Resume Upload                | Upload your resume (PDF) for analysis                                      |
| 📋 JD Input                     | Paste or upload job descriptions                                           |
| 🧠 Skill Extraction             | NLP-based keyword extraction from resume and JD                            |
| 📊 Match % Calculation          | Uses TF-IDF + cosine similarity for match scoring                          |
| ❌ Missing Skills Detection     | Highlights the gaps in your resume                                         |
| 🔍 Smart Project Recommender   | Suggests tailored projects from a JSON knowledge base                      |
| 📈 Visual Insights              | Live heatmaps and gauge charts for visual understanding                    |

---

## 🧱 Project Architecture

```bash
careercompass_project/
├── app/
│   ├── jd_parser.py          # JD keyword extractor
│   ├── matcher.py            # Match % calculator
│   ├── recommender.py        # Recommends projects
│   ├── resume_parser.py      # Resume keyword extractor
│   └── utils.py              # Reusable NLP helpers
│
├── assets/
│   └── project_ideas.json    # 250 curated project suggestions
│
├── main.py                   # 🚀 Streamlit app entry point
├── requirements.txt          # All dependencies
├── runtime.txt               # Python version pin
├── README.md                 # You're reading it!
└── .gitignore
