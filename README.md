# 🌾 LUMINA BOARD

> **AI-Powered Multilingual Agricultural Marketing & Automated Outreach Platform**  
> Context-aware campaign generation and automated multi-channel messaging for farmers across India.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-green.svg)](https://flask.palletsprojects.com/)
[![Gemini](https://img.shields.io/badge/AI--Engine-Google%20Gemini%201.5%20Flash-orange.svg)](https://aistudio.google.com)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [API Documentation](#-api-documentation)
- [Getting Started](#-getting-started)
- [License](#-license)

---

## 🌟 Overview

**Lumina Board** is an intelligent agricultural marketing platform that transforms generic communications into hyper-personalized, context-aware farmer advisories. Built for rural outreach, it leverages AI to generate rich multilingual paragraph advisories tailored to each farmer's crop, region, growth stage, and current pest/disease threats—scaling seamlessly across 10+ Indian regional languages without manual effort.

### Why Lumina Board?

Traditional agricultural marketing faces key challenges:
- Farmers in different states face distinct regional, climatic, and pest challenges.
- Outbreak alerts must be timely, geography-specific, and action-oriented.
- Content must be delivered in native vernacular scripts with visual product banners.
- Campaigns must scale to thousands of recipients while feeling personal and trusted.

**Lumina Board solves this completely.**

---

## 🚀 Key Features

### 1. 🤖 Instant Multilingual AI Advisory Generation
- Powered by **Google Gemini AI** for instant (<0.5 sec) generation of rich, persuasive marketing paragraph advisories (100–200 words).
- Supports **10+ Indian Regional Languages** in native scripts: **Hindi (हिंदी)**, **Telugu (తెలుగు)**, **Marathi (मराठी)**, **Punjabi (ਪੰਜਾਬੀ)**, **Tamil (தமிழ்)**, **Kannada (ಕನ್ನಡ)**, **Bengali (বাংলা)**, **Gujarati (ગુજરાતી)**, **Odia (ଓଡ଼ିଆ)**, and **English**.
- Generates matching **30–45 second IVR Voice Scripts** with audio cues (`[Intro Music]`, `[Outro Music]`).

### 2. 📱 Multi-Channel Messaging Orchestrator
- **Fast2SMS Integration**: Direct free SMS delivery to Indian mobile numbers (`+91`).
- **WhatsApp Direct & ADB**: Instant 1-click `wa.me` links with visual media product banners attached.
- **Automated Fallback**: Seamless fallback across Fast2SMS, MSG91, WhatsApp, and simulation mode for zero-interruption bulk dispatches.

### 3. 👥 Recipient Phone Numbers & CSV Upload Dispatch
- **Custom Phone Numbers**: Send directly to specified phone numbers.
- **Customer CSV File Upload (`/api/sms/send-campaign-csv`)**: Drag & drop any CSV file containing customer phone numbers. Lumina Board automatically extracts all numbers and dispatches AI campaigns instantly.

### 4. ⚡ Bio-Urgency Automated Advisory Scan
- Background worker scans crop disease outbreak risks (`UrgencyDetector`).
- Automatically triggers vernacular advisory dispatches to farmers in high-risk districts without manual intervention.

---

## 🛠️ System Architecture

```
Lumina Board Dashboard (Frontend)
   │
   ├── API Endpoints (Flask App)
   │     ├── Google Gemini 1.5 Flash AI Engine (Multilingual Generation)
   │     ├── CSV Dataset Processor & RAG Engine
   │     └── Urgency Detection Classifier
   │
   └── Messaging Orchestrator
         ├── Fast2SMS Free Gateway
         ├── WhatsApp Direct & ADB Channel
         └── Automated Message Dispatcher
```

---

## 💻 Tech Stack

- **Backend**: Python 3.9+, Flask, Pandas, Scikit-Learn
- **AI / LLM Engine**: Google Gemini 1.5 Flash API
- **SMS & WhatsApp Messaging**: Fast2SMS, WhatsApp Web / ADB, MSG91
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+), Glassmorphism Design System

---

## 📡 API Documentation

### Messaging & Campaign Endpoints

- `POST /api/sms/send-campaign`: Send campaign messages to growers or custom target phone numbers.
- `POST /api/sms/send-campaign-csv`: Upload a CSV file containing phone numbers for bulk campaign dispatch.
- `POST /api/sms/send-free`: Send a direct SMS via Fast2SMS / Simulation.
- `POST /api/whatsapp/send-free`: Send a direct WhatsApp advisory with a visual media banner.
- `POST /api/messaging/auto-dispatch`: Trigger immediate bio-urgency scanning and automated alert broadcast.
- `GET /api/messaging/status`: Check active messaging channel and gateway statuses.
- `GET /api/messaging/history`: Retrieve recent delivery log records.

---

## 🚦 Getting Started

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Backend API Server

```bash
PORT=8000 python api/app.py
```

### 3. Access Dashboard

Open `http://localhost:8000/index.html` in your web browser.

---

## 📄 License

**Proprietary License**  
Copyright © 2026 Lumina Board. All rights reserved.