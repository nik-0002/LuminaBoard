# 🌾 LUMINA BOARD — Enhanced Command Center

> **AI-Powered Multilingual Agricultural Marketing, Live Threat Mapping & Automated Outreach Platform**  
> Context-aware campaign generation and automated multi-channel messaging for farmers across India.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-green.svg)](https://flask.palletsprojects.com/)
[![Gemini](https://img.shields.io/badge/AI--Engine-Google%20Gemini%201.5%20Flash-orange.svg)](https://aistudio.google.com)
[![Leaflet](https://img.shields.io/badge/Mapping-Leaflet.js-green.svg)](https://leafletjs.com/)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features & How to Use Them](#-key-features--how-to-use-them)
  - [1. 🗺️ Interactive Geospatial Threat Map](#1--interactive-geospatial-threat-map)
  - [2. ⚙️ Settings & API Configurator](#2--settings--api-configurator)
  - [3. 🤖 AI Multilingual Campaign Generator](#3--ai-multilingual-campaign-generator)
  - [4. 🎙️ Vernacular Audio Synthesis](#4--vernacular-audio-synthesis)
  - [5. 📱 Multi-Channel Messaging Orchestrator](#5--multi-channel-messaging-orchestrator)
  - [6. ⚡ Bio-Urgency Automated Advisory Scan](#6--bio-urgency-automated-advisory-scan)
- [System Architecture](#-system-architecture)
- [Getting Started](#-getting-started)
- [License](#-license)

---

## 🌟 Overview

**Lumina Board** is a state-of-the-art agricultural marketing platform and command center. It transforms generic communications into hyper-personalized, context-aware farmer advisories. Built for rural outreach, it leverages AI to generate rich multilingual text and audio advisories tailored to each farmer's crop, region, growth stage, and current pest/disease threats.

---

## 🚀 Key Features & How to Use Them

### 1. 🗺️ Interactive Geospatial Threat Map
Transform your dashboard into a live intelligence command center. The map visualizes agricultural threats and campaign outreach across India in real-time.
- **Features**: 
  - **Pulsating Heat Spots (🔴/🟠)**: Indicates regions with active biological threats (e.g., pests, drought). The intensity of the pulse matches the urgency score.
  - **Campaign Markers (📍)**: Teal pins denote regions where outreach campaigns have been deployed.
  - **Clickable Tooltips**: Click any marker to view specific data (Crop type, Urgency Score, Estimated Farmer Reach).
- **How to Use**: 
  1. Click the **🗺️ Map View** tab in the top navigation bar.
  2. Pan and zoom across the "Dark Matter" basemap of India.
  3. Click on the glowing circles and pins to inspect live threat data.

### 2. ⚙️ Settings & API Configurator
Securely manage your API keys without exposing secrets in your codebase.
- **Features**: 
  - Allows you to save your **Fast2SMS API Key** and **Google Gemini API Key** directly in the browser's local storage.
  - Features a beautiful, premium glassmorphic **User Profile** interface showcasing your Administrator status.
- **How to Use**: 
  1. Click the **⚙️ Settings** tab in the top navigation bar.
  2. Enter your Fast2SMS API key (required to dispatch live SMS to Indian numbers).
  3. Enter your Gemini API key (optional; overrides the system default).
  4. Click **💾 Save Configurations**. The keys are instantly synced with the backend API and will persist across browser reloads.

### 3. 🤖 AI Multilingual Campaign Generator
Instantly draft culturally relevant, persuasive marketing advisories using Gemini AI.
- **Features**: 
  - Generates paragraphs (100–200 words) matching the crop, state, and specific threat context.
  - Automatically translates into 10+ Indian Regional Languages in native scripts: Hindi, Telugu, Marathi, Punjabi, Tamil, Kannada, Bengali, Gujarati, Odia, and English.
- **How to Use**: 
  1. Click the **🚀 Campaign** tab.
  2. Fill out the form fields: Campaign Type, State (e.g., "Andhra Pradesh"), Product (e.g., "Amistar Top"), and Crop.
  3. Select your target languages and click **Generate Messages**.
  4. The AI will return fully written, formatted SMS scripts and WhatsApp messages instantly.

### 4. 🎙️ Vernacular Audio Synthesis
Not all farmers read text advisories. Lumina Board converts generated text scripts into highly accurate vernacular audio streams.
- **Features**: 
  - Backend integration using `gTTS` to stream native `.mp3` audio directly to the browser.
  - Features an embedded HTML5 audio player inside every generated campaign card.
- **How to Use**: 
  1. Generate a campaign in the **🚀 Campaign** tab.
  2. In the resulting campaign cards, look for the **🎧 Play Audio** embedded control bar.
  3. Click Play to listen to the exact regional pronunciation of the advisory.

### 5. 📱 Multi-Channel Messaging Orchestrator
Execute bulk dispatches to custom numbers or CSV uploads.
- **Features**: 
  - **Fast2SMS Integration**: Sends live SMS directly to Indian numbers (`+91`). *(Requires ₹100 minimum wallet transaction on Fast2SMS)*.
  - **WhatsApp Direct**: Provides 1-click `wa.me` links to send the generated advisory (with visual product banners) directly via WhatsApp Web/Desktop.
  - **CSV Dispatch**: Drop a CSV of phone numbers to automatically iterate and dispatch campaigns.
- **How to Use**: 
  1. In the **🚀 Campaign** tab, enter a test phone number in the custom phone field (e.g., `8978518496`).
  2. Click **WhatsApp** to open the chat window, or click **Device SMS** to open your phone's native SMS app.
  3. Click **API Send** to trigger the Fast2SMS gateway routing through the backend.

### 6. ⚡ Bio-Urgency Automated Advisory Scan
An automated background engine that scans regional data for anomalies and triggers alerts.
- **Features**:
  - Classifies threats (High, Medium, Low) and recommends immediate product interventions.
- **How to Use**:
  1. Click the **Threats** tab in the top navigation bar.
  2. Review the detected threats in the left panel.
  3. Click "Trigger Alert Broadcast" to have the system automatically draft and dispatch a campaign for that specific threat.

---

## 🛠️ System Architecture

```text
Lumina Board Dashboard (Frontend - HTML5/CSS3/Vanilla JS)
   │
   ├── API Endpoints (Flask Backend)
   │     ├── Google Gemini 1.5 Flash (Multilingual Generation)
   │     ├── Audio Stream Synthesis Endpoint (gTTS)
   │     └── Mock Geospatial Metric Engine
   │
   └── Messaging Orchestrator
         ├── Fast2SMS Free Gateway
         ├── WhatsApp Direct Channel
         └── Automated Message Dispatcher
```

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

> **Tip:** Start by visiting the **⚙️ Settings** tab to configure your Fast2SMS API key!

---

## 📄 License

**Proprietary License**  
Copyright © 2026 Lumina Board. All rights reserved.