"""
Lumina Board - Multilingual Campaign Message Generator
Generates SMS messages and audio scripts for Indian agricultural campaigns
in multiple Indian languages using Qwen2.5 LLM with template fallback.

Languages supported:
Hindi, Marathi, Punjabi, Telugu, Tamil, Kannada, Bengali, Gujarati,
Odia, Malayalam, Assamese, English

No hallucination: product/crop/state data comes from CSV, language selection
from actual grower language distribution.
"""

import os
import logging
from typing import Dict, List, Optional
import requests

logger = logging.getLogger("lumina.messaging")

# ─── Language Config ──────────────────────────────────────────────────────────
LANGUAGE_META = {
    "Hindi": {
        "code": "hi",
        "script": "Devanagari",
        "greeting": "नमस्ते किसान भाई",
        "sms_limit": 160,
        "audio_wpm": 90  # words per minute for audio scripts
    },
    "Marathi": {
        "code": "mr",
        "script": "Devanagari",
        "greeting": "नमस्कार शेतकरी बंधू",
        "sms_limit": 160,
        "audio_wpm": 85
    },
    "Punjabi": {
        "code": "pa",
        "script": "Gurmukhi",
        "greeting": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਕਿਸਾਨ ਜੀ",
        "sms_limit": 160,
        "audio_wpm": 90
    },
    "Telugu": {
        "code": "te",
        "script": "Telugu",
        "greeting": "నమస్కారం రైతు అన్నా",
        "sms_limit": 160,
        "audio_wpm": 85
    },
    "Tamil": {
        "code": "ta",
        "script": "Tamil",
        "greeting": "வணக்கம் விவசாயி நண்பரே",
        "sms_limit": 160,
        "audio_wpm": 85
    },
    "Kannada": {
        "code": "kn",
        "script": "Kannada",
        "greeting": "ನಮಸ್ಕಾರ ರೈತ ಬಂಧು",
        "sms_limit": 160,
        "audio_wpm": 80
    },
    "Bengali": {
        "code": "bn",
        "script": "Bengali",
        "greeting": "নমস্কার কৃষক ভাই",
        "sms_limit": 160,
        "audio_wpm": 90
    },
    "Gujarati": {
        "code": "gu",
        "script": "Gujarati",
        "greeting": "નમસ્તે ખેડૂત ભાઈ",
        "sms_limit": 160,
        "audio_wpm": 85
    },
    "Odia": {
        "code": "or",
        "script": "Odia",
        "greeting": "ନମସ୍କାର ଚାଷୀ ଭାଇ",
        "sms_limit": 160,
        "audio_wpm": 80
    },
    "Malayalam": {
        "code": "ml",
        "script": "Malayalam",
        "greeting": "നമസ്കാരം കർഷക സഹോദരാ",
        "sms_limit": 160,
        "audio_wpm": 80
    },
    "English": {
        "code": "en",
        "script": "Latin",
        "greeting": "Dear Farmer",
        "sms_limit": 160,
        "audio_wpm": 130
    }
}

# ─── Campaign Templates (fallback when LLM unavailable) ──────────────────────
TEMPLATES = {
    "product_launch": {
        "Hindi": (
            "{greeting},\n"
            "{state} के {crop} किसानों के लिए Lumina Board लाया है एक क्रांतिकारी फसल सुरक्षा समाधान — {product}। "
            "{crop} की फसल में फफूंद जनित रोगों और कीटों का खतरा हमेशा बना रहता है। "
            "{product} अपनी उन्नत सिस्टमिक तकनीक से पौधे को अंदर से मजबूत बनाता है और लंबे समय तक सुरक्षा प्रदान करता है। "
            "अनुशंसित मात्रा में सही समय पर छिड़काव करें और अपनी फसल की पैदावार बढ़ाएं। "
            "अधिक जानकारी के लिए आज ही अपने निकटतम अधिकृत डीलर से संपर्क करें।\n"
            "Lumina Board — आपका भरोसेमंद खेती साथी।"
        ),
        "Telugu": (
            "{greeting},\n"
            "{state}లోని {crop} రైతులకు Lumina Board అందించే విప్లవాత్మక పంట సంరక్షణ పరిష్కారం — {product}। "
            "{crop} పంటలో తెగుళ్లు మరియు సిలీంధ్ర వ్యాధుల ముప్పు నివారణకు {product} అద్భుతంగా పనిచేస్తుంది. "
            "ఇది మొక్కకు లోపలి నుండి బలమైన రక్షణనిచ్చి పంట దిగుబడిని మరియు నాణ్యతను విశేషంగా పెంచుతుంది. "
            "ఎకరాకు సిఫార్సు చేసిన మోతాదులో సమయానుకూలంగా పిచికారీ చేయండి. "
            "మరిన్ని వివరాలకు నేడే మీ సమీప డీలర్‌ను సంప్రదించండి.\n"
            "Lumina Board — మీ నమ్మకమైన వ్యవసాయ భాగస్వామి."
        ),
        "Marathi": (
            "{greeting},\n"
            "{state} मधील {crop} उत्पादक शेतकऱ्यांसाठी Lumina Board घेऊन आले आहे अत्यंत प्रभावी औषध — {product}। "
            "पिकावरील बुरशीजन्य रोग आणि कीटकांच्या प्रादुर्भावावर {product} त्वरित आणि दीर्घकाळ टिकणारे संरक्षण प्रदान करते. "
            "योग्य वेळी आणि शिफारस केलेल्या मात्रेत फवारणी करून पिकाची गुणवत्ता सुधारा. "
            "अधिक माहितीसाठी व खरेदीसाठी आजच तुमच्या जवळच्या अधिकृत डीलरशी संपर्क साधा.\n"
            "Lumina Board — शेतकऱ्यांचा विश्वासू सोबती."
        ),
        "Punjabi": (
            "{greeting},\n"
            "{state} ਦੇ {crop} ਕਿਸਾਨ ਵੀਰੋ, ਆਪਣੀ ਫਸਲ ਨੂੰ ਬੀਮਾਰੀਆਂ ਤੋਂ ਬਚਾਉਣ ਲਈ Lumina Board ਲਿਆਇਆ ਹੈ ਨਵਾਂ ਉਤਪਾਦ — {product}। "
            "{product} ਫਸਲ ਨੂੰ ਫੰਗਸ ਅਤੇ ਬੀਮਾਰੀਆਂ ਤੋਂ ਅੰਦਰੋਂ ਮਜ਼ਬੂਤ ਸੁਰੱਖਿਆ ਦਿੰਦਾ ਹੈ ਅਤੇ ਫਸਲ ਦਾ ਵਿਕਾਸ ਵਧਾਉਂਦਾ ਹੈ। "
            "ਸਹੀ ਸਮੇਂ 'ਤੇ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਖੁਰਾਕ ਅਨੁਸਾਰ ਛਿੜਕਾਅ ਕਰੋ। "
            "ਵਧੇਰੇ ਜਾਣਕਾਰੀ ਲਈ ਅੱਜ ਹੀ ਆਪਣੇ ਨੇੜਲੇ ਡੀਲਰ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।\n"
            "Lumina Board — ਕਿਸਾਨਾਂ ਦਾ ਭਰੋਸੇਯੋਗ ਸਾਥੀ।"
        ),
        "Tamil": (
            "{greeting},\n"
            "{state} மாநில {crop} விவசாயிகளுக்கு Lumina Board வழங்கும் சிறந்த பயிர் பாதுகாப்பு தீர்வு — {product}। "
            "{crop} பயிரில் ஏற்படும் நோய் தாக்குதல்களை கட்டுப்படுத்தி, பயிரின் வளர்ச்சியை துரிதப்படுத்தி அதிக மகசூல் பெற {product} உதவுகிறது. "
            "பரிந்துரைக்கப்பட்ட அளவில் சரியான நேரத்தில் தெளித்து பயன்பெறுங்கள். "
            "கூடுதல் தகவல்களுக்கு இன்றே உங்கள் அருகில் உள்ள Lumina Board டீலரை அணுகவும்.\n"
            "Lumina Board India — உங்கள் நம்பிக்கைக்குரிய விவசாய பங்குதாரர்."
        ),
        "Kannada": (
            "{greeting},\n"
            "{state} ರಾಜ್ಯದ {crop} ರೈತ ಬಂಧುಗಳೇ, ನಿಮ್ಮ ಬೆಳೆಯನ್ನು ರೋಗಗಳಿಂದ ರಕ್ಷಿಸಿ ಹೆಚ್ಚಿನ ಇಳುವರಿ ಪಡೆಯಲು Lumina Board ತಂದಿದೆ ಅತ್ಯುತ್ತಮ ಉತ್ಪನ್ನ — {product}। "
            "ಇದು ಬೆಳೆಗೆ ಸಂಪೂರ್ಣ ರಕ್ಷಣೆ ನೀಡಿ ಸಸ್ಯದ ಬೆಳವಣಿಗೆಯನ್ನು ಉತ್ತಮಗೊಳಿಸುತ್ತದೆ. "
            "ಸೂಕ್ತ ಸಮಯದಲ್ಲಿ ಶಿಫಾರಸು ಮಾಡಿದ ಪ್ರಮಾಣದಲ್ಲಿ ಸಿಂಪಡಿಸಿ. "
            "ಹೆಚ್ಚಿನ ಮಾಹಿತಿಗಾಗಿ ಇಂದೇ ನಿಮ್ಮ ಹತ್ತಿರದ Lumina Board ಡೀಲರ್ ಅನ್ನು ಸಂಪರ್ಕಿಸಿ.\n"
            "Lumina Board India — ನಿಮ್ಮ ನಂಬಿಕಸ್ಥ ಕೃಷಿ ಸಂಗಾತಿ."
        ),
        "Bengali": (
            "{greeting},\n"
            "{state}-এর {crop} চাষী ভাইদের জন্য Lumina Board নিয়ে এসেছে এক দুর্দান্ত ফসল সুরক্ষা সমাধান — {product}। "
            "{crop} ফসলে ছত্রাকজনিত রোগ প্রতিরোধে এবং ফলন বৃদ্ধিতে {product} অত্যন্ত কার্যকরী। "
            "সঠিক সময়ে সঠিক মাত্রায় স্প্রে করে আপনার ফসলের সুরক্ষা নিশ্চিত করুন। "
            "বিস্তারিত জানতে আজই আপনার নিকটস্থ অনুমোদিত Lumina Board ডিলারের সাথে যোগাযোগ করুন।\n"
            "Lumina Board India — আপনার বিশ্বস্ত কৃষি বন্ধু।"
        ),
        "Gujarati": (
            "{greeting},\n"
            "{state} ના {crop} ખેડૂત ભાઈઓ, તમારી પાકને રોગમુક્ત રાખવા અને બમ્પર ઉત્પાદન મેળવવા માટે Lumina Board લાવ્યું છે — {product}। "
            "આ દવા પાકને ફૂગ અને રોગો સામે લાંબા સમય સુધી રક્ષણ આપે છે. "
            "યોગ્ય સમયે ભલામણ કરેલ માત્રા મુજબ છંટકાવ કરો. "
            "વધુ માહિતી માટે આજે જ તમારા નજીકના Lumina Board ડીલરનો સંપર્ક કરો.\n"
            "Lumina Board India — તમારો વિશ્વાસુ ખેતી સાથી."
        ),
        "Odia": (
            "{greeting},\n"
            "{state}ର {crop} ଚାଷୀ ଭାଇମାନଙ୍କ ପାଇଁ Lumina Board ଆଣିଛି ଏକ ଅତ୍ୟାଧୁନିକ ଫସଲ ସୁରକ୍ଷା ସମାଧାନ — {product}। "
            "{crop} ଫସଲରେ ଛତୁ ଜନିତ ରୋଗ ଓ କୀଟ ସମସ୍ୟା ଦୂର କରିବାକୁ {product} ଅତ୍ୟନ୍ତ ପ୍ରଭାବଶାଳୀ। "
            "ଏହା ଗଛକୁ ଭିତରୁ ମଜବୁତ କରି ଦୀର୍ଘ ସମୟ ସୁରକ୍ଷା ଦେଇଥାଏ। "
            "ସଠିକ ସମୟରେ ସିଫାରିସ ମାତ୍ରାରେ ସ୍ପ୍ରେ କରନ୍ତୁ ଓ ଫସଲର ଉତ୍ପାଦନ ବଢ଼ାନ୍ତୁ। "
            "ଅଧିକ ତଥ୍ୟ ପାଇଁ ଆଜି ନିକଟତମ Lumina Board ଡିଲରଙ୍କ ସହ ଯୋଗାଯୋଗ କରନ୍ତୁ।\n"
            "Lumina Board India — ଆପଣଙ୍କ ବିଶ୍ୱସ୍ତ କୃଷି ସାଥୀ।"
        ),
        "Malayalam": (
            "{greeting},\n"
            "{state}ലെ {crop} കർഷകർക്കായി Lumina Board അവതരിപ്പിക്കുന്നു ഒരു വിപ്ലവകരമായ വിള സംരക്ഷണ പരിഹാരം — {product}. "
            "{crop} വിളയിൽ കുമിൾ രോഗങ്ങളും കീടബാധയും തടയാൻ {product} അത്യന്തം ഫലപ്രദമാണ്. "
            "ഇത് ചെടിക്ക് ഉള്ളിൽ നിന്ന് ശക്തമായ സംരക്ഷണം നൽകുകയും വിളവ് വർദ്ധിപ്പിക്കുകയും ചെയ്യുന്നു. "
            "ശുപാർശ ചെയ്ത അളവിൽ ശരിയായ സമയത്ത് തളിക്കുക. "
            "കൂടുതൽ വിവരങ്ങൾക്ക് ഇന്ന് തന്നെ അടുത്തുള്ള Lumina Board ഡീലറെ ബന്ധപ്പെടുക.\n"
            "Lumina Board India — നിങ്ങളുടെ വിശ്വസ്ത കൃഷി പങ്കാളി."
        ),
        "English": (
            "{greeting},\n"
            "Lumina Board presents an advanced crop protection solution for {crop} growers in {state} — {product}. "
            "Protect your crop against severe fungal diseases and pest attacks with the power of {product}. "
            "It offers fast systemic action, ensuring greener leaves, healthier crops, and higher yield. "
            "Apply at the recommended dosage during key growth stages for maximum disease control. "
            "Visit your nearest authorized Lumina Board dealer today or call our helpline for expert advisory.\n"
            "Lumina Board India — Your Trusted Farming Partner."
        )
    },
    "urgency_offer": {
        "Hindi": (
            "{greeting},\nसीमित समय ऑफर! {product} पर विशेष छूट।"
            " {crop} की सुरक्षा के लिए अभी संपर्क करें। ऑफर {state} में मात्र 3 दिन।\nLumina Board"
        ),
        "English": (
            "{greeting},\nLimited time offer! Special discount on {product}."
            " Protect your {crop} crop now. Offer valid in {state} for 3 days only.\nLumina Board"
        ),
        "Marathi": (
            "{greeting},\nमर्यादित वेळाची ऑफर! {product} वर विशेष सूट।"
            " {crop} संरक्षणासाठी आत्ताच संपर्क साधा. {state} मध्ये फक्त ३ दिवस.\nLumina Board"
        ),
        "Punjabi": (
            "{greeting},\nਸੀਮਤ ਸਮੇਂ ਦੀ ਪੇਸ਼ਕਸ਼! {product} 'ਤੇ ਵਿਸ਼ੇਸ਼ ਛੋਟ।"
            " {crop} ਸੁਰੱਖਿਆ ਲਈ ਹੁਣੇ ਸੰਪਰਕ ਕਰੋ। {state} ਵਿੱਚ ਸਿਰਫ਼ 3 ਦਿਨ।"
        ),
        "Telugu": (
            "{greeting},\nపరిమిత కాల ఆఫర్! {product}పై ప్రత్యేక తగ్గింపు।"
            " {crop} రక్షణకు ఇప్పుడే సంప్రదించండి। {state}లో 3 రోజులు మాత్రమే।"
        ),
        "Tamil": (
            "{greeting},\nவரையறுக்கப்பட்ட நேர சலுகை! {product}பை சிறப்பு தள்ளுபடி।"
            " {crop} பாதுகாப்பிற்கு இப்போதே தொடர்பு கொள்ளுங்கள்। {state}ல் 3 நாட்கள் மட்டுமே।"
        ),
        "Kannada": (
            "{greeting},\nಸೀಮಿತ ಸಮಯದ ಆಫರ್! {product}ಗೆ ವಿಶೇಷ ರಿಯಾಯಿತಿ।"
            " {crop} ರಕ್ಷಣೆಗಾಗಿ ಈಗಲೇ ಸಂಪರ್ಕಿಸಿ। {state}ದಲ್ಲಿ 3 ದಿನಗಳು ಮಾತ್ರ।"
        ),
        "Bengali": (
            "{greeting},\nসীমিত সময়ের অফার! {product}-এ বিশেষ ছাড়।"
            " {crop} সুরক্ষার জন্য এখনই যোগাযোগ করুন। {state}-এ মাত্র ৩ দিন।"
        ),
        "Gujarati": (
            "{greeting},\nમર્યાદિત સમય ઓફર! {product} પર વિશેષ છૂટ।"
            " {crop} સુરક્ષા માટે અત્યારે સંપર્ક કરો। {state}માં માત્ર ૩ દિવસ।"
        ),
        "Odia": (
            "{greeting},\nସୀମିତ ସମୟ ଅଫର! {product} ଉପରେ ବିଶେଷ ଛାଡ଼।"
            " {crop} ସୁରକ୍ଷା ପାଇଁ ଏବେ ଯୋଗାଯୋଗ କରନ୍ତୁ। {state}ରେ ମାତ୍ର ୩ ଦିନ।\nLumina Board"
        ),
        "Malayalam": (
            "{greeting},\nപരിമിത സമയ ഓഫർ! {product}ന് പ്രത്യേക കിഴിവ്."
            " {crop} സംരക്ഷണത്തിനായി ഇപ്പോൾ ബന്ധപ്പെടുക. {state}ൽ 3 ദിവസം മാത്രം.\nLumina Board"
        )
    },
    "season_reminder": {
        "Hindi": (
            "{greeting},\n{crop} की बुवाई का सही समय आ गया है।"
            " Lumina Board के {product} से फसल सुरक्षित रखें। {state} के किसान आज ही डीलर से मिलें।"
        ),
        "English": (
            "{greeting},\nIt's the right time to sow {crop}."
            " Protect your crop with Lumina Board's {product}. Meet your dealer in {state} today."
        ),
        "Marathi": (
            "{greeting},\n{crop} पेरणीचा योग्य वेळ आला आहे।"
            " Lumina Board च्या {product} ने पीक सुरक्षित ठेवा। {state} मधील शेतकरी आज डीलरला भेटा।"
        ),
        "Punjabi": (
            "{greeting},\n{crop} ਬੀਜਣ ਦਾ ਸਹੀ ਸਮਾਂ ਆ ਗਿਆ ਹੈ।"
            " Lumina Board ਦੇ {product} ਨਾਲ ਫਸਲ ਸੁਰੱਖਿਅਤ ਰੱਖੋ। {state} ਦੇ ਕਿਸਾਨ ਅੱਜ ਡੀਲਰ ਨਾਲ ਮਿਲੋ।"
        ),
        "Telugu": (
            "{greeting},\n{crop} విత్తడానికి సరైన సమయం వచ్చింది।"
            " Lumina Board యొక్క {product}తో పంట సురక్షితంగా ఉంచుకోండి। {state} రైతులు నేడే డీలర్‌ని కలవండి।"
        ),
        "Tamil": (
            "{greeting},\n{crop} விதைக்க சரியான நேரம் வந்துவிட்டது।"
            " Lumina Board இன் {product} மூலம் பயிரை பாதுகாக்கவும்। {state} விவசாயிகள் இன்றே டீலரை சந்தியுங்கள்।"
        ),
        "Kannada": (
            "{greeting},\n{crop} ಬಿತ್ತನೆಯ ಸರಿಯಾದ ಸಮಯ ಬಂದಿದೆ।"
            " Lumina Board ಯ {product}ನಿಂದ ಬೆಳೆ ರಕ್ಷಿಸಿಕೊಳ್ಳಿ। {state} ರೈತರು ಇಂದೇ ಡೀಲರ್ ಅನ್ನು ಭೇಟಿ ಮಾಡಿ।"
        ),
        "Bengali": (
            "{greeting},\n{crop} বপনের সঠিক সময় এসেছে।"
            " Lumina Board-র {product} দিয়ে ফসল সুরক্ষিত রাখুন। {state}-এর কৃষকরা আজই ডিলারের সাথে দেখা করুন।"
        ),
        "Gujarati": (
            "{greeting},\n{crop} વાવણીનો સાચો સમય આવ્યો છે।"
            " Lumina Board ના {product} થી પાક સુરક્ષિત રાખો। {state} ના ખેડૂતો આજે જ ડીલરને મળો।"
        ),
        "Odia": (
            "{greeting},\n{crop} ବୁଣିବାର ସଠିକ ସମୟ ଆସିଗଲା।"
            " Lumina Board ର {product} ଦ୍ୱାରା ଫସଲ ସୁରକ୍ଷିତ ରଖନ୍ତୁ। {state}ର ଚାଷୀ ଆଜି ଡିଲରଙ୍କୁ ଭେଟନ୍ତୁ।"
        ),
        "Malayalam": (
            "{greeting},\n{crop} വിതയ്ക്കാനുള്ള ശരിയായ സമയം വന്നു."
            " Lumina Board ന്റെ {product} ഉപയോഗിച്ച് വിള സംരക്ഷിക്കുക. {state}ലെ കർഷകരേ, ഇന്ന് ഡീലറെ കാണുക."
        )
    }
}

AUDIO_SCRIPT_TEMPLATES = {
    "Hindi": """
[ऑडियो स्क्रिप्ट - {duration} सेकंड]
[शुरुआत - उत्साहवर्धक संगीत]

"किसान भाइयों और बहनों, नमस्कार!

{campaign_body}

याद रखें — Lumina Board के साथ आपकी फसल, आपका भविष्य सुरक्षित है।
अधिक जानकारी के लिए अपने नजदीकी Lumina Board डीलर से संपर्क करें।

लुमिना बोर्ड इंडिया — किसान का साथी।"

[संगीत - फेड आउट]
""",
    "English": """
[AUDIO SCRIPT - {duration} seconds]
[Intro - uplifting agricultural jingle]

"Hello farmers of {state}!

{campaign_body}

Remember — with Lumina Board, your crop and future are protected.
For more information, contact your nearest Lumina Board dealer.

Lumina Board India — Your Farming Partner."

[Music - fade out]
""",
    "Marathi": """
[ऑडिओ स्क्रिप्ट - {duration} सेकंद]
[सुरुवात - उत्साहवर्धक संगीत]

"शेतकरी बंधू आणि भगिनींनो, नमस्कार!

{campaign_body}

लक्षात ठेवा — Lumina Board सोबत तुमचे पीक, तुमचे भविष्य सुरक्षित आहे.
अधिक माहितीसाठी तुमच्या जवळच्या Lumina Board डीलरशी संपर्क साधा.

लुमिना बोर्ड इंडिया — शेतकऱ्याचा साथीदार।"

[संगीत - फेड आउट]
""",
    "Punjabi": """
[ਆਡੀਓ ਸਕ੍ਰਿਪਟ - {duration} ਸਕਿੰਟ]
[ਸ਼ੁਰੂਆਤ - ਉਤਸ਼ਾਹਜਨਕ ਸੰਗੀਤ]

"ਕਿਸਾਨ ਭੈਣਾਂ ਅਤੇ ਭਰਾਵੋ, ਸਤ ਸ੍ਰੀ ਅਕਾਲ!

{campaign_body}

ਯਾਦ ਰੱਖੋ — Lumina Board ਨਾਲ ਤੁਹਾਡੀ ਫਸਲ, ਤੁਹਾਡਾ ਭਵਿੱਖ ਸੁਰੱਖਿਅਤ ਹੈ।
ਹੋਰ ਜਾਣਕਾਰੀ ਲਈ ਆਪਣੇ ਨੇੜੇ ਦੇ Lumina Board ਡੀਲਰ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।

Lumina Board India — ਕਿਸਾਨ ਦਾ ਸਾਥੀ।"

[ਸੰਗੀਤ - ਫੇਡ ਆਊਟ]
""",
    "Telugu": """
[ఆడియో స్క్రిప్ట్ - {duration} సెకన్లు]
[ప్రారంభం - ఉత్సాహపూరిత సంగీతం]

"రైతు అన్నదమ్ములారా, నమస్కారం!

{campaign_body}

గుర్తుంచుకోండి — Lumina Board తో మీ పంట, మీ భవిష్యత్తు సురక్షితం.
మరిన్ని వివరాలకు మీ సమీప Lumina Board డీలర్‌ను సంప్రదించండి.

Lumina Board India — రైతు బంధు."

[సంగీతం - ఫేడ్ అవుట్]
""",
    "Tamil": """
[ஒலிப்பதிவு - {duration} வினாடிகள்]
[தொடக்கம் - உற்சாகமான இசை]

"விவசாயி நண்பர்களே, வணக்கம்!

{campaign_body}

நினைவில் கொள்ளுங்கள் — Lumina Board உடன் உங்கள் பயிர், உங்கள் எதிர்காலம் பாதுகாப்பாக.
கூடுதல் தகவலுக்கு உங்கள் அருகிலுள்ள Lumina Board டீலரை அணுகவும்.

Lumina Board India — விவசாயியின் நம்பிக்கை."

[இசை - மெல்ல முடிவு]
""",
    "Kannada": """
[ಆಡಿಯೋ ಸ್ಕ್ರಿಪ್ಟ್ - {duration} ಸೆಕೆಂಡುಗಳು]
[ಪ್ರಾರಂಭ - ಉತ್ಸಾಹಜನಕ ಸಂಗೀತ]

"ರೈತ ಬಂಧುಗಳೇ, ನಮಸ್ಕಾರ!

{campaign_body}

ನೆನಪಿಡಿ — Lumina Board ಜೊತೆ ನಿಮ್ಮ ಬೆಳೆ, ನಿಮ್ಮ ಭವಿಷ್ಯ ಸುರಕ್ಷಿತ.
ಹೆಚ್ಚಿನ ಮಾಹಿತಿಗಾಗಿ ಹತ್ತಿರದ Lumina Board ಡೀಲರ್ ಅನ್ನು ಸಂಪರ್ಕಿಸಿ.

Lumina Board India — ರೈತನ ನಂಬಿಕಸ್ಥ ಸಂಗಾತಿ."

[ಸಂಗೀತ - ಫೇಡ್ ಔಟ್]
""",
    "Bengali": """
[অডিও স্ক্রিপ্ট - {duration} সেকেন্ড]
[শুরু - উদ্দীপনামূলক সংগীত]

"কৃষক ভাই ও বোনেরা, নমস্কার!

{campaign_body}

মনে রাখবেন — Lumina Board-র সাথে আপনার ফসল, আপনার ভবিষ্যৎ সুরক্ষিত।
আরও তথ্যের জন্য আপনার নিকটস্থ Lumina Board ডিলারের সাথে যোগাযোগ করুন।

Lumina Board India — কৃষকের সাথী।"

[সংগীত - ফেড আউট]
""",
    "Gujarati": """
[ઓડિયો સ્ક્રિપ્ટ - {duration} સેકન્ડ]
[શરૂઆત - ઉત્સાહવર્ધક સંગીત]

"ખેડૂત ભાઈઓ અને બહેનો, નમસ્તે!

{campaign_body}

યાદ રાખો — Lumina Board સાથે તમારો પાક, તમારું ભવિષ્ય સુરક્ષિત છે.
વધુ માહિતી માટે તમારા નજીકના Lumina Board ડીલરનો સંપર્ક કરો.

Lumina Board India — ખેડૂતનો સાથી."

[સંગીત - ફેડ આઉટ]
""",
    "Odia": """
[ଅଡ଼ିଓ ସ୍କ୍ରିପ୍ଟ - {duration} ସେକେଣ୍ଡ]
[ଆରମ୍ଭ - ଉତ୍ସାହଜନକ ସଙ୍ଗୀତ]

"ଚାଷୀ ଭାଇ ଓ ଭଉଣୀମାନେ, ନମସ୍କାର!

{campaign_body}

ମନେ ରଖନ୍ତୁ — Lumina Board ସହ ଆପଣଙ୍କ ଫସଲ, ଆପଣଙ୍କ ଭବିଷ୍ୟତ ସୁରକ୍ଷିତ।
ଅଧିକ ତଥ୍ୟ ପାଇଁ ନିକଟତମ Lumina Board ଡିଲରଙ୍କ ସହ ଯୋଗାଯୋଗ କରନ୍ତୁ।

Lumina Board India — ଚାଷୀଙ୍କ ସାଥୀ।"

[ସଙ୍ଗୀତ - ଫେଡ଼ ଆଉଟ]
""",
    "Malayalam": """
[ഓഡിയോ സ്ക്രിപ്റ്റ് - {duration} സെക്കൻഡ്]
[തുടക്കം - ആവേശകരമായ സംഗീതം]

"കർഷക സഹോദരങ്ങളേ, നമസ്കാരം!

{campaign_body}

ഓർക്കുക — Lumina Board ഉള്ളപ്പോൾ നിങ്ങളുടെ വിള, നിങ്ങളുടെ ഭാവി സുരക്ഷിതം.
കൂടുതൽ വിവരങ്ങൾക്ക് അടുത്തുള്ള Lumina Board ഡീലറെ ബന്ധപ്പെടുക.

Lumina Board India — കർഷകന്റെ ചങ്ങാതി."

[സംഗീതം - ഫേഡ് ഔട്ട്]
"""
}


class CampaignMessageGenerator:
    """
    Generates multilingual SMS messages and audio scripts.
    Uses Google Gemini API or Qwen2.5 for high-quality LLM generation; falls back to templates.
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.model = os.environ.get("LLM_MODEL", "qwen2.5:7b")
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key or str(self.gemini_api_key).startswith("${"):
            try:
                import yaml
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                config_path = os.path.join(base_dir, "config", "api_keys.yaml")
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                k = cfg.get("gemini", {}).get("api_key", "")
                if k and not k.startswith("${"):
                    self.gemini_api_key = k
            except Exception:
                pass

    def update_gemini_api_key(self, api_key: str):
        """Update Gemini API Key dynamically"""
        self.gemini_api_key = api_key
        if api_key:
            logger.info("Gemini API key configured for Campaign Message Generator")

    def _call_gemini_api(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        """Call Google Gemini API for instant multilingual LLM generation"""
        api_key = self.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key or len(str(api_key).strip()) < 5:
            return None
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": max_tokens
            }
        }

        for model in ["gemini-flash-latest", "gemini-2.0-flash", "gemini-pro-latest"]:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key.strip()}"
            for attempt in range(2):
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        logger.info(f"Successfully generated campaign message via Google Gemini API ({model})")
                        return text
                    elif resp.status_code == 429:
                        logger.warning(f"Gemini API rate limit 429 for {model} (attempt {attempt+1}), pausing 2.0s...")
                        import time
                        time.sleep(2.0)
                    else:
                        logger.warning(f"Gemini model {model} returned status {resp.status_code}: {resp.text[:100]}")
                        break
                except Exception as e:
                    logger.error(f"Gemini API error ({model}): {e}")
                    break
        return None

    def generate_multilingual(
        self,
        campaign_type: str,
        product: str,
        crop: str,
        state: str,
        languages: List[str],
        context: str = "",
        segment_stats: Optional[Dict] = None
    ) -> Dict[str, Dict]:
        """
        Generate SMS + audio script for each language.
        Returns: {language: {sms, audio_script, char_count, estimated_duration_sec}}
        """
        import time
        results = {}

        for idx_l, lang in enumerate(languages):
            if lang not in LANGUAGE_META:
                logger.warning(f"Unsupported language: {lang}, skipping")
                continue

            if idx_l > 0 and not os.environ.get("VERCEL"):
                time.sleep(0.5)  # Short pause between Gemini API requests

            try:
                sms = self._generate_sms(campaign_type, product, crop, state, lang, context, segment_stats)
                audio = self._generate_audio_script(campaign_type, product, crop, state, lang, sms)

                results[lang] = {
                    "sms": sms,
                    "audio_script": audio,
                    "char_count": len(sms),
                    "sms_parts": max(1, -(-len(sms) // 160)),  # ceil division
                    "estimated_audio_duration_sec": self._estimate_duration(audio, lang),
                    "script": LANGUAGE_META[lang]["script"],
                    "language_code": LANGUAGE_META[lang]["code"]
                }
            except Exception as e:
                logger.error(f"Generation failed for {lang}: {e}")
                results[lang] = {"error": str(e)}

        return results

    def _generate_sms(
        self,
        campaign_type: str,
        product: str,
        crop: str,
        state: str,
        lang: str,
        context: str,
        segment_stats: Optional[Dict]
    ) -> str:
        """Generate SMS using Gemini API, Ollama, or template fallback."""
        # Try Gemini API first
        meta = LANGUAGE_META.get(lang, {})
        script_name = meta.get("script", "Native")
        stats_str = ""
        if segment_stats:
            stats_str = (
                f"Target segment: {segment_stats.get('total_growers', '?')} growers, "
                f"avg farm {segment_stats.get('avg_farm_size', '?')} acres, "
                f"dominant device: {segment_stats.get('dominant_device', '?')}"
            )

        prompt = f"""You are a master agricultural marketing copywriter for Lumina Board India.
Generate a detailed, highly persuasive marketing advisory message in {lang} language for Indian farmers.

CAMPAIGN PARAMETERS:
- Campaign Type: {campaign_type}
- Recommended Product: {product}
- Target Crop: {crop}
- Target State/Region: {state}
- Target Language: {lang} ({script_name} Script)
{stats_str}
{f"Additional Context: {context}" if context else ""}

CRITICAL LANGUAGE REQUIREMENTS:
You MUST write the main message in {lang} using {script_name} script.
Do NOT use English transliteration (romanized text) for the main content.
You MAY use English letters ONLY for brand names (like Lumina Board, {product}) or if a specific technical term has no common translation.

SCRIPT REFERENCE (write your output ONLY in one of these):
- Hindi → Devanagari: नमस्ते किसान भाई, आपकी फसल की सुरक्षा
- Telugu → Telugu: నమస్కారం రైతు అన్నా, మీ పంట సంరక్షణ
- Marathi → Devanagari: नमस्कार शेतकरी बंधू, तुमच्या पिकाची सुरक्षा
- Punjabi → Gurmukhi: ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਕਿਸਾਨ ਜੀ, ਤੁਹਾਡੀ ਫਸਲ ਦੀ ਸੁਰੱਖਿਆ
- Tamil → Tamil: வணக்கம் விவசாயி, உங்கள் பயிர் பாதுகாப்பு
- Kannada → Kannada: ನಮಸ್ಕಾರ ರೈತ ಬಂಧು, ನಿಮ್ಮ ಬೆಳೆಯ ರಕ್ಷಣೆ
- Bengali → Bengali: নমস্কার কৃষক ভাই, আপনার ফসল সুরক্ষা
- Gujarati → Gujarati: નમસ્તે ખેડૂત ભાઈ, તમારા પાકની સુરક્ષા
- Odia → Odia: ନମସ୍କାର ଚାଷୀ ଭାଇ, ଆପଣଙ୍କ ଫସଲ ସୁରକ୍ଷା
- Malayalam → Malayalam: നമസ്കാരം കർഷക സഹോദരാ, നിങ്ങളുടെ വിള സംരക്ഷണം
- English → Latin: Dear Farmer, your crop protection

CONTENT STRUCTURE (write in {lang} native script only):
1. Greeting: {meta.get('greeting', 'Dear Farmer')}
2. Threat Context: Describe pest/disease/weather risk facing {crop} growers in {state}
3. Product Solution: Introduce {product}, highlight systemic action and yield enhancement
4. Application Guidance: Recommended dosage and spray timing
5. Call to Action: Visit nearest authorized dealer
6. Closing: "Lumina Board — Your Trusted Farming Partner" (translated to {lang})

LENGTH: Rich paragraph, 100-200 words in {lang} script.
OUTPUT: ONLY the message text in {lang} {script_name} script. No English text, no title, no translation notes."""

        gemini_result = self._call_gemini_api(prompt, max_tokens=600)
        if gemini_result:
            # Validate: check if the output actually contains native script characters
            # If the result is mostly English/Latin for a non-English language, fall back to template
            if lang != "English":
                validated = self._validate_native_script(gemini_result, lang)
                if validated:
                    return gemini_result
                else:
                    logger.warning(f"Gemini output for {lang} failed native script validation, using template fallback")
            else:
                return gemini_result

        # Skip Ollama LLM for languages where Qwen2.5 hallucination rate is too high
        skip_ollama_langs = ["Tamil", "Kannada", "Bengali", "Gujarati", "Odia"]
        if lang not in skip_ollama_langs:
            # Try Ollama LLM next
            llm_result = self._llm_generate_sms_prompt(prompt)
            if llm_result:
                if lang != "English":
                    if self._validate_native_script(llm_result, lang):
                        return llm_result
                else:
                    return llm_result

        # Template fallback
        return self._template_sms(campaign_type, product, crop, state, lang)

    def _llm_generate_sms_prompt(self, prompt: str) -> Optional[str]:
        """Call Ollama LLM with prompt"""
        try:
            resp = requests.post(
                f"{self.ollama_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 200
                },
                timeout=(2, 10)
            )
            resp.raise_for_status()
            sms = resp.json()["choices"][0]["message"]["content"].strip()
            if len(sms) > 320:
                sms = sms[:317] + "..."
            return sms
        except Exception:
            return None

    def _translate_term(self, term: str, lang: str, term_type: str = "crop") -> str:
        if not term or lang == "English":
            return term
        term_lower = term.lower().strip()
        
        # Extremely simple translation dict for common terms used in these campaigns
        crop_dict = {
            "wheat": {"Hindi": "गेहूँ", "Marathi": "गहू", "Punjabi": "ਕਣਕ", "Telugu": "గోధుమ", "Tamil": "கோதுமை", "Kannada": "ಗೋಧಿ", "Bengali": "গম", "Gujarati": "ઘઉં", "Odia": "ଗହମ", "Malayalam": "ഗോതമ്പ്"},
            "rice": {"Hindi": "धान", "Marathi": "भात", "Punjabi": "ਝੋਨਾ", "Telugu": "వరి", "Tamil": "நெல்", "Kannada": "ಭತ್ತ", "Bengali": "ধান", "Gujarati": "ડાંગર", "Odia": "ଧାନ", "Malayalam": "നെല്ല്"},
            "cotton": {"Hindi": "कपास", "Marathi": "कापूस", "Punjabi": "ਕਪਾਹ", "Telugu": "పత్తి", "Tamil": "பருத்தி", "Kannada": "ಹತ್ತಿ", "Bengali": "তুলা", "Gujarati": "કપાસ", "Odia": "କପା", "Malayalam": "പരുത്തി"},
            "soybean": {"Hindi": "सोयाबीन", "Marathi": "सोयाबीन", "Punjabi": "ਸੋਇਆਬੀਨ", "Telugu": "సోయాబీన్", "Tamil": "சோயாபீன்", "Kannada": "ಸೋಯಾಬೀನ್", "Bengali": "সয়াবিন", "Gujarati": "સોયાબીન", "Odia": "ସୋୟାବିନ୍", "Malayalam": "സോയാബീൻ"},
            "sugarcane": {"Hindi": "गन्ना", "Marathi": "ऊस", "Punjabi": "ਗੰਨਾ", "Telugu": "చెరకు", "Tamil": "கரும்பு", "Kannada": "ಕಬ್ಬು", "Bengali": "আখ", "Gujarati": "શેરડી", "Odia": "ଆଖୁ", "Malayalam": "കരിമ്പ്"}
        }
        
        state_dict = {
            "punjab": {"Hindi": "पंजाब", "Marathi": "पंजाब", "Punjabi": "ਪੰਜਾਬ", "Telugu": "పంజాబ్", "Tamil": "பஞ்சாப்", "Kannada": "ಪಂಜಾಬ್", "Bengali": "পাঞ্জাব", "Gujarati": "પંજાબ", "Odia": "ପଞ୍ଜାବ", "Malayalam": "പഞ്ചാബ്"},
            "maharashtra": {"Hindi": "महाराष्ट्र", "Marathi": "महाराष्ट्र", "Punjabi": "ਮਹਾਰਾਸ਼ਟਰ", "Telugu": "మహారాష్ట్ర", "Tamil": "மகாராஷ்டிரா", "Kannada": "ಮಹಾರಾಷ್ಟ್ರ", "Bengali": "মহারাষ্ট্র", "Gujarati": "મહારાષ્ટ્ર", "Odia": "ମହାରାଷ୍ଟ୍ର", "Malayalam": "മഹാരാഷ്ട്ര"},
            "andhra pradesh": {"Hindi": "आंध्र प्रदेश", "Marathi": "आंध्र प्रदेश", "Punjabi": "ਆਂਧਰਾ ਪ੍ਰਦੇਸ਼", "Telugu": "ఆంధ్రప్రదేశ్", "Tamil": "ஆந்திரப் பிரதேசம்", "Kannada": "ಆಂಧ್ರಪ್ರದೇಶ", "Bengali": "অন্ধ্রপ্রদেশ", "Gujarati": "આંધ્ર પ્રદેશ", "Odia": "ଆନ୍ଧ୍ରପ୍ରଦେଶ", "Malayalam": "ആന്ധ്രാപ്രദേശ്"},
            "telangana": {"Hindi": "तेलंगाना", "Marathi": "तेलंगणा", "Punjabi": "ਤੇਲੰਗਾਨਾ", "Telugu": "తెలంగాణ", "Tamil": "தெலுங்கானா", "Kannada": "ತೆಲಂಗಾಣ", "Bengali": "তেলেঙ্গানা", "Gujarati": "તેલંગાણા", "Odia": "ତେଲେଙ୍ଗାନା", "Malayalam": "തെലങ്കാന"},
            "karnataka": {"Hindi": "कर्नाटक", "Marathi": "कर्नाटक", "Punjabi": "ਕਰਨਾਟਕ", "Telugu": "కర్ణాటక", "Tamil": "கர்நாடகா", "Kannada": "ಕರ್ನಾಟಕ", "Bengali": "কর্ণাটক", "Gujarati": "કર્ણાટક", "Odia": "କର୍ଣ୍ଣାଟକ", "Malayalam": "കർണാടക"},
            "tamil nadu": {"Hindi": "तमिलनाडु", "Marathi": "तामिळनाडू", "Punjabi": "ਤਾਮਿਲਨਾਡੂ", "Telugu": "తమిళనాడు", "Tamil": "தமிழ்நாடு", "Kannada": "ತಮಿಳುನಾಡು", "Bengali": "তামিলনাড়ু", "Gujarati": "તમિલનાડુ", "Odia": "ତାମିଲନାଡୁ", "Malayalam": "തമിഴ്നാട്"},
            "gujarat": {"Hindi": "गुजरात", "Marathi": "गुजरात", "Punjabi": "ਗੁਜਰਾਤ", "Telugu": "గుజరాత్", "Tamil": "குஜராத்", "Kannada": "ಗುಜರಾತ್", "Bengali": "গুজরাট", "Gujarati": "ગુજરાત", "Odia": "ଗୁଜରାଟ", "Malayalam": "ഗുജറാത്ത്"}
        }
        
        dictionary = crop_dict if term_type == "crop" else state_dict
        if term_lower in dictionary and lang in dictionary[term_lower]:
            return dictionary[term_lower][lang]
        return term

    def _template_sms(self, campaign_type: str, product: str, crop: str, state: str, lang: str) -> str:
        """Use pre-built template."""
        templates = TEMPLATES.get(campaign_type, TEMPLATES.get("product_launch", {}))
        # Find template for language or fall back to English
        template = templates.get(lang) or templates.get("English", "{greeting}, Lumina Board {product} for {crop} in {state}.")

        meta = LANGUAGE_META.get(lang, {})
        
        # Translate crop and state
        t_crop = self._translate_term(crop, lang, "crop") if crop else ("" if lang != "English" else "crop")
        t_state = self._translate_term(state, lang, "state") if state else ("" if lang != "English" else "region")
        
        return template.format(
            greeting=meta.get("greeting", "Dear Farmer"),
            product=product or "Amistar Top",
            crop=t_crop,
            state=t_state
        )

    def _generate_audio_script(
        self, campaign_type: str, product: str, crop: str, state: str, lang: str, sms: str
    ) -> str:
        """Generate 30-60 second audio/IVR script based on SMS content."""
        # Try LLM
        llm_audio = self._llm_generate_audio(product, crop, state, lang, sms)
        if llm_audio:
            return llm_audio

        # Template fallback
        base_template = AUDIO_SCRIPT_TEMPLATES.get(lang) or AUDIO_SCRIPT_TEMPLATES.get("English", "")
        campaign_body = self._expand_sms_to_audio_body(sms, lang, product, crop)

        return base_template.format(
            duration=45,
            state=state or "India",
            campaign_body=campaign_body
        )

    def _llm_generate_audio(self, product, crop, state, lang, sms_text) -> Optional[str]:
        """Call Gemini API or Qwen2.5 for audio script generation."""
        prompt = f"""Create a 30-45 second IVR/radio audio script in {lang} for Indian farmers.

Based on this SMS campaign:
"{sms_text}"

Product: {product}, Crop: {crop}, State: {state}

REQUIREMENTS:
- Write in {lang} language (native script)  
- Include [intro music] and [outro music] stage directions
- Warm, trusted voice tone — like talking to a neighbor
- Natural speech rhythm — not too fast
- 30-45 seconds when read aloud at normal pace (~{LANGUAGE_META.get(lang, {}).get('audio_wpm', 90)} wpm)
- Include: greeting → problem (pest/disease/yield risk) → solution ({product}) → call to action → Lumina Board tagline
- Stage directions in [square brackets]

Output only the script text."""

        gemini_audio = self._call_gemini_api(prompt, max_tokens=400)
        if gemini_audio:
            return gemini_audio

        try:
            resp = requests.post(
                f"{self.ollama_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                    "max_tokens": 400
                },
                timeout=(2, 10)
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    def _expand_sms_to_audio_body(self, sms: str, lang: str, product: str, crop: str) -> str:
        """Expand SMS to 3-4 sentence audio body."""
        # Simple expansion: repeat key points with pauses
        return sms.replace("।", "।\n").replace(".", ".\n")

    def _validate_native_script(self, text: str, lang: str) -> bool:
        """
        Validate that the generated text actually contains native script characters.
        Returns True if at least 30% of non-space, non-punctuation characters are
        in the expected Unicode range for the language.
        """
        import re
        # Unicode ranges for each Indian language script
        script_ranges = {
            "Hindi": r'[\u0900-\u097F]',      # Devanagari
            "Marathi": r'[\u0900-\u097F]',     # Devanagari
            "Telugu": r'[\u0C00-\u0C7F]',      # Telugu
            "Punjabi": r'[\u0A00-\u0A7F]',     # Gurmukhi
            "Tamil": r'[\u0B80-\u0BFF]',       # Tamil
            "Kannada": r'[\u0C80-\u0CFF]',     # Kannada
            "Bengali": r'[\u0980-\u09FF]',     # Bengali
            "Gujarati": r'[\u0A80-\u0AFF]',    # Gujarati
            "Odia": r'[\u0B00-\u0B7F]',        # Odia
            "Malayalam": r'[\u0D00-\u0D7F]',   # Malayalam
        }
        
        pattern = script_ranges.get(lang)
        if not pattern:
            return True  # Unknown language, skip validation
        
        # Count native script characters vs total non-whitespace characters
        clean_text = re.sub(r'[\s\d.,!?;:\-—–\'"()\[\]{}@#$%^&*+=/<>\\|~`\n\r\t]', '', text)
        # Also remove common brand names that will be in Latin
        clean_text = re.sub(r'[A-Za-z]+', '', clean_text)
        
        if len(clean_text) < 80:
            logger.warning(f"Native script validation for {lang} failed: output too short ({len(clean_text)} chars)")
            return False
        
        native_chars = len(re.findall(pattern, text))
        # At least 20% of all characters should be in native script
        total_chars = len(re.sub(r'\s', '', text))
        if total_chars == 0:
            return False
        
        ratio = native_chars / total_chars
        is_valid = ratio >= 0.50
        
        if not is_valid:
            logger.warning(f"Native script validation for {lang}: ratio={ratio:.2f} ({native_chars}/{total_chars} native chars) - FAILED")
        
        return is_valid

    def _estimate_duration(self, script: str, lang: str) -> int:
        """Estimate audio duration in seconds based on word count."""
        wpm = LANGUAGE_META.get(lang, {}).get("audio_wpm", 90)
        # Rough word count (works for both Latin and Indic scripts)
        word_count = len(script.split())
        # Subtract stage directions (in brackets)
        import re
        clean = re.sub(r'\[.*?\]', '', script)
        clean_words = len(clean.split())
        return max(20, int(clean_words / wpm * 60))