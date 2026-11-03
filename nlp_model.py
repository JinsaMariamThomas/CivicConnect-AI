
from langdetect import detect, LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import re

# ----------------------------------
# CIVICCONNECT AI - MULTILINGUAL NLP
# ----------------------------------

training_data = {
    "Water": [
        # English
        "water leakage in my street",
        "no water supply",
        "drinking water is dirty",
        "water pipe is broken",
        "water shortage",
        "there is no drinking water in our area",
        "the water supply has stopped",
        "dirty water is coming from the tap",
        "water pipeline is leaking",
        "we have not received water for three days",

        # Malayalam
        "വെള്ളം വരുന്നില്ല",
        "വാട്ടർ പൈപ്പ് പൊട്ടി",
        "കുടിവെള്ളം മലിനമാണ്",
        "ഞങ്ങളുടെ പ്രദേശത്ത് വെള്ളമില്ല",
        "മൂന്ന് ദിവസമായി വെള്ളം വരുന്നില്ല",
        "കുടിവെള്ളത്തിന് ക്ഷാമമുണ്ട്",
        "പൈപ്പിൽ നിന്ന് വെള്ളം ചോരുന്നു",
        "വെള്ളത്തിന്റെ വിതരണം നിലച്ചു",

        # Hindi
        "पानी नहीं आ रहा है",
        "पानी की पाइप टूट गई है",
        "पीने का पानी गंदा है",
        "हमारे इलाके में पानी नहीं है",
        "तीन दिनों से पानी नहीं आ रहा है",
        "पानी की कमी हो रही है",
        "पानी की पाइपलाइन में रिसाव है",

        # Tamil
        "தண்ணீர் வரவில்லை",
        "குடிநீர் மாசடைந்துள்ளது",
        "தண்ணீர் குழாய் உடைந்துவிட்டது",
        "எங்கள் பகுதியில் தண்ணீர் இல்லை",
        "மூன்று நாட்களாக தண்ணீர் வரவில்லை",
        "குடிநீர் பற்றாக்குறை உள்ளது",

        # Kannada
        "ನಮ್ಮ ಪ್ರದೇಶದಲ್ಲಿ ನೀರು ಬರುತ್ತಿಲ್ಲ",
        "ಕುಡಿಯುವ ನೀರು ಕಲುಷಿತವಾಗಿದೆ",
        "ನೀರಿನ ಪೈಪ್ ಒಡೆದಿದೆ",
        "ಮೂರು ದಿನಗಳಿಂದ ನೀರು ಬರುತ್ತಿಲ್ಲ"
    ],

    "Roads": [
        # English
        "road is damaged",
        "large potholes on the road",
        "repair the broken road",
        "street has no proper road",
        "the road is full of holes",
        "road construction is incomplete",
        "the main road needs repair",
        "vehicles cannot travel because of bad roads",
        "the road surface is damaged",
        "there is a big pothole near the school",

        # Malayalam
        "റോഡ് തകർന്നു",
        "റോഡിൽ വലിയ കുഴികളുണ്ട്",
        "റോഡ് നന്നാക്കണം",
        "ഞങ്ങളുടെ റോഡ് വളരെ മോശമാണ്",
        "റോഡിൽ അപകടകരമായ കുഴികളുണ്ട്",
        "റോഡ് നിർമ്മാണം പൂർത്തിയായിട്ടില്ല",
        "പ്രധാന റോഡ് അറ്റകുറ്റപ്പണി നടത്തണം",

        # Hindi
        "सड़क टूटी हुई है",
        "सड़क पर गड्ढे हैं",
        "सड़क की मरम्मत करनी चाहिए",
        "हमारे इलाके की सड़क खराब है",
        "सड़क पर बड़े गड्ढे हैं",
        "सड़क निर्माण अधूरा है",
        "मुख्य सड़क की मरम्मत जरूरी है",

        # Tamil
        "சாலை சேதமடைந்துள்ளது",
        "சாலையில் பெரிய பள்ளங்கள் உள்ளன",
        "சாலையை சரிசெய்ய வேண்டும்",
        "எங்கள் பகுதி சாலை மிகவும் மோசமாக உள்ளது",
        "சாலை அமைக்கும் பணி முடிக்கப்படவில்லை",
        "முக்கிய சாலையை சீரமைக்க வேண்டும்",

        # Kannada
        "ರಸ್ತೆ ಹಾಳಾಗಿದೆ",
        "ರಸ್ತೆಯಲ್ಲಿ ದೊಡ್ಡ ಗುಂಡಿಗಳಿವೆ",
        "ರಸ್ತೆಯನ್ನು ದುರಸ್ತಿ ಮಾಡಬೇಕು",
        "ನಮ್ಮ ಪ್ರದೇಶದ ರಸ್ತೆ ತುಂಬಾ ಕೆಟ್ಟದಾಗಿದೆ"
    ],

    "Electricity": [
        # English
        "there is no electricity",
        "power cut in my area",
        "electric pole is damaged",
        "street lights are not working",
        "electric wire is hanging",
        "electricity supply is interrupted",
        "there is a frequent power outage",
        "the transformer is not working",
        "power has been unavailable since morning",
        "a damaged electric wire is hanging on the road",

        # Malayalam
        "വൈദ്യുതി ഇല്ല",
        "വൈദ്യുതി തടസ്സം",
        "തെരുവ് വിളക്കുകൾ പ്രവർത്തിക്കുന്നില്ല",
        "വൈദ്യുതി ലൈൻ തകരാറിലാണ്",
        "ഞങ്ങളുടെ പ്രദേശത്ത് വൈദ്യുതി മുടങ്ങി",
        "വൈദ്യുതി പോസ്റ്റ് തകർന്നു",
        "ട്രാൻസ്ഫോർമർ പ്രവർത്തിക്കുന്നില്ല",
        "വൈദ്യുതി വയർ താഴെ തൂങ്ങിക്കിടക്കുന്നു",

        # Hindi
        "बिजली नहीं है",
        "बिजली की तार टूट गई है",
        "हमारे इलाके में बिजली कटौती हो रही है",
        "स्ट्रीट लाइट काम नहीं कर रही है",
        "बिजली की आपूर्ति बंद है",
        "ट्रांसफार्मर काम नहीं कर रहा है",
        "बिजली का खंभा टूट गया है",

        # Tamil
        "மின்சாரம் இல்லை",
        "தெரு விளக்குகள் வேலை செய்யவில்லை",
        "மின்சார கம்பி அறுந்துவிட்டது",
        "எங்கள் பகுதியில் மின்தடை ஏற்பட்டுள்ளது",
        "மின்மாற்றி வேலை செய்யவில்லை",
        "மின்சார கம்பம் சேதமடைந்துள்ளது",

        # Kannada
        "ನಮ್ಮ ಪ್ರದೇಶದಲ್ಲಿ ವಿದ್ಯುತ್ ಇಲ್ಲ",
        "ಬೀದಿ ದೀಪಗಳು ಕೆಲಸ ಮಾಡುತ್ತಿಲ್ಲ",
        "ವಿದ್ಯುತ್ ತಂತಿ ತುಂಡಾಗಿದೆ",
        "ವಿದ್ಯುತ್ ಸರಬರಾಜು ಸ್ಥಗಿತಗೊಂಡಿದೆ"
    ],

    "Waste Management": [
        # English
        "garbage is not collected",
        "dustbins are overflowing",
        "waste is dumped on the road",
        "bad smell from garbage",
        "clean our streets",
        "garbage has not been collected for days",
        "there is too much waste in our area",
        "the public dustbin is full",
        "waste is scattered near our houses",
        "the garbage collection vehicle is not coming",

        # Malayalam
        "മാലിന്യം നീക്കം ചെയ്യുന്നില്ല",
        "മാലിന്യം റോഡിൽ കിടക്കുന്നു",
        "ചവറ്റുകുട്ട നിറഞ്ഞു",
        "ഞങ്ങളുടെ പ്രദേശത്ത് മാലിന്യം കെട്ടിക്കിടക്കുന്നു",
        "ദിവസങ്ങളായി മാലിന്യം ശേഖരിക്കുന്നില്ല",
        "റോഡിൽ മാലിന്യം തള്ളുന്നു",
        "മാലിന്യത്തിൽ നിന്ന് ദുർഗന്ധം വരുന്നു",
        "മാലിന്യ വണ്ടി വരുന്നില്ല",

        # Hindi
        "कचरा नहीं उठाया जा रहा है",
        "सड़क पर कचरा पड़ा है",
        "कूड़ेदान भर गया है",
        "हमारे इलाके में कचरा जमा हो रहा है",
        "कई दिनों से कचरा नहीं उठाया गया है",
        "कचरे से बदबू आ रही है",
        "कचरा गाड़ी नहीं आ रही है",

        # Tamil
        "குப்பைகள் அகற்றப்படவில்லை",
        "தெருவில் குப்பைகள் கிடக்கின்றன",
        "குப்பைத்தொட்டி நிரம்பியுள்ளது",
        "எங்கள் பகுதியில் குப்பைகள் தேங்கியுள்ளன",
        "பல நாட்களாக குப்பைகள் சேகரிக்கப்படவில்லை",
        "குப்பையிலிருந்து துர்நாற்றம் வருகிறது",
        "குப்பை வண்டி வரவில்லை",

        # Kannada
        "ಕಸವನ್ನು ತೆಗೆದುಹಾಕುತ್ತಿಲ್ಲ",
        "ರಸ್ತೆಯಲ್ಲಿ ಕಸ ಬಿದ್ದಿದೆ",
        "ಕಸದ ಬುಟ್ಟಿ ತುಂಬಿದೆ",
        "ನಮ್ಮ ಪ್ರದೇಶದಲ್ಲಿ ಕಸ ಸಂಗ್ರಹವಾಗಿದೆ"
    ],

    "Public Safety": [
        # English
        "dangerous open manhole",
        "unsafe public area",
        "open drainage is dangerous",
        "fallen electric pole",
        "there is an open manhole near the road",
        "street lights are missing in a dark area",
        "a damaged bridge is unsafe",
        "there is a dangerous open drain",
        "a tree has fallen across the road",
        "the public area is not safe for residents",

        # Malayalam
        "അപകടകരമായ തുറന്ന മാൻഹോൾ",
        "പ്രദേശം സുരക്ഷിതമല്ല",
        "തുറന്ന ഓട അപകടകരമാണ്",
        "വൈദ്യുതി പോസ്റ്റ് വീണു",
        "റോഡിൽ തുറന്ന മാൻഹോൾ ഉണ്ട്",
        "പൊട്ടിയ പാലം അപകടകരമാണ്",
        "മരം റോഡിലേക്ക് വീണു",
        "തുറന്ന ഓടയിൽ അപകടസാധ്യതയുണ്ട്",

        # Hindi
        "खुला मैनहोल खतरनाक है",
        "सार्वजनिक स्थान सुरक्षित नहीं है",
        "खुली नाली खतरनाक है",
        "बिजली का खंभा गिर गया है",
        "सड़क पर खुला मैनहोल है",
        "टूटा हुआ पुल खतरनाक है",
        "एक पेड़ सड़क पर गिर गया है",

        # Tamil
        "திறந்த சாக்கடை ஆபத்தானது",
        "பொது இடம் பாதுகாப்பற்றது",
        "திறந்த கழிவுநீர் கால்வாய் ஆபத்தானது",
        "மின்கம்பம் கீழே விழுந்துவிட்டது",
        "சாலையில் திறந்த நிலையில் பாதாள சாக்கடை உள்ளது",
        "உடைந்த பாலம் ஆபத்தானது",
        "மரம் சாலையில் விழுந்துள்ளது",

        # Kannada
        "ತೆರೆದ ಮ್ಯಾನ್‌ಹೋಲ್ ಅಪಾಯಕಾರಿಯಾಗಿದೆ",
        "ಸಾರ್ವಜನಿಕ ಸ್ಥಳ ಸುರಕ್ಷಿತವಾಗಿಲ್ಲ",
        "ತೆರೆದ ಚರಂಡಿ ಅಪಾಯಕಾರಿಯಾಗಿದೆ",
        "ವಿದ್ಯುತ್ ಕಂಬ ಬಿದ್ದಿದೆ"
    ]
}

departments = {
    "Water": "Water Supply Department",
    "Roads": "Public Works Department (PWD)",
    "Electricity": "Electricity Department",
    "Waste Management": "Municipal Waste Management Department",
    "Public Safety": "Public Safety Department"
}

# Prepare training data
X = []
y = []

for category, complaints in training_data.items():
    for complaint in complaints:
        X.append(complaint.lower())
        y.append(category)

# Build NLP model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        sublinear_tf=True
    )),
    ("classifier", LinearSVC())
])

# Train model
model.fit(X, y)


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def get_priority(text):
    urgent_words = [
        # English
        "emergency", "danger", "accident", "fire",
        "urgent", "unsafe", "life threatening",

        # Malayalam
        "അപകടം", "അടിയന്തിരം", "അപകടകരം",

        # Hindi
        "खतरा", "आपातकाल", "दुर्घटना",

        # Tamil
        "ஆபத்து", "அவசரம்", "விபத்து",

        # Kannada
        "ಅಪಾಯ", "ತುರ್ತು", "ಅಪಘಾತ"
    ]

    text = text.lower()

    if any(word in text for word in urgent_words):
        return "High"

    return "Normal"


def analyze_complaint(text):

    if not text or not text.strip():
        return None

    text = re.sub(r"\s+", " ", text.strip())

    category = model.predict([text])[0]

    language = detect_language(text)

    department = departments[category]

    priority = get_priority(text)

    return {
        "complaint": text,
        "language": language,
        "category": category,
        "department": department,
        "priority": priority
    }


# Run model independently for testing
if __name__ == "__main__":

    print("\n================================")
    print("       CIVICCONNECT AI")
    print("    MULTILINGUAL NLP SYSTEM")
    print("================================")

    while True:

        complaint = input("\nEnter your complaint (or exit): ")

        if complaint.lower() == "exit":
            print("Thank you for using CivicConnect AI!")
            break

        result = analyze_complaint(complaint)

        if result:
            print("\n--- AI ANALYSIS ---")

            for key, value in result.items():
                print(f"{key.title()}: {value}")
