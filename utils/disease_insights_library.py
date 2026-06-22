"""
MediMap AI — Static Disease Insights Library
=============================================
Fallback health insights used when the Gemini API is unavailable.
Covers all 41 disease classes in the training dataset.

Each entry contains:
  - description : 2-sentence plain-english overview
  - precautions : list of 4 safe at-home precautions
"""

DISEASE_INSIGHTS: dict[str, dict] = {
    "(vertigo) Paroymsal  Positional Vertigo": {
        "description": (
            "Benign Paroxysmal Positional Vertigo (BPPV) is an inner ear disorder "
            "that causes brief episodes of intense dizziness triggered by head movements. "
            "It occurs when tiny calcium crystals in the ear become dislodged and move into the ear canals."
        ),
        "precautions": [
            "Move slowly when sitting up, lying down, or turning your head to avoid triggering episodes.",
            "Sleep with your head slightly elevated using two pillows to reduce crystal displacement.",
            "Avoid sudden head movements and bending over quickly.",
            "Try the Epley manoeuvre (a series of guided head movements) — look up a tutorial or ask your doctor.",
        ],
    },
    "AIDS": {
        "description": (
            "AIDS (Acquired Immunodeficiency Syndrome) is the advanced stage of HIV infection, "
            "where the immune system is severely damaged and unable to fight off infections. "
            "It is caused by the Human Immunodeficiency Virus (HIV) and is managed with antiretroviral therapy."
        ),
        "precautions": [
            "Take all prescribed antiretroviral medications consistently and never miss a dose.",
            "Eat a nutritious, balanced diet to support your weakened immune system.",
            "Avoid contact with people who are sick, as your body is less able to fight infections.",
            "Get regular medical check-ups to monitor your CD4 count and viral load.",
        ],
    },
    "Acne": {
        "description": (
            "Acne is a common skin condition where hair follicles become clogged with oil and dead skin cells, "
            "leading to pimples, blackheads, and whiteheads. "
            "It most commonly affects the face, forehead, chest, shoulders, and upper back."
        ),
        "precautions": [
            "Wash your face gently twice daily with a mild, non-comedogenic cleanser.",
            "Avoid touching or picking at pimples, as this can cause scarring and spread bacteria.",
            "Use oil-free, non-comedogenic moisturisers and sunscreen.",
            "Change pillowcases frequently and keep hair products away from your face.",
        ],
    },
    "Alcoholic hepatitis": {
        "description": (
            "Alcoholic hepatitis is inflammation of the liver caused by excessive and prolonged alcohol consumption. "
            "It can range from mild to severe and, if untreated, may progress to liver failure or cirrhosis."
        ),
        "precautions": [
            "Stop consuming alcohol immediately — this is the single most important step.",
            "Eat small, frequent, high-protein meals to support liver recovery.",
            "Stay well hydrated by drinking plenty of water throughout the day.",
            "Avoid all over-the-counter pain medications like paracetamol (Tylenol), which can further damage the liver.",
        ],
    },
    "Allergy": {
        "description": (
            "An allergy is an immune system reaction to a foreign substance such as pollen, pet dander, food, "
            "or insect stings that is normally harmless to most people. "
            "Symptoms can range from mild (sneezing, itching) to severe (anaphylaxis)."
        ),
        "precautions": [
            "Identify and avoid your known allergy triggers as much as possible.",
            "Keep windows closed during high pollen seasons and use air purifiers indoors.",
            "Take over-the-counter antihistamines to relieve mild symptoms like itching and sneezing.",
            "If you have a known severe allergy, always carry an epinephrine auto-injector (EpiPen).",
        ],
    },
    "Arthritis": {
        "description": (
            "Arthritis is inflammation of one or more joints, causing pain, swelling, stiffness, and reduced range of motion. "
            "There are over 100 types, with osteoarthritis and rheumatoid arthritis being the most common."
        ),
        "precautions": [
            "Apply warm compresses or heating pads to stiff joints to ease pain and improve flexibility.",
            "Perform gentle low-impact exercises like swimming or walking to keep joints mobile.",
            "Maintain a healthy weight to reduce excess pressure on weight-bearing joints.",
            "Rest affected joints during flare-ups and use supportive braces or splints if needed.",
        ],
    },
    "Bronchial Asthma": {
        "description": (
            "Bronchial asthma is a chronic respiratory condition where the airways become inflamed and narrow, "
            "making it difficult to breathe. "
            "Triggers include allergens, cold air, exercise, and respiratory infections."
        ),
        "precautions": [
            "Always keep your prescribed rescue inhaler (e.g., salbutamol) close at hand.",
            "Identify and avoid personal asthma triggers such as dust, smoke, or pet dander.",
            "Keep indoor air clean by vacuuming regularly and using allergen-proof pillow covers.",
            "Monitor your breathing daily and seek emergency care if your rescue inhaler provides no relief.",
        ],
    },
    "Cervical spondylosis": {
        "description": (
            "Cervical spondylosis is age-related wear and tear of the spinal discs in the neck, "
            "which can lead to neck pain, stiffness, and sometimes numbness or tingling in the arms. "
            "It is very common and worsens gradually with age."
        ),
        "precautions": [
            "Maintain good posture, especially when sitting at a desk or using a phone.",
            "Apply a warm or cold pack to the neck for 15–20 minutes to reduce pain and stiffness.",
            "Perform gentle neck stretches and strengthening exercises daily.",
            "Use a supportive, ergonomic pillow that keeps your neck in a neutral position while sleeping.",
        ],
    },
    "Chicken pox": {
        "description": (
            "Chickenpox is a highly contagious viral infection caused by the varicella-zoster virus, "
            "resulting in an itchy, blister-like rash all over the body. "
            "It is most common in children but can affect adults and is generally more severe in older patients."
        ),
        "precautions": [
            "Stay home and isolate from others, especially pregnant women, newborns, and immunocompromised people.",
            "Apply calamine lotion or take cool oatmeal baths to relieve itching.",
            "Avoid scratching the blisters to prevent scarring and bacterial infection.",
            "Take paracetamol for fever — never give aspirin to children with chickenpox.",
        ],
    },
    "Chronic cholestasis": {
        "description": (
            "Chronic cholestasis is a condition where bile flow from the liver is reduced or blocked over a long period, "
            "causing bile to build up in the liver and bloodstream. "
            "Symptoms include persistent itching, jaundice, and fatigue."
        ),
        "precautions": [
            "Follow a strict low-fat diet to reduce strain on the liver and bile system.",
            "Avoid alcohol completely, as it further impairs liver function.",
            "Take cool showers and use mild moisturisers to relieve skin itching.",
            "Take all prescribed medications and attend all follow-up appointments consistently.",
        ],
    },
    "Common Cold": {
        "description": (
            "The common cold is a viral infection of the upper respiratory tract, primarily caused by rhinoviruses. "
            "It typically causes a runny nose, sore throat, sneezing, and mild fever, and usually resolves within 7–10 days."
        ),
        "precautions": [
            "Rest as much as possible to allow your immune system to fight the virus.",
            "Stay well hydrated with water, warm teas, and clear broths to soothe the throat.",
            "Use saline nasal sprays or steam inhalation to relieve nasal congestion.",
            "Take paracetamol or ibuprofen for fever and body aches.",
        ],
    },
    "Dengue": {
        "description": (
            "Dengue fever is a mosquito-borne viral disease common in tropical and subtropical regions, "
            "causing high fever, severe headache, pain behind the eyes, and muscle and joint pain. "
            "Severe dengue can be life-threatening and requires immediate medical attention."
        ),
        "precautions": [
            "Rest completely and drink large amounts of fluids (water, oral rehydration salts, coconut water) to prevent dehydration.",
            "Take paracetamol for fever — never take aspirin or ibuprofen, as they increase bleeding risk.",
            "Use mosquito repellents and nets to prevent further mosquito bites.",
            "Monitor for warning signs of severe dengue: severe abdominal pain, persistent vomiting, or bleeding — and go to hospital immediately.",
        ],
    },
    "Diabetes": {
        "description": (
            "Diabetes is a chronic metabolic disease characterised by high blood sugar levels, "
            "either because the body doesn't produce enough insulin (Type 1) or doesn't use insulin effectively (Type 2). "
            "If poorly managed, it can lead to serious complications affecting the heart, kidneys, eyes, and nerves."
        ),
        "precautions": [
            "Monitor your blood glucose levels regularly and keep a log of your readings.",
            "Follow a low-sugar, low-refined-carbohydrate diet with plenty of vegetables and fibre.",
            "Take all prescribed medications or insulin injections at the correct times.",
            "Exercise for at least 30 minutes daily, as physical activity helps lower blood sugar naturally.",
        ],
    },
    "Dimorphic hemmorhoids(piles)": {
        "description": (
            "Haemorrhoids (piles) are swollen and inflamed veins in the rectum and anus that cause discomfort, "
            "bleeding, and itching. "
            "They are commonly caused by straining during bowel movements, chronic constipation, or prolonged sitting."
        ),
        "precautions": [
            "Eat a high-fibre diet (fruits, vegetables, whole grains) and drink plenty of water to soften stools.",
            "Avoid straining during bowel movements — take your time and never hold your breath.",
            "Take warm sitz baths (sitting in warm water) for 10–15 minutes several times a day to relieve pain.",
            "Use over-the-counter haemorrhoid creams or medicated wipes to reduce itching and swelling.",
        ],
    },
    "Drug Reaction": {
        "description": (
            "A drug reaction is an unintended and harmful response to a medication, ranging from mild skin rashes "
            "to severe anaphylaxis. "
            "Reactions can occur immediately after taking a drug or develop over days or weeks."
        ),
        "precautions": [
            "Stop taking the suspected medication immediately and contact your doctor or pharmacist.",
            "If experiencing difficulty breathing, severe swelling, or a rapid pulse, call emergency services immediately.",
            "Document the name of the medication, the dose, and all symptoms experienced.",
            "Never take the suspected medication again without explicit medical clearance.",
        ],
    },
    "Fungal infection": {
        "description": (
            "Fungal infections are caused by microscopic fungi that thrive in warm, moist environments, "
            "commonly affecting the skin, nails, and mucous membranes. "
            "Common examples include athlete's foot, ringworm, and oral thrush."
        ),
        "precautions": [
            "Keep affected skin areas clean and completely dry, especially skin folds.",
            "Apply over-the-counter antifungal creams or powders to the affected area as directed.",
            "Wear loose-fitting, breathable clothing and change socks daily.",
            "Avoid sharing towels, clothing, or footwear, as fungal infections are contagious.",
        ],
    },
    "GERD": {
        "description": (
            "Gastro-oesophageal Reflux Disease (GERD) is a chronic digestive condition where stomach acid "
            "frequently flows back into the oesophagus, causing heartburn, chest pain, and regurgitation. "
            "It is triggered by certain foods, obesity, and lying down after meals."
        ),
        "precautions": [
            "Avoid trigger foods such as spicy food, fatty food, caffeine, alcohol, and citrus fruits.",
            "Eat smaller meals more frequently and avoid lying down for at least 2–3 hours after eating.",
            "Elevate the head of your bed by 15–20 cm (6–8 inches) to reduce nighttime reflux.",
            "Take over-the-counter antacids or proton pump inhibitors as directed to neutralise stomach acid.",
        ],
    },
    "Gastroenteritis": {
        "description": (
            "Gastroenteritis is inflammation of the stomach and intestines, usually caused by a viral or bacterial "
            "infection, leading to diarrhoea, vomiting, nausea, and stomach cramps. "
            "It is commonly called 'stomach flu' and typically resolves within a few days."
        ),
        "precautions": [
            "Stay hydrated by sipping water, clear broths, or oral rehydration salts (ORS) frequently.",
            "Rest your stomach by eating bland foods (toast, rice, bananas, boiled potatoes) when you feel ready to eat.",
            "Avoid dairy products, fatty foods, alcohol, and caffeine until fully recovered.",
            "Wash your hands thoroughly and frequently to prevent spreading the infection to others.",
        ],
    },
    "Heart attack": {
        "description": (
            "A heart attack (myocardial infarction) occurs when blood flow to a part of the heart muscle is blocked, "
            "usually by a blood clot, causing that part of the heart to begin dying. "
            "It is a medical emergency requiring immediate treatment."
        ),
        "precautions": [
            "Call emergency services (911/999) immediately — do not drive yourself to hospital.",
            "Chew a 325mg aspirin immediately if you are not allergic to it and if available.",
            "Sit or lie down in a comfortable position and stay as calm as possible.",
            "Loosen any tight clothing and do not eat or drink anything while waiting for emergency services.",
        ],
    },
    "Hepatitis A": {
        "description": (
            "Hepatitis A is a highly contagious liver infection caused by the hepatitis A virus, "
            "typically spread through contaminated food and water. "
            "Most people recover fully within a few weeks without long-term liver damage."
        ),
        "precautions": [
            "Rest extensively and avoid all physical exertion to support liver recovery.",
            "Avoid alcohol and all medications metabolised by the liver unless prescribed by a doctor.",
            "Eat small, frequent, low-fat meals to reduce strain on the liver.",
            "Practice strict hand hygiene and do not prepare food for others while infectious.",
        ],
    },
    "Hepatitis B": {
        "description": (
            "Hepatitis B is a serious viral infection that attacks the liver and can cause both acute and chronic disease. "
            "It is spread through contact with infected blood, unprotected sex, and from mother to child during childbirth."
        ),
        "precautions": [
            "Take all prescribed antiviral medications consistently as directed by your doctor.",
            "Avoid alcohol completely, as it accelerates liver damage.",
            "Eat a healthy, balanced diet rich in fruits, vegetables, and lean proteins.",
            "Inform all close contacts so they can be tested and vaccinated.",
        ],
    },
    "Hepatitis C": {
        "description": (
            "Hepatitis C is a viral infection that causes liver inflammation, sometimes leading to serious liver damage. "
            "It is primarily spread through contact with infected blood and can often be cured with modern antiviral treatments."
        ),
        "precautions": [
            "Never share needles, syringes, razors, or toothbrushes with anyone.",
            "Avoid alcohol as it dramatically worsens liver damage from Hepatitis C.",
            "Take prescribed antiviral medications consistently for the full treatment course.",
            "Get regular blood tests to monitor your viral load and liver function.",
        ],
    },
    "Hepatitis D": {
        "description": (
            "Hepatitis D is a serious liver disease caused by the hepatitis D virus (HDV), "
            "which only infects people who are already infected with Hepatitis B. "
            "It can cause severe and rapid liver damage and is the most serious form of viral hepatitis."
        ),
        "precautions": [
            "Take all prescribed medications and attend all specialist appointments.",
            "Avoid alcohol completely to protect your already stressed liver.",
            "Eat a liver-friendly diet: low in fat, high in fresh vegetables and lean protein.",
            "Prevent spread to others by avoiding sharing sharp objects or unprotected contact.",
        ],
    },
    "Hepatitis E": {
        "description": (
            "Hepatitis E is a liver disease caused by the hepatitis E virus, primarily spread through "
            "drinking water contaminated with faecal matter. "
            "It usually resolves on its own in 4–6 weeks but can be dangerous for pregnant women."
        ),
        "precautions": [
            "Rest completely and avoid strenuous physical activity.",
            "Drink only clean, boiled, or bottled water and avoid raw or undercooked food.",
            "Avoid alcohol and all unnecessary medications to reduce liver strain.",
            "Pregnant women must seek immediate hospital care, as Hepatitis E can be life-threatening during pregnancy.",
        ],
    },
    "Hypertension": {
        "description": (
            "Hypertension (high blood pressure) is a long-term condition where the force of blood against artery walls "
            "is consistently too high, putting strain on the heart and blood vessels. "
            "It is a major risk factor for heart attack, stroke, and kidney disease."
        ),
        "precautions": [
            "Reduce salt intake significantly — aim for less than 5g (one teaspoon) of salt per day.",
            "Exercise regularly with at least 30 minutes of moderate activity (walking, cycling) most days.",
            "Take all prescribed blood pressure medications at the same time every day without skipping.",
            "Monitor your blood pressure at home daily and keep a log to share with your doctor.",
        ],
    },
    "Hyperthyroidism": {
        "description": (
            "Hyperthyroidism is a condition where the thyroid gland produces too much thyroid hormone, "
            "speeding up the body's metabolism and causing symptoms like rapid heartbeat, weight loss, and anxiety. "
            "It is most commonly caused by Graves' disease."
        ),
        "precautions": [
            "Take all prescribed anti-thyroid medications at the same time daily without missing doses.",
            "Avoid caffeine and stimulants, as they can worsen palpitations and anxiety.",
            "Rest adequately and avoid intense exercise until your hormone levels are controlled.",
            "Eat a calcium and vitamin D-rich diet to protect bones, as hyperthyroidism can cause bone loss.",
        ],
    },
    "Hypoglycemia": {
        "description": (
            "Hypoglycaemia is a condition where blood sugar (glucose) levels drop dangerously low, "
            "causing symptoms such as shakiness, sweating, confusion, and in severe cases, loss of consciousness. "
            "It is most common in people with diabetes who take insulin or certain medications."
        ),
        "precautions": [
            "If conscious, immediately consume 15–20g of fast-acting sugar: glucose tablets, fruit juice, or regular soft drink.",
            "Re-check blood sugar after 15 minutes and repeat if still low.",
            "Always carry fast-acting sugar snacks or glucose tablets with you at all times.",
            "Do not drive or operate machinery until your blood sugar has fully recovered to a safe level.",
        ],
    },
    "Hypothyroidism": {
        "description": (
            "Hypothyroidism is a condition where the thyroid gland does not produce enough thyroid hormone, "
            "slowing down the body's metabolism and causing fatigue, weight gain, cold intolerance, and depression. "
            "It is managed effectively with daily thyroid hormone replacement medication."
        ),
        "precautions": [
            "Take prescribed levothyroxine tablets on an empty stomach every morning at the same time.",
            "Avoid taking calcium, iron supplements, or antacids within 4 hours of your thyroid medication, as they block absorption.",
            "Eat a balanced diet rich in iodine (fish, dairy, eggs) to support thyroid function.",
            "Exercise regularly to combat fatigue and weight gain associated with the condition.",
        ],
    },
    "Impetigo": {
        "description": (
            "Impetigo is a highly contagious bacterial skin infection, most common in children, "
            "that causes red sores which rupture, ooze, and form a honey-coloured crust. "
            "It is usually caused by Staphylococcus or Streptococcus bacteria."
        ),
        "precautions": [
            "Wash the affected area gently with soap and water and keep it clean and dry.",
            "Apply prescribed antibiotic cream directly to the sores as directed.",
            "Cover the sores loosely with a clean dressing to prevent spreading.",
            "Wash hands frequently and avoid sharing towels, clothing, or bedding with others.",
        ],
    },
    "Jaundice": {
        "description": (
            "Jaundice is a yellowing of the skin and the whites of the eyes caused by a build-up of bilirubin "
            "in the blood, indicating a problem with the liver, bile ducts, or red blood cells. "
            "It is a symptom, not a disease itself, and the underlying cause must be treated."
        ),
        "precautions": [
            "Rest completely and avoid any alcohol or unnecessary medications that strain the liver.",
            "Stay well hydrated by drinking plenty of water and fresh juices.",
            "Eat small, easily digestible meals that are low in fat.",
            "Seek medical care urgently to identify the underlying cause, as some causes are serious.",
        ],
    },
    "Malaria": {
        "description": (
            "Malaria is a life-threatening disease caused by parasites transmitted to humans through the bites "
            "of infected female Anopheles mosquitoes. "
            "Symptoms include cyclical high fevers, chills, and flu-like illness, and it requires prompt treatment."
        ),
        "precautions": [
            "Take all prescribed antimalarial medications for the full course without stopping early.",
            "Rest and stay hydrated with plenty of fluids to manage fever and prevent dehydration.",
            "Use mosquito nets (preferably insecticide-treated) at night to prevent further bites.",
            "Take paracetamol for fever — never aspirin or ibuprofen — and monitor for severe symptoms.",
        ],
    },
    "Migraine": {
        "description": (
            "A migraine is a neurological condition characterised by intense, debilitating headaches, "
            "often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. "
            "Attacks can last from a few hours to several days."
        ),
        "precautions": [
            "Lie down in a quiet, dark room and rest until the migraine passes.",
            "Apply a cold or warm compress to your forehead or neck for relief.",
            "Take prescribed or over-the-counter pain relief (paracetamol, ibuprofen, triptans) at the very first sign of a migraine.",
            "Identify and avoid your personal migraine triggers such as certain foods, stress, dehydration, or lack of sleep.",
        ],
    },
    "Osteoarthristis": {
        "description": (
            "Osteoarthritis is the most common form of arthritis, occurring when the protective cartilage "
            "that cushions the ends of bones wears down over time. "
            "It most commonly affects joints in the hands, knees, hips, and spine."
        ),
        "precautions": [
            "Exercise regularly with low-impact activities like swimming, cycling, or walking to maintain joint function.",
            "Maintain a healthy weight to reduce excess mechanical stress on weight-bearing joints.",
            "Apply warm compresses for stiffness or cold packs for swelling and acute pain.",
            "Use over-the-counter pain relief (paracetamol, topical NSAIDs) as directed for pain management.",
        ],
    },
    "Paralysis (brain hemorrhage)": {
        "description": (
            "Paralysis from a brain haemorrhage occurs when bleeding in or around the brain damages motor pathways, "
            "resulting in loss of movement or sensation in parts of the body. "
            "It is a medical emergency that requires immediate treatment to minimise permanent damage."
        ),
        "precautions": [
            "Call emergency services immediately if you suspect someone is having a brain haemorrhage.",
            "Keep the patient still, calm, and warm while waiting for help to arrive.",
            "Do not give food, water, or any medications to someone who may be having a stroke or brain bleed.",
            "Follow all physiotherapy and rehabilitation plans strictly after hospitalisation to regain function.",
        ],
    },
    "Peptic ulcer diseae": {
        "description": (
            "Peptic ulcer disease involves open sores that develop on the inner lining of the stomach "
            "and the upper portion of the small intestine, causing burning stomach pain. "
            "It is most commonly caused by H. pylori bacteria or long-term use of anti-inflammatory drugs (NSAIDs)."
        ),
        "precautions": [
            "Avoid NSAIDs (ibuprofen, aspirin, naproxen) — use paracetamol instead for pain relief.",
            "Avoid alcohol, spicy foods, caffeine, and acidic foods that irritate the stomach lining.",
            "Eat small, regular meals and avoid skipping meals or going long periods without eating.",
            "Take all prescribed antibiotics and acid-reducing medications for the complete course.",
        ],
    },
    "Pneumonia": {
        "description": (
            "Pneumonia is an infection that inflames the air sacs in one or both lungs, "
            "which may fill with fluid or pus, causing cough with phlegm, fever, chills, and difficulty breathing. "
            "It can be caused by bacteria, viruses, or fungi and ranges from mild to life-threatening."
        ),
        "precautions": [
            "Rest completely and avoid any strenuous physical activity to allow your lungs to recover.",
            "Stay well hydrated with water and warm fluids to help loosen and thin mucus.",
            "Take all prescribed antibiotics for the full course, even if you start feeling better.",
            "Sleep propped up on pillows to make breathing easier and monitor for worsening breathlessness.",
        ],
    },
    "Psoriasis": {
        "description": (
            "Psoriasis is a chronic autoimmune condition that causes a rapid build-up of skin cells, "
            "resulting in scaling on the skin surface, red patches, and itching. "
            "It is not contagious and tends to follow a cycle of flare-ups and remissions."
        ),
        "precautions": [
            "Moisturise your skin thoroughly every day using thick, fragrance-free creams or ointments.",
            "Avoid triggers such as stress, smoking, excessive alcohol, and skin injuries.",
            "Apply prescribed topical treatments (steroid creams, vitamin D creams) consistently.",
            "Take short, lukewarm showers instead of hot baths to avoid drying out the skin.",
        ],
    },
    "Tuberculosis": {
        "description": (
            "Tuberculosis (TB) is a potentially serious infectious disease that mainly affects the lungs, "
            "caused by the bacterium Mycobacterium tuberculosis and spread through the air when infected people cough. "
            "It is treatable and curable with a 6-month course of antibiotics."
        ),
        "precautions": [
            "Take all prescribed TB medications every single day for the full 6-month course — stopping early causes drug resistance.",
            "Cover your mouth and nose with a tissue when coughing or sneezing, then dispose of the tissue safely.",
            "Ensure good ventilation in your home by opening windows frequently.",
            "Wear a surgical mask in enclosed spaces and inform close contacts so they can be tested.",
        ],
    },
    "Typhoid": {
        "description": (
            "Typhoid fever is a life-threatening illness caused by the bacterium Salmonella Typhi, "
            "spread through contaminated food and water. "
            "It causes sustained high fever, stomach pain, headache, and sometimes a rash."
        ),
        "precautions": [
            "Drink only boiled or bottled water and avoid ice cubes made from tap water.",
            "Eat only thoroughly cooked food and avoid raw salads or street food.",
            "Rest completely and take all prescribed antibiotics for the full course.",
            "Take paracetamol for fever and stay well hydrated with ORS or boiled water.",
        ],
    },
    "Urinary tract infection": {
        "description": (
            "A urinary tract infection (UTI) is an infection in any part of the urinary system including the kidneys, "
            "bladder, and urethra, most commonly causing a painful, burning sensation when urinating. "
            "UTIs are much more common in women and are usually treated effectively with antibiotics."
        ),
        "precautions": [
            "Drink at least 2–3 litres of water daily to help flush bacteria out of the urinary tract.",
            "Take all prescribed antibiotics for the full course, even if symptoms resolve early.",
            "Urinate frequently and do not hold urine for long periods.",
            "Wipe front to back after using the toilet to prevent bacteria from entering the urethra.",
        ],
    },
    "Varicose veins": {
        "description": (
            "Varicose veins are enlarged, twisted, bluish-purple veins that appear just under the skin surface, "
            "usually in the legs, caused by weakened or damaged vein valves. "
            "They cause aching, heaviness, and swelling and are worsened by prolonged standing."
        ),
        "precautions": [
            "Elevate your legs above heart level for 15–20 minutes several times a day to reduce swelling.",
            "Wear prescribed graduated compression stockings to improve blood flow.",
            "Exercise regularly, especially walking, to strengthen calf muscles and promote circulation.",
            "Avoid sitting or standing for prolonged periods — take short walks every 30 minutes.",
        ],
    },
    "hepatitis A": {
        "description": (
            "Hepatitis A is a highly contagious liver infection caused by the hepatitis A virus, "
            "typically spread through contaminated food and water. "
            "Most people recover fully within a few weeks without long-term liver damage."
        ),
        "precautions": [
            "Rest extensively and avoid all physical exertion to support liver recovery.",
            "Avoid alcohol and all medications metabolised by the liver unless prescribed by a doctor.",
            "Eat small, frequent, low-fat meals to reduce strain on the liver.",
            "Practice strict hand hygiene and do not prepare food for others while infectious.",
        ],
    },
}


def get_fallback_insights(disease_name: str) -> dict | None:
    """
    Return static insights for a given disease name.
    Performs a case-insensitive lookup with fuzzy matching as backup.

    Returns a dict with keys 'description' and 'precautions', or None if not found.
    """
    # Exact match first
    if disease_name in DISEASE_INSIGHTS:
        return DISEASE_INSIGHTS[disease_name]

    # Case-insensitive match
    for key in DISEASE_INSIGHTS:
        if key.lower() == disease_name.lower():
            return DISEASE_INSIGHTS[key]

    # Partial match fallback
    for key in DISEASE_INSIGHTS:
        if disease_name.lower() in key.lower() or key.lower() in disease_name.lower():
            return DISEASE_INSIGHTS[key]

    return None
