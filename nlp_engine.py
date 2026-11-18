import re
from rapidfuzz import process, fuzz

agri_dictionary = {'tamatar': 'tomato', 'tomatto': 'tomato', 'tomatoo': 'tomato', 'gehun': 'wheat', 'gehu': 'wheat', 'godhi': 'wheat', 'chawal': 'rice', 'bhattada': 'rice', 'mirchi': 'chilli', 'menasinakayi': 'chilli', 'patta': 'leaf', 'patte': 'leaves', 'pattey': 'leaves', 'ele': 'leaf', 'eleya': 'leaf', 'gida': 'plant', 'jad': 'root', 'moola': 'root', 'peela': 'yellow', 'peele': 'yellow', 'haladi': 'yellow', 'sukha': 'dry', 'murjha': 'wilting', 'baaduttide': 'wilting', 'kale': 'spots', 'keeda': 'pest', 'keede': 'insects', 'keeta': 'pest', 'roga': 'disease', 'bimari': 'disease', 'gobbara': 'fertilizer', 'khad': 'fertilizer', 'dawa': 'medicine', 'aushadhi': 'medicine', 'belege': 'crop', 'belegge': 'crop', 'gidakke': 'plant', 'gidada': 'plant', 'mele': 'on', 'yavudu': 'which', 'yava': 'which', 'bandide': 'affected', 'agide': 'is', 'ide': 'is', 'leaf': 'leaf', 'leaves': 'leaves', 'spots': 'spots', 'plant': 'plant', 'crop': 'crop', 'fasal': 'crop', 'aphids': 'aphids', 'patton': 'leaves', 'bhattakke': 'rice', 'gidalli': 'plant', 'daali': 'attack', 'aphid': 'aphid', 'blight': 'blight', 'disease': 'disease', 'fertilizer': 'fertilizer', 'mildew': 'mildew', 'powdery': 'powdery', 'rust': 'rust', 'spray': 'spray', 'ragi': 'ragi', 'tomato': 'tomato', 'brown': 'brown', 'yellow': 'yellow', 'rice': 'rice', 'wheat': 'wheat', 'chilli': 'chilli', 'spot': 'spot'}

multiword_dictionary = {'powdery mildew': 'powdery mildew', 'leaf spot': 'leaf spot', 'brown leaves': 'brown leaves', 'yellow leaves': 'yellow leaves', 'leaf spots': 'leaf spots'}

context_rules = {'keeta bandide': 'pest affected', 'keeta daali madide': 'pest attack', 'aphid hai': 'aphid infestation', 'aphids ive': 'aphid infestation', 'belegge spray': 'crop spray'}

entity_dictionary = {'crop': {'tomato': ['tomato', 'tamatar'], 'wheat': ['wheat', 'gehun', 'gehu', 'godhi'], 'rice': ['rice', 'chawal', 'bhattada', 'bhattakke'], 'chilli': ['chilli', 'mirchi'], 'ragi': ['ragi']}, 'plant_part': {'plant': ['plant', 'gida'], 'leaf': ['leaf', 'leaves', 'ele', 'eleya'], 'root': ['root', 'jad', 'moola'], 'stem': ['stem'], 'fruit': ['fruit']}, 'disease': {'blight': ['blight'], 'rust': ['rust'], 'powdery mildew': ['powdery mildew', 'mildew'], 'leaf spot': ['leaf spot', 'leaf spots'], 'disease': ['disease', 'roga', 'bimari']}, 'pest': {'insect': ['insect', 'insects', 'keede'], 'aphid': ['aphid', 'aphids'], 'caterpillar': ['caterpillar'], 'worm': ['worm'], 'pest': ['pest', 'keeta']}, 'symptom': {'yellow': ['yellow', 'peela', 'peele', 'haladi'], 'brown': ['brown'], 'wilting': ['wilting', 'murjha', 'baaduttide'], 'spots': ['spots', 'spot', 'kale']}, 'agricultural_action': {'spray': ['spray'], 'fertilizer': ['fertilizer', 'gobbara', 'khad'], 'medicine': ['medicine', 'dawa', 'aushadhi'], 'treatment': ['treatment']}}

def fuzzy_normalize(word, dictionary, threshold=85):
    word = word.lower()

    if word in dictionary:
        return dictionary[word]

    match = process.extractOne(
        word,
        dictionary.keys(),
        scorer=fuzz.ratio
    )

    if match:
        matched_word, score, index = match

        if score >= threshold:
            return dictionary[matched_word]

    return word


def normalize_phrases(query, phrase_dictionary):
    query = query.lower()

    for phrase, normalized_phrase in sorted(
        phrase_dictionary.items(),
        key=lambda item: len(item[0]),
        reverse=True
    ):
        query = query.replace(phrase, normalized_phrase)

    return query


def apply_context_rules(query):
    query = query.lower()

    for phrase, replacement in sorted(
        context_rules.items(),
        key=lambda item: len(item[0]),
        reverse=True
    ):
        query = query.replace(phrase, replacement)

    return query


def improved_normalize_query(query, dictionary):

    # Step 1: Apply context-aware agricultural rules
    query = apply_context_rules(query)

    # Step 2: Normalize multiword agricultural phrases
    query = normalize_phrases(
        query, multiword_dictionary
    )

    # Step 3: Normalize individual words
    words = query.lower().split()
    normalized_words = []

    for word in words:
        normalized_word = fuzzy_normalize(
            word, dictionary, threshold=85
        )
        normalized_words.append(normalized_word)

    return " ".join(normalized_words)


def extract_entities(query, original_query=None):

    combined_text = (
        str(query) + " " + str(original_query or "")
    ).lower()

    entities = {}

    for category, entity_terms in entity_dictionary.items():

        found_entities = []

        for canonical, aliases in entity_terms.items():

            for term in aliases:

                pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"

                if re.search(pattern, combined_text):
                    found_entities.append(canonical)
                    break

        entities[category] = found_entities

    # Remove generic disease label when a specific disease is detected
    specific_diseases = [
        "blight", "rust", "powdery mildew", "leaf spot"
    ]

    if any(d in entities["disease"] for d in specific_diseases):
        entities["disease"] = [
            d for d in entities["disease"]
            if d != "disease"
        ]

    return entities


def classify_intent(query):
    query = query.lower()

    if any(word in query for word in [
        "yellow", "brown", "spots", "wilting",
        "peele", "sukha", "kale"
    ]):
        return "symptom_identification"

    elif any(word in query for word in [
        "pest", "insect", "keede", "keeta",
        "aphid", "worm"
    ]):
        return "pest_identification"

    elif any(word in query for word in [
        "disease", "bimari", "roga",
        "blight", "rust", "mildew"
    ]):
        return "disease_identification"

    elif any(word in query for word in [
        "fertilizer", "gobbara", "treatment",
        "medicine", "spray"
    ]):
        return "treatment_advice"

    else:
        return "unknown"


def agricultural_nlp_pipeline(query):

    normalized_query = improved_normalize_query(
        query, agri_dictionary
    )

    entities = extract_entities(normalized_query)

    intent = classify_intent(normalized_query)

    return {
        "original_query": query,
        "normalized_query": normalized_query,
        "entities": entities,
        "intent": intent
    }


