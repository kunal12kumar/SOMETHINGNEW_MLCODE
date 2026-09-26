"""Hand-written formatting conventions used by normalisation.

Everything here is general spelling/abbreviation knowledge (legal forms, street
types, postal state codes). Nothing is derived from individual training records
and nothing is fetched from outside sources.
"""

# Legal / company-form words, mapped to one canonical token.
LEGAL_FORMS = {
    # English / US
    "llc": "llc", "l.l.c": "llc", "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "ltd": "ltd", "limited": "ltd", "lp": "lp", "llp": "llp", "pllc": "pllc",
    "plc": "plc", "pc": "pc", "pa": "pa", "esq": "esq", "lc": "lc",
    # India
    "pvt": "pvt", "private": "pvt", "pvtltd": "pvt ltd", "opc": "opc",
    # Romanised Indic spellings after transliteration
    "praivet": "pvt", "praiveta": "pvt", "prayivet": "pvt", "praibhet": "pvt",
    "limiteda": "ltd", "elaelapi": "llp", "elelpi": "llp",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa",
    "sci": "sci", "snc": "snc", "ei": "ei", "earl": "earl", "selarl": "selarl",
    "scop": "scop", "gie": "gie",
}

# Two-token abbreviations that mean a legal form ("pra li" = Hindi "pvt ltd").
LEGAL_BIGRAMS = {("pra", "li"): "pvt ltd", ("p", "ltd"): "pvt ltd"}

# Forms of address that can prefix a business name.
HONORIFICS = {"mr", "mrs", "ms", "dr", "messrs", "m/s"}

# Alias markers: text on either side is an alternative name.
ALIAS_MARKERS = {
    "aka", "a/k/a", "dba", "d/b/a", "t/a", "fka", "f/k/a", "o/a",
    "trading as", "doing business as", "operating as", "also known as", "formerly known as", "formerly",
}

# Words that mean the same thing written differently, mapped to one token in
# both names and addresses. "St" is "Street" in US addresses but "Saint" in
# place names ("St.-Nazaire" = "Saint-Nazaire"); one shared token keeps both
# spellings equal without having to guess which is meant.
SHARED_TOKENS = {"street": "st", "saint": "st", "sainte": "ste"}

# Address abbreviations -> full word. Street types, building terms, French forms.
ADDRESS_ABBREV = {
    "str": "st", "ave": "avenue", "av": "avenue", "crs": "cours", "bvd": "boulevard",
    "rd": "road", "dr": "drive", "ln": "lane", "blvd": "boulevard",
    "bd": "boulevard", "bld": "boulevard", "ct": "court", "cir": "circle",
    "hwy": "highway", "pkwy": "parkway", "pky": "parkway", "pl": "place",
    "trl": "trail", "ter": "terrace", "terr": "terrace", "sq": "square",
    "cres": "crescent", "expy": "expressway", "fwy": "freeway", "mt": "mount",
    "ft": "fort", "stn": "station", "jn": "junction", "jct": "junction",
    "bldg": "building", "apts": "apartment", "apt": "apartment",
    "fl": "floor", "flr": "floor", "opp": "opposite", "nr": "near",
    "clny": "colony", "cly": "colony", "ngr": "nagar", "mkt": "market",
    # French ("r" = rue is handled in code: only directly after a house number)
    "imp": "impasse", "all": "allee", "che": "chemin",
    "ch": "chemin", "rte": "route", "fbg": "faubourg", "qu": "quai",
    "pte": "porte", "sq.": "square",
}

# Words that only label a number ("H.No 12", "Plot No 5"); the number is kept.
NUMBER_MARKERS = {
    "no", "nos", "num", "number", "h", "hno", "house", "door", "plot", "pno",
    "p", "building", "flat", "unit", "suite", "ste", "room", "shop", "survey",
    "sy", "kh", "khasra", "cts", "m", "ward", "gat", "gaht", "block", "n",
}

# Tokens that carry no location information.
ADDRESS_NOISE = {"na", "n/a", "township", "twp", "tonwship", "cdp", "nil", "none"}

# Common modern names for renamed Indian cities.
CITY_ALIASES = {
    "bangalore": "bengaluru", "bombay": "mumbai", "calcutta": "kolkata",
    "madras": "chennai", "gurgaon": "gurugram", "poona": "pune",
    "trivandrum": "thiruvananthapuram", "cochin": "kochi", "mysore": "mysuru",
    "baroda": "vadodara", "banaras": "varanasi", "benares": "varanasi",
    "pondicherry": "puducherry", "calicut": "kozhikode",
}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "dc": "district of columbia", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana",
    "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan",
    "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon",
    "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "pr": "puerto rico",
}

INDIA_STATES = {
    "an": "andaman and nicobar islands", "ap": "andhra pradesh",
    "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "ch": "chandigarh", "cg": "chhattisgarh", "ct": "chhattisgarh",
    "dn": "dadra and nagar haveli", "dd": "daman and diu", "dl": "delhi",
    "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
    "jk": "jammu and kashmir", "jh": "jharkhand", "ka": "karnataka",
    "kl": "kerala", "la": "ladakh", "ld": "lakshadweep",
    "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur",
    "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha",
    "or": "odisha", "py": "puducherry", "pb": "punjab", "rj": "rajasthan",
    "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana", "ts": "telangana",
    "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand",
    "ut": "uttarakhand", "wb": "west bengal",
}

# Other spellings of state names, including romanised native-script forms.
STATE_NAME_ALIASES = {
    "dilli": "delhi", "nct of delhi": "delhi", "dhamilnadhu": "tamil nadu",
    "pascim bangal": "west bengal",
    "orissa": "odisha", "odisa": "odisha", "tamilnatu": "tamil nadu",
    "tamilnadu": "tamil nadu", "pascimabanga": "west bengal",
    "pashchim banga": "west bengal", "paschimbanga": "west bengal",
    "uttaranchal": "uttarakhand", "pondicherry": "puducherry",
    "maharastra": "maharashtra", "karnatak": "karnataka",
    "telamgana": "telangana", "keralam": "kerala",
}

FRANCE_REGIONS = [
    "auvergne-rhone-alpes", "bourgogne-franche-comte", "bretagne",
    "centre-val de loire", "corse", "grand est", "hauts-de-france",
    "ile-de-france", "normandie", "nouvelle-aquitaine", "occitanie",
    "pays de la loire", "provence-alpes-cote d'azur",
]
