import pytest

from ber.normalize import normalize_address, normalize_name


@pytest.mark.parametrize(
    "raw, norm, core",
    [
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "ram marketing pvt ltd", "ram marketing"),
        ("ब्लैक कंस्ट्रक्शन प्रा. लि.", "blaik kanstraksan pvt ltd", "blaik kanstraksan"),
        ("-- Holloway Peak Inc Seafood", "holloway peak inc seafood", "holloway peak seafood"),
        ("5umit (india) Limited", "sumit india ltd", "sumit"),
        ("Manchester H0rizon Assets Harbor Inc", "manchester horizon assets harbor inc", "manchester horizon assets harbor"),
        ("Premier Technology Pvt-Limited", "premier technology pvt ltd", "premier technology"),
        ("Mr Snehal Engineering Private  Limited", "mr snehal engineering pvt ltd", "snehal engineering"),
        ("Etoile SAS  et Associés", "etoile sas et associes", "etoile et associes"),
        ("Aromatic & Brothers Pvt. Ltd.", "aromatic and brothers pvt ltd", "aromatic and brothers"),
        ("Ridge LP [Center] #17196", "ridge lp center 17196", "ridge"),
        ("wilfordhancock.com", "wilfordhancock", "wilfordhancock"),
    ],
)
def test_name_forms(raw, norm, core):
    out = normalize_name(raw)
    assert out["name_norm"] == norm
    assert out["name_core"] == core


def test_same_business_written_differently_gets_same_core():
    a = normalize_name("Sharma Electronics Pvt. Ltd.")
    b = normalize_name("SHARMA ELECTRONICS PRIVATE LIMITED")
    assert a["name_norm"] == b["name_norm"] == "sharma electronics pvt ltd"


def test_distinct_businesses_are_not_collapsed():
    a = normalize_name("ABC Holdings LLC")
    b = normalize_name("ABC Services LLC")
    assert a["name_core"] != b["name_core"]


def test_alias_is_split_out_and_both_parts_searchable():
    out = normalize_name("Ciraonyx a/k/a VL Creations Private Limited")
    assert out["name_norm"] == "ciraonyx vl creations pvt ltd"
    assert out["name_core"] == "ciraonyx"
    assert out["name_alias"] == "vl creations"


def test_codes_are_not_treated_as_typos():
    assert normalize_name("cs1464.com")["name_core"] == "cs1464"
    assert normalize_name("Cl0guh LLC")["name_core"] == "cloguh"


def test_dangling_connector_removed_from_core():
    assert normalize_name("Achintya & Co")["name_core"] == "achintya"


def test_initials():
    assert normalize_name("M R & X Sun Inc.")["name_initials"] == "mrxs"


@pytest.mark.parametrize(
    "raw, country, state, city, numbers",
    [
        ("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India", "delhi", "west delhi", "13 570"),
        ("GREENSBORO, NC, 19 1/2 STARDUST TRAIL", "US", "north carolina", "greensboro", "1 2 19"),
        ("KS, 0549 OAK STREET, COLUMBUS", "US", "kansas", "columbus", "549"),
        ("SURVEY NO. 376, MORBI-JETPURA ROAD, SHAPAR, MORBI, RAJKOT, ગુજરાત", "India", "gujarat", "rajkot", "376"),
        ("NO 96 J - 141 SAKET, NEW DELHI, दिल्ली", "India", "delhi", "new delhi", "96 141"),
        ("Nouvelle-Aquitaine, PESSAC, 18 AV PIERRE CASTAING", "France", "nouvelle-aquitaine", "pessac", "18"),
        ("No 752 40A, Calcutta, Howrah, WB", "India", "west bengal", "howrah", "40 752"),
    ],
)
def test_address_parts(raw, country, state, city, numbers):
    out = normalize_address(raw, country)
    assert out["addr_state"] == state
    assert out["addr_city"] == city
    assert out["addr_numbers"] == numbers


def test_abbreviations_expand_the_same_way():
    a = normalize_address("105 ELM ST, MORGANTON, NC", "US")
    b = normalize_address("105 Elm Street, Morganton, North Carolina", "US")
    assert a["addr_norm"] == b["addr_norm"] == "105 elm st morganton"
    assert a["addr_state"] == b["addr_state"] == "north carolina"


def test_saint_and_st_agree():
    a = normalize_address("R. RENÉ GUILLOUZO, ST.-NAZAIRE, Pays de la Loire", "France")
    b = normalize_address("4 Rue René Guillouzo, Saint-Nazaire, Pays de la Loire", "France")
    assert a["addr_city"] == b["addr_city"] == "st nazaire"


def test_trading_as_is_an_alias():
    out = normalize_name("Novixylonyla trading as Infirmiers Comite EURL")
    assert out["name_core"] == "novixylonyla"
    assert out["name_alias"] == "infirmiers comite"


@pytest.mark.parametrize("raw", [
    "Dovadovadrex formerly known as Satterwhite and Massengill PLLC",
    "Dovadovadrex Formerly Satterwhite and Massengill PLLC",
    "Dovadovadrex doing business as Satterwhite and Massengill PLLC",
])
def test_formerly_and_long_markers(raw):
    out = normalize_name(raw)
    assert out["name_core"] == "dovadovadrex"
    assert out["name_alias"] == "satterwhite and massengill"


def test_cours_abbreviation():
    assert "cours" in normalize_address("No. 29 Crs François Bart, Dunkerque, Nord", "France")["addr_norm"]


def test_french_rue_only_after_number():
    assert normalize_address("18 R de Picardie, Roubaix", "France")["addr_norm"] == "18 rue de picardie roubaix"
    assert "rue" not in normalize_address("H.no 701 Wing R, Kharadi", "India")["addr_norm"]


def test_number_label_removed_value_kept():
    out = normalize_address("N° 161 BOULEVARD ROBERT SCHUMAN, NANTES", "France")
    assert out["addr_norm"] == "161 boulevard robert schuman nantes"


def test_unknown_country_label_works():
    out = normalize_address("12 Some Road, Springfield", "Atlantis")
    assert out["addr_norm"] == "12 some road springfield"
    assert out["addr_missing"] is False


def test_empty_address():
    assert normalize_address("", "US")["addr_missing"] is True
