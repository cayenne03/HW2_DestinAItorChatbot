from spacy.tokens import Token, Span
from spacy.language import Language
from mylogger import get_logger

logger = get_logger(__name__)

def setup_entity_patterns(nlp):
    """Setup custom entity patterns for spaCy model"""
    
    # register custom extensions
    if not Token.has_extension("id"):
        Token.set_extension("id", default=None)
    if not Span.has_extension("id"):
        Span.set_extension("id", default=None)

    # setup entity ruler
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")

    PATTERNS = {
        "FLIGHT_TYPE": [
            {"pattern": "one-way", "id": "oneway"},
            {"pattern": "oneway", "id": "oneway"},
            {"pattern": "one way", "id": "oneway"},
            {"pattern": "no return", "id": "oneway"},
            {"pattern": "1 way", "id": "oneway"},
            {"pattern": "single", "id": "oneway"},
            {"pattern": "round trip", "id": "round_trip"},
            {"pattern": "roundtrip", "id": "round_trip"},
            {"pattern": "round-trip", "id": "round_trip"},
            {"pattern": "return flight", "id": "round_trip"},
            {"pattern": "two way", "id": "round_trip"},
            {"pattern": "2 way", "id": "round_trip"},
            {"pattern": "with return", "id": "round_trip"},
        ],
        "LOCATION_INDICATOR": [
            # basic from/to
            {"pattern": [{"LOWER": "from"}], "id": "departure"},
            {"pattern": [{"LOWER": "to"}], "id": "arrival"},
            
            # with flight/flights
            {"pattern": [{"LOWER": "flight"}, {"LOWER": "from"}], "id": "departure"},
            {"pattern": [{"LOWER": "flight"}, {"LOWER": "to"}], "id": "arrival"},
            {"pattern": [{"LOWER": "flights"}, {"LOWER": "from"}], "id": "departure"},
            {"pattern": [{"LOWER": "flights"}, {"LOWER": "to"}], "id": "arrival"},
            {"pattern": [{"TEXT": "->"}, {"IS_SPACE": True}], "id": "arrival"},
            {"pattern": [{"TEXT": "→"}, {"IS_SPACE": True}], "id": "arrival"},
            
            # book/booking variations
            {"pattern": [{"LOWER": "book"}, {"LOWER": "from"}], "id": "departure"},
            {"pattern": [{"LOWER": "book"}, {"LOWER": "to"}], "id": "arrival"},
            {"pattern": [{"LOWER": "booking"}, {"LOWER": "from"}], "id": "departure"},
            {"pattern": [{"LOWER": "booking"}, {"LOWER": "to"}], "id": "arrival"},
            
            # travel variations
            {"pattern": [{"LOWER": "travelling"}, {"LOWER": "from"}], "id": "departure"},
            {"pattern": [{"LOWER": "travelling"}, {"LOWER": "to"}], "id": "arrival"},
            {"pattern": [{"LOWER": "traveling"}, {"LOWER": "from"}], "id": "departure"},
            {"pattern": [{"LOWER": "traveling"}, {"LOWER": "to"}], "id": "arrival"},
            {"pattern": [{"LOWER": "travel"}, {"LOWER": "from"}], "id": "departure"},
            {"pattern": [{"LOWER": "travel"}, {"LOWER": "to"}], "id": "arrival"},
            
            # movement variations
            {"pattern": [{"LOWER": "departing"}, {"LOWER": "from"}], "id": "departure"},
            {"pattern": [{"LOWER": "departure"}, {"LOWER": "from"}], "id": "departure"},
            {"pattern": [{"LOWER": "starting"}, {"LOWER": "from"}], "id": "departure"},
            {"pattern": [{"LOWER": "leaving"}, {"LOWER": "from"}], "id": "departure"},
            {"pattern": [{"LOWER": "going"}, {"LOWER": "from"}], "id": "departure"},
            {"pattern": [{"LOWER": "going"}, {"LOWER": "to"}], "id": "arrival"},
            {"pattern": [{"LOWER": "flying"}, {"LOWER": "from"}], "id": "departure"},
            {"pattern": [{"LOWER": "flying"}, {"LOWER": "to"}], "id": "arrival"},
            
            # arrival variations
            {"pattern": [{"LOWER": "arriving"}, {"LOWER": "at"}], "id": "arrival"},
            {"pattern": [{"LOWER": "arriving"}, {"LOWER": "in"}], "id": "arrival"},
            {"pattern": [{"LOWER": "landing"}, {"LOWER": "in"}], "id": "arrival"},
            {"pattern": [{"LOWER": "destination"}], "id": "arrival"},
            
            # arrow indicators
            {"pattern": [{"TEXT": "->"}, {"IS_SPACE": True}], "id": "arrival"},
            {"pattern": [{"TEXT": "→"}, {"IS_SPACE": True}], "id": "arrival"},
            {"pattern": [{"TEXT": "=>"}, {"IS_SPACE": True}], "id": "arrival"},
            {"pattern": [{"TEXT": ">"}, {"IS_SPACE": True}], "id": "arrival"},
        ],
        "PASSENGERS": [
            {"pattern": [{"LIKE_NUM": True}, {"LOWER": "passengers"}]},
            {"pattern": [{"LIKE_NUM": True}, {"LOWER": "people"}]},
            {"pattern": [{"LIKE_NUM": True}, {"LOWER": "persons"}]},
            {"pattern": [{"LOWER": "for"}, {"LIKE_NUM": True}]},
        ],
        # add some new DATE patterns
        "DATE": [
            {"pattern": [{"SHAPE": "dddd-dd-dd"}], "id": "iso_date"},
            {"pattern": [{"SHAPE": "dd/dd/dddd"}], "id": "slash_date"},
            {"pattern": [{"SHAPE": "dd-dd-dddd"}], "id": "dash_date"},
            {"pattern": [
                {"IS_DIGIT": True, "LENGTH": 4},
                {"TEXT": "-"},
                {"IS_DIGIT": True, "LENGTH": 2},
                {"TEXT": "-"},
                {"IS_DIGIT": True, "LENGTH": 2}
            ], "id": "iso_date_explicit"}
        ],
    }

    # convert patterns to ruler format
    ruler_patterns = []
    for label, patterns in PATTERNS.items():
        for pattern in patterns:
            ruler_patterns.append({"label": label, "pattern": pattern["pattern"], 
                                 "id": pattern.get("id")})

    ruler.add_patterns(ruler_patterns)

    @Language.component("set_entity_ids")
    def set_entity_ids(doc):
        for ent in doc.ents:
            for pattern in ruler_patterns:
                if pattern["label"] == ent.label_ and "id" in pattern:
                    ent._.id = pattern["id"]
                    break
        return doc

    if "set_entity_ids" not in nlp.pipe_names:
        nlp.add_pipe("set_entity_ids", after="entity_ruler")

    logger.info("Entity patterns and extensions added successfully")
    return nlp
