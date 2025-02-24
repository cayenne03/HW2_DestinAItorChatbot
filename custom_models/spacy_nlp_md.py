import spacy
from custom_models.spacy_entity_patterns import setup_entity_patterns
from mylogger import get_logger

logger = get_logger(__name__)

class SpacyNLPManager:
    _instance = None
    _nlp = None

    @classmethod
    def get_nlp(cls):
        if cls._nlp is None:
            try:
                logger.info("Initializing spaCy model...")
                cls._nlp = spacy.load("en_core_web_md")
                setup_entity_patterns(cls._nlp)
                logger.info("SpaCy model initialized with custom entity patterns")
            except Exception as e:
                logger.error(f"Error initializing spaCy model: {str(e)}")
                raise
        return cls._nlp

# create a single instance to be imported
try:
    nlp = SpacyNLPManager.get_nlp()
except Exception as e:
    logger.error(f"Failed to initialize NLP: {str(e)}")
    raise