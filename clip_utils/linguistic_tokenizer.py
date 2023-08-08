TAGS = dict(
    NUMERALS='numerals',
    NEGATION='negation',
    PASSIVE='passive',
    ROOT_VERB='root_verb',
    TRANSITIVITY='transitivity',
)

TAGS_SYMBOLS = dict(
    NUMERALS='NMR',
    NEGATION='NEG',
    PASSIVE='PAS',
    ROOT_VERB='RVB',
    TRANSITIVITY='TRN',
)

TAG_START = '<#'
TAG_END = '#> '


def create_tag(tag) -> str:
    return TAG_START + TAGS_SYMBOLS[tag] + TAG_END


def create_tokenizer(gpt2_type: str):
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': [create_tag(tag) for tag in TAGS]}
    )
    return tokenizer

def save_tokenizer(tokenizer, path):
    tokenizer.save_pretrained(path)

def load_tokenizer(path):
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    return tokenizer
