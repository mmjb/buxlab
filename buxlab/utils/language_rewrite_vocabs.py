from typing_extensions import Final
import gin

from .vocabulary import Vocabulary


PYTHON_OPERATOR_REWRITES: Final = frozenset(
    {
        "+",
        "-",
        "*",
        "/",
        "**",
        "//",
        "%",
        "@",
        "<<",
        ">>",
        "|",
        "&",
        "^",
        "+=",
        "-=",
        "*=",
        "/=",
        "**=",
        "//=",
        "%=",
        "@=",
        "<<=",
        ">>=",
        "|=",
        "&=",
        "^=",
        "=",
        "<",
        "<=",
        ">",
        ">=",
        "==",
        "!=",
        " in ",
        " not in ",
        " is ",
        " is not ",
        "0",
        "1",
        "2",
        "-1",
        "-2",
        "and",
        "or",
        "not ",
        "",
        "True",
        "False",
    }
)

PYTHON_OPERATOR_VOCABULARY = Vocabulary.create_vocabulary(
    PYTHON_OPERATOR_REWRITES,
    max_size=len(PYTHON_OPERATOR_REWRITES),
    count_threshold=0,
    add_unk=False,
)


@gin.configurable
def get_language_vocab(lang: str) -> Vocabulary:
    if lang.lower() == "python":
        return PYTHON_OPERATOR_VOCABULARY

    raise ValueError(f"Unknown language {lang}")
