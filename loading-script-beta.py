import datasets
import pyarrow.parquet as pq

_DESCRIPTION = """\
This is the DeCaBERT corpus, consisting of languages from the widely-discredited Dené-Caucasian language macrofamily.
"""
_HOMEPAGE = "https://github.com/olworth/decabert"

_CITATION = ""

_VERSION= "0.0.1"

_LANGCODES = ["abk", "ahk", "apw", "bod", "cdo", "che", "csy", "dzo",
            "eus", "gan", "gwi", "kac", "ksw", "kbd", "lhu", "lus",
            "mya", "nan", "nav", "new", "suz", "wuu", "yue", "zho"]

_URLS = {
    langcode: {
            "train":f"https://huggingface.co/datasets/homersimpson/DeCaBERT-dataset-2/blob/main/{langcode}/{langcode}.parquet"
    } for langcode in _LANGCODES    
}

class DeCaBERTCorpusConfig(datasets.BuilderConfig):
    def __init__(self, version, name, description):
        self.version = version
        self.name = name
        self.description = description

class DeCaBERTCorpus(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
#        (DeCaBERTCorpusConfig(
#            version=_VERSION,
#            name="all",
#            description=_DESCRIPTION
#    ),
        DeCaBERTCorpusConfig(
            version=_VERSION,
            name=langcode,
            description=f"{langcode} subsection of the DeCaBERT corpus."
        ) for langcode in _LANGCODES
    ]
    #DEFAULT_CONFIG_NAME = "all"
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                },
            ),
            supervised_keys = None,
            homepage = _HOMEPAGE,
            citation = _CITATION,
        )

    def _split_generators(self, dl_manager):
      downloaded_files = dl_manager.download_and_extract(_URLS[self.config.name])
      return [
        datasets.SplitGenerator(
          name=datasets.Split.TRAIN,
          gen_kwargs={
            "filepath": downloaded_files['train']
          }
        )
      ]
        
    def _generate_examples(self, filepath):
      with open(filepath) as f:
            for index, line in enumerate(f):
                entry = (
                    index,
                    {
                        "id": index,
                        "text": line,
                    },
                )
                yield entry
