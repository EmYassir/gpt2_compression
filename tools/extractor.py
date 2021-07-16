import os
import re
from itertools import chain
import datasets


_CITATION = """\
@misc{Gokaslan2019OpenWeb,
  title={OpenWebText Corpus},
  author={Aaron Gokaslan*, Vanya Cohen*, Ellie Pavlick, Stefanie Tellex},
  howpublished{\\url{http://Skylion007.github.io/OpenWebTextCorpus}},
  year={2019}
}
"""

_DESCRIPTION = """\
An open-source replication of the WebText dataset from OpenAI.
"""

class OpenWebText(datasets.GeneratorBasedBuilder):
    """The Open WebText dataset."""
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            description="Plain text",
            version=datasets.Version("1.0.0"),
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"text": datasets.Value("string")}),
            homepage="https://skylion007.github.io/OpenWebTextCorpus/",
            citation=_CITATION,
        )

    def get_text_files(self, dl_manager, filespath):
        subset_xzs = [
            os.path.join(filespath, file_name)
            for file_name in sorted(os.listdir(filespath))
            if file_name.endswith("xz")  # filter out everything else
        ]
        ex_dirs = dl_manager.extract(subset_xzs, num_proc=round(os.cpu_count() * 0.75))
        nested_txt_files = [
            [
                os.path.join(ex_dir, txt_file_name)
                for txt_file_name in sorted(os.listdir(ex_dir))
                if txt_file_name.endswith("txt")
            ]
            for ex_dir in ex_dirs
        ]
        txt_files = chain(*nested_txt_files)
        return txt_files

    def generate_examples(self, txt_files):
        """ Yields examples. """
        for idx, filepath in enumerate(txt_files):
            with open(filepath, encoding="utf-8") as f:
                yield idx, {"text": re.sub("\n\n\n+", "\n\n", f.read()).strip()}
    
    def merge_files(self, txt_files, final_output):
        """ Merges files. """
        with open(final_output, "w", encoding="utf-8") as out:
            for idx, filepath in enumerate(txt_files):
                with open(filepath, encoding="utf-8") as f:
                    stream = re.sub("\n\n\n+", "\n\n", f.read()).strip()
                    out.write(stream + '\n\n')
                

if __name__ == "__main__":
    # execute only if run as a script
    print("### Loading extractor...")
    owt_cls = OpenWebText()
    print("### Extracting text files...")
    text_files = owt_cls.get_text_files(datasets.DownloadManager(), '/media/data/openwebtext/archives/')
    print("### Merging text data...")
    owt_cls.merge_files(text_files, '/media/data/openwebtext/all-merged-fixed.txt')
    print("### Done!!!")

