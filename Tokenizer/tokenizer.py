import sentencepiece as spm
from typing import List, Union


class Tokenizer:
    def __init__(self, model_path: str, data_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.model_path = model_path
        self.data_path = data_path
        try:
            self.loader()
        except Exception as e:
            self.trainner()
            self.loader()
        self.sp.SetEncodeExtraOptions("bos:eos")

    def trainner(self):
        
        spm.SentencePieceTrainer.Train(
            f'--input={self.data_path} '
            '--model_prefix=bpe '
            '--vocab_size=16000 '
            '--model_type=bpe '
            '--character_coverage=1.0 '

            '--pad_id=0 '
            '--unk_id=1 '
            '--bos_id=2 '
            '--eos_id=3 '

            # Custom tokens go AFTER the system tokens automatically
            '--control_symbols=<start>,<end>'
        )


    def loader(self):
        self.sp.load(f"{self.model_path}/bpe.model")
        print("SentencePiece model loaded successfully!")

    def encode(self, sentence: str, encode_type=''):

        if encode_type == 'tokens':
            return self.sp.EncodeAsIds(sentence)
        elif encode_type == 'pieces':
            return self.sp.EncodeAsPieces(sentence)
        else:
            return ValueError("encode_type must be either 'tokens' or 'pieces'")

    def decode(self, encode_file: Union[List[int], List[str]]):

        if isinstance(encode_file[0], str):
            return self.sp.DecodePieces(encode_file)
        return self.sp.DecodeIds(encode_file)

    def encode_batch(self, sentences, encode_type='tokens'):
        """
        sentences: list of str
        returns: list of list[int] or list of list[str]
        """
        if encode_type == 'tokens':
            return self.sp.EncodeAsIds(sentences)
        elif encode_type == 'pieces':
            return self.sp.EncodeAsPieces(sentences)
        else:
            raise ValueError("encode_type must be either 'tokens' or 'pieces'")
    
    
if __name__ == "__main__":
    # for checking purpose
    token = Tokenizer(model_path="..", data_path="../../Data/dev_test/dev.all")

    sample_sentence = "पुलिस की प्रारंभिक जांच में सामने आया है कि कोमल आर्थिक तंगी से परेशान थी व इसी वजह से उसने यह कदम उठाया।"

    print(token.encode(sample_sentence, encode_type='tokens'))
    print(token.encode(sample_sentence, encode_type='pieces'))
