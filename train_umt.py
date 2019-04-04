from data_provider.data_provider import DataProvider
from net.seq2seq import Seq2Seq

provider = DataProvider(64)

seq_seq = Seq2Seq(provider=provider)

seq_seq.train()
