from data_provider.data_provider import DataProvider
from net.seq2seq import Seq2Seq

provider = DataProvider(1)

seq_seq = Seq2Seq(provider=provider)

seq_seq.predict(["hace mucho frio aqui."])
