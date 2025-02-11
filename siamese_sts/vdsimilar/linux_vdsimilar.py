from torch.functional import norm
from siamese_sts.dataLoader import VDSData
from siamese_sts.siamese_net import SiameseBiLSTMAttention
from siamese_sts.trainer.train import train_model
from gensim.models import keyedvectors
import torch
torch.cuda.empty_cache()
from torch import nn
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def main():
    ## define configurations and hyperparameters
    columns_mapping = {
        "vul1": "sentence_A",
        "vul2": "sentence_B",
        "label": "1",
    }
    dataset_name = "new_linux_data"
    linux_data = VDSData(dataset_name=dataset_name)
    sick_dataloaders = linux_data.cross_validation()
    batch_size = 64
    output_size = 1
    hidden_size = 32
    ''' word2vec_model = keyedvectors.load_word2vec_format('siamese_sts\dataLoader\word2vec.model')
    # 创建一个空的 PyTorch 嵌入层
    embedding_layer = torch.nn.Embedding(word2vec_model.vector_size, word2vec_model.vocab_size)

    # 将预训练的词嵌入权重复制到 PyTorch 嵌入层
    embedding_layer.weight.data.copy_(torch.from_numpy(word2vec_model.vectors))'''
    vocab_size = len(linux_data.vectorizer.embeddings)
    embedding_size = 64
    embedding_weights = torch.from_numpy(linux_data.vectorizer.embeddings.vectors).float()
    lstm_layers = 4
    learning_rate = 0.001
    fc_hidden_size = 64
    max_epochs = 150
    bidirectional = True
    device = torch.device("cpu")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## self attention config
    self_attention_config = {
        "hidden_size": 150,  ## refers to variable 'da' in the ICLR paper
        "output_size": 20,  ## refers to variable 'r' in the ICLR paper
        "penalty": 0.6,  ## refers to penalty coefficient term in the ICLR paper
    }

    ## init siamese lstm
    siamese_lstm_attention = SiameseBiLSTMAttention(
        batch_size=batch_size,
        output_size=output_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        embedding_weights=embedding_weights,
        #embedding_layer=embedding_layer,
        lstm_layers=lstm_layers,
        self_attention_config=self_attention_config,
        fc_hidden_size=fc_hidden_size,
        device=device,
        bidirectional=bidirectional,
    )

    ## define optimizer and loss function
    optimizer = torch.optim.Adamax(params=siamese_lstm_attention.parameters(), lr=learning_rate)

    train_model(
        model=siamese_lstm_attention,
        optimizer=optimizer,
        dataloader=sick_dataloaders,
        data=linux_data,
        max_epochs=max_epochs,
        config_dict={
            "device": device,
            "model_name": "siamese_lstm_attention",
            "self_attention_config": self_attention_config,
        },
    )


if __name__ == "__main__":
    main()
