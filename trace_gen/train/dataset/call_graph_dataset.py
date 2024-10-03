import logging
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe


logger = logging.getLogger(__name__)

class CallGraphDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.epoch = 0
        self.current_size = 0
        self.content_field = "content"
        self.max_buffer_size = seq_length * num_of_sequences

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        cnt = 0
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    cg_str = next(iterator)[self.content_field]
                    cnt += 1
                    buffer.append(cg_str)
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                        logger.info(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input[1:])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield torch.tensor(input_ids)

    def shuffle(self, buffer_size=1000):
        return ShufflerIterDataPipe(self, buffer_size=buffer_size)
