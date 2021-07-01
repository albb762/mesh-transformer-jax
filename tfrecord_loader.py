import jax
import tensorflow as tf
import numpy as np
from transformers import GPT2TokenizerFast
import itertools


class TFRecordLoader:
    def __init__(self, index_fname, batch_size, parse_fn, map_fn=None, restore_state=None):
        if restore_state is not None:
            self.file_idx = restore_state["file_idx"]
            self.file_idx_init = False
            self.used = restore_state["used"]
        else:
            self.file_idx = 0
            self.file_idx_init = True
            self.used = []

        self.index = open(index_fname).read().splitlines()
        self.clean_index = list(filter(lambda x: x not in self.used, self.index))
        self.bs = batch_size
        # self.seq = sample_size
        self.parse_fn = parse_fn

        if map_fn:
            self.map_fn = map_fn
        else:
            self.map_fn = lambda x: x

        self.sample_fn = self.sample_once()

    def reset(self):
        self.file_idx = 0
        self.file_idx_init = True
        self.used = []

        self.clean_index = list(filter(lambda x: x not in self.used, self.index))
        self.sample_fn = self.sample_once()

    def sample_once(self):
        for i in self.clean_index:
            

            file = tf.data.TFRecordDataset(i, compression_type='GZIP').map(self.parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
            file = file.apply(tf.data.experimental.dense_to_ragged_batch(np.prod(self.bs), drop_remainder=True))
            file = file.prefetch(10)

            for file_idx, data in enumerate(file):
                data = jax.tree_map(lambda x: x.numpy(), data)
                data = self.map_fn(data)


                self.file_idx_init = True
                self.file_idx = file_idx
                yield jax.tree_map(lambda x: x.reshape(self.bs + x.shape[1:]), data)
            self.used.append(i)
            self.file_idx = 0

    # this loops infinitely, use .sample_once to get an iterator for validation
    def get_samples(self):
        try:
            return next(self.sample_fn)
        except StopIteration:
            self.reset()
            return self.get_samples()

    def get_state(self):
        return {
            "used": self.used,
            "file_idx": self.file_idx
        }


class TFRecordNewInputs(TFRecordLoader):
    def __init__(self, index_fname, batch_size, sample_size, restore_state=None):
        def tf_parse(example_proto):
            features = {
                'input': tf.io.FixedLenFeature([168], tf.float32)
            }
            data = tf.io.parse_single_example(example_proto, feature_description)
            inp = tf.cast(data['input'][:128],tf.int32)
            return inp,inp[49:]

        super().__init__(index_fname, batch_size, tf_parse, restore_state=restore_state)





if __name__ == "__main__":
    d = TFRecordNewInputs("data/pile.val.index", (8, 32), 2048)
    for idx, i in enumerate(d.sample_once()):
        print(i)
        break

    

    print()
