import jax
import tensorflow as tf
import numpy as np
from transformers import GPT2TokenizerFast
import itertools
import os


class TFRecordLoader:
    def __init__(self, path, batch_size, parse_fn, ):


        self.file_list = [os.path.join(path,f) for f in list(tf.io.gfile.listdir(path))]
        self.bs = batch_size
        # self.seq = sample_size
        self.parse_fn = parse_fn



        self.sample_fn = self.sample_once()

    def reset(self):
       pass

    def sample_once(self):
        for i in self.file_list:
 

            file = tf.data.TFRecordDataset(i, compression_type='GZIP').map(self.parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
            file = file.apply(tf.data.experimental.dense_to_ragged_batch(np.prod(self.bs), drop_remainder=True))
            file = file.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            for file_idx, data in enumerate(file):
                data = jax.tree_map(lambda x: x.numpy(), data)
                yield jax.tree_map(lambda x: x.reshape(self.bs + x.shape[1:]), data)


    # this loops infinitely, use .sample_once to get an iterator for validation
    def get_samples(self):
        try:
            return next(self.sample_fn)
        except StopIteration:
            self.reset()
            return self.get_samples()




class TFRecordNewInputs(TFRecordLoader):
    def __init__(self, index_fname, batch_size):
        def tf_parse(example_proto):
            feature_description = {
                'input': tf.io.FixedLenFeature([168], tf.float32),  
            }
            data = tf.io.parse_single_example(example_proto, feature_description)
            inp = tf.cast(data['input'][:128],tf.int32)
            return inp[:-1],inp[1:]

        super().__init__(index_fname, batch_size, tf_parse, )





if __name__ == "__main__":
    d = TFRecordNewInputs('gs://tf_cloud001/cookbook-train3/', (8, 32))
    for idx, i in enumerate(d.sample_once()):
        print(i[1].shape)
        break

