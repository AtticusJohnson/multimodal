# coding=utf-8
# Copyright (c) 2019 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
sys.path.append("../..")
import os
import tensorflow as tf
from easytransfer.losses import softmax_cross_entropy
from easytransfer import Config
from easytransfer import base_model
from easytransfer import layers, FLAGS
from easytransfer import model_zoo
from easytransfer import preprocessors
from easytransfer.datasets import CSVReader, OdpsTableReader, CSVWriter, OdpsTableWriter
from easytransfer.evaluators import classification_eval_metrics



class Application(base_model):
    def __init__(self, **kwargs):
        super(Application, self).__init__(**kwargs)

    def build_logits(self, features, mode=None):

        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path, user_defined_config=self.config)
        model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)
        dense = layers.Dense(self.num_labels, kernel_initializer=layers.get_initializer(0.02), name='dense')

        input_ids, input_mask, segment_ids, label_ids = preprocessor(features)
        outputs = model([input_ids, input_mask, segment_ids], mode=mode)
        pooled_output = outputs[1]
        logits = dense(pooled_output)
        return logits, label_ids

    def build_loss(self, logits, labels):
        return softmax_cross_entropy(labels, self.num_labels, logits)

    def build_eval_metrics(self, logits, labels):
        return classification_eval_metrics(logits, labels, self.num_labels)
    
    def build_predictions(self, output):
        logits, _ = output
        predictions = dict()
        predictions["predictions"] = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return predictions

    
def main(_):
    app = Application()
    if FLAGS.mode == "train_and_evaluate_on_the_fly":
        if "PAI" in tf.__version__:
            train_reader = OdpsTableReader(input_glob=app.train_input_fp, is_training=True, input_schema=app.input_schema, batch_size=app.train_batch_size)
            eval_reader = OdpsTableReader(input_glob=app.eval_input_fp, is_training=False, input_schema=app.input_schema, batch_size=app.eval_batch_size)
        else:
            train_reader = CSVReader(input_glob=app.train_input_fp, is_training=True, input_schema=app.input_schema, batch_size=app.train_batch_size)
            eval_reader = CSVReader(input_glob=app.eval_input_fp, is_training=False, input_schema=app.input_schema, batch_size=app.eval_batch_size)
        app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)
        
    if FLAGS.mode == "predict_on_the_fly":
        if "PAI" in tf.__version__:
            pred_reader = OdpsTableReader(input_glob=app.predict_input_fp, input_schema=app.input_schema, is_training=False, batch_size=app.predict_batch_size)
            pred_writer = OdpsTableWriter(output_glob=app.predict_output_fp, output_schema=app.output_schema, slice_id=0, input_queue=None)
        else:
            pred_reader = CSVReader(input_glob=app.predict_input_fp, is_training=False, input_schema=app.input_schema, batch_size=app.predict_batch_size)
            pred_writer = CSVWriter(output_glob=app.predict_output_fp, output_schema=app.output_schema)
        app.run_predict(reader=pred_reader, writer=pred_writer, checkpoint_path=app.predict_checkpoint_path)
        

if __name__ == '__main__':
    tf.app.run()
