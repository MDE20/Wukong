import model

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import json
import random
import sys
import tensorflow as tf

def eprint(s):
	sys.stderr.write(s + '\n')
	sys.stderr.flush()

def oprint(s):
	sys.stdout.write(s + '\n')
	sys.stdout.flush()

# 打印传递的参数
eprint("NOWnownow")
eprint(f"Arguments: {sys.argv}")
num_outputs = int(sys.argv[1])
model_path = sys.argv[2]
eprint(model_path)
# eprint(model_path0)

# model_path = tf.train.latest_checkpoint(model_path0)

# if model_path:
#     eprint(f"RNN model path: {model_path}")
# else:
#     eprint("No checkpoints found in the directory")
#     exit(1)


m = model.Model(num_outputs=num_outputs)
config = tf.ConfigProto(
	device_count={'GPU': 0}
)
session = tf.Session(config=config)
m.saver.restore(session, model_path)

# 尝试恢复检查点
# try:
#     m.saver.restore(session, model_path)
#     eprint("RNNModel restored successfully")
# except ValueError as e:
#     eprint("Error restoring checkpoint:", str(e))
#     exit(1)

while True:
	line = sys.stdin.readline()
	if not line:
		break
	tracks = json.loads(line.strip())
	outputs = []
	for i in range(0, len(tracks), model.BATCH_SIZE):
		if i % 1024 == 0:
			eprint('... {}/{}'.format(i, len(tracks)))
		batch = tracks[i:i+model.BATCH_SIZE]
		batch = [model.pad_track(track) for track in batch]
		batch_outputs = session.run(m.outputs, feed_dict={
			m.is_training: False,
			m.inputs: [t[0] for t in batch],
			m.lengths: [t[1] for t in batch],
		})
		outputs.extend(batch_outputs.tolist())
	s = json.dumps(outputs)
	oprint(s)
