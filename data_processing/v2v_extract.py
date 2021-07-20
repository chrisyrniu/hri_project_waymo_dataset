import os
import tensorflow as tf

# Features of other agents.
state_features = {
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/objects_of_interest':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None)
}

features_description = {}
features_description.update(state_features)

start = 0
end = 1000
idx_file = open("file_idx.txt","w")
idx_file.close()

for i in range(start, end):
    cmd = f'/home/denso/yniu/google_cloud_sdk/google-cloud-sdk/bin/gsutil cp gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/training/training_tfexample.tfrecord-0{i:04d}-of-01000 ./data'
    os.system(cmd)

    FILENAME = f'./data/training_tfexample.tfrecord-0{i:04d}-of-01000'
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    data = next(dataset.as_numpy_iterator())
    parsed = tf.io.parse_single_example(data, features_description)
    ia_idx = tf.where((parsed['state/objects_of_interest']==1)).numpy()
    
    if ia_idx.shape[0] == 2:
        print(parsed['state/type'][ia_idx[0][0]])
        print(parsed['state/type'][ia_idx[1][0]])
        if parsed['state/type'][ia_idx[0][0]] == 1 and parsed['state/type'][ia_idx[1][0]] == 1:
            idx_file = open("file_idx.txt","a")
            idx_file.write(f'{i} \n')
            idx_file.close()
            continue

    rm_cmd = f'rm ./data/training_tfexample.tfrecord-0{i:04d}-of-01000'
    os.system(rm_cmd)