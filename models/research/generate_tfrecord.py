import os
import tensorflow as tf
from object_detection.utils import dataset_util
from lxml import etree
import io

def xml_to_tf_example(xml_path, img_dir, label_map):
    with tf.io.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    image_path = os.path.join(img_dir, data['filename'])
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes_text = []
    classes = []

    for obj in data['object']:
        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        classes_text.append(obj['name'].encode('utf8'))

        # Map class name to numeric ID using the label_map
        if obj['name'] in label_map:
            classes.append(label_map[obj['name']])
        else:
            raise ValueError(f"Class name '{obj['name']}' not found in label map.")

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/format': dataset_util.bytes_feature(b'jpeg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tf_record(annotations_dir, img_dir, output_path, label_map):
    writer = tf.io.TFRecordWriter(output_path)
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith('.xml'):
            continue  # Skip non-XML files
        xml_path = os.path.join(annotations_dir, xml_file)
        try:
            tf_example = xml_to_tf_example(xml_path, img_dir, label_map)
            writer.write(tf_example.SerializeToString())
        except ValueError as e:
            print(f"Error processing {xml_file}: {e}")
    writer.close()

# Example usage
label_map = {
    'b': 1,
    'champaka': 2,
    'chitrak': 3,
    'common lanthana': 4,
    'daisy': 5,
    'dandelion': 6,
    'hibiscus': 7,
    'honeysuckle': 8,
    'indian mallow': 9,
    'jatropha': 10,
    'lily': 11,
    'malabar melastome': 12,
    'marigold': 13,
    'orchid': 14,
    'rose': 15,
    'shankupushpam': 16,
    'spider lily': 17,
    'sunflower': 18,
    'tulip': 19
}

create_tf_record('C:/Github-Project_file/Comvision/Lab/models/research/data/train/annotations', 'C:/Github-Project_file/Comvision/Lab/models/research/data/train/images', 'C:/Github-Project_file/Comvision/Lab/models/ssd_mobilenet_v2/train/train.record', label_map)
create_tf_record('C:/Github-Project_file/Comvision/Lab/models/research/data/val/annotations', 'C:/Github-Project_file/Comvision/Lab/models/research/data/val/images', 'C:/Github-Project_file/Comvision/Lab/models/ssd_mobilenet_v2/train/val.record', label_map)

