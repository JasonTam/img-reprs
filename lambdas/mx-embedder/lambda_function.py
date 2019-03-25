import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import numpy as np
import boto3
import json
from PIL import Image
from pathlib import Path
from datetime import datetime
from decimal import Decimal

W = 224
H = 224
BATCH_SIZE = 32

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

DYNAMO_TABLE = 'img-reprs'
DEFAULT_BUCKET = 'jason-garbage'
DEFAULT_PREFIX = 'images'


def batcher(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_model(
        model_name='densenet121',
        tap='model_flatten0_flatten0_output',
        tmpdir='/tmp/gluon',
        ):
    ctx = mx.cpu()
    model = getattr(vision, model_name)(
            pretrained=True, ctx=ctx,
            prefix='model_',
            root=tmpdir)

    inputs = mx.sym.var('data')
    out = model(inputs)
    internals = out.get_internals()

    outputs = [internals[tap]]
    model_trunc = gluon.SymbolBlock(
        outputs, inputs, params=model.collect_params())

    return model_trunc


print('Getting and cutting model...')
feat_model = get_model()


def write_repr(img_id, repr_vec):
    table = dynamodb.Table(DYNAMO_TABLE)
    item = {
            'id': img_id,
            'repr': repr_vec,
            'meta': {
                'ts_updated': datetime.utcnow().isoformat(),
            }
        }
    response = table.put_item(Item=item)
    return response


def get_s3_img(bucket, key):
    tic = datetime.now()
    response = s3.get_object(Bucket=bucket, Key=str(key))
    img_orig = Image.open(response['Body'])
    # print('img_orig size: ', img_orig.size)

    img_arr = np.asarray(img_orig)
    print('img_arr shape: ', img_arr.shape)

    print(f'[{datetime.now()-tic}] Pre-processing image...')
    img = mx.nd.array(img_arr)
    img = mx.image.imresize(img, W, H)  # resize
    img = img.transpose((2, 0, 1))  # Channel first
    img = img.astype('float32')  # for gpu context
    return img


def process_imgs(imgs, img_ids):
    tic = datetime.now()
    print(f'[{datetime.now()-tic}] Forward Inference...')
    feats = feat_model(nd.stack(*imgs, axis=0))
    resps = []
    for img_id, feat in zip(img_ids, feats):
        # Consider batch write if we actually have huge batches
        feat_compat = [
            Decimal(str(x))
            for x in feat.squeeze().asnumpy().tolist()]
        print(f'[{datetime.now()-tic}] Writing reprs to db...')
        resp = write_repr(img_id, feat_compat)
        resps.append(resp)
    return resps


def is_s3_trigger(record):
    return record.get('eventSource') == 'aws:s3' and \
           record.get('eventName') == 'ObjectCreated:Put'


def is_cw_trigger(event):
    return event.get('source') == 'aws.events'


def process_record(record):
    img_ids = []
    imgs = []
    if is_s3_trigger(record):
        bucket = record['s3']['bucket']['name']
        key = Path(record['s3']['object']['key'])
        key_name = key.name  # ex) 'cat.jpg'
        key_base = key.stem  # ex) 'cat'
        path_data = f's3://{bucket}/{key}'
        print(f'Bucket: {bucket}')
        print(f'Key: {key}', '\t', f'Key name: {key_name}')
        print(f'Path: {path_data}')

        print(f'Streaming in image from S3...')
        img = get_s3_img(bucket, key)
        imgs.append(img)
        img_ids.append(key_base)
    else:
        raise ValueError('Only s3 records supported')

    process_imgs(imgs, img_ids)


def process_dir(bucket, prefix):

    paginator = s3.get_paginator('list_objects')
    ops_params = {'Bucket': bucket,
                  'Prefix': prefix,
                  # 'PaginationConfig': {'MaxItems': 100},
                  }
    page_iter = paginator.paginate(**ops_params)

    print(f'Discovering keys in {bucket} / {prefix}')
    keys = []
    for page in page_iter:
        for obj in page['Contents']:
            if obj['Size'] > 0:
                key = Path(obj['Key'])
                keys.append(key)

    # Batch Process the list of keys
    for ii, batch in enumerate(batcher(keys, BATCH_SIZE)):
        print(f'Processing batch {ii}')
        imgs_batch = []
        ids_batch = []
        for key in batch:
            img = get_s3_img(bucket, key)
            imgs_batch.append(img)
            ids_batch.append(key.stem)

        process_imgs(imgs_batch, ids_batch)


def lambda_handler(event, context):
    tic = datetime.now()

    if 'Records' in event:
        for record in event['Records']:
            # Note: typically, there is only 1 record
            process_record(record)
    elif is_cw_trigger(event):
        print('Scheduled CW event trigger')
        process_dir(DEFAULT_BUCKET, DEFAULT_PREFIX)

    print(f'[{datetime.now()-tic}] Returning!')
    return {
        'statusCode': 200,
        'body': json.dumps('Done!')
    }
