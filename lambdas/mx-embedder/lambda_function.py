from pathlib import Path
from datetime import datetime
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import numpy as np
import boto3
import json
from PIL import Image

W = 224
H = 224

s3 = boto3.client('s3')


def get_model():
    ctx = mx.cpu()
    model = vision.densenet121(pretrained=True, ctx=ctx,
                               prefix='model_',
                               root='/tmp/gluon')

    inputs = mx.sym.var('data')
    out = model(inputs)
    internals = out.get_internals()

    outputs = [internals['model_flatten0_flatten0_output']]
    feat_model = gluon.SymbolBlock(
        outputs, inputs, params=model.collect_params())

    return feat_model


def lambda_handler(event, context):
    tic = datetime.now()

    # `s3:ObjectCreated:Put` can only ever create 1 record. Take head.
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = Path(event['Records'][0]['s3']['object']['key'])
    key_name = key.name  # ex) 'cat.jpg'
    key_base = key.stem
    path_data = f's3://{bucket}/{key}'
    print(f'Bucket: {bucket}')
    print(f'Key: {key}', '\t', f'Key name: {key_name}')
    print(f'Path: {path_data}')

    print(f'[{datetime.now()-tic}] Streaming in image from S3...')

    response = s3.get_object(Bucket=bucket, Key=str(key))
    img_orig = Image.open(response['Body'])
    print('img_orig size: ', img_orig.size)

    img_arr = np.asarray(img_orig)
    print('img_arr shape: ', img_arr.shape)

    print('Pre-processing image...')
    img = mx.nd.array(img_arr)
    img = mx.image.imresize(img, W, H)  # resize
    img = img.transpose((2, 0, 1))  # Channel first
    img = img.expand_dims(axis=0)  # batchify
    img = img.astype('float32')  # for gpu context

    print('Getting and cutting model...')
    feat_model = get_model()

    print('Forward Inference...')
    feats = feat_model(img)

    print(feats)

    print(f'[{datetime.now()-tic}] Returning!')
    return {
        'statusCode': 200,
        'body': json.dumps('Done!')
    }
