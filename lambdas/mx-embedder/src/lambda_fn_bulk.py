import boto3
import json
from pathlib import Path
from datetime import datetime
from common import (
    batcher,
    get_model,
    get_s3_img,
    process_imgs,
    is_cw_trigger,
    is_s3_trigger,
)

BATCH_SIZE = 32

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

DYNAMO_TABLE = 'img-reprs'
DEFAULT_BUCKET = 'jason-garbage'
DEFAULT_PREFIX = 'images'


print('Getting and cutting model...')
feat_model = get_model()


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
            img = get_s3_img(s3, bucket, key)
            imgs_batch.append(img)
            ids_batch.append(key.stem)

        process_imgs(imgs_batch, ids_batch, feat_model, dynamodb, DYNAMO_TABLE)


def lambda_handler(event, context):
    tic = datetime.now()

    if is_cw_trigger(event):
        print('Scheduled CW event trigger')
        process_dir(DEFAULT_BUCKET, DEFAULT_PREFIX)

    print(f'[{datetime.now()-tic}] Returning!')
    return {
        'statusCode': 200,
        'body': json.dumps('Done!')
    }
