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


s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

DYNAMO_TABLE = 'img-reprs'
DEFAULT_BUCKET = 'jason-garbage'
DEFAULT_PREFIX = 'images'


print('Getting and cutting model...')
feat_model = get_model()


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
        img = get_s3_img(s3, bucket, key)
        imgs.append(img)
        img_ids.append(key_base)
    else:
        raise ValueError('Only s3 records supported')

    process_imgs(imgs, img_ids, feat_model, dynamodb, DYNAMO_TABLE)


def lambda_handler(event, context):
    tic = datetime.now()

    if 'Records' in event:
        for record in event['Records']:
            # Note: typically, there is only 1 record
            process_record(record)

    print(f'[{datetime.now()-tic}] Returning!')
    return {
        'statusCode': 200,
        'body': json.dumps('Done!')
    }
