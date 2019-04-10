from google.cloud import bigquery as bq
import boto3
import s3fs
import json
import requests
from datetime import datetime
from common import (
    get_model,
    pp_img_obj,
    process_imgs,
)
from PIL import Image
from io import BytesIO


BATCH_SIZE = 32
# number of batches before re-uploading
# `processed_ids` to s3
INDEX_UPLOAD_PERIOD = 5

s3 = s3fs.S3FileSystem()
dynamodb = boto3.resource('dynamodb')

DYNAMO_TABLE = 'img-reprs'
DEFAULT_BUCKET = 'mo-ml-dev'
PATH_PROCESSED_SET = 's3://mo-ml-dev/processed_ids.txt'
QUALITY = 'medium'

id_col = 'variant_id'
img_url_col = 'product_primary_image'

q = f"""
SELECT DISTINCT {id_col}
, {img_url_col} 
FROM `moda-operandi-dw.dw_production.dw__dim_sku` 
WHERE product_primary_image != ''
"""


print('Getting and cutting model...')
feat_model = get_model()


def run():
    tic = datetime.now()

    # Get processed ids (assuming ids are int)
    if s3.exists(PATH_PROCESSED_SET):
        with s3.open(PATH_PROCESSED_SET, 'rb') as f:
            processed_ids = set(
                int(b) for b in f.read().splitlines())
    else:
        processed_ids = set()

    # BQ Img Urls
    client_bq = bq.Client()
    query_job = client_bq.query(q)

    # Go through and batch process results from BQ
    # TODO: not pretty
    header = 'images/products/'
    ids_batch = []
    imgs_batch = []
    n_batches_processed = 0
    n_imgs_processed = 0

    for row in query_job:
        if row[id_col] not in processed_ids:
            if header not in row[img_url_col]:
                continue
            cdn, key = row[img_url_col].split(header)
            p_id, v_id, name = key.split('/')
            img_url = f'{cdn}{header}{p_id}/{v_id}/{QUALITY}_{name}'
            resp = requests.get(img_url)
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content))
                img_pp = pp_img_obj(img)
                if img_pp is None:
                    continue
                ids_batch.append(row[id_col])
                imgs_batch.append(img_pp)

        if len(ids_batch) >= BATCH_SIZE:
            process_imgs(
                imgs_batch, ids_batch, feat_model, dynamodb, DYNAMO_TABLE)
            processed_ids = processed_ids.union(ids_batch)
            n_imgs_processed += len(ids_batch)
            n_batches_processed += 1
            ids_batch = []
            imgs_batch = []
            print(f'[{datetime.now()-tic}] #processed: {n_imgs_processed}')
            if (n_batches_processed % INDEX_UPLOAD_PERIOD) == 0:
                print(f'[{datetime.now()-tic}] Updating processed set on s3')
                with s3.open(PATH_PROCESSED_SET, 'wb') as f:
                    for processed_id in processed_ids:
                        f.write(f'{processed_id}\n'.encode())

    # Process remaining stuff in queue
    # TODO: repeated code
    process_imgs(
        imgs_batch, ids_batch, feat_model, dynamodb, DYNAMO_TABLE)
    processed_ids = processed_ids.union(ids_batch)
    n_imgs_processed += len(ids_batch)
    n_batches_processed += 1
    print(f'[{datetime.now()-tic}] #processed: {n_imgs_processed}')
    print(f'[{datetime.now()-tic}] Updating processed set on s3')
    with s3.open(PATH_PROCESSED_SET, 'wb') as f:
        for processed_id in processed_ids:
            f.write(f'{processed_id}\n'.encode())

    return n_imgs_processed


def lambda_handler(event, context):
    tic = datetime.now()

    n_processed = run()

    print(f'[{datetime.now()-tic}] Processed: {n_processed}')
    return {
        'statusCode': 200,
        'body': json.dumps('Done!')
    }
