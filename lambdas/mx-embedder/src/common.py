import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import numpy as np
from PIL import Image
from datetime import datetime

W = 224
H = 224


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


def write_repr(dynamodb, table, img_id, repr_vec):
    table = dynamodb.Table(table)
    item = {
        'id': img_id,
        'repr': repr_vec,
        'meta': {
            'ts_updated': datetime.utcnow().isoformat(),
        }
    }
    response = table.put_item(Item=item)
    return response


def write_repr_batch(dynamodb, table, img_ids, repr_vecs):
    tic = datetime.now()
    table = dynamodb.Table(table)
    items = [{
        'id': img_id,
        'repr': repr_vec,
        'meta': {
            'ts_updated': datetime.utcnow().isoformat(),
        }
    } for img_id, repr_vec in zip(img_ids, repr_vecs)]

    with table.batch_writer() as batch:
        for item in items:
            print(f'[{datetime.now()-tic}] Batch Q {item["id"]}...')
            batch.put_item(Item=item)

    return 'Batch write queued'


def get_s3_img(s3, bucket, key):
    response = s3.get_object(Bucket=bucket, Key=str(key))
    img_orig = Image.open(response['Body'])
    # print('img_orig size: ', img_orig.size)

    # Ensure there is no 4th alpha channel
    img_arr = np.asarray(img_orig)[...,:3]
    # print('img_arr shape: ', img_arr.shape)

    img = mx.nd.array(img_arr)
    img = mx.image.imresize(img, W, H)  # resize
    img = img.transpose((2, 0, 1))  # Channel first
    img = img.astype('float32')  # for gpu context
    return img


def process_imgs(imgs, img_ids, feat_model, dynamodb, table):
    tic = datetime.now()
    print(f'[{datetime.now()-tic}] Forward Inference...')
    feats = feat_model(nd.stack(*imgs, axis=0))
    print(f'feats shape: {feats.shape}')
    resps = []
    print(f'[{datetime.now()-tic}] Writing reprs to db...')
    feat_bytes = [feat.squeeze().asnumpy().tobytes()
                  for feat in feats]

    resp = write_repr_batch(dynamodb, table,
                            img_ids, feat_bytes)
    resps.append(resp)
    return resps


def is_s3_trigger(record):
    return record.get('eventSource') == 'aws:s3' and \
           record.get('eventName') == 'ObjectCreated:Put'


def is_cw_trigger(event):
    return event.get('source') == 'aws.events'
