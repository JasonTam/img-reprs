import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import numpy as np
from PIL import Image
from datetime import datetime
from decimal import Decimal

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


def get_s3_img(s3, bucket, key):
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


def process_imgs(imgs, img_ids, feat_model, dynamodb, table):
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
        resp = write_repr(dynamodb, table, img_id, feat_compat)
        resps.append(resp)
    return resps


def is_s3_trigger(record):
    return record.get('eventSource') == 'aws:s3' and \
           record.get('eventName') == 'ObjectCreated:Put'


def is_cw_trigger(event):
    return event.get('source') == 'aws.events'
