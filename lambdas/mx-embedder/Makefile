
FN_AWS_NAME='img-emb-test'
FN_FILE_NAME='lambda_function.py'
ZIP_NAME='function.zip'

install-libs:
	mkdir -p package \
	&& cd package \
	&& pip install mxnet --target . \
	&& pip install Pillow --target .

build:
	cd package \
	&& zip -r9 ../${ZIP_NAME} . \
	&& cd ../ \
	&& zip -g ${ZIP_NAME} ${FN_FILE_NAME}

push-s3:
	aws s3 cp \
	--profile lambda-pub \
	function.zip \
	s3://jason-garbage/deploys/img-emb-test/function.zip


publish-local:
	aws lambda \
	--profile lambda-pub \
	update-function-code --function-name ${FN_AWS_NAME} \
	--zip-file fileb://${ZIP_NAME}


publish-s3: push-s3
	aws lambda \
	--profile lambda-pub \
	update-function-code --function-name ${FN_AWS_NAME} \
	--s3-bucket jason-garbage \
	--s3-key deploys/img-emb-test/function.zip