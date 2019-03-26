
FN_AWS_NAME='img-emb-test'
FN_SRC='src'
ZIP_NAME='function.zip'

pip_args := install \
		--platform manylinux1_x86_64 \
		--only-binary=:all: \
		--target . 

install-libs:
	mkdir -p package \
	&& cd package \
	&& pip ${pip_args} mxnet \
	&& pip ${pip_args} Pillow

build:
	cd package \
	&& zip -r9 ../${ZIP_NAME} . \
	&& cd ../${FN_SRC} \
	&& zip -g ../${ZIP_NAME} ./* \
	&& cd ..

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
