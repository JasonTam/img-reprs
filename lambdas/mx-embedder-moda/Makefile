DIR := ${CURDIR}

FN_AWS_NAME='img-embedder'
FN_DEPS='${DIR}/../mx-embedder/src'
FN_SRC='${DIR}/src'
ZIP_BASE='${DIR}/function_base.zip'
ZIP_PATH='${DIR}/function.zip'
G_CREDS_PATH='${HOME}/google.json'
S3_BUCKET='mo-ml-dev'

pip_args := install \
		--platform manylinux1_x86_64 \
		--only-binary=:all: \
		--target .

pip_args_min := install \
		--target .

install-libs:
	mkdir -p package \
	&& cd package \
	&& pip ${pip_args} mxnet \
	&& pip ${pip_args} Pillow \
	&& pip ${pip_args} s3fs \
	&& pip ${pip_args_min} google-cloud-bigquery

build-base:
	cd package \
	&& zip -r9 ${ZIP_BASE} . \
	&& cd ${DIR}

build-code:
	cp ${ZIP_BASE} ${ZIP_PATH} \
	&& cd ${FN_DEPS} \
	&& zip -g ${ZIP_PATH} ./* \
	&& cd ${FN_SRC} \
	&& zip -g ${ZIP_PATH} ./* \
	&& zip -g -j ${ZIP_PATH} ${G_CREDS_PATH} \
	&& cd ${DIR}

build: build-base build-code

push-s3:
	aws s3 cp \
	function.zip \
	s3://${S3_BUCKET}/deploys/${FN_AWS_NAME}/function.zip


publish-local:
	aws lambda \
	--profile lambda-pub \
	update-function-code --function-name ${FN_AWS_NAME} \
	--zip-file fileb://${ZIP_PATH}


publish-s3: push-s3
	aws lambda \
	--profile lambda-pub \
	update-function-code --function-name ${FN_AWS_NAME} \
	--s3-bucket ${S3_BUCKET} \
	--s3-key deploys/${FN_AWS_NAME}/function.zip
