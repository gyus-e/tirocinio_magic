* Prerequisites

- python 3.11
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- postgres


* Optional: create venv

```sh
python -m venv .venv
source .venv/bin/activate
```

* Install requirements

```sh
pip install -r requirements.txt
pip install -r requirements_torch.txt
```

* Set your environment variables

Rename `.env.example` to `.env` and change the values accordingly.
- HF_TOKEN is required to download models from HuggingFace.
- DOCUMENTS_DIR specifies the directory containing the documents.
- HF_HOME specifies the directory to which the models will be downloaded. You can leave it to the default value.
- DB_DRIVER, DB_NAME and DB_PASSWORD are used to connect to the relational database. Check the file `environ.py` to see how the URL is built.

* Run test

- Optional: you can change the model and system prompt used to run the tests in `test_utils/configuration_mock.py`
- Optional: you can change the test questions in `test_utils/questions_mock.py`

```sh
python test.py
```

* Run webapp

```sh
python app.py
```