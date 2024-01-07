# mSUsIE: multimodal Search Using Image Encoders (Group 15)
## Team members
- Benedikt Kantz: Model Search, Dataset preparation  \
- Corinna Kindlhofer: Evaluation, Metrics

## How to run
- Start by running the Docker containers for the Milvus database (`docker-compose up -d`)
- The first model (clip-ViT-B-32) in `data_loading.ipynb` is ready to use without any additional setup.
- For the second model (clip-resnet-101-visual-float32), follow the steps:
  - Download the [visual](https://huggingface.co/mlunar/clip-variants/blob/main/models/clip-resnet-101-visual-float32.onnx) and [textual](https://huggingface.co/mlunar/clip-variants/blob/main/models/clip-resnet-101-textual-float32.onnx) model from [Hugging Face](https://huggingface.co/mlunar/clip-variants/tree/main/models)
  - Place the downloaded models in the `/models` folder

## Running the Web Interface

- Run the Docker containers (`docker-compose up -d`)
- Make sure that `flask` & all other dependencies are installed (run the `visual_querying.ipynb` first)
- Run the `visual_querying.ipynb` notebook to load the embeddings into the database
- Place the data into the `service/static` folder
- Run the app `flask --app hello run` and go to `http://127.0.0.1:5000/index.html`


## Dataset

The dataset is private and can thus not be shared. You can, however, place your png's into the `data`-folder and then run and try the visual querying. The dataset was simplified and generated using `prepare-queries.ipynb`