
from PIL import Image
import pandas as pd

import regex as rg
import glob
from tqdm.notebook import tqdm

def openImg(p:str):
    try:
        img=Image.open(p)
        return img
    except Exception as e:
        print(e)
        return None
    
def create_ground_truth_dicts():
    qrels_df = pd.read_csv('data/qrels_simplified.csv')
    queries_df = pd.read_csv('data/queries.csv')
    merged_df = pd.merge(qrels_df, queries_df, on='qid')

    merged_df['path'] = merged_df['path'].apply(lambda p:" ".join(rg.split(r'[\\|/_\\.-]', p)).strip())

    ground_truth_de = {}
    ground_truth_en = {}

    for index, row in merged_df.iterrows():
        query_de = row['query_de']
        query_en = row['query_en']
        path = row['path']
        ground_truth_de.setdefault(query_de, []).append(path)
        ground_truth_en.setdefault(query_en, []).append(path)

    return ground_truth_de, ground_truth_en

def check_for_invalid_imgs():
    img_paths=glob.glob("data/Cliparts/01_Kate Hadfield/**/*.png", recursive=True)

    for file in tqdm(img_paths):
        try:
            img = Image.open(file)
        except Exception as e:
            print("Bad file: ", file)

def precision_air(target, pred, k=10):
    rel_set = set(target)
    # print(rel_set)
    doc_set = set(pred[:k])
    tp = len(doc_set.intersection(rel_set))  # docs that are in both -relevant docs
    fp = len(
        doc_set.difference(rel_set)
    )  # docs that are not in relevant set - irrelevant docs (false positiv)
    fn = len(
        rel_set.difference(doc_set)
    )  # relevant docs that are not present in doc set - missing docs
    if tp == 0:
        return 0
    precision = tp / (tp + fp)
    return precision

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
cliparts = None
cliparts_txt = None
def milvus_connect():
    connections.connect("default", host="localhost", port="19530")

def setup_collections(emb_dims=512):
    global cliparts
    global cliparts_txt
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="img_embeddings", dtype=DataType.FLOAT_VECTOR, dim=emb_dims),
        FieldSchema(name="text", dtype=DataType.VARCHAR, dim=emb_dims, max_length=1024),
        FieldSchema(name="path", dtype=DataType.VARCHAR, dim=emb_dims, max_length=1024),
        FieldSchema(name='txt_embeddings', dtype=DataType.ARRAY, element_type=DataType.FLOAT, max_capacity=emb_dims)
    ]
    schema = CollectionSchema(fields, "Clipart collection")
    cliparts = Collection("cliparts", schema,
            enable_dynamic_field=True)
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="txt_embeddings", dtype=DataType.FLOAT_VECTOR, dim=emb_dims),
        FieldSchema(name="text", dtype=DataType.VARCHAR, dim=emb_dims, max_length=1024),
        FieldSchema(name="path", dtype=DataType.VARCHAR, dim=emb_dims, max_length=1024),
        FieldSchema(name='img_embeddings', dtype=DataType.ARRAY, element_type=DataType.FLOAT, max_capacity=emb_dims)
    ]
    schema = CollectionSchema(fields, "Clipart collection (text vectors)")
    cliparts_txt = Collection("cliparts_txt", schema,
            enable_dynamic_field=True)