import os
import pickle
import numpy as np
from numpy.linalg import norm
from PIL import Image
import streamlit as st
from sentence_transformers import SentenceTransformer
import glob
import zipfile

# 1. Load pretrained image model (CLIP backbone)
model = SentenceTransformer("clip-ViT-B-32")

# 2. Function: convert image → embedding
def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    emb = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
    return emb

# 3. Load dataset (with saved embeddings)
ZIP_PATH = r"C:\Documents\Visual Product Matcher\dataset.zip"
DATASET_DIR = r"C:\Documents\Visual Product Matcher\dataset"
EMBED_FILE = "embeddings.pkl"

# 🔹 If only zip exists, extract it
if not os.path.exists(DATASET_DIR) and os.path.exists(ZIP_PATH):
    print("📦 Extracting dataset.zip ...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
    print("✅ Extraction complete!")

if os.path.exists(EMBED_FILE):
    # load pre-computed embeddings
    with open(EMBED_FILE, "rb") as f:
        product_db = pickle.load(f)
    print("✅ Loaded saved embeddings:", len(product_db))
else:
    # build embeddings if not already saved
    image_paths = glob.glob(os.path.join(DATASET_DIR, "**", "*.jpg"), recursive=True) \
               + glob.glob(os.path.join(DATASET_DIR, "**", "*.jpeg"), recursive=True) \
               + glob.glob(os.path.join(DATASET_DIR, "**", "*.png"), recursive=True)

    print(f"🔍 Found {len(image_paths)} images")
    product_db = []

    for i, path in enumerate(image_paths, 1):
        try:
            emb = get_embedding(path)
            product_db.append({"file": os.path.basename(path), "path": path, "embedding": emb})

            if i % 50 == 0:   # progress log every 50 images
                print(f"📸 Processed {i}/{len(image_paths)} images")
        except Exception as e:
            print(f"⚠️ Error processing {path}: {e}")

    # save embeddings for next time
    with open(EMBED_FILE, "wb") as f:
        pickle.dump(product_db, f)

    print("✅ Built & saved embeddings:", len(product_db))

# 4. Similarity search (cosine similarity)
def search_similar(query_img, product_db, top_k=5):
    q_emb = get_embedding(query_img)
    sims = []
    for p in product_db:
        sim = np.dot(q_emb, p["embedding"]) / (norm(q_emb) * norm(p["embedding"]))
        sims.append((p, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

# 5. Streamlit UI
st.title("🛍️ Visual Product Matcher")

uploaded_file = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    query_path = "query.jpg"
    with open(query_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(query_path, caption="Query Image", width=200)

    st.markdown(
    """
    <title> 🛍️ Visual Product Matcher</title>
    <style>
    div[data-testid="stHorizontalBlock"] {
        justify-content: flex-start;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    results = search_similar(query_path, product_db, top_k=5)
    st.subheader("Top Matches:")
    cols = st.columns(len(results)) 
    
    for i, (match, score) in enumerate(results):
        with cols[i]:
            base = os.path.splitext(match["file"])[0]
            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = os.path.join("images", base + ext)
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"{match['file']}\nScore={score:.3f}")
                    break









