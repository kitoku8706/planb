# app.py (Streamlit UI for Hazard Classifier: awl/knife/scissor)
# %%writefile app.py
import os, io, json, time, itertools
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# =================== 자동 다운로드 + 튼튼한 로더 ===================
def _get_secret(*keys, env_key=None, default=""):
    """
    st.secrets['a']['b'] 혹은 st.secrets['A_B'] → 실패 시 os.environ[env_key] → default
    """
    # 계층형
    try:
        d = st.secrets
        for k in keys:
            d = d[k]
        if d:
            return str(d)
    except Exception:
        pass
    # 평평한
    try:
        flat_key = "_".join(keys).upper()
        v = st.secrets.get(flat_key, "")
        if v:
            return str(v)
    except Exception:
        pass
    # 환경변수
    if env_key:
        v = os.environ.get(env_key, "")
        if v:
            return str(v)
    return default

def ensure_file_via_gdown(local_path: str, file_id: str) -> str | None:
    """
    local_path 없고 file_id 있으면 Google Drive에서 다운로드
    """
    if os.path.exists(local_path):
        return local_path
    if not file_id:
        return None
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, local_path, quiet=False)
        return local_path if os.path.exists(local_path) else None
    except Exception as e:
        st.error(f"gdown 다운로드 실패: {e}")
        return None

def load_model_robust(path: str):
    """
    .keras(Keras3) 우선 → 실패 시 tf.keras 로더(compose=False)
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"모델 경로가 존재하지 않습니다: {path}")
    try:
        import keras
        return keras.saving.load_model(path)
    except Exception:
        return tf.keras.models.load_model(path, compile=False)
# ================================================================

# --------------- UI 기본 설정 ---------------
st.set_page_config(page_title="Hazard Classifier UI", layout="wide")
st.title("🔪 Hazard Classifier (10종)")
st.caption("ResNet50 / MobileNetV2 모델 체크포인트로 예측 · 시각화 · 리포트")

# --------------- 사이드바 설정 ---------------
with st.sidebar:
    st.header("⚙️ 설정")

    backbone = st.selectbox(
        "백본(전처리 선택)",
        options=["ResNet50", "MobileNetV2"],
        index=0,
        help="모델이 어떤 전처리를 썼는지에 따라 선택"
    )

    # ✅ Streamlit 환경에서는 현재 작업 디렉토리 기준이 가장 안전
    default_model = "./hazard_resnet50_eye_6k.keras"
    default_labelmap = "./class_to_idx.json"
    if backbone == "MobileNetV2":
        default_model = "./hazard_mobilenetv2.keras"
        default_labelmap = "./class_to_idx.json"

    model_path = st.text_input(
        "모델 경로(.keras) (비워두면 자동 다운로드 시도)",
        value=default_model
    )
    labelmap_path = st.text_input(
        "라벨맵 경로(class_to_idx.json) (비워두면 자동 다운로드 시도)",
        value=default_labelmap
    )

    thresh = st.slider("불확실 임계치(↓면 과감, ↑면 보수)", min_value=0.0, max_value=0.99, value=0.75, step=0.01)
    topk = st.slider("Top-K 확률 표시", min_value=1, max_value=10, value=10, step=1)

    st.markdown("---")
    st.subheader("📂 폴더 일괄 예측 (선택)")
    batch_dir = st.text_input("폴더 경로(이미지들, 앱 서버 파일시스템 경로)", value="")
    show_grid = st.checkbox("그리드로 이미지/결과 미리보기", value=True)

    st.markdown("---")
    st.subheader("🧪 Test 폴더 리포트 (선택)")
    test_dir = st.text_input("Test 폴더 루트 (class별 하위폴더 구조)", value="")

    st.markdown("---")
    st.caption("Tip: ResNet50은 resnet50 전처리, MobileNetV2는 mobilenet_v2 전처리를 사용해야 결과가 정확합니다.")
    st.caption("Secrets에 gdrive.model_file_id / labelmap_file_id 를 넣으면, 경로가 비었을 때 자동 다운로드합니다.")

# --------------- 전처리 함수 ---------------
@st.cache_resource(show_spinner=False)
def get_preprocess(backbone_name: str):
    if backbone_name == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    else:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input

# 라벨맵 로더
@st.cache_resource(show_spinner=False)
def load_labelmap_safe(path: str):
    if not path or not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        _ = [idx_to_class[i] for i in sorted(idx_to_class.keys())]  # 검증
        return class_to_idx, idx_to_class
    except Exception as e:
        st.error(f"라벨맵 로드 실패: {e}")
        return None, None

def is_image_file(name: str):
    name = name.lower()
    return any(name.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"])

def safe_open_image(fp: str):
    # 손상/숨김 파일 안전 핸들
    base = os.path.basename(fp)
    if base.startswith(".") or base.startswith("._"):
        raise UnidentifiedImageError("hidden/meta file")
    if not os.path.isfile(fp) or os.path.getsize(fp) == 0:
        raise UnidentifiedImageError("empty or not file")
    with Image.open(fp) as im:
        im.verify()  # 헤더 검증
    im = Image.open(fp).convert("RGB")     # 실제 로딩
    return im

def preprocess_image(im: Image.Image, img_size=(224,224), preprocess=None):
    im_resized = im.resize(img_size)
    x = tf.keras.preprocessing.image.img_to_array(im_resized)
    x = preprocess(x)
    x = np.expand_dims(x, axis=0)
    return x

def predict_image(model, preprocess, im: Image.Image, idx_to_class, threshold=0.75, topk=10):
    x = preprocess_image(im, (224,224), preprocess)
    prob = model.predict(x, verbose=0)[0]  # (num_classes,)
    order = np.argsort(prob)[::-1]
    top = [(idx_to_class[i], float(prob[i])) for i in order[:topk]]
    best_idx = int(order[0]); best_cls = idx_to_class[best_idx]; best_conf = float(prob[best_idx])
    label = best_cls if best_conf >= threshold else "uncertain"
    return label, best_conf, top, prob

# --------------- 모델/라벨맵 로드 ---------------
preprocess = get_preprocess(backbone)

# Secrets → File ID 불러오기
model_file_id = _get_secret("gdrive", "model_file_id", env_key="MODEL_FILE_ID", default="")
labelmap_file_id = _get_secret("gdrive", "labelmap_file_id", env_key="LABELMAP_FILE_ID", default="")

# 모델/라벨맵 경로가 없으면 자동 다운로드 시도
effective_model_path = model_path if (model_path and os.path.exists(model_path)) else None
effective_labelmap_path = labelmap_path if (labelmap_path and os.path.exists(labelmap_path)) else None

if effective_model_path is None:
    # 백본에 따라 파일명이 다를 수 있으나, 기본적으로 아래 이름으로 받는다
    fallback_model_name = "hazard_resnet50_eye_6k.keras" if backbone == "ResNet50" else "hazard_mobilenetv2.keras"
    auto_model = ensure_file_via_gdown(f"./{fallback_model_name}", model_file_id)
    if auto_model:
        st.info(f"모델 자동 다운로드 완료: {auto_model}")
        effective_model_path = auto_model

if effective_labelmap_path is None and labelmap_file_id:
    auto_map = ensure_file_via_gdown("./class_to_idx.json", labelmap_file_id)
    if auto_map:
        st.info(f"라벨맵 자동 다운로드 완료: {auto_map}")
        effective_labelmap_path = auto_map

# 모델 로드
if effective_model_path is None:
    st.error("모델 경로가 없고, 자동 다운로드도 실패했습니다. 모델 경로를 입력하거나 Secrets에 gdrive.model_file_id (또는 MODEL_FILE_ID)를 설정하세요.")
    model = None
else:
    try:
        model = load_model_robust(effective_model_path)
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        model = None

# 라벨맵 로드
class_to_idx, idx_to_class = load_labelmap_safe(effective_labelmap_path)
if idx_to_class is None and model is not None:
    st.warning("라벨맵이 없어 클래스 이름 매핑이 불완전할 수 있습니다. class_to_idx.json을 제공하는 것을 권장합니다.")

# --------------- 단일 이미지 업로드 예측 ---------------
# st.header("단일 이미지 예측")
# uploaded_files = st.file_uploader("이미지 업로드 (여러 장 가능)", type=["jpg","jpeg","png","bmp","gif","webp"], accept_multiple_files=True)

# if model and idx_to_class and uploaded_files:
#     cols = st.columns(3)
#     for i, uf in enumerate(uploaded_files):
#         try:
#             img = Image.open(io.BytesIO(uf.read())).convert("RGB")
#             label, conf, top, _ = predict_image(model, preprocess, img, idx_to_class, threshold=thresh, topk=topk)
#             with cols[i % 3]:
#                 st.image(img, caption=f"{uf.name}", use_container_width=True)
#                 st.markdown(f"**Pred:** `{label}`  |  **conf:** `{conf:.3f}`")
#                 st.markdown("Top-{}:".format(topk))
#                 for cls, p in top:
#                     st.caption(f"- {cls}: {p:.3f}")
#         except Exception as e:
#             st.warning(f"{uf.name} 처리 실패: {e}")

st.header("단일 이미지 예측")

uploaded_files = st.file_uploader(
    "이미지 업로드 (여러 장 가능)", 
    type=["jpg", "jpeg", "png", "bmp", "gif", "webp"], 
    accept_multiple_files=True
)

if model and idx_to_class and uploaded_files:
    cols = st.columns(3)
    for i, uf in enumerate(uploaded_files):
        try:
            # 이미지 로드
            img = Image.open(io.BytesIO(uf.read())).convert("RGB")
            
            # 예측 실행
            label, conf, top, _ = predict_image(
                model, preprocess, img, idx_to_class,
                threshold=thresh, topk=topk
            )
            
            # np.int64 방지용 형변환
            label = str(label)
            conf = float(conf)  # ← 여기 추가!
            top = [(str(cls), float(p)) for cls, p in top]

            # UI 표시
            with cols[i % 3]:
                st.image(img, caption=f"{uf.name}", use_container_width=True)
                st.markdown(f"**Pred:** `{label}`  |  **conf:** `{conf:.3f}`")
                st.markdown(f"Top-{topk}:")
                for cls, p in top:
                    st.caption(f"- {cls}: {p:.3f}")

        except Exception as e:
            st.warning(f"{uf.name} 처리 실패: {e}")

# --------------- 폴더 일괄 예측 ---------------
st.header("📂 폴더 일괄 예측")
if model and idx_to_class and batch_dir and os.path.isdir(batch_dir):
    paths = [os.path.join(batch_dir, n) for n in os.listdir(batch_dir) if is_image_file(n)]
    paths.sort()
    st.write(f"이미지 {len(paths)}장 발견")

    preds = []
    grid_imgs, grid_caps = [], []
    start = time.time()
    for p in paths:
        try:
            im = safe_open_image(p)
            label, conf, top, _ = predict_image(model, preprocess, im, idx_to_class, threshold=thresh, topk=topk)
            preds.append({
                "path": p,
                "pred": label,
                "conf": conf,
                **{f"top{i+1}_cls": t[0] for i, t in enumerate(top)},
                **{f"top{i+1}_prob": t[1] for i, t in enumerate(top)}
            })
            if show_grid and len(grid_imgs) < 24:
                grid_imgs.append(im.copy())
                grid_caps.append(f"{os.path.basename(p)}\n→ {label} ({conf:.2f})")
        except Exception as e:
            preds.append({"path": p, "pred": "error", "conf": 0.0})
    dur = time.time() - start
    st.success(f"완료: {len(paths)}장 / {dur:.1f}s")

    if show_grid and grid_imgs:
        cols = st.columns(6)
        for i, (im, cap) in enumerate(zip(grid_imgs, grid_caps)):
            with cols[i % 6]:
                st.image(img, caption="입력 이미지", use_container_width=True)

    # 결과 테이블 & 다운로드
    import pandas as pd
    df = pd.DataFrame(preds)
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 결과 CSV 다운로드", data=csv, file_name="batch_predictions.csv", mime="text/csv")

# --------------- Test 폴더 리포트 ---------------
st.header("Test 정확도/리포트")
if model and class_to_idx and idx_to_class and test_dir and os.path.isdir(test_dir):
    classes = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    img_paths, y_true = [], []
    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for name in os.listdir(cls_dir):
            p = os.path.join(cls_dir, name)
            if is_image_file(p):
                try:
                    safe_open_image(p)  # 검증
                    img_paths.append(p)
                    y_true.append(class_to_idx[cls])
                except Exception:
                    pass  # 손상/숨김은 건너뜀

    if len(img_paths) == 0:
        st.warning("유효한 이미지가 없습니다.")
    else:
        y_pred = []
        for p in img_paths:
            im = Image.open(p).convert("RGB")
            label, conf, top, _ = predict_image(model, preprocess, im, idx_to_class, threshold=0.0, topk=topk)  # 임계치 없이 순수 예측
            if label == "uncertain":
                label = top[0][0]
            # 클래스명을 index로 변환
            if label in class_to_idx:
                y_pred.append(class_to_idx[label])
            else:
                y_pred.append(class_to_idx[top[0][0]])

        # 리포트
        report_text = classification_report(y_true, y_pred, target_names=classes, digits=4)
        st.text("Classification Report\n" + report_text)

        # 정확도
        acc = (np.array(y_true) == np.array(y_pred)).mean()
        st.metric("Test Accuracy", f"{acc:.4f}")

        # 혼동행렬
        cm = confusion_matrix(y_true, y_pred, labels=[class_to_idx[c] for c in classes])

        fig = plt.figure(figsize=(6,5))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        ticks = np.arange(len(classes))
        plt.xticks(ticks, classes, rotation=45, ha="right")
        plt.yticks(ticks, classes)
        th = cm.max()/2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], ha="center", color="white" if cm[i, j] > th else "black")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        st.pyplot(fig)

# --------------- 푸터 ---------------
st.markdown("---")
st.caption("✅ 모델은 과적합 시작 직전(Val 최고점) 체크포인트 사용을 권장합니다.")
