# =============================
# Import Library Yang diperlukan
# =============================
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import re

# =============================
# CONFIG PAGE
# =============================
st.set_page_config(
    page_title="HotelMatch — Rekomendasi Hotel",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f0f13; color: #e8e4dc; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0rem !important; }

.navbar {
    background: rgba(15,15,19,0.95);
    border-bottom: 1px solid rgba(212,175,100,0.25);
    padding: 14px 40px;
    display: flex; align-items: center; justify-content: space-between;
    backdrop-filter: blur(12px); margin-bottom: 0;
}
.navbar-brand { font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 700; color: #d4af64; }
.navbar-brand span { color: #e8e4dc; }
.navbar-user {
    font-size: 0.78rem; color: #888;
    background: rgba(212,175,100,0.1); border: 1px solid rgba(212,175,100,0.2);
    padding: 5px 14px; border-radius: 20px; font-weight: 500;
}
.navbar-user b { color: #d4af64; }

.stButton > button {
    background: transparent !important; border: 1px solid rgba(212,175,100,0.3) !important;
    color: #c8c0b0 !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 7px 18px !important; border-radius: 6px !important;
    transition: all 0.2s ease !important; letter-spacing: 0.03em !important;
}
.stButton > button:hover {
    background: rgba(212,175,100,0.12) !important;
    border-color: #d4af64 !important; color: #d4af64 !important;
}
/* Tombol logout merah */
.logout-btn > button {
    border-color: rgba(239,68,68,0.4) !important;
    color: #fca5a5 !important;
}
.logout-btn > button:hover {
    background: rgba(239,68,68,0.1) !important;
    border-color: #ef4444 !important; color: #ef4444 !important;
}

.hero-section {
    min-height: 86vh; display: flex; flex-direction: column;
    justify-content: center; padding: 60px 60px 40px 60px;
    position: relative; overflow: hidden;
}
.hero-label { font-size: 0.72rem; letter-spacing: 0.18em; color: #d4af64; text-transform: uppercase; margin-bottom: 20px; font-weight: 600; }
.hero-title { font-family: 'Playfair Display', serif; font-size: clamp(2.8rem, 5vw, 4.8rem); font-weight: 700; line-height: 1.08; color: #f0ebe0; margin-bottom: 24px; max-width: 700px; }
.hero-title em { font-style: italic; color: #d4af64; }
.hero-desc { font-size: 1.05rem; color: #888; max-width: 480px; line-height: 1.7; margin-bottom: 40px; font-weight: 300; }
.hero-bg-text { position: absolute; right: -20px; top: 50%; transform: translateY(-50%); font-family: 'Playfair Display', serif; font-size: 18rem; font-weight: 700; color: rgba(212,175,100,0.04); pointer-events: none; user-select: none; }
.hero-stats { display: flex; gap: 48px; margin-top: 56px; padding-top: 32px; border-top: 1px solid rgba(255,255,255,0.07); }
.hero-stat-num { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: #d4af64; }
.hero-stat-label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px; }

.section-header { font-family: 'Playfair Display', serif; font-size: 1.9rem; font-weight: 700; color: #f0ebe0; margin-bottom: 6px; }
.section-sub { font-size: 0.88rem; color: #666; margin-bottom: 28px; }

.hotel-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 24px 26px; margin-bottom: 14px;
    position: relative; overflow: hidden; transition: border-color 0.25s;
}
.hotel-card:hover { border-color: rgba(212,175,100,0.25); }
.hotel-card::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: #d4af64; opacity: 0; transition: opacity 0.25s; }
.hotel-card:hover::before { opacity: 1; }
.hotel-rank { position: absolute; top: 14px; right: 20px; font-family: 'Playfair Display', serif; font-size: 2.5rem; font-weight: 700; color: rgba(212,175,100,0.08); line-height: 1; }
.hotel-name { font-family: 'Playfair Display', serif; font-size: 1.15rem; font-weight: 600; color: #f0ebe0; margin-bottom: 2px; }
.hotel-id-tag { font-size: 0.7rem; color: #555; font-family: 'DM Sans', monospace; margin-bottom: 8px; letter-spacing: 0.04em; }
.hotel-location { font-size: 0.8rem; color: #777; margin-bottom: 10px; }
.hotel-stars { color: #d4af64; font-size: 0.85rem; margin-bottom: 12px; }

.pill { font-size: 0.7rem; padding: 3px 10px; border-radius: 20px; font-weight: 600; letter-spacing: 0.04em; display: inline-block; }
.pill-hybrid  { background: rgba(138,92,246,0.18); color: #a78bfa; border: 1px solid rgba(138,92,246,0.25); }
.pill-cf      { background: rgba(59,130,246,0.15);  color: #93c5fd; border: 1px solid rgba(59,130,246,0.25); }
.pill-cbf     { background: rgba(34,197,94,0.12);   color: #86efac; border: 1px solid rgba(34,197,94,0.2); }
.pill-popular { background: rgba(251,191,36,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }

.score-bar-bg  { background: rgba(255,255,255,0.06); border-radius: 4px; height: 5px; margin-top: 6px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #d4af64, #f0d090); }

.metric-small { font-size: 0.72rem; color: #666; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 3px; }
.metric-val   { font-family: 'Playfair Display', serif; font-size: 1.3rem; font-weight: 700; color: #e8e4dc; }
.metric-val-gold { color: #d4af64; }

.eval-card { background: rgba(212,175,100,0.06); border: 1px solid rgba(212,175,100,0.2); border-radius: 10px; padding: 18px 22px; text-align: center; }
.eval-label { font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px; }
.eval-value { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: #d4af64; }
.eval-note  { font-size: 0.72rem; color: #555; margin-top: 4px; }

/* Cold start banner */
.cold-start-banner {
    background: rgba(251,191,36,0.07);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 10px;
    padding: 16px 22px;
    margin-bottom: 24px;
    font-size: 0.88rem;
    color: #d4a800;
}
.cold-start-banner b { color: #fbbf24; }

.login-box { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 48px 44px; max-width: 440px; width: 100%; }
.login-title { font-family: 'Playfair Display', serif; font-size: 1.9rem; font-weight: 700; color: #f0ebe0; margin-bottom: 6px; }
.login-sub   { font-size: 0.85rem; color: #666; margin-bottom: 32px; }

.stTextInput > div > div > input { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 8px !important; color: #e8e4dc !important; font-family: 'DM Sans', sans-serif !important; }
.stTextInput > div > div > input:focus { border-color: rgba(212,175,100,0.5) !important; box-shadow: 0 0 0 2px rgba(212,175,100,0.1) !important; }
.stTextInput label { color: #999 !important; font-size: 0.8rem !important; text-transform: uppercase !important; }
.stSelectbox > div > div { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #e8e4dc !important; border-radius: 8px !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# LOAD & PREPROCESSING (cached)
# =====================================================================
USER_CSV = 'Dataset_SudahPreprocessing/dataUser.csv'

def load_df_users():
    """Baca dataUser.csv langsung dari disk (tidak di-cache).
    Dipanggil setiap render sehingga selalu up-to-date setelah pendaftaran."""
    return pd.read_csv(USER_CSV)

@st.cache_data
def load_and_preprocess():
    path_Hotels  = 'Dataset_SudahPreprocessing/dataHotel.csv'
    path_Reviews = 'Dataset_SudahPreprocessing/dataReview_2.csv'
    # df_users SENGAJA tidak di-cache — dikelola via load_df_users()

    df_hotels_raw = pd.read_csv(path_Hotels)
    df_reviews    = pd.read_csv(path_Reviews)

    df_reviews = df_reviews.dropna(subset=['user_id', 'offering_id', 'overall'])

    avg_rating   = df_reviews.groupby('offering_id')['overall'].mean().reset_index()
    avg_rating.columns = ['id', 'avg_rating']
    review_count = df_reviews.groupby('offering_id')['overall'].count().reset_index()
    review_count.columns = ['id', 'review_count']

    df_hotels = df_hotels_raw.copy()
    df_hotels = df_hotels.merge(avg_rating,   on='id', how='left')
    df_hotels = df_hotels.merge(review_count, on='id', how='left')
    df_hotels['avg_rating']   = df_hotels['avg_rating'].fillna(0)
    df_hotels['review_count'] = df_hotels['review_count'].fillna(0)

    scaler = MinMaxScaler()
    df_hotels[['hotel_class', 'avg_rating', 'review_count']] = scaler.fit_transform(
        df_hotels[['hotel_class', 'avg_rating', 'review_count']]
    )
    return df_hotels_raw, df_hotels, df_reviews


@st.cache_data
def build_cf_structures(_df_reviews):
    user_codes = _df_reviews['user_id'].astype('category').cat.codes
    item_codes = _df_reviews['offering_id'].astype('category').cat.codes
    ratings    = _df_reviews['overall']

    user_item_matrix = coo_matrix((ratings, (user_codes, item_codes))).tocsr()
    user_mapping  = dict(enumerate(_df_reviews['user_id'].astype('category').cat.categories))
    item_mapping  = dict(enumerate(_df_reviews['offering_id'].astype('category').cat.categories))
    user_to_index = {v: k for k, v in user_mapping.items()}
    item_to_code  = {v: k for k, v in item_mapping.items()}
    return user_item_matrix, user_mapping, item_mapping, user_to_index, item_to_code


@st.cache_data
def build_cbf_structures(_df_hotels):
    hotel_features = _df_hotels[['id','hotel_class','locality','region','avg_rating','review_count']]
    hotel_encoded  = pd.get_dummies(hotel_features, columns=['locality','region'])
    hotel_encoded  = hotel_encoded.astype(float).set_index('id')
    hotel_similarity    = cosine_similarity(hotel_encoded)
    similarity_hotel_df = pd.DataFrame(hotel_similarity, index=hotel_encoded.index, columns=hotel_encoded.index)
    return hotel_encoded, similarity_hotel_df


# ---- Jalankan sekali ----
df_hotels_raw, df_hotels, df_reviews = load_and_preprocess()
# df_users dibaca fresh setiap render — tidak ikut cache
df_users = load_df_users()
user_item_matrix, user_mapping, item_mapping, user_to_index, item_to_code = build_cf_structures(df_reviews)
hotel_encoded, similarity_hotel_df = build_cbf_structures(df_hotels)

user_ratings       = df_reviews[['user_id', 'offering_id', 'overall']]
global_mean_rating = df_reviews['overall'].mean()

# Helper: ambil username dari user_id — selalu baca fresh dari disk
def get_username(uid):
    df = load_df_users()
    row = df[df['user_id'] == uid]
    if len(row) > 0:
        return row.iloc[0]['username']
    return uid  # fallback ke user_id jika tidak ada di dataUser


# =====================================================================
# FUNGSI REKOMENDASI
# =====================================================================

def get_similar_users(user_index, user_item_matrix, top_k=10):
    user_vector   = user_item_matrix[user_index]
    similarities  = cosine_similarity(user_vector, user_item_matrix).flatten()
    similar_users = similarities.argsort()[::-1][1:top_k+1]
    return similar_users, similarities[similar_users]


# =====================================================================
# [DIPERBARUI] recommend_items — CF dengan normalisasi per-item
# Perubahan utama:
#   LAMA: sum_similarity = np.sum(scores)  ← pembagi global (semua user)
#   BARU: sum_similarity_per_item = np.dot(scores, rated_mask)
#         ← pembagi hanya dari user yang benar-benar merating item tersebut
# Dampak: prediksi lebih akurat, tidak under-estimated
# =====================================================================
def recommend_items(user_index, user_item_matrix, similar_users, scores, top_n=20):
    user_vector    = user_item_matrix[user_index].toarray().flatten()
    similar_matrix = user_item_matrix[similar_users].toarray()

    # Weighted sum per item
    weighted_sum = np.dot(scores, similar_matrix)

    # [BARU] Mask: 1 jika user similar merating item, 0 jika tidak
    rated_mask = (similar_matrix > 0).astype(float)

    # [BARU] Sum similarity hanya untuk user yang merating item tersebut
    sum_similarity_per_item = np.dot(scores, rated_mask)

    # Prediksi rating: hindari pembagian nol
    predicted_ratings = np.where(
        sum_similarity_per_item > 0,
        weighted_sum / sum_similarity_per_item,
        0
    )

    # Hapus item yang sudah dirating user target
    predicted_ratings[user_vector > 0] = 0

    candidate_items = np.where(predicted_ratings > 0)[0]
    if len(candidate_items) == 0:
        return pd.DataFrame(columns=['hotel_id', 'prediksi_CF'])

    candidate_scores = predicted_ratings[candidate_items]
    top_indices      = candidate_scores.argsort()[::-1][:top_n]
    recommended_items = candidate_items[top_indices]
    item_scores       = candidate_scores[top_indices]

    return pd.DataFrame({
        'hotel_id':    [item_mapping[idx] for idx in recommended_items],
        'prediksi_CF':  item_scores
    })


def reccomend_items_cbf(user_data):
    """CBF: kembalikan prediksi untuk SEMUA hotel yang belum dirating (tanpa top_n)."""
    cbf_predictions = {}
    user_mean       = user_data['overall'].mean()
    SIM_THRESHOLD   = 0.1

    for hotel in hotel_encoded.index:
        if hotel in user_data['offering_id'].values:
            continue
        numerator = 0; denominator = 0

        for _, row in user_data.iterrows():
            rated_hotel = row['offering_id']
            rating      = row['overall']
            if rated_hotel not in similarity_hotel_df.columns:
                continue
            sim = similarity_hotel_df.loc[hotel, rated_hotel]
            if sim > SIM_THRESHOLD:
                numerator   += sim * (rating - user_mean)
                denominator += sim

        pred = (user_mean + numerator / denominator) if denominator != 0 else df_hotels['avg_rating'].mean()
        cbf_predictions[hotel] = pred

    return pd.DataFrame(list(cbf_predictions.items()), columns=['hotel_id', 'prediksi_CBF'])


def dynamicWeightHybrid(user_data, k=10):
    num_rated = user_data.shape[0]
    Au        = min((num_rated / k), 1) * 0.9
    return Au, 1 - Au   # bobot_cf, bobot_cbf


def dynamicReccomendation(cf_df, cbf_df, bobot_cf, bobot_cbf, top_n=10):
    df = pd.merge(cf_df, cbf_df, on='hotel_id', how='outer')
    df['hybrid_predict'] = (
        bobot_cf  * df['prediksi_CF'].fillna(0) +
        bobot_cbf * df['prediksi_CBF'].fillna(0)
    )
    return df.sort_values('hybrid_predict', ascending=False).head(top_n).reset_index(drop=True)


# ── Cold Start: Popularity-Based + preferensi lokasi user ─────────────
def popularityBased(df_review, df_hotel_raw, preferred_locality=None, preferred_region=None, top_n=10):
    """
    Rekomendasi untuk user cold start.
    - Profil popularitas diambil dari top-3 hotel terbanyak dirating.
    - Locality & Region menggunakan pilihan user (bukan dari popularitas).
    - Skor: class(0.5) + locality(0.3) + region(0.2)
    """
    hotel_popularity = df_review.groupby('offering_id').size().reset_index(name='jumlah_rating')
    top3 = hotel_popularity.sort_values('jumlah_rating', ascending=False).head(3)
    popular_features = df_hotel_raw[df_hotel_raw['id'].isin(top3['offering_id'])]

    # Profile class dari popularitas
    profile_class = popular_features['hotel_class'].mode()[0] if len(popular_features) > 0 else 3

    # Locality & Region dari preferensi user
    profile_locality = [preferred_locality] if preferred_locality else popular_features['locality'].unique().tolist()
    profile_region   = [preferred_region]   if preferred_region   else popular_features['region'].unique().tolist()

    w_class    = 0.5
    w_locality = 0.3
    w_region   = 0.2

    scores = []
    for _, row in df_hotel_raw.iterrows():
        class_score    = 1 if row['hotel_class'] >= profile_class else 0
        locality_score = 1 if row['locality'] in profile_locality else 0
        region_score   = 1 if row['region']   in profile_region   else 0
        score = class_score * w_class + locality_score * w_locality + region_score * w_region
        scores.append({'hotel_id': row['id'], 'score_popularity': score})

    df_scores = pd.DataFrame(scores).sort_values('score_popularity', ascending=False)
    return df_scores.head(top_n).reset_index(drop=True)


# =====================================================================
# [DIPERBARUI] get_cf_prediction_for_item — untuk LOOCV (MAE & RMSE)
# Perubahan utama:
#   LAMA: sum_of_similarities = np.sum(scores)  ← global
#   BARU: sum_similarity_per_item = np.dot(scores, rated_mask)
#         ← hanya user yang merating item tersebut
# Konsisten dengan perubahan pada recommend_items di atas
# =====================================================================
def get_cf_prediction_for_item(target_user_idx, target_item_code,
                                user_item_matrix_full, similar_users_indices, similarity_scores):
    # Validasi item
    if target_item_code >= user_item_matrix_full.shape[1]:
        return global_mean_rating

    # Rating item dari similar users
    similar_users_ratings_for_item = (
        user_item_matrix_full[similar_users_indices, target_item_code]
        .toarray()
        .flatten()
    )

    # Weighted sum
    weighted_sum_ratings = np.dot(similarity_scores, similar_users_ratings_for_item)

    # [BARU] Mask: hanya user yang merating item ini
    rated_mask = (similar_users_ratings_for_item > 0).astype(float)

    # [BARU] Normalisasi per item — bukan global
    sum_similarity_per_item = np.dot(similarity_scores, rated_mask)

    # Hindari pembagian nol → fallback ke global mean
    if sum_similarity_per_item == 0:
        return global_mean_rating

    predicted_rating = weighted_sum_ratings / sum_similarity_per_item
    return predicted_rating


def loocProcess(target_user_id, user_index):
    user_data           = user_ratings[user_ratings['user_id'] == target_user_id]
    bobot_cf, bobot_cbf = dynamicWeightHybrid(user_data)
    all_actual          = []
    all_predicted       = []

    target_ratings              = df_reviews[df_reviews['user_id'] == target_user_id]
    similar_users_cf, scores_cf = get_similar_users(user_index, user_item_matrix, top_k=10)

    for _, row in target_ratings.iterrows():
        left_out_id   = row['offering_id']
        actual_rating = row['overall']

        left_out_item_code = item_to_code.get(left_out_id)
        if left_out_item_code is None:
            continue

        cf_pred = np.clip(
            get_cf_prediction_for_item(user_index, left_out_item_code,
                                        user_item_matrix, similar_users_cf, scores_cf),
            1.0, 5.0
        )

        user_data_loo = target_ratings[target_ratings['offering_id'] != left_out_id]
        user_mean_loo = user_data_loo['overall'].mean() if not user_data_loo.empty else global_mean_rating

        if left_out_id not in hotel_encoded.index:
            cbf_pred = global_mean_rating
        else:
            num, den = 0, 0
            for _, ud in user_data_loo.iterrows():
                rh = ud['offering_id']
                if rh in similarity_hotel_df.columns:
                    sim = similarity_hotel_df.loc[left_out_id, rh]
                    if sim > 0.1:
                        num += sim * (ud['overall'] - user_mean_loo)
                        den += sim
            cbf_pred = (user_mean_loo + num / den) if den != 0 else global_mean_rating

        cbf_pred    = np.clip(cbf_pred, 1.0, 5.0)
        hybrid_pred = bobot_cf * cf_pred + bobot_cbf * cbf_pred

        all_actual.append(actual_rating)
        all_predicted.append(hybrid_pred)

    if len(all_actual) > 0:
        mae  = mean_absolute_error(all_actual, all_predicted)
        rmse = np.sqrt(mean_squared_error(all_actual, all_predicted))
        return round(mae, 4), round(rmse, 4)
    return None, None


def enrich_recommendations(df_reco):
    avg_r = df_reviews.groupby('offering_id')['overall'].mean().reset_index()
    avg_r.columns = ['id', 'avg_rating_display']
    hotel_info = df_hotels_raw[['id', 'name', 'hotel_class', 'locality', 'region']].copy()
    hotel_info = hotel_info.merge(avg_r, on='id', how='left')
    hotel_info['avg_rating_display'] = hotel_info['avg_rating_display'].fillna(0)
    return df_reco.merge(hotel_info, left_on='hotel_id', right_on='id', how='left')


def enrich_cold_start(df_reco):
    avg_r = df_reviews.groupby('offering_id')['overall'].mean().reset_index()
    avg_r.columns = ['id', 'avg_rating_display']
    hotel_info = df_hotels_raw[['id', 'name', 'hotel_class', 'locality', 'region']].copy()
    hotel_info = hotel_info.merge(avg_r, on='id', how='left')
    hotel_info['avg_rating_display'] = hotel_info['avg_rating_display'].fillna(0)
    return df_reco.merge(hotel_info, left_on='hotel_id', right_on='id', how='left')


# =====================================================================
# SESSION STATE
# =====================================================================
defaults = {
    "page":             "home",
    "user_id":          None,
    "username":         None,
    "is_new_user":      False,
    "cold_locality":    None,
    "cold_region":      None,
    "recommendations":  None,
    "bobot_cf":         None,
    "bobot_cbf":        None,
    "mae":              None,
    "rmse":             None,
    "eval_done":        False,
    "cold_start_reco":  None,
}
for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =====================================================================
# NAVBAR
# =====================================================================
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">Hotel<span>Match</span></div>
</div>
""", unsafe_allow_html=True)

is_logged_in = st.session_state.user_id is not None
nc = st.columns([1, 0.6, 0.9, 0.6, 0.6, 2.2])

with nc[1]:
    if st.button("🏠 Home"):
        st.session_state.page = "home"; st.rerun()
with nc[2]:
    if st.button("🏨 Rekomendasi"):
        st.session_state.page = "rekomendasi" if is_logged_in else "login"
        st.rerun()

with nc[3]:
    if is_logged_in:
        st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.session_state.signup_done = False   # reset flag signup
            st.session_state.page = "home"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        if st.button("🔑 Login"):
            st.session_state.page = "login"; st.rerun()

with nc[4]:
    if not is_logged_in:
        if st.button("📝 Daftar"):
            st.session_state.page = "signup"; st.rerun()

with nc[5]:
    if is_logged_in:
        uname     = st.session_state.username or st.session_state.user_id
        uid_short = st.session_state.user_id[:14] + "…"
        st.markdown(
            f'<div class="navbar-user">👤 <b>{uname}</b> &nbsp;·&nbsp; <span style="color:#555">{uid_short}</span></div>',
            unsafe_allow_html=True
        )

st.markdown(
    "<div style='height:1px;background:rgba(212,175,100,0.12);margin:0 0 24px 0'></div>",
    unsafe_allow_html=True
)


# =====================================================================
# PAGE: HOME
# =====================================================================
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero-section">
        <div class="hero-bg-text">H</div>
        <div class="hero-label">✦ Sistem Rekomendasi Cerdas</div>
        <div class="hero-title">Temukan Hotel<br>yang <em>Tepat</em><br>untuk Anda</div>
        <div class="hero-desc">
            Sistem rekomendasi hybrid menggabungkan Collaborative Filtering
            dan Content-Based Filtering dengan pembobotan dinamis —
            menghadirkan rekomendasi yang semakin personal.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("✦  Dapatkan Rekomendasi Sekarang", use_container_width=True):
            st.session_state.page = "rekomendasi" if is_logged_in else "login"
            st.rerun()

    total_users   = df_reviews['user_id'].nunique()
    total_hotels  = df_hotels_raw['id'].nunique()
    total_reviews = len(df_reviews)
    st.markdown(f"""
    <div class="hero-stats">
        <div><div class="hero-stat-num">{total_hotels:,}</div><div class="hero-stat-label">Hotel Tersedia</div></div>
        <div><div class="hero-stat-num">{total_users:,}</div><div class="hero-stat-label">Pengguna Aktif</div></div>
        <div><div class="hero-stat-num">{total_reviews:,}</div><div class="hero-stat-label">Ulasan Terkumpul</div></div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# PAGE: LOGIN
# =====================================================================
elif st.session_state.page == "login":
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    _, col_c, _ = st.columns([1, 1.4, 1])
    with col_c:
        st.markdown("""
        <div class="login-box">
            <div class="login-title">Selamat Datang</div>
            <div class="login-sub">Masukkan User ID Anda untuk melanjutkan</div>
        </div>
        """, unsafe_allow_html=True)

        user_input = st.text_input("USER ID", placeholder="Masukkan user ID kamu…")

        if st.button("🔑  Masuk Sekarang", use_container_width=True):
            uid_in = user_input.strip()
            df_users_fresh = load_df_users()   # baca ulang CSV — tangkap user baru
            if uid_in in df_users_fresh["user_id"].values:
                uname = get_username(uid_in)
                st.session_state.user_id         = uid_in
                st.session_state.username        = uname
                st.session_state.is_new_user     = uid_in not in df_reviews["user_id"].values
                st.session_state.recommendations = None
                st.session_state.eval_done       = False
                st.session_state.mae             = None
                st.session_state.rmse            = None
                st.session_state.cold_start_reco = None
                st.success(f"✅ Login berhasil! Halo, {uname}!")
                st.session_state.page = "rekomendasi"
                st.rerun()
            else:
                st.error("❌ User ID tidak ditemukan. Silakan coba lagi atau daftar terlebih dahulu.")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Belum punya akun?")
        with c2:
            if st.button("Daftar di sini"):
                st.session_state.page = "signup"; st.rerun()


# =====================================================================
# PAGE: SIGN UP
# =====================================================================
elif st.session_state.page == "signup":
    # ── Inisialisasi flag pendaftaran di session state ──────────────
    # Flag ini memastikan proses tulis CSV hanya terjadi SATU KALI,
    # tidak berulang meski Streamlit melakukan re-render.
    if "signup_done" not in st.session_state:
        st.session_state.signup_done = False

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    _, col_c, _ = st.columns([1, 1.4, 1])
    with col_c:
        st.markdown("""
        <div class="login-box">
            <div class="login-title">Buat Akun Baru</div>
            <div class="login-sub">Daftarkan diri kamu untuk mendapatkan rekomendasi hotel personal</div>
        </div>
        """, unsafe_allow_html=True)

        new_uid      = st.text_input("USER ID BARU",  placeholder="Contoh: JOHN123  (huruf besar, tanpa spasi)")
        new_username = st.text_input("USERNAME",       placeholder="Nama tampilan kamu")

        st.markdown("""
        <div style="font-size:0.75rem;color:#555;margin-top:-8px;margin-bottom:16px;">
        ⚠️ User ID harus <b style="color:#888">huruf besar semua</b> dan <b style="color:#888">tanpa spasi</b>.
        </div>
        """, unsafe_allow_html=True)

        if st.button("📝  Daftar Sekarang", use_container_width=True):
            uid_in  = new_uid.strip()
            uname   = new_username.strip()
            errors  = []

            # ── Validasi input ────────────────────────────────────
            if not uid_in:
                errors.append("User ID tidak boleh kosong.")
            elif ' ' in uid_in:
                errors.append("User ID tidak boleh mengandung spasi.")
            elif uid_in != uid_in.upper():
                errors.append("User ID harus seluruhnya huruf BESAR (uppercase).")
            elif uid_in in load_df_users()["user_id"].values:
                errors.append("User ID sudah digunakan. Pilih User ID lain.")

            if not uname:
                errors.append("Username tidak boleh kosong.")

            if errors:
                for e in errors:
                    st.error(f"❌ {e}")

            # ── Hanya tulis jika belum pernah diproses (cegah duplikat) ──
            elif not st.session_state.signup_done:
                USER_CSV = 'Dataset_SudahPreprocessing/dataUser.csv'
                try:
                    import csv, os

                    # [FIX 1] Baca ulang TEPAT sebelum tulis — cegah race condition
                    df_check = load_df_users()
                    if uid_in in df_check["user_id"].values:
                        st.error("❌ User ID sudah terdaftar (terdeteksi saat pengecekan akhir).")
                        st.stop()

                    # [FIX 2] Tandai DULU sebelum tulis — blokir render kedua
                    st.session_state.signup_done = True

                    file_exists = os.path.isfile(USER_CSV)
                    with open(USER_CSV, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['user_id', 'username'])
                        writer.writerow([uname,uid_in])

                    # Set session state lain SETELAH tulis berhasil
                    st.session_state.user_id         = uid_in
                    st.session_state.username        = uname
                    st.session_state.is_new_user     = True
                    st.session_state.recommendations = None
                    st.session_state.cold_start_reco = None
                    st.session_state.eval_done       = False
                    st.success(f"✅ Akun berhasil dibuat! Selamat datang, {uname}!")
                    st.session_state.page = "rekomendasi"
                    st.rerun()

                except PermissionError:
                    # ── Panduan spesifik jika file sedang dikunci ─────────
                    st.error(
                        "❌ **Gagal menyimpan akun — file dataUser.csv sedang terkunci.**\n\n"
                        "Kemungkinan penyebab:\n"
                        "- File sedang dibuka di **Excel** atau aplikasi lain → tutup dulu.\n"
                        "- Proses Streamlit lain sedang mengaksesnya → restart terminal.\n\n"
                        "Setelah file ditutup, klik tombol Daftar lagi."
                    )
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat menyimpan: {e}")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if st.button("← Sudah punya akun? Login"):
            st.session_state.signup_done = False   # reset flag saat pindah halaman
            st.session_state.page = "login"; st.rerun()


# =====================================================================
# PAGE: REKOMENDASI
# =====================================================================
elif st.session_state.page == "rekomendasi":
    if not st.session_state.user_id:
        st.warning("Silakan login terlebih dahulu.")
        st.session_state.page = "login"; st.rerun()

    uid      = st.session_state.user_id
    uname    = st.session_state.username or uid
    user_data = user_ratings[user_ratings['user_id'] == uid]
    num_rated = user_data.shape[0]

    bobot_cf, bobot_cbf = dynamicWeightHybrid(user_data, k=10)
    is_cold_start = (bobot_cf == 0)

    st.markdown(f"""
    <div class="section-header">Rekomendasi untuk {uname}</div>
    <div class="section-sub">
        Halo <b style="color:#d4af64">{uname}</b>
        <span style="color:#444"> · ID: {uid[:24]}</span> —
        {"Anda belum memiliki riwayat ulasan." if is_cold_start else f"berdasarkan <b>{num_rated}</b> ulasan Anda, berikut hotel terbaik yang kami temukan."}
    </div>
    """, unsafe_allow_html=True)

    # ====================================================
    # COLD START PATH
    # ====================================================
    if is_cold_start:
        st.markdown("""
        <div class="cold-start-banner">
            🌟 <b>Rekomendasi untuk Pengguna Baru</b><br>
            Karena Anda belum memiliki riwayat ulasan, sistem menggunakan
            <b>Popularity-Based Recommendation</b> yang disesuaikan dengan preferensi lokasi Anda.
            Mulailah memberi ulasan agar rekomendasi menjadi lebih personal!
        </div>
        """, unsafe_allow_html=True)

        all_localities = sorted(df_hotels_raw['locality'].dropna().unique().tolist())
        all_regions    = sorted(df_hotels_raw['region'].dropna().unique().tolist())

        pref_col1, pref_col2 = st.columns(2)
        with pref_col1:
            chosen_locality = st.selectbox(
                "Preferensi Kota / Lokasi",
                ["Tidak ada preferensi"] + all_localities,
                key="cold_loc"
            )
        with pref_col2:
            chosen_region = st.selectbox(
                "Preferensi Region / Negara Bagian",
                ["Tidak ada preferensi"] + all_regions,
                key="cold_reg"
            )

        loc_val = None if chosen_locality == "Tidak ada preferensi" else chosen_locality
        reg_val = None if chosen_region   == "Tidak ada preferensi" else chosen_region

        pref_key = (loc_val, reg_val)
        if (st.session_state.cold_start_reco is None or
                (st.session_state.cold_locality, st.session_state.cold_region) != pref_key):
            with st.spinner("🔄 Menyusun rekomendasi populer untuk Anda…"):
                cold_df = popularityBased(df_reviews, df_hotels_raw,
                                           preferred_locality=loc_val,
                                           preferred_region=reg_val,
                                           top_n=10)
                cold_enriched = enrich_cold_start(cold_df)
                st.session_state.cold_start_reco = cold_enriched
                st.session_state.cold_locality   = loc_val
                st.session_state.cold_region     = reg_val

        cold_enriched = st.session_state.cold_start_reco
        st.markdown(
            f"<div class='section-sub'>Menampilkan <b>{len(cold_enriched)}</b> hotel populer"
            + (f" di <b>{loc_val}</b>" if loc_val else "")
            + (f", <b>{reg_val}</b>" if reg_val else "")
            + "</div>",
            unsafe_allow_html=True
        )

        max_score_cs = cold_enriched['score_popularity'].max() if len(cold_enriched) > 0 else 1
        for i, row in cold_enriched.iterrows():
            rank         = i + 1
            hotel_name   = row.get('name',               'Nama tidak tersedia')
            hotel_id_val = row.get('hotel_id',            '-')
            locality     = row.get('locality',            '-')
            region_val   = row.get('region',              '-')
            avg_r        = row.get('avg_rating_display',   0)
            score        = row.get('score_popularity',     0)

            hc_raw    = df_hotels_raw[df_hotels_raw['id'] == hotel_id_val]['hotel_class'].values
            stars_raw = int(hc_raw[0]) if len(hc_raw) > 0 else 0
            stars_str = "★" * stars_raw + "☆" * max(0, 5 - stars_raw)
            bar_pct   = (score / max_score_cs * 100) if max_score_cs > 0 else 0

            with st.container():
                st.markdown(f"""
                <div class="hotel-card">
                    <div class="hotel-rank">#{rank}</div>
                    <div class="hotel-name">{hotel_name}</div>
                    <div class="hotel-id-tag">ID Hotel: {hotel_id_val}</div>
                    <div class="hotel-location">📍 {locality}, {region_val}</div>
                    <div class="hotel-stars">{stars_str}&nbsp;
                        <span style="font-size:0.75rem;color:#666">{stars_raw} Bintang</span>
                    </div>
                """, unsafe_allow_html=True)

                cs1, cs2 = st.columns([2, 2])
                with cs1:
                    st.markdown(f"""
                    <div class="metric-small">Skor Popularitas</div>
                    <div class="metric-val metric-val-gold">{score:.2f}</div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{bar_pct:.1f}%"></div>
                    </div>""", unsafe_allow_html=True)
                with cs2:
                    st.markdown(f"""
                    <div class="metric-small">Rating Rata-rata</div>
                    <div class="metric-val">{avg_r:.2f}
                        <span style='font-size:0.8rem;color:#666'> / 5.0</span>
                    </div>""", unsafe_allow_html=True)

                st.markdown(
                    '<div style="margin-top:12px"><span class="pill pill-popular">Popularity-Based</span></div></div>',
                    unsafe_allow_html=True
                )

    # ====================================================
    # NORMAL PATH
    # ====================================================
    else:
        if uid not in user_to_index:
            st.warning("User ini belum ada di data rating. Silakan beri ulasan hotel terlebih dahulu.")
            st.stop()

        user_index = user_to_index[uid]

        if st.session_state.recommendations is None:
            with st.spinner("🔄 Menghitung rekomendasi untuk Anda…"):
                similar_users, scores = get_similar_users(user_index, user_item_matrix, top_k=10)
                cf_df   = recommend_items(user_index, user_item_matrix, similar_users, scores, top_n=500)
                cbf_df  = reccomend_items_cbf(user_data)
                reco_df = dynamicReccomendation(cf_df, cbf_df, bobot_cf, bobot_cbf, top_n=5000)
                enriched = enrich_recommendations(reco_df)

                st.session_state.recommendations = enriched
                st.session_state.bobot_cf        = bobot_cf
                st.session_state.bobot_cbf       = bobot_cbf

        if not st.session_state.eval_done:
            with st.spinner("📊 Menghitung evaluasi MAE & RMSE (LOOCV)…"):
                mae, rmse = loocProcess(uid, user_index)
                st.session_state.mae       = mae
                st.session_state.rmse      = rmse
                st.session_state.eval_done = True

        enriched  = st.session_state.recommendations   # pool besar (~5000)
        bobot_cf  = st.session_state.bobot_cf
        bobot_cbf = st.session_state.bobot_cbf
        mae       = st.session_state.mae
        rmse      = st.session_state.rmse

        # --- Eval cards (tidak berubah) ---
        e1, e2, e3, e4, e5 = st.columns(5)
        # ... (kode eval_cards tetap sama) ...
        eval_cards = [
        ("Bobot CF (α)",    f"{bobot_cf:.2f}",                 "Collaborative Filtering"),
        ("Bobot CBF (1-α)", f"{bobot_cbf:.2f}",                "Content-Based Filtering"),
        ("Hotel Dirating",  str(num_rated),                     "oleh user ini"),
        ("MAE (LOOCV)",     f"{mae:.4f}"  if mae  else "N/A",  "Mean Absolute Error"),
        ("RMSE (LOOCV)",    f"{rmse:.4f}" if rmse else "N/A",  "Root Mean Square Error"),]
        for col, (label, value, note) in zip([e1, e2, e3, e4, e5], eval_cards):
            with col:
                st.markdown(f"""
                <div class="eval-card">
                    <div class="eval-label">{label}</div>
                    <div class="eval-value">{value}</div>
                    <div class="eval-note">{note}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

        # --- Filter & Sort UI ---
        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])   # tambah 1 kolom untuk locality

        with fc1:
            # [PERUBAHAN 3a] Tambah filter locality (seperti cold start)
            locality_opts   = ["Semua"] + sorted(df_hotels_raw['locality'].dropna().unique().tolist())
            locality_filter = st.selectbox("Filter Kota / Locality", locality_opts)

        with fc2:
            region_opts   = ["Semua"] + sorted(df_hotels_raw['region'].dropna().unique().tolist())
            region_filter = st.selectbox("Filter Region", region_opts)

        with fc3:
            sort_by = st.selectbox(
                "Urutkan berdasarkan",
                ["Skor Hybrid (Default)", "Rating Rata-rata", "Bintang Hotel"]
            )
        with fc4:
            sort_asc = st.radio("Urutan", ["Tertinggi", "Terendah"], horizontal=True)

        # [PERUBAHAN 3b] Filter dari pool besar, BARU ambil head(10)
        df_show = enriched.copy()

        if locality_filter != "Semua":
            df_show = df_show[df_show['locality'] == locality_filter]

        if region_filter != "Semua":
            df_show = df_show[df_show['region'] == region_filter]

        sort_col_map = {
            "Skor Hybrid (Default)": "hybrid_predict",
            "Rating Rata-rata":      "avg_rating_display",
            "Bintang Hotel":         "hotel_class",
        }
        sc      = sort_col_map[sort_by]
        asc     = sort_asc == "Terendah"

        # [PERUBAHAN 3c] Sort dulu, BARU head(10) — bukan sebaliknya
        df_show = (
            df_show
            .sort_values(sc, ascending=asc)
            .head(10)                          # ambil 10 terbaik SETELAH filter
            .reset_index(drop=True)
        )

        # Pesan informatif jika filter terlalu ketat
        if len(df_show) == 0:
            st.info(
                "🔍 Tidak ada hotel yang cocok dengan filter ini dalam pool rekomendasi Anda. "
                "Coba ubah filter **Kota** atau **Region**, atau pilih **'Semua'** untuk melihat "
                "seluruh rekomendasi."
            )
        else:
            st.markdown(
                f"<div class='section-sub'>Menampilkan <b>{len(df_show)}</b> hotel"
                + (f" di <b>{locality_filter}</b>" if locality_filter != "Semua" else "")
                + (f", <b>{region_filter}</b>"     if region_filter   != "Semua" else "")
                + " — diurutkan dari <b>skor hybrid tertinggi</b></div>",
                unsafe_allow_html=True
            )

        max_score = df_show['hybrid_predict'].max() if len(df_show) > 0 else 1

        for i, row in df_show.iterrows():
            rank         = i + 1
            hotel_name   = row.get('name',               'Nama tidak tersedia')
            hotel_id_val = row.get('hotel_id',            '-')
            locality     = row.get('locality',            '-')
            region_val   = row.get('region',              '-')
            avg_r        = row.get('avg_rating_display',   0)
            hybrid_score = row.get('hybrid_predict',       0)
            cf_score     = row.get('prediksi_CF',          None)
            cbf_score    = row.get('prediksi_CBF',         None)

            hc_raw    = df_hotels_raw[df_hotels_raw['id'] == hotel_id_val]['hotel_class'].values
            stars_raw = int(hc_raw[0]) if len(hc_raw) > 0 else 0
            stars_str = "★" * stars_raw + "☆" * max(0, 5 - stars_raw)

            def _valid(v):
                return v is not None and not (isinstance(v, float) and np.isnan(v)) and v > 0

            has_cf  = _valid(cf_score)
            has_cbf = _valid(cbf_score)

            if has_cf and has_cbf:
                badge = '<span class="pill pill-hybrid">Hybrid</span>'
            elif has_cf:
                badge = '<span class="pill pill-cf">Collaborative</span>'
            else:
                badge = '<span class="pill pill-cbf">Content-Based</span>'

            bar_pct = (hybrid_score / max_score * 100) if max_score > 0 else 0

            with st.container():
                st.markdown(f"""
                <div class="hotel-card">
                    <div class="hotel-rank">#{rank}</div>
                    <div class="hotel-name">{hotel_name}</div>
                    <div class="hotel-id-tag">ID Hotel: {hotel_id_val}</div>
                    <div class="hotel-location">📍 {locality}, {region_val}</div>
                    <div class="hotel-stars">{stars_str}&nbsp;
                        <span style="font-size:0.75rem;color:#666">{stars_raw} Bintang</span>
                    </div>
                """, unsafe_allow_html=True)

                mc1, mc2, mc3, mc4 = st.columns([2.2, 1.5, 1.5, 1.5])
                with mc1:
                    st.markdown(f"""
                    <div class="metric-small">Skor Hybrid (Tertinggi = Terbaik)</div>
                    <div class="metric-val metric-val-gold">{hybrid_score:.4f}</div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{bar_pct:.1f}%"></div>
                    </div>""", unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"""
                    <div class="metric-small">Rating Rata-rata</div>
                    <div class="metric-val">{avg_r:.2f}
                        <span style='font-size:0.8rem;color:#666'> / 5.0</span>
                    </div>""", unsafe_allow_html=True)
                with mc3:
                    cf_disp = f"{cf_score:.4f}" if has_cf else "—"
                    st.markdown(f"""
                    <div class="metric-small">Prediksi CF</div>
                    <div class="metric-val" style="font-size:1.1rem">{cf_disp}</div>
                    """, unsafe_allow_html=True)
                with mc4:
                    cbf_disp = f"{cbf_score:.4f}" if has_cbf else "—"
                    st.markdown(f"""
                    <div class="metric-small">Prediksi CBF</div>
                    <div class="metric-val" style="font-size:1.1rem">{cbf_disp}</div>
                    """, unsafe_allow_html=True)

                st.markdown(
                    f'<div style="margin-top:12px">{badge}</div></div>',
                    unsafe_allow_html=True
                )

        if len(df_show) == 0:
            st.info("Tidak ada hotel yang cocok dengan filter yang dipilih.")
