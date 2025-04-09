import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# 1. قراءة الداتا
data = pd.read_excel(r"D:\project_movie\data\Code\egypt_tourism_data_5000.xlsx")

# 2. تحضير الداتا
features = data[['Preferred_Destination', 'Spending_USD', 'Accommodation_Type']]
features_encoded = pd.get_dummies(features, columns=['Accommodation_Type'], prefix='Accommodation')
destination_features = features_encoded.groupby('Preferred_Destination').mean().reset_index()

# 3. حساب مصفوفة التشابه
feature_matrix = destination_features.drop('Preferred_Destination', axis=1)
similarity_matrix = cosine_similarity(feature_matrix)

# 4. دالة التوصية
def recommend_destinations(tourist_id, data, destination_features, similarity_matrix, top_n=3):
    preferred_dest = data[data['Tourist_ID'] == tourist_id]['Preferred_Destination'].values[0]
    dest_idx = destination_features[destination_features['Preferred_Destination'] == preferred_dest].index[0]
    similar_scores = similarity_matrix[dest_idx]
    similar_indices = similar_scores.argsort()[::-1]
    recommendations = [
        destination_features.iloc[idx]['Preferred_Destination'] 
        for idx in similar_indices if idx != dest_idx
    ][:top_n]
    return preferred_dest, recommendations

# 5. واجهة Streamlit
st.title("نظام توصية السياحة في مصر")
st.write("ادخل رقم السائح (Tourist_ID) للحصول على توصيات بالوجهات السياحية")

# اختيار Tourist_ID من قائمة منسدلة
tourist_ids = data['Tourist_ID'].unique()
tourist_id = st.selectbox("اختر رقم السائح:", tourist_ids)

# زر لعرض التوصيات
if st.button("اعرض التوصيات"):
    preferred_dest, recommendations = recommend_destinations(tourist_id, data, destination_features, similarity_matrix)
    st.write(f"الوجهة المفضلة للسائح {tourist_id}: **{preferred_dest}**")
    st.write("التوصيات:")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")

st.write("تم تطوير هذا النظام باستخدام Content-Based Filtering")