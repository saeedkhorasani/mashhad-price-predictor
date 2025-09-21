import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from num2fawords import words

# بارگذاری مدل و ابزارها
model = joblib.load("xgb_model.pkl")
encoder = joblib.load("smart_encoder.pkl")
encoderAddress = joblib.load("address.pkl")
feature_list = joblib.load("model_features.pkl")
valid_addresses = encoderAddress.mapping_.index.tolist()
price_mean = encoderAddress.price_mean_

st.markdown("""
<style>
@font-face {
    font-family: 'BTitr';
    src: url('https://cdn.fontcdn.ir/Font/Persian/BTitr/BTitr.woff') format('woff');
}
@font-face {
    font-family: 'BNazanin';
    src: url('https://cdn.fontcdn.ir/Font/Persian/BNazanin/BNazanin.woff') format('woff');
}
body, div, textarea {
    font-family: 'BNazanin', 'BTitr', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="پیش‌بینی قیمت آپارتمان", layout="centered")

# عنوان اصلی
st.markdown("""
<div dir="rtl" style='text-align:center; font-family:B Titr; font-size:30px;'>
    <p><b>پیش‌بینی قیمت هر متر مربع آپارتمان در مشهد</b></p>
</div>
""", unsafe_allow_html=True)

# فرم ورودی در صفحه اصلی
with st.form("main_form"):
    st.markdown("""
    <div dir="rtl" style='text-align:center; font-family:B Titr; font-size:20px; color:#831f29;'>
        <b>لطفاً اطلاعات آپارتمان را وارد کنید</b>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        address = st.selectbox("منطقه", options=valid_addresses)
        area = st.number_input("متراژ (متر مربع)", min_value=40, max_value=1000, value=100)
        floor = st.number_input("طبقه", min_value=0, max_value=20, value=1)
        rooms = st.number_input("تعداد اتاق", min_value=0, max_value=5, value=1)
        has_balcony = st.checkbox("بالکن دارد؟", value=False)
        has_warehouse = st.checkbox("انباری دارد؟", value=False)
        has_parking = st.checkbox("پارکینگ دارد؟", value=False)
    with col2:
        construction_year = st.number_input("سال ساخت", min_value=1300, max_value=1403, value=1390)
        cooling_system = st.selectbox("سیستم سرمایشی", ['split', 'water_cooler', 'duct_split', 'air_conditioner', 'fan_coil'])
        heating_system = st.selectbox("سیستم گرمایشی", ['shoofaj', 'heater', 'duct_split', 'floor_heating', 'fan_coil', 'air_conditioner'])
        floor_material = st.selectbox("جنس کف", ['ceramic', 'parquet', 'stone', 'carpet', 'laminate'])
        has_elevator = st.checkbox("آسانسور دارد؟", value=True)
        is_rebuilt = st.checkbox("بازسازی شده؟", value=True)

    submitted = st.form_submit_button("پیش‌بینی قیمت")

# پردازش و پیش‌بینی
if submitted:
    df = pd.DataFrame([{
        'area': area,
        'floor': floor,
        'rooms': rooms,
        'cooling_system': cooling_system,
        'heating_system': heating_system,
        'floor_material': floor_material,
        'has_balcony': has_balcony,
        'has_warehouse': has_warehouse,
        'has_parking': has_parking,
        'construction_year': construction_year,
        'has_elevator': has_elevator,
        'is_rebuilt': is_rebuilt,
        'address': address,
    }])

    st.markdown("""
    <div dir="rtl" style='background-color:#FADBD8; padding:10px; border-radius:10px; font-family:"B Titr"; font-size:18px; text-align:right;'>
    <b>مشخصات وارد شده:</b>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df)

    # مهندسی ویژگی‌ها
    df['property_age'] = 1403 - df['construction_year']
    df['elevator_penalty'] = np.where(df['has_elevator'], 0, np.where(df['floor'] >= 3, 1, 0.3))
    df['rebuilt_penalty'] = np.where(df['is_rebuilt'], 0, np.where(df['property_age'] > 20, 1, np.where(df['property_age'] > 10, 0.5, 0)))
    df['neigh_encoded'] = encoderAddress.transform(df)['neigh_encoded']
    df = df.drop(columns=['address', 'construction_year', 'has_elevator', 'is_rebuilt'])
    df = encoder.transform(df)
    df = df.reindex(columns=feature_list)

    # تبدیل عدد به حروف فارسی
    def number_to_persian_words(n):
        if n == 0:
            return "صفر"
        if n < 0:
            return "منفی " + number_to_persian_words(-n)
        units = ["", "هزار", "میلیون", "میلیارد", "تریلیون"]
        parts = []
        while n > 0:
            parts.append(n % 1000)
            n //= 1000
        def three_digit_to_words(num):
            return words(int(num))
        result = []
        for i, part in enumerate(parts):
            if part != 0:
                word = three_digit_to_words(part)
                unit = units[i]
                result.insert(0, f"{word} {unit}".strip())
        return " و ".join(result)

    # پیش‌بینی قیمت
    predicted_price = model.predict(df)[0] * 10e5
    total_price = int(area * predicted_price)

    st.markdown("""
    <div dir="rtl" style='background-color:#D5F5E3; padding:10px; border-radius:10px; font-family:"B Titr"; font-size:18px; text-align:right;'>
    <b>قیمت هر متر مربع:</b> {:,.0f} تومان<br>
    <b>میانگین منطقه:</b> {:,.0f} تومان<br>
    <b>قیمت کل:</b> {:,.0f} تومان<br>
    <b>به حروف:</b> {} تومان
    </div>
    """.format(predicted_price, price_mean[address] * 10e5, total_price, number_to_persian_words(total_price)), unsafe_allow_html=True)

    # تحلیل SHAP تصویری
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df)
    shap.plots.waterfall(shap_values[0], show=False)
    fig = plt.gcf()
    st.markdown("""
    <div dir="rtl" style='text-align:center; font-family:B Titr; font-size:20px; color:#831f29;'>
        <b>تحلیل تصویری ویژگی‌ها</b>
    </div>
    """, unsafe_allow_html=True)
    st.pyplot(fig)

    # تحلیل متنی SHAP
    shap_vals = shap_values[0].values
    features = df.columns
    impact_df = pd.DataFrame({'ویژگی': features, 'اثر': shap_vals})
    impact_df['جهت'] = np.where(impact_df['اثر'] > 0, 'افزایش قیمت', 'کاهش قیمت')
    impact_df = impact_df.sort_values(by='اثر', key=abs, ascending=False)

    top_features = impact_df.head(5)
    report_lines = []
    for i, row in top_features.iterrows():
        line = f"ویژگی «{row['ویژگی']}» باعث {row['جهت']} شده و تأثیر آن حدوداً {row['اثر']:.2f} میلیون تومان بوده است."
        report_lines.append(line)

    report_text = "\n".join(report_lines)

    st.markdown("""
    <style>
    textarea {
        direction: rtl;
        text-align: right;
        font-family: BNazanin;
        font-size: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.text_area(label='', value=report_text, height=200)
