import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ضبط إعدادات الواجهة الاحترافية
st.set_page_config(
    page_title="المنصة الوطنية لإدارة أزمات الكهرباء - العراق",
    page_icon="⚡",
    layout="wide"
)


# تحميل البيانات والموديل المطور
@st.cache_resource
def load_assets():
    model = joblib.load("nlp_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    df = pd.read_csv("kaggle_processed_iraq.csv")
    return model, vectorizer, df


model, vectorizer, df = load_assets()

# القائمة الجانبية لنظام الصفحات التنفيذية
st.sidebar.title("🏢 مركز التحكم والسيطرة")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "انتقل إلى:",
    [
        "📊 لوحة التحليلات التنفيذية",
        "🤖 مركز تصنيف الشكاوى (AI)",
        "🚨 غرفة عمليات التوجيه الفوري",
        "🗺️ الخريطة التفاعلية للأعطال"
    ]
)

st.sidebar.markdown("---")
selected_city = st.sidebar.selectbox("فلترة حسب المحافظة:", ["الكل"] + list(df["City"].unique()))

filtered_df = df if selected_city == "الكل" else df[df["City"] == selected_city]

# Header المنصة
st.title("⚡ المنصة الوطنية الموحدة لإدارة وأعطال شبكات الكهرباء")
st.caption(f"نظام الذكاء الاصطناعي للمراقبة والاستجابة - إجمالي السجلات المعالجة: {len(filtered_df):,} بلاغ")
st.markdown("---")

# 1. الصفحة الأولى: التحليلات التنفيذية
if page == "📊 لوحة التحليلات التنفيذية":
    st.header("📊 المؤشرات الرئيسية والأداء (KPIs)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("إجمالي البلاغات", f"{len(filtered_df):,}")
    col2.metric("الحالات الحرجة جداً", f"{len(filtered_df[filtered_df['Severity'] == 'عالي جداً']):,}")
    col3.metric("معدل درجات الحرارة", f"{int(filtered_df['Temperature_C'].mean())}°C")
    col4.metric("المستشفيات التأثرة", f"{filtered_df['Hospitals_Nearby'].sum():,}")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        fig_pie = px.pie(filtered_df, names="Fault_Type", title="توزيع الأعطال حسب التصنيف الذكي", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_bar = px.histogram(filtered_df, x="City", color="Severity",
                               title="كثافة البلاغات ومستوى الخطورة حسب المحافظات")
        st.plotly_chart(fig_bar, use_container_width=True)

# 2. الصفحة الثانية: مركز تصنيف الشكاوى
elif page == "🤖 مركز تصنيف الشكاوى (AI)":
    st.header("🤖 محرك التحليل اللغوي للشكاوى (NLP Engine)")
    st.write("اختبار تصنيف الشكاوى لحظياً باستخدام النموذج المكتشف على داتا كاجل:")

    user_input = st.text_area("أدخل نص الشكوى أو البلاغ الوارد من المواطن:", height=100)

    if st.button("تحليل الشكوى عبر الذكاء الاصطناعي"):
        if user_input.strip() != "":
            vec = vectorizer.transform([user_input])
            pred = model.predict(vec)[0]

            st.success(f"🎯 التصنيف المتوقع لنوع العطل: **{pred}**")
            st.info("تم توجيه البلاغ تلقائياً إلى القسم المختص بناءً على هذا التصنيف.")
        else:
            st.warning("الرجاء إدخال نص الشكوى أولاً.")

# 3. الصفحة الثالثة: غرفة عمليات التوجيه الفوري
elif page == "🚨 غرفة عمليات التوجيه الفوري":
    st.header("🚨 قائمة الأولوية لتوجيه فرق الصيانة (Dispatch Priority)")
    st.write("ترتيب تلقائي للبلاغات بناءً على وجود المستشفيات بالقرب من العطل وشدة الخطورة:")

    priority_df = filtered_df.sort_values(
        by=["Hospitals_Nearby", "Severity", "Outage_Duration_Hrs"],
        ascending=[False, False, False]
    )

    st.dataframe(
        priority_df[["Complaint_ID", "City", "Fault_Type", "Severity", "Hospitals_Nearby", "Outage_Duration_Hrs"]],
        use_container_width=True,
        height=450
    )

# 4. الصفحة الرابعة: الخريطة التفاعلية
elif page == "🗺️ الخريطة التفاعلية للأعطال":
    st.header("🗺️ التوزيع الجغرافي المباشر للبلاغات والأعطال")

    map_data = filtered_df[["Latitude", "Longitude"]].rename(
        columns={"Latitude": "lat", "Longitude": "lon"}
    )
    st.map(map_data, zoom=5 if selected_city == "الكل" else 9)