import streamlit as st
import pandas as pd
import pyodbc
import plotly.express as px
from PIL import Image
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

@st.cache_data(ttl=600)
def load_data():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=db22845.public.databaseasp.net;'
        'DATABASE=db22845;'
        'UID=db22845;'
        'PWD=12345678'
    )
    df = pd.read_sql("""
        SELECT F.*, 
               B.Branch_name AS Branch, 
               C.Certificate_Name AS Certificate,
               G.pro_name AS Project, 
               G.team_number AS Team,
               S.FullName AS Student
        FROM Fact_Student_Activity F
        LEFT JOIN dimbranch B ON F.BranchKey = B.BranchKey
        LEFT JOIN dimcertificate C ON F.CertificateKey = C.CertificateKey
        LEFT JOIN dimgraduation G ON F.GraduationKey = G.GraduationKey
        LEFT JOIN dimstudent S ON F.StudentKey = S.StudentKey
    """, conn)
    return df

def improved_salary_chart(df):
    employed_df = df[df['salary'].notna()]
    
    fig = px.histogram(employed_df, 
                      x='salary',
                      nbins=15,
                      color_discrete_sequence=["#900d06"],
                      opacity=0.8,
                      labels={'salary': 'Monthly Salary (USD)', 'count': 'Number of Students'},
                      title='<b>📊 Salary Distribution Analysis</b><br><span style="font-size:14px">Detailed breakdown of graduate salaries</span>')
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x',
        bargap=0.1,
        xaxis=dict(
            title='Salary Range (USD)',
            tickprefix='$',
            gridcolor='lightgray',
            range=[employed_df['salary'].min()-5000, employed_df['salary'].max()+5000]
        ),
        yaxis=dict(
            title='Number of Students',
            gridcolor='lightgray'
        ),
        font=dict(
            family="Arial",
            size=12,
            color="black"
        ),
        title_x=0.5,
        title_font_size=20
    )
    
    mean_salary = employed_df['salary'].mean()
    fig.add_vline(x=mean_salary, line_width=2, line_dash="dash", line_color="yellow",
                 annotation_text=f"Mean: ${mean_salary:,.0f}", 
                 annotation_position="top right")
    
    fig.add_annotation(
        x=0.95,
        y=0.95,
        xref='paper',
        yref='paper',
        text=f"Total Students: {len(employed_df)}<br>Min: ${employed_df['salary'].min():,.0f}<br>Max: ${employed_df['salary'].max():,.0f}",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig

def employment_dashboard(df):
    st.subheader("📈 Employment Analytics")

    employed_df = df[df['salary'].notna()]
    col1, col2, col3 = st.columns(3)
    col1.metric("Employed Students", f"{employed_df['StudentKey'].nunique()}")
    col2.metric("Average Salary", f"${employed_df['salary'].mean():,.0f}")
    col3.metric("Total Salary", f"${employed_df['salary'].sum():,.0f}")

    st.divider()
    st.subheader("💰 Salary Distribution")
    st.plotly_chart(improved_salary_chart(df), use_container_width=True)

    st.divider()
    st.subheader("🏢 Hiring Insights")
    col1, col2 = st.columns(2)
    with col1:
        top_companies = employed_df['company_name'].value_counts().nlargest(10).reset_index()
        top_companies.columns = ['Company', 'Students']
        fig = px.bar(top_companies, x='Company', y='Students', color='Students', 
                     title="Top 10 Hiring Companies",
                     color_continuous_scale=['gray', '#B22222'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        hiring_df = df[df['hiring_date'].notna()].copy()
        hiring_df['hiring_date'] = pd.to_datetime(hiring_df['hiring_date'], errors='coerce')
        timeline = hiring_df.groupby(hiring_df['hiring_date'].dt.to_period("M")).size().reset_index(name='Count')
        timeline['hiring_date'] = timeline['hiring_date'].astype(str)
        fig = px.line(timeline, x='hiring_date', y='Count', 
                      title="Monthly Hiring Trends", 
                      markers=True, color_discrete_sequence=['#B22222'])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("👔 Job Type Breakdown")
    job_counts = df['job_type'].value_counts().reset_index()
    job_counts.columns = ['Job Type', 'Count']
    fig = px.pie(job_counts, names='Job Type', values='Count', 
                 title="Job Type Distribution",
                 color_discrete_sequence=['gray', '#B22222'])
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def certificate_dashboard(df):
    st.subheader("📜 Certificates Analytics")

    cert_df = df[df['Certificate'].notna()]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Certificates", f"{cert_df['Certificate'].count()}")
    col2.metric("Students with Certificates", f"{cert_df['StudentKey'].nunique()}")
    col3.metric("Unique Certificates", f"{cert_df['Certificate'].nunique()}")

    st.divider()
    st.subheader("🏆 Top Certificates")
    cert_counts = cert_df['Certificate'].value_counts().nlargest(10).reset_index()
    cert_counts.columns = ['Certificate', 'Count']
    fig = px.bar(cert_counts, x='Certificate', y='Count', color='Count',
                 title="Top 10 Certificates",
                 color_continuous_scale=['gray', "#D90909"])
    st.plotly_chart(fig, use_container_width=True)

def freelance_dashboard(df):
    st.subheader("💼 Freelance Analytics")

    freelance_df = df[df['FreelanceKey'].notna()]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tasks", f"{freelance_df['FreelanceKey'].count()}")
    col2.metric("Total Earnings", f"${freelance_df['freelance_cost'].sum():,.0f}")
    col3.metric("Average Task Income", f"${freelance_df['freelance_cost'].mean():,.0f}")

    st.divider()
    st.subheader("🏅 Top Earning Tasks")
    top_tasks = freelance_df[['Student', 'freelance_cost']].sort_values(
        by='freelance_cost', ascending=False).head(10)
    st.dataframe(top_tasks.reset_index(drop=True))

def graduation_dashboard(df):
    st.subheader("🎓 Graduation Projects Analytics")

    grad_df = df[df['Project'].notna()]
    col1, col2 = st.columns(2)
    col1.metric("Total Projects", f"{grad_df['Project'].nunique()}")
    col2.metric("Total Teams", f"{grad_df['Team'].nunique()}")

    st.divider()
    st.subheader("🚀 Popular Projects")
    project_counts = grad_df['Project'].value_counts().nlargest(10).reset_index()
    project_counts.columns = ['Project', 'Count']
    fig = px.bar(project_counts, x='Project', y='Count', color='Count',
                 title="Top 10 Projects",
                 color_continuous_scale=['gray', '#B22222'])
    st.plotly_chart(fig, use_container_width=True)
# ... (جميع الدوال السابقة تبقى كما هي)

def prediction_dashboard(df):
    st.subheader("🔮 Career Prediction")
    
    # تحضير البيانات
    df['Employed'] = df['salary'].notna().astype(int)
    model_df = df[['Branch', 'job_type', 'Certificate', 'freelance_cost', 'Project', 'Employed']].dropna()
    
    # ترميز البيانات
    le_branch = LabelEncoder()
    le_job = LabelEncoder()
    le_cert = LabelEncoder()
    le_proj = LabelEncoder()
    
    model_df['Branch_enc'] = le_branch.fit_transform(model_df['Branch'])
    model_df['Job_enc'] = le_job.fit_transform(model_df['job_type'])
    model_df['Cert_enc'] = le_cert.fit_transform(model_df['Certificate'])
    model_df['Proj_enc'] = le_proj.fit_transform(model_df['Project'])
    
    # تدريب النموذج
    X = model_df[['Branch_enc', 'Job_enc', 'Cert_enc', 'freelance_cost', 'Proj_enc']]
    y = model_df['Employed']
    model = RandomForestClassifier().fit(X, y)
    
    # واجهة المستخدم
    student_names = df['Student'].dropna().unique()
    selected_student = st.selectbox("Select Student", sorted(student_names))
    
    student_row = df[df['Student'] == selected_student].iloc[0]
    
    st.markdown("#### 📋 Student Profile")
    st.write(student_row[['Branch', 'job_type', 'Certificate', 'freelance_cost', 'Project']])
    
    try:
        input_data = [
            le_branch.transform([student_row['Branch']])[0],
            le_job.transform([student_row['job_type']])[0],
            le_cert.transform([student_row['Certificate']])[0],
            student_row['freelance_cost'],
            le_proj.transform([student_row['Project']])[0]
        ]
        pred = model.predict([input_data])
        st.success(f"Prediction Result: {'✅ Employed' if pred[0] == 1 else '❌ Not Employed'}")
    except Exception as e:
        st.warning(f"Cannot make prediction: {str(e)}")

def main():
    st.set_page_config(layout="wide")
    
    # Display logo and title
    col1, col2 = st.columns([1, 6])
    with col1:
        logo = Image.open("logo.jpg")
        st.image(logo, width=70)
    with col2:
        st.title("ITI Student Analytics Dashboard")

    df = load_data()

    # Create tabs - تم إضافة تبويب التنبؤ هنا
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Employment", 
        "📜 Certificates", 
        "💼 Freelance", 
        "🎓 Graduation",
        "🔮 Prediction"  # تمت إضافته
    ])

    with tab1:
        employment_dashboard(df)
    with tab2:
        certificate_dashboard(df)
    with tab3:
        freelance_dashboard(df)
    with tab4:
        graduation_dashboard(df)
    with tab5:  # القسم الجديد
        prediction_dashboard(df)

if __name__ == "__main__":
    main()
