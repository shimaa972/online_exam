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

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    processed_data = output.getvalue()
    return processed_data

def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to bottom, #ffffff, #f7f7f7);
        }
        </style>
        """, unsafe_allow_html=True
    )

def dropdown_filter(df, column, label):
    options = ["All"] + sorted(df[column].dropna().unique().tolist())
    choice = st.selectbox(label, options)
    if choice != "All":
        df = df[df[column] == choice]
    return df

def employment_dashboard(df):
    st.subheader("Employment Dashboard")

    employed_df = df[df['salary'].notna()]
    col1, col2, col3 = st.columns(3)
    col1.metric("Employed Students", f"{employed_df['StudentKey'].nunique()}")
    col2.metric("Average Salary", f"${employed_df['salary'].mean():,.0f}")
    col3.metric("Total Salary", f"${employed_df['salary'].sum():,.0f}")

    st.divider()
    st.subheader("Hiring Insights")
    col1, col2 = st.columns(2)
    with col1:
        top_companies = employed_df['company_name'].value_counts().nlargest(10).reset_index()
        top_companies.columns = ['Company', 'Students']
        fig = px.bar(top_companies, x='Company', y='Students', color='Students', title="Top Hiring Companies",
                     color_continuous_scale=['gray', '#B22222'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        hiring_df = df[df['hiring_date'].notna()].copy()
        hiring_df['hiring_date'] = pd.to_datetime(hiring_df['hiring_date'], errors='coerce')
        timeline = hiring_df.groupby(hiring_df['hiring_date'].dt.to_period("M")).size().reset_index(name='Count')
        timeline['hiring_date'] = timeline['hiring_date'].astype(str)
        fig = px.line(timeline, x='hiring_date', y='Count', title="Hiring Timeline", markers=True,
                      color_discrete_sequence=['#B22222'])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Salary Distribution")
    fig = px.histogram(employed_df, x='salary', nbins=20, title="Salary Histogram",
                       color_discrete_sequence=['#B22222'])
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Job Type Distribution")
    job_counts = df['job_type'].value_counts().reset_index()
    job_counts.columns = ['Job Type', 'Count']
    fig = px.pie(job_counts, names='Job Type', values='Count', title="Job Type Share",
                 color_discrete_sequence=['gray', '#B22222'])
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def certificate_dashboard(df):
    st.subheader("Certificate Dashboard")

    cert_df = df[df['Certificate'].notna()]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Certificates", f"{cert_df['Certificate'].count()}")
    col2.metric("Students with Certificates", f"{cert_df['StudentKey'].nunique()}")
    col3.metric("Unique Certificates", f"{cert_df['Certificate'].nunique()}")

    st.divider()
    st.subheader("Top Certificates")
    cert_counts = cert_df['Certificate'].value_counts().nlargest(10).reset_index()
    cert_counts.columns = ['Certificate', 'Count']
    fig = px.bar(cert_counts, x='Certificate', y='Count', color='Count', title="Most Common Certificates",
                 color_continuous_scale=['gray', '#B22222'])
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    if 'Website' in cert_df.columns and cert_df['Website'].notna().any():
        st.subheader("Certificate Providers (Websites)")
        website_counts = cert_df['Website'].value_counts().reset_index()
        website_counts.columns = ['Website', 'Count']
        fig = px.bar(website_counts, x='Count', y='Website', orientation='h',
                     color='Count', color_continuous_scale=['gray', '#B22222'],
                     title="Certificate Sources by Count")
        st.plotly_chart(fig, use_container_width=True)
    # Removed the "No website data available" part

def freelance_dashboard(df):
    st.subheader("Freelance Dashboard")

    freelance_df = df[df['FreelanceKey'].notna()]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tasks", f"{freelance_df['FreelanceKey'].count()}")
    col2.metric("Total Earnings", f"${freelance_df['freelance_cost'].sum():,.0f}")
    col3.metric("Average Task Income", f"${freelance_df['freelance_cost'].mean():,.0f}")

    st.divider()
    st.subheader("Top Tasks by Cost")
    top_tasks = freelance_df[['StudentKey', 'freelance_cost', 'FreelanceKey']].sort_values(
        by='freelance_cost', ascending=False).head(10)
    st.dataframe(top_tasks.reset_index(drop=True))

def graduation_dashboard(df):
    st.subheader("Graduation Projects Dashboard")

    grad_df = df[df['Project'].notna()]
    col1, col2 = st.columns(2)
    col1.metric("Total Projects", f"{grad_df['Project'].nunique()}")
    col2.metric("Total Teams", f"{grad_df['Team'].nunique()}")

    st.divider()
    st.subheader("Most Frequent Projects")
    project_counts = grad_df['Project'].value_counts().nlargest(10).reset_index()
    project_counts.columns = ['Project Name', 'Count']
    fig = px.bar(project_counts, x='Project Name', y='Count', color='Count',
                 color_continuous_scale=['gray', '#B22222'], title="Top Projects")
    st.plotly_chart(fig, use_container_width=True)

def prediction_dashboard(df):
    st.subheader("🔮 Career Prediction")

    df['Employed'] = df['salary'].notna().astype(int)
    model_df = df[['Branch', 'job_type', 'Certificate', 'freelance_cost', 'Project', 'Employed']].dropna()
    le_branch = LabelEncoder()
    le_job = LabelEncoder()
    le_cert = LabelEncoder()
    le_proj = LabelEncoder()

    model_df['Branch_enc'] = le_branch.fit_transform(model_df['Branch'])
    model_df['Job_enc'] = le_job.fit_transform(model_df['job_type'])
    model_df['Cert_enc'] = le_cert.fit_transform(model_df['Certificate'])
    model_df['Proj_enc'] = le_proj.fit_transform(model_df['Project'])

    X = model_df[['Branch_enc', 'Job_enc', 'Cert_enc', 'freelance_cost', 'Proj_enc']]
    y = model_df['Employed']
    model = RandomForestClassifier().fit(X, y)

    student_names = df['Student'].dropna().unique()
    selected_student = st.selectbox("Select Student", sorted(student_names))

    student_row = df[df['Student'] == selected_student].iloc[0]

    st.markdown("#### 📋 Student Data Used in Prediction")
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
        st.write("🔮 Career Outcome Prediction: ", "Employed" if pred[0] == 1 else "Not Employed")
    except KeyError:
        st.warning("Prediction data is incomplete for the selected student.")

def main():
    set_background()
    
    # Display logo next to title
    col1, col2 = st.columns([1, 6])  # Adjust columns size
    with col1:
        logo = Image.open("logo.jpg")  # Update with your actual path to logo.jpg
        st.image(logo, width=70)  # Adjust the width as needed
    with col2:
        st.title("ITI Online Exam Dashboard")

    df = load_data()

    st.sidebar.title("Select Dashboard")
    dashboard_selection = st.sidebar.radio("Choose a Dashboard", ["Employment", "Certificates", "Freelance", "Graduation", "Career Prediction"])

    if dashboard_selection == "Employment":
        employment_dashboard(df)
    elif dashboard_selection == "Certificates":
        certificate_dashboard(df)
    elif dashboard_selection == "Freelance":
        freelance_dashboard(df)
    elif dashboard_selection == "Graduation":
        graduation_dashboard(df)
    else:
        prediction_dashboard(df)

if __name__ == "__main__":
    main()
