import streamlit as st
import numpy as np
import pandas as pd
import base64

def set_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Use your local file path here
set_bg_from_local("C:\\Users\\Krishnapriya\\Downloads\\download.jpeg")

st.title("My First Streamlit App")


# Adding background gradient
#st.markdown(
#    """
#    <style>
#    .stApp {
#        background: linear-gradient(to right, rgba(0,176,155,0.5), rgba(150,201,61,0.5));
#        background-attachment: fixed;
#        background-size: cover;
#    }
#    </style>
#    """,
#    unsafe_allow_html=True
#)


# Creating tabs for organizing content
tab1, tab2, tab3, tab4 = st.tabs(["Basics", "Interactive Widgets", "Visualizations","LLM Interaction"])
with tab1:
	# Basic Streamlit elements
	# Setting the title and headers
	st.title("My First Streamlit App")
	st.header("This is a Header")
	st.subheader("This is a Subheader")
	
	st.write("This is a simple text display")
	
	# Creating a simple DataFrame
	df = pd.DataFrame({
		'first column': [1, 2, 3, 4],
		'second column': [10, 20, 30, 40]
	})
	
	# Writing an expander
	with st.expander("About AI"):
		st.write("Here is more info: " \
		"AI stands for Artificial Intelligence, " \
		"which refers to the simulation of human intelligence in machines. " \
		"These machines are programmed to think like humans and mimic their actions.")
	
	
	# Display the DataFrame
	st.write("Here is the DataFrame:")
	st.write(df)
	
	# Create a line chart
	chart_data=pd.DataFrame(
		np.random.randn(20,3),columns=['a','b','c']
	)
	st.line_chart(chart_data)

# Interactive Widgets
with tab2: 
	st.write("Below are a few interactive widgets:")
	
	# Adding a button and some interactivity
	st.write("button:")
	if st.button('Say hello'):
		st.write('Hello!')
	
	st.button('Say goodbye')
	
	# Adding a text input
	user_input = st.text_input("Your input here", "Type something...")
	if user_input:
		st.write(f"You wrote: {user_input}")
	
		# Adding a button to submit the text
		if st.button('Submit'):
			st.write(f"Thank you for your input!")
	
	# Adding a number input
	number = st.number_input("Insert a number", min_value=0, max_value=100, value=50)
	st.write(f"You selected: {number}")
	
	# Adding a slider
	slider_value = st.slider("Select a range", 0, 100, (25, 75))
	st.write(f"You selected the range: {slider_value}")
	
	# Adding a selectbox
	option = st.selectbox("Choose an option: ", ["None", "AI", "Machine Learning", "Deep Learning", "Data Science"])
	st.write(f"You selected: {option}")

	# Adding a multiselect
	options = st.multiselect("Select multiple options: ", ["Python", "Java", "C++", "JavaScript"])
	st.write(f"You selected: {options}")
	
	# Adding a checkbox
	checkbox_value = st.checkbox("I agree to the terms and conditions.")
	if checkbox_value:
		st.write("Thank you for agreeing to the terms and conditions!")
	
	# Adding a radio button
	radio_value = st.radio("Choose one option: ", ["None", "LLM response", "Evaluation", "Feedback"])
	st.write(f"You selected: {radio_value}")
	
	# Adding date picker
	date = st.date_input("Select a date")
	st.write(f"You selected: {date}")
	
	# Adding time input
	time = st.time_input("Select a time")
	st.write(f"You selected: {time}")
	
	# Adding a file uploader
	uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt"])
	if uploaded_file is not None:
		if uploaded_file.type == "text/csv":
			df_uploaded = pd.read_csv(uploaded_file)
	
			st.write("Here is the uploaded CSV file:")
			st.write(df_uploaded)
	
	
	# Adding a divider
	st.divider()
	
	st.title("Survey Form")
	col1, col2 = st.columns(2)

	with col1: 
		name = st.text_input("Name")
		gender = st.radio("Gender",["Male","Female","Others"])
	
	with col2:
		age = st.slider("Age", 0, 100)
		fav_color = st.selectbox("Favorite Color", ["Red", "Blue", "Green"])
	
	if st.button("Submit Survey"):
		st.success(f"{name} ({gender}, {age}) likes the color {fav_color}")
		
with tab3: 
	# Visualizations
	# importing images
	from PIL import Image
	
	# Displaying a image from local file
	image = Image.open("image_1776.png")
	st.image(image, caption="Sample Image", use_container_width=True)

	# Displaying images from URLs
	st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/800px-Fronalpstock_big.jpg", caption="Sample Image from URL", use_container_width=True)
	

	# Plotting using matplotlib
	import matplotlib.pyplot as plt
	x = [1, 2, 3, 4, 5]
	y = [10, 20, 25, 30, 40]
	plt.plot(x, y)
	st.pyplot(plt)
	st.write("This is a simple line plot using Matplotlib.")
	
	# Plotting using seaborn
	import seaborn as sns
	tip = sns.load_dataset("tips")
	sns_plot = sns.scatterplot(data=tip, x="total_bill", y="tip", hue="day")
	st.pyplot(sns_plot.figure)
	st.write("This is a scatter plot using Seaborn.")
	
	# Plotting using plotly
	import plotly.express as px
	fig = px.scatter(tip, x="total_bill", y="tip", color="day", title="Total Bill vs Tip")
	st.plotly_chart(fig)
	st.write("This is an interactive scatter plot using Plotly.")

	# Plotting using Altair
	import altair as alt
	alt_chart = alt.Chart(tip).mark_circle(size=60).encode(
		x='total_bill',
		y='tip',
		color='day',
		tooltip=['total_bill', 'tip', 'day']
	).interactive()     
	st.altair_chart(alt_chart, use_container_width=True)
	st.write("This is an interactive scatter plot using Altair.")
	
with tab4: 
	import streamlit as st
	import google.generativeai as genai
	from google.generativeai.types import GenerationConfig

	# 🔐 Configure Gemini API key
	genai.configure(api_key="AIzaSyAp8dYX_4kErPhgBQbULuKRtCksbN3075I")  # Replace with your real key

	# 🧠 Initialize the model once (not inside function or button)
	@st.cache_resource
	def load_model():
		return genai.GenerativeModel(
			model_name="gemini-2.5-flash",
			generation_config=GenerationConfig(
				temperature=0.7,
				top_p=0.9
			)
		)
	
	model = load_model()

	# 🎛️ Streamlit UI
	st.title("💬 Gemini 2.5 Chatbot with Streamlit")
	prompt = st.text_area("📝 Enter your prompt:", height=100)

	# 💬 Generate on button click
	if st.button("Generate Response"):
		if prompt.strip() == "":
			st.warning("⚠️ Please enter a valid prompt.")
		else:
			with st.spinner("Generating..."):
				try:
					response = model.generate_content(prompt)
					st.markdown("### 🤖 Gemini says:")
					st.write(response.text)
				except Exception as e:
					st.error(f"❌ Error occurred: {e}")