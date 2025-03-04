import asyncio
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import PyPDF2
import openai
import os
import base64
from io import BytesIO
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime

#### Move to Agents.py #####
# Define the Mentor agent
@st.cache_resource 
def create_mentor_agent():
    prompt_template = """
    Act as a Mentor for the candidate. 
    Provide supportive feedback, suggest improvements, and highlight the candidate's strengths.
    Be more positive in your assessment. Think step-by-step when needed.
    {context_data}
    """
    prompt = PromptTemplate(
        input_variables=["context_data"],
        template=prompt_template
    )
    memory = ConversationBufferMemory()
    # Use a smaller, faster model with appropriate parameters
    llm = ChatOpenAI(model="gpt-4o-mini", seed=100, temperature=0.0)
    return LLMChain(llm=llm, prompt=prompt, memory=memory)

# Define the Recruiter agent
@st.cache_resource 
def create_recruiter_agent():
    prompt_template = """
    Act as a Recruiter hiring for specific job opening. 
    Critically evaluate the candidate's profile, identify gaps, and assess fit for the role.
    Be more critical in your assessment. Think step-by-step when needed.
    {context_data}. 
    """
    prompt = PromptTemplate(
        input_variables=["context_data"],
        template=prompt_template
    )
    memory = ConversationBufferMemory()
    # Use a smaller, faster model with appropriate parameters
    llm = ChatOpenAI(model="gpt-4o-mini", seed=100, temperature=0.0)
    return LLMChain(llm=llm, prompt=prompt, memory=memory)

# Define the Supervisor agent
@st.cache_resource 
def create_supervisor_agent():
    prompt_template = """
    Act as a Supervisor in a Multi-Agent System that includes: (1) Supervisor (2) Recruiter (3) Mentor agents. 
    Your name is Vira.
    Summarize the following responses from the Mentor and Recruiter concisely.
    {agents_response}
    """
    prompt = PromptTemplate(
        input_variables=["agents_response"],
        template=prompt_template
    )
    memory = ConversationBufferMemory()
    # Use a smaller, faster model with appropriate parameters
    llm = ChatOpenAI(model="gpt-4o-mini", seed=100, temperature=0.0)
    return LLMChain(llm=llm, prompt=prompt, memory=memory)


# Define the Supervisor agent
@st.cache_resource 
def create_general_agent():
    prompt_template = """
    Act as a AI recruiter. Compare the given candidate's resume with the job description and help the user address the user question.
    {agent_input}
    Do not make the response long and text-heavy unless the user wants a detailed response with examples or concrete reasoning.
    Instead of ordered or unordered lists use a table when needed. 
    When highlighting positive points, use a check emoji, and when highlighting negative points, use a cross emoji.
    """
    prompt = PromptTemplate(
        input_variables=["agent_input"],
        template=prompt_template
    )
    memory = ConversationBufferMemory()
    # Use a smaller, faster model with appropriate parameters
    llm = ChatOpenAI(model="gpt-4o-mini", seed=100, temperature=0.0)
    return LLMChain(llm=llm, prompt=prompt, memory=memory)

########### Agents.py #####################

############# Move to Utils.py ##################
# Function to extract text from PDF - cached for performance
@st.cache_data
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
    return text

# Function to display PDF
def display_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%%" height="600px"></iframe>'
    return pdf_display

@st.cache_data
def convert_json_response_mentor(mentor_response):
    # Parse the JSON response
    try:
        json_score = json.loads(mentor_response)
        eval = json_score.get("evaluation", "unable to evaluate")
        suggestions = json_score.get("suggestion", "no suggestion")
        supervisor_feedback = json_score.get("detailed_feedback", "unable to assess")
        positives = json_score.get("positives", "no positives found")
        
        return {
            "evaluation" : eval,
            "suggestion" : suggestions,
            "supervisor_feedback" : supervisor_feedback,
            "positives" : positives
        }
    except Exception as e:
        print(e)
        return {
            "evaluation" : "unable to evaluate",
            "suggestion" : "no suggestion",
            "supervisor_feedback" : "unable to assess",
            "positives" : "no positives found"
        }

@st.cache_data
def convert_json_response_recruiter(recruiter_response):
    # Parse the JSON response
    try:
        json_score = json.loads(recruiter_response)
        eval = json_score.get("evaluation", "unable to evaluate")
        suggestions = json_score.get("suggestion", "no suggestion")
        supervisor_feedback = json_score.get("detailed_feedback", "unable to assess")
        negatives = json_score.get("negatives", "no weakness found")
        
        return {
            "evaluation" : eval,
            "suggestion" : suggestions,
            "supervisor_feedback" : supervisor_feedback,
            "negatives" : negatives
        }
    except Exception as e:
        print(e)
        return {
            "evaluation" : "unable to evaluate",
            "suggestion" : "no suggestion",
            "supervisor_feedback" : "unable to assess",
            "negatives" : "no weakness found"
        }

# Function to create an interactive doughnut chart
@st.cache_data
def create_interactive_doughnut(score):
    # Determine color based on score
    if score >= 80:
        color = 'green'
    elif 50 <= score < 80:
        color = 'orange'
    else:
        color = 'red'

    # Create the Doughnut Chart
    fig = go.Figure()

    # Add main doughnut chart
    fig.add_trace(go.Pie(
        values=[score, 100 - score],
        labels=["Match Rate", "Miss Rate"],  # Hide labels
        marker=dict(colors=[color, "lightgrey"]),
        hole=0.8,  # Creates the "doughnut" effect
        direction="clockwise",
        textinfo="none",
        showlegend=False
    ))

    # Add text inside the doughnut
    fig.add_annotation(
        text=f"<b>{score}%<br>Match</b>",
        x=0.5, y=0.5,  # Position in the center
        font=dict(size=22, color=color),
        showarrow=False
    )

    # Update layout for no margins & transparent background
    fig.update_layout(
        width=200,  # Reduce width
        height=200,  # Reduce height
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
    )

    return fig

# Function to display quick insights
def display_quick_insights(score, insights):
    st.sidebar.markdown("### Quick Insights")
    left, middle, right = st.sidebar.columns((3, 5, 3))
    with middle:
        st.plotly_chart(create_interactive_doughnut(score), use_container_width=False)

    # Display insights
    for insight in insights:
        st.sidebar.markdown(f"- {insight}")
######### Utils.py #################

# Function to get match score from GPT
@st.cache_data
def get_match_score(resume_text, jd_text):
    prompt = f"""
    You are an AI recruiter. Compare the given candidate's resume with the job description and provide a match score out of 100 based on these criteria:
    
    1. Minimum skills required for the opening is met by the candidate (out of 20)
    2. Minimum years of experience mentioned in the job description is met by the candidate (out of 20)
    3. Minimum educational qualification is met by the candidate (out of 20)
    4. Rate the candidate on leadership skills (both people management or tech leadership, or thought leadership: also out of 20)
    5. Rate based on additional strengths like awards, recognitions, certifications, publications, patents, or any other achievements (out of 20)

    Think step-by-step when assigning the scores. But if something is completely missing, then only go for a low score.

    Resume:
    {resume_text}

    Job Description:
    {jd_text}

    Provide the individual scores for each criterion and the total match score. Format the output as:
    Skill Match: X/20
    Experience Match: X/20
    Education Match: X/20
    Leadership Match: X/20
    Additional Strengths: X/20

    The "total_score" should be sum of all individual scores and should be out of 100.

    You will also have to give explanations to justify the total score. Along with "total_score", the output should have three explanations:
    1. "exp1" : The AI recruiter thinks you are a **(good if total_score >=80/moderate if total_score > 50 and < 80/bad if total_score <=50) match** for this role because ... (reason).
    2. "exp2" : Explain what type of expectations are not present in the candidate profile (for example, not meeting the experience level or minimum educational requirements or core skills needed)
    3. "exp3" : Explain what can be better highlighted in the profile to have a higher score.

    ** The output should be in JSON format strictly. **
    Example: If total score is 90/100, then output will be:
    ###
    Output =
    {{
    "total_score" : 90,
    "exp1" : "The AI recruiter thinks you are a **good match** for this role because you have almost all the required skills and educational qualifications.",
    "exp2" : "You will have a better match score if you can elaborate on your **leadership skills** and **people management experience** in your profile.",
    "exp3" : "During the interview, be prepared for questions on **advanced statistics**, as this role requires knowledge of **applied statistics**."
    }}
    ###
    **In the explanations, make keywords bold by enclosing them with '**'
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0,  # Lower temperature for more deterministic output
        seed=100,
        max_tokens=500  # Limit token usage
    )

    op_score = response.choices[0].message.content.strip()
    # Remove the triple backticks and the 'json' annotation
    op_score_cleaned = op_score.strip('`').replace('json', '').strip()
    # Parse the JSON response
    try:
        json_score = json.loads(op_score_cleaned)
        final_score = json_score.get("total_score", 0) # Extract integer score
        exp1 = json_score.get("exp1", "No explanation") # Extract string score
        exp2 = json_score.get("exp2", "No explanation") # Extract string score
        exp3 = json_score.get("exp3", "No explanation") # Extract string score
        return int(final_score), [exp1, exp2, exp3]
    except Exception as e:
        print(e)
        return 0, ["Unfortunately, the agent could not generate explanations due to an error"]  # Default score if JSON parsing fails

####### Move to StremlitUI.py #########

# Streamlit UI
st.set_page_config(layout="wide")
st.markdown(" <style> div[class^='st-emotion-cache-10oheav'] { padding-top: 0rem; } </style> ", unsafe_allow_html=True)

# Sidebar: Role Selection Dropdown
avatar="👩🏻‍💼"
st.title(f"Ask Vira {avatar}")

# Initialize agents
mentor_agent = create_mentor_agent()
recruiter_agent = create_recruiter_agent()
supervisor_agent = create_supervisor_agent()
general_agent = create_general_agent()

# Async function to run agents in parallel
async def run_agents_in_parallel(context_data):
    # Create tasks for concurrent execution
    with ThreadPoolExecutor(max_workers=2) as executor:
        mentor_future = executor.submit(mentor_agent.predict, context_data=context_data)
        recruiter_future = executor.submit(recruiter_agent.predict, context_data=context_data)
        
        # Wait for both to complete
        mentor_response = mentor_future.result()
        recruiter_response = recruiter_future.result()
        
    return mentor_response, recruiter_response

st.sidebar.subheader("Upload Candidate Profile and Job Description")
# Create two equal-width columns inside the sidebar
col_uploaded_resume, col_uploaded_jd = st.sidebar.columns(2)
with col_uploaded_resume:
    uploaded_resume = st.file_uploader("Upload Resume", type=["pdf"], key="resume_loader")
    if uploaded_resume is not None:
        # Save and display the PDF
        with open("temp_resume.pdf", "wb") as f:
            f.write(uploaded_resume.getbuffer())
        # Extract text
        resume_text = extract_text_from_pdf(uploaded_resume)
    else:
        resume_text = ""
    

with col_uploaded_jd:
    uploaded_jd = st.file_uploader("Upload Job Description", type=["pdf"], key="jd_loader")
    if uploaded_jd is not None:
        # Save and display the PDF
        with open("temp_jd.pdf", "wb") as f:
            f.write(uploaded_jd.getbuffer())
        # Extract text
        jd_text = extract_text_from_pdf(uploaded_jd)
    else:
        jd_text = ""

# Sidebar: Section Divider
st.sidebar.markdown(
    """
    <hr style="margin: 0; padding: 0; border: 1px solid #CCC;">
    """,
    unsafe_allow_html=True
)


if uploaded_resume and uploaded_jd:
    # Call the function in the sidebar
    # Set a sample score (You can modify this dynamically)
    match_score, insights = get_match_score(resume_text, jd_text)  # Change this value dynamically as needed
    display_quick_insights(match_score, insights)
    # Sidebar: Section Divider
    st.sidebar.markdown(
        """
        <hr style="margin: 0; padding: 0; border: 1px solid #CCC;">
        """,
        unsafe_allow_html=True
    )
    # Create two equal-width columns inside the sidebar
    col_display_resume, col_display_jd = st.sidebar.columns(2)
    with col_display_resume:
        if st.session_state.display_resume:
            st.sidebar.subheader("Candidate Profile")
            st.sidebar.markdown(display_pdf("temp_resume.pdf"), unsafe_allow_html=True)
            # Sidebar: Section Divider
            st.sidebar.markdown("---")  # Adds a horizontal line
    with col_display_jd:
        if st.session_state.display_resume:
            st.sidebar.subheader("Job Description")
            st.sidebar.markdown(display_pdf("temp_jd.pdf"), unsafe_allow_html=True)
            # Sidebar: Section Divider
            st.sidebar.markdown(
                """
                <hr style="margin: 0; padding: 0; border: 1px solid #CCC;">
                """,
                unsafe_allow_html=True
            )
    
# Cache the PDF text to avoid reprocessing
@st.cache_data
def get_pdf_text(resume_text, jd_text):
    return f"""
    This is the candidate profile: {resume_text}. This is job description document: {jd_text}.
    """

pdf_text = get_pdf_text(resume_text, jd_text) if resume_text and jd_text else ""

# Sidebar: Configurations Section
st.sidebar.subheader("Configurations")
st.sidebar.markdown("Would you like me to display the following?")

# Create a 2x2 grid for switch buttons
col1, col2 = st.sidebar.columns(2)

# Define toggles in session_state to persist changes
if "display_quick_questions" not in st.session_state:
    st.session_state.display_quick_questions = True
if "display_resume" not in st.session_state:
    st.session_state.display_resume = False
if "display_mentor_response" not in st.session_state:
    st.session_state.display_mentor_response = True
if "display_recruiter_response" not in st.session_state:
    st.session_state.display_recruiter_response = True


# Toggle Buttons
st.session_state.display_quick_questions = col1.toggle("Quick Questions", value=st.session_state.display_quick_questions)
st.session_state.display_resume = col2.toggle("Resume and JD", value=st.session_state.display_resume)
st.session_state.display_mentor_response = col1.toggle("Mentor Perspective", value=st.session_state.display_mentor_response)
st.session_state.display_recruiter_response = col2.toggle("Recruiter Perspective", value=st.session_state.display_recruiter_response)


st.markdown(
    """
    <style>
        /* Ensure sidebar remains open */
        [data-testid="stSidebar"] {
            min-width: 40% !important;
            max-width: 40% !important;
        }
        
        /* Hide the sidebar close button */
        [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }

        .block-container {
                    padding-top: 3rem;
                    padding-bottom: 0rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

########## Move to ChatbotUI.py ##############

# Chatbot UI
if uploaded_resume is None or uploaded_jd is None:
    st.warning("Please upload the resume and the job description in pdf format to start chatting.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # save st.session_state.messages
        now = datetime.now()
        filename = now.strftime("%d%m%y_%H%M.json")
        filepath = "logs/" + filename  # You can add a path here if needed, e.g., "data/" + filename
        if not os.path.exists(filepath):
            open(filepath, 'w').close()
            st.session_state.chat_logs_name = filepath
    
    # Display pre-filled questions in a 2x2 grid
    st.info(f"👋 Hello! I am your Virtual AI Recruiter. How can I help you today? Choose a question below or type your own.")
    
    # Scrollable chat history container
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
    
    # User input at the bottom
    button_pressed = False
    sample_questions = [
        "Considering my current profile, am I a good fit for this job opening?",
        "How many key skills are matching between my profile and the job requirement?",
        "What are my stengths and weaknesses of my profile for this role?",
        "How can I improve my resume to increase the chances of getting selected?"
    ]
    tooltips_info = sample_questions.copy()  # Use the same text for tooltips
    
    user_input = st.chat_input("Type your message...")

    # Add a container with the special class for the suggested questions
    if st.session_state.display_quick_questions:
            # Add division
            st.markdown(
                """
                <hr style="margin: 0; padding: 0; border: 1px solid #CCC;">
                """,
                unsafe_allow_html=True
            ) # Adds a horizontal line
            st.markdown("Quick Questions:")
            cols = st.columns(2)
            for i, question in enumerate(sample_questions):
                if cols[i % 2].button(question, help=tooltips_info[i]):
                    button_pressed = question
    
    if query := (user_input or button_pressed):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        query_classification = supervisor_agent.run(
            f"""Analyze this user query: "{query}"
            
            Is this a simple greeting, general question about the system, or small talk that doesn't require detailed resume and job description analysis?
            
            If yes, please respond with "SIMPLE" followed by an appropriate direct response in one sentence only.
            If no, respond with "COMPLEX" only.
            """
        )

        print(query_classification)

        # Create context data once
        context_data = f"""
            The candidate's resume is: {resume_text}. 
            The job description is: {jd_text}.
            Now, answer the following user query considering your role, resume data and job description data: 
            {query}
        """   

        # Show "Typing..." message
        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            # Add spinner while preparing response
            with typing_placeholder.container():
                with st.spinner(f"{avatar} Hmmm ... let me think and answer...", show_time=True):
                    # Check supervisor's decision
                    if "SIMPLE" in query_classification:
                        # Extract the direct response from supervisor
                        direct_response = query_classification.replace("SIMPLE", avatar, 1).strip()
                        supervisor_response = direct_response
                    else:
                        if (not st.session_state.display_recruiter_response) or (not st.session_state.display_mentor_response):
                            supervisor_response = general_agent.run(agent_input=context_data)
                        else:
                            # Run mentor and recruiter agents in parallel
                            mentor_response, recruiter_response = asyncio.run(run_agents_in_parallel(context_data))
                            
                            # Prepare data for supervisor
                            agents_response = f""" 
                            Mentor: {mentor_response}. 
                            Recruiter: {recruiter_response}.

                            Output Format: 
                            - Prepare a short answer first to address the user query in one paragraph as a recruiter.  
                            - Then, justify the response and try to give the responses as a table whenever possible instead of ordered or unordered lists.
                            - When highlighting positive points, use a green check emoji, and when highlighting negative points, use a red cross emoji. 
                            - Also, do not make the response long and text-heavy unless the user wants a detailed response with examples or concrete reasoning.
                            
                            if {st.session_state.display_recruiter_response}, then in short paragraphs summarize recruiters feedback.
                            if {st.session_state.display_mentor_response}, then in short paragraphs summarize mentor feedback.

                            if no, ask a follow-up question to help the job applicant.
                            """
                            # Get supervisor response
                            supervisor_response = supervisor_agent.run(agents_response=agents_response)
            displayed_response = ""
            for char in supervisor_response:
                displayed_response += char
                typing_placeholder.markdown(displayed_response)
                # Control the typing speed (adjust as needed)
                time.sleep(0.00005)  # 5ms delay between characters
            
            # Store assistant response
            st.session_state.messages.append({"role": "assistant", "content": supervisor_response})
            # save st.session_state.messages
            try:
                with open(st.session_state.chat_logs_name, 'w') as f:
                    json.dump(st.session_state.messages, f, indent=4)  # indent for readability
                print(f"Messages saved to {filepath}")
            except Exception as e:
                print(f"Error saving messages: {e}")
            
            # Don't use st.rerun() as it's expensive - let Streamlit handle updates