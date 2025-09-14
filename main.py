import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv
import io
import os
from openai import OpenAI
from daytona import Daytona, DaytonaConfig, SessionExecuteRequest

load_dotenv()

st.title("AutoML Agent")
st.markdown("Upload a CSV file to get started")

uploaded_file = st.file_uploader("uplaod a csv file", type=["csv"])

def summarize_data(dataframe: pd.DataFrame) -> str:
    # prompt = f"""
    #     You are an expert data scientist, specifically in the field of data summarization.
    #     You are given a dataframe and you need to summarize the data.
    #     The dataframe is as follows:
    #     {dataframe.head()}
    # """
    try:
        buffer = io.StringIO()
        sample_rows = min(30, len(dataframe))
        dataframe.head(sample_rows).to_csv(buffer, index=False)
        sample_csv = buffer.getvalue()
        dtypes = dataframe.dtypes.astype(str).to_dict()
        non_null_counts = dataframe.notnull().sum().to_dict()
        null_counts = dataframe.notnull().sum().to_dict()
        nunique = dataframe.nunique(dropna=True).to_dict()
        numer_cols = [c for c in dataframe.columns if pd.api.
        types.is_numeric_dtype(dataframe[c])]

        #generate descriptive statstistcs of numeric columns
        desc = dataframe[numer_cols].describe().to_dict() if numer_cols else {}

        lines = []
        lines.append("Scheme (dtype):")
        for k,v in dtypes.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

        lines.append("Null/Non-Null counts:")
        null_counts = dataframe.isnull().sum().to_dict()
        non_null_counts = dataframe.notnull().sum().to_dict()
        for c in dataframe.columns:
            lines.append(f"  - {c}: nulls={int(null_counts[c])}, non_nulls={int(non_null_counts[c])}")
        lines.append("")

        # Section 3: Cardinality (uniqueness) information
        lines.append("Cardinality (nunique):")
        for k, v in nunique.items():
            lines.append(f"  - {k}: {int(v)}")
        lines.append("")

        # Section 4: Statistical summaries for numeric columns (if any exist)
        if desc:
            lines.append("Numeric summary stats (describe):")
            for col, stats in desc.items():
                # Format each statistic with rounding and handle NaN values
                stat_line = ", ".join(
                    [f"{s}:{round(float(val), 4) if pd.notnull(val) else 'nan'}" for s, val in stats.items()]
                )
                lines.append(f"  - {col}: {stat_line}")
        lines.append("")

        lines.append("Sample row (CSV head):")
        lines.append(sample_csv)

        return "\n".join(lines)

    except Exception as e:
        return f"Error summarizing data: {e}"






def build_cleaning_prompt(df, selected_column):
    data_summary = summarize_data(df)
    prompt = f"""
        You an expert data scientist, specificially in the field of
        data cleaning.
        You are given a dataframe and you need to clean the data. 

        The data summary is as follows:
        {data_summary}

        Please clean the data and return the cleaned data. 
        Make sure to handle the following:
        -Missing values
        -Duplicate values
        -Outliers
        -standardize the data accordingly
        -Use one-hot-encoding for categorical variables
        -Make sure to drop the column {selected_column} from the data 

       
        write a Python script to clean the data, based on the data
        summary provided, in a json property called "script".
        - DO NOT PRINT to stdout or stderr. 

        ##IMPORTANT
        - MAKE SURE THE PYTHON SCRIPT IS STANDALONE
        - Make sure to load in the data from a csv file called "input.csv"
        - This script should be a Python script that can be executed to clean the data.
        - Make sure to save the cleaned data to a new csv file called "cleaned.csv".
        - print inital shape and final shape of data 
    """
    return prompt

def get_openai_script(prompt: str) -> str:

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior data scientist. "
                        "Always return a strict JSON object "
                        "matching the user's requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        if not resp or not getattr(resp, "choices", None):
            return None

        text = resp.choices[0].message.content or ""

        # Expect a JSON object like {"script": "..."}
        try:
            data = json.loads(text)
            script_val = data.get("script")
            if isinstance(script_val, str) and script_val.strip():
                return script_val.strip()
        except Exception as e:
            print(f"Error parsing script: {text} {e}")
    except Exception as e:
        print(f"Error getting script: {e}")
        return None

def execute_in_daytona(script: str, csv_bytes: bytes):
    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        raise ValueError("DAYTONA_API_KEY is not set")

    client = Daytona(DaytonaConfig(api_key=api_key))

    # Clean up old sandboxes first
    try:
        existing_sandboxes = client.list()
        for sandbox_info in existing_sandboxes:
            try:
                client.delete(sandbox_info.id)
                print(f"Deleted sandbox: {sandbox_info.id}")
            except Exception as e:
                print(f"Could not delete sandbox {sandbox_info.id}: {e}")
    except Exception as e:
        print(f"Error listing/deleting sandboxes: {e}")
    
    sandbox = client.create()
    exec_info = {}
    try:
        sandbox.fs.upload_file(csv_bytes, "input.csv")

        cmd = "python -u - <<'PY'\n" + script + "\nPY"
        result = sandbox.process.exec(cmd,timeout=600,env={"PYTHONUNBUFFERED": "1"})
        exec_info = {}
        exec_info["exit_code"] = getattr(result, "exit_code", None)
        exec_info["stdout"] = getattr(result, "result", "")
        exec_info["stderr"] = getattr(result, "stderr", "")

        try:
            cleaned_bytes = sandbox.fs.download_file("cleaned.csv")
            return cleaned_bytes, exec_info
        except Exception as e:
            print(f"Error downloading cleaned file: {e}")
            return None, exec_info
    except Exception as e:
        print(f"Error executing in Daytona: {e}")
        return None, exec_info
    finally:
        # Always clean up the sandbox we just created
        try:
            client.delete(sandbox.id)
        except Exception as e:
            print(f"Error deleting current sandbox: {e}")



if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    selected_column = st.selectbox("Select a column to predict",
    df.columns.tolist(),
    help="the column to predict")

    button = st.button("Run AutoML")

    if button:
       with st.spinner("Running AutoML..."):
            clearning_prompt = build_cleaning_prompt(df, selected_column)
            with st.expander("Cleaning Prompt"):    
                st.write(clearning_prompt)
            script = get_openai_script(clearning_prompt)
            with st.expander("Script"):
                st.code(script)
            with st.spinner("Excecuting in Daytona..."):
                input_csv_bytes = df.to_csv(index=False).encode("utf-8")
                cleaned_bytes, exec_info = execute_in_daytona(script, input_csv_bytes)

                with st.expander("Execution Info"):
                    st.write(f"Exit code: {exec_info.get('exit_code')}")
                    st.write(f"Stdout: {exec_info.get('stdout')}")
                    st.write(f"Stderr: {exec_info.get('stderr')}")
    
                    # Add debug info
                    if cleaned_bytes is None:
                        st.error("⚠️ No cleaned data returned from Daytona")
                    else:
                        st.write(f"✅ Cleaned data size: {len(cleaned_bytes)} bytes")

                with st.expander("Cleaned Data"):
                    if cleaned_bytes and len(cleaned_bytes) > 0:
                        try:
                            # Check if it's valid CSV content
                            content_preview = cleaned_bytes[:100].decode('utf-8', errors='ignore')
                            st.write(f"Content preview: {content_preview}")
            
                            cleaned_df = pd.read_csv(io.BytesIO(cleaned_bytes))
                            st.dataframe(cleaned_df)
                        except Exception as e:
                            st.error(f"Error reading cleaned data: {e}")
                            st.write("Raw cleaned bytes:", cleaned_bytes[:200])
                    else:
                        st.error("No cleaned data received or empty file")


           