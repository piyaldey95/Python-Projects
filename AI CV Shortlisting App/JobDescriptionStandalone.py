from groq import Groq
import json
import os
from charset_normalizer import from_path

# Load API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


def load_job_description(file_path):
    """
    Reads the job description from the specified file and automatically detects the encoding.
    """
    try:
        # Automatically detect encoding and return content as a string
        detected = from_path(file_path).best()
        if detected is None:
            raise RuntimeError("Failed to detect file encoding.")
        return str(detected)  # Converts the content to a string
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found.")
    except Exception as e:
        raise RuntimeError(f"Error reading file: {e}")


def analyze_job_description(job_description):
    """
    Analyzes the job description using the Groq API.
    """
    structure_template = """{
      "jobTitle": "",
      "company": "",
      "location": "",
      "keyResponsibilities": [
        "", "", ""
      ],
      "requiredSkills": [
        "", "", ""
      ],
      "preferredQualifications": [
        "", "", ""
      ]
    }"""

    # API call for analysis
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an HR Assistant. You will analyze Job Descriptions for the necessary skills, responsibilities, and qualifications needed for the position."
            },
            {
                "role": "user",
                "content": f"Analyze the following Job Description: {job_description} and extract the necessary details using this structure: {structure_template}. Just output an JSON, we dont need any other text ! DO NOT INVENT THINGS!"
            }
        ],
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    # Collect the output
    response_content = ""
    for chunk in completion:
        response_content += chunk.choices[0].delta.content or ""





    # Parse JSON data
    try:
        analysis_result = json.loads(response_content)
        return analysis_result
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw_response": response_content}


def save_to_json_file(data, output_file_path):
    """
    Saves the analysis results to a JSON file.
    """
    try:
        with open(output_file_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
        print(f"Output successfully saved to {output_file_path}")
    except Exception as e:
        raise RuntimeError(f"Error saving JSON file: {e}")


if __name__ == "__main__":
    # File paths
    input_file = "Job.txt"
    output_file = "JobAnalyzed.json"

    try:
        # Load the job description
        job_description = load_job_description(input_file)

        # Analyze the job description
        analysis_result = analyze_job_description(job_description)

        # Save the analysis results
        save_to_json_file(analysis_result, output_file)

    except Exception as e:
        print(f"Error: {e}")
