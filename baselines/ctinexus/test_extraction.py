import os
from dotenv import load_dotenv
from ctinexus import process_cti_report
import json

load_dotenv()

text = "APT29 used PowerShell to download additional malware from command-and-control server at 192.168.1.100."

try:
    # Use gpt-4o-mini if gpt-5-mini is not available
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    print(f"Using model: {model}")
    
    result = process_cti_report(
        text=text,
        provider="openai",
        model=model,
    )
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Error: {e}")
