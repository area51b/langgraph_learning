from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def main():
    print("Hello from langgraph-learning!")


if __name__ == "__main__":
    main()
