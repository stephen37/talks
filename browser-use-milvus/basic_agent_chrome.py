from browser_use import Agent, Browser, BrowserConfig
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# Configure the browser to connect to your Chrome instance
browser = Browser(
    config=BrowserConfig(
        # Specify the path to your Chrome executable
        chrome_instance_path='/Applications/Arc.app/Contents/MacOS/Arc',  # macOS path
    )
)

# Create the agent with your configured browser
agent = Agent(
    task="Go on CommonRoom and find the most active users that have been active in the last 72 hours",
    llm=ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', google_api_key=api_key),
    browser=browser,
)

async def main():
    await agent.run()

    input('Press Enter to close the browser...')
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
