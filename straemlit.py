import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def crawl_contractors(max_pages=1):
    base_url = "https://muqawil.org/en/contractors"
    links = []
    all_contractors = []
    needed_boxes = ["Membership Number" , "Organization Email" , "Organization Mobile Number" , "City"]
    results = {}

    for page in range(1, max_pages + 1):
        # Construct URL with pagination
        url = f"{base_url}?page={page}"

        try:
            # Send GET request with headers to mimic browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise exception for bad status codes

            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find contractor cards
            contractor_cards = soup.find_all('div', class_='has-action')

            print(f"Found {len(contractor_cards)} cards on page {page}")

            # Extract information from each card
            for card in contractor_cards:
              # to get the card that have the link to the needed infomation
                botton = card.find('div', class_='col-md-3 col-sm-6')
                link = botton.find('a')['href']
                print(f"Extracted info for {link}")
                links.append(link)
                response = requests.get(link, headers=headers, timeout=10)
                response.raise_for_status()  # Raise exception for bad status codes

                # Parse HTML content
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find contractor cards
                contractor_cards = soup.find_all('div', class_='container')

                print(f"Found {len(contractor_cards)} cards on page {page}")
                cont = 1
                for card in contractor_cards:
                  if cont == 1:
                      title = soup.find('h3', class_='card-title')
                      results['contractor_name'] = title.text.strip()
                      vld_info = soup.find('div', class_='info-box-wrapper')
                      infos = soup.find_all('div', class_='col-md-6 col-lg-4')
                      print(len(infos))
                      # Loop through the info boxes
                      for info in infos:
                          # Find the div with class 'info-name'
                          name_div = info.find('div', class_='info-name')
                          if name_div and name_div.text.strip() in needed_boxes:
                              # Get the corresponding 'info-value'
                              value_div = info.find('div', class_='info-value')
                              if value_div:
                                  # Add to results dictionary
                                  results[name_div.text.strip()] = value_div.text.strip()

                      # Print the extracted results
                      for key, value in results.items():
                          print(f"{key}: {value}")

                  #Here claude write your code!!
                  section_cards = card.find_all('div', class_='section-card')
                  for section in section_cards:
                      title = section.find('h3', class_='card-title')
                      if title and title.text.strip() == 'Interests':
                          # Extract all text content from this section
                          interests_text = ' '.join(section.stripped_strings)
                          print(f"Interests Text: {interests_text}")
                          results["Interests"] = interests_text


                  cont = cont + 1
                all_contractors.append(results)





        except requests.RequestException as e:
            print(f"Error crawling page {page}: {e}")
            continue

    # Convert to DataFrame and export to Excel
    df = pd.DataFrame(all_contractors)
    df.to_excel('contractors_data.xlsx', index=False)

    return all_contractors , df

# # Run the crawler
# contractors , df = crawl_contractors()
# print(f"Extracted {len(contractors)} contractor records")

    def _save_results(self):
        """Save scraped data to Excel."""
        df = pd.DataFrame(self.all_contractors)
        df.to_excel('contractors_data.xlsx', index=False)
        st.success(f"Extracted {len(self.all_contractors)} contractor records")
        return self.all_contractors, df

def create_documents(df):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000000)
    marked_text = []
    for index, row in df.iterrows():
        Membership_Number = row['Membership Number']
        Organization_Mobile_Number = row['Organization Mobile Number']
        Organization_Email = row['Organization Email']
        City = row['City']
        Interests = row['Interests']
        contractor_name = row['contractor_name']
        marked_text.append((f'the information is contractor_name: {contractor_name} ,  Membership_Number: {Membership_Number} , Organization_Mobile_Number : {Organization_Mobile_Number}  , Organization_Email : {Organization_Email} , City : {City}  now the Interests : {Interests}'))
    return splitter.create_documents(marked_text)

def create_embedding(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embeddings)

# Streamlit App
def main():
    st.title("Bab Chatbot")
    
    # Web Scraping Section
    if st.button("Scrape Contractor Data"):
        contractors, df = crawl_contractors(max_pages=1)
        st.session_state.df = df
    
    # Check if data has been scraped
    if 'df' not in st.session_state:
        st.warning("Please scrape contractor data first!")
        return
    
    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Create vector database
    documents = create_documents(st.session_state.df)
    arabic_VDB = create_embedding(documents)
    
    # Initialize LLM
    llm = ChatGroq(
        temperature=0.8, 
        groq_api_key="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",  # Use Streamlit secrets
        model_name="llama3-70b-8192"
    )
    
    # Prompt Template
    prompt_template = """
    <|SYSTEM|>
    You are an AI model have information about construction workers you should give the user what they need form the information you have.
    <|END_SYSTEM|>
    <|CONTEXT|>
    RAG context
    {context}
    HISTORY context
    {history}
    <|END_CONTEXT|>
    <|USER|>
    {query}
    <|END_USER|>
    <|ASSISTANT|>
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['query', "context", "history"]
    )
    
    # LLM Chain
    MODEL = LLMChain(llm=llm, prompt=prompt, verbose=False)
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask about construction workers"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare context for RAG
        results = arabic_VDB.similarity_search_with_score(prompt, k=1) # you can adjust it 
        context_text = "\n\n".join([doc.page_content for doc, score in results if score > 0.8])
        
        # Prepare history
        history = "\n".join([f"User: {msg['content']}" for msg in st.session_state.messages if msg["role"] == "user"])
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = MODEL.invoke({
                    "query": prompt, 
                    "context": context_text, 
                    "history": history
                })
                full_response = response.get("text", "I couldn't find a relevant response.")
                st.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()