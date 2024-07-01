# PDF_Chatbot_without_external_APIs
PDF Chatbot on Local computer without any External Dependencies using Streamlit , GPT2 Model , Faiss, Sentence Transformers, HDF5 file format, pdfminer


Description : 

Architecture Overview:

The Architecture of the application can be broken down into several stages, each involving specific functions and models. The primary goal is to build a chatbot that can answer questions based on the content of multiple PDF documents using embeddings and a language model.

Components:


PDF Processing:

Extract Text from PDFs:
                        This involves reading specified PDF files and extracting their text content using 'pdfminer'. The extracted text is stored in a dictionary for further processing.
                        
Generate Embeddings:
                     Using a 'SentenceTransformer' model, the extracted text from each PDF is converted into embeddings. These embeddings are dense vector representations that capture the semantic meaning and context of the text.

Store Embeddings in Faiss Index:
                                  The embeddings are added to a 'Faiss' index, which is a library optimized for efficient similarity search and clustering of dense vectors. Metadata (such as document names) is also stored alongside the embeddings.
 
Save Index and Metadata: 
                        The Faiss index and metadata are serialized and saved to an 'HDF5' file for persistent storage.


    
Chatbot Interface:

Load Faiss Index and Metadata: 
                               The Faiss index and metadata are deserialized from the HDF5 file, making them available for query processing.
   
Query Processing: 
                   When a user submits a query, it is encoded using the SentenceTransformer model. The encoded query is then used to search the Faiss index for similar document embeddings. The most relevant documents are retrieved and their contexts are provided to the 'GPT2' Large language model.
   
Streamlit UI: 
              The user interface is built using 'Streamlit' , which handles user interactions and displays chat history. Users can submit queries, and the chatbot's responses are displayed in the UI.
