"""RAG related data"""
import json
import os
from dataclasses import dataclass
from typing import List, Optional
from bs4 import BeautifulSoup
from markdown import markdown
import spacy
from typing import List, Dict, Any, Optional
# import config
from src.config import RAG_SETTINGS


# Constants for document chunking and retrieval
DEFAULT_CHUNK_SIZE = RAG_SETTINGS.DEFAULT_CHUNK_SIZE
DEFAULT_CHUNK_OVERLAP = RAG_SETTINGS.DEFAULT_CHUNK_OVERLAP
DEFAULT_TOP_K = RAG_SETTINGS.DEFAULT_TOP_K
DEFAULT_RERANK_TOP_K = RAG_SETTINGS.DEFAULT_RERANK_TOP_K



class DocumentChunk:
    """Represents a chunk of a document with metadata and embeddings"""
    def __init__(
        self,
        text: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        bm25_tokens: List[str] = None,
        node_id: Optional[str] = None
    ):
        self.text = text
        self.metadata = metadata
        self.embedding = embedding
        self.bm25_tokens = bm25_tokens or []
        self.node_id = node_id
    
    def to_dict(self):
        return {
            "text": self.text,
            "metadata": self.metadata,
            "node_id": self.node_id
        }



class MarkdownParser:
    """Parser for extracting text and structure from markdown files"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
    
    def parse_markdown(self, content: str, file_path: str) -> List[DocumentChunk]:
        """Parse markdown content into document chunks with metadata"""
        # Convert markdown to HTML for easier parsing
        html = markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract text
        text = soup.get_text()
        
        # Create document chunks
        chunks = []
        doc = self.nlp(text)
        
        # Extract headers and create a hierarchical structure
        headers = self._extract_headers(soup)
        
        # Get chunks with metadata about their location in the document structure
        text_chunks = self._create_chunks(doc, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
        
        for i, chunk_text in enumerate(text_chunks):
            # Find the header this chunk belongs to
            header_context = self._get_header_context(chunk_text, headers)
            
            chunks.append(DocumentChunk(
                text=chunk_text,
                metadata={
                    "source": file_path,
                    "chunk_id": i,
                    "header_context": header_context
                }
            ))
            
        return chunks
    
    def _extract_headers(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract headers with their hierarchy from the HTML soup"""
        headers = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(tag.name[1])
            headers.append({
                "text": tag.get_text(),
                "level": level,
                "position": len(headers)
            })
        return headers
    
    def _get_header_context(self, text: str, headers: List[Dict]) -> List[str]:
        """Get hierarchical header context for a text chunk"""
        # This is a simplified implementation
        # A real implementation would track chunk positions relative to headers
        context = []
        current_level = 6  # Start with the highest header level (h6)
        
        for header in reversed(headers):
            if header["level"] < current_level:
                context.append(header["text"])
                current_level = header["level"]
                
            # Stop at h1 to avoid going too high in the hierarchy
            if current_level == 1:
                break
                
        return list(reversed(context))
    
    def _create_chunks(self, doc, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create overlapping chunks from a spaCy document"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_tokens = len(sent)
            
            # If adding this sentence would exceed the chunk size,
            # save the current chunk and start a new one with overlap
            if current_length + sent_tokens > chunk_size and current_length > 0:
                chunks.append(" ".join(current_chunk))
                
                # Create overlap by keeping some sentences from the previous chunk
                overlap_size = 0
                overlap_chunk = []
                
                for s in reversed(current_chunk):
                    s_tokens = len(self.nlp(s))
                    if overlap_size + s_tokens <= chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += s_tokens
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_length = overlap_size
            
            current_chunk.append(sent_text)
            current_length += sent_tokens
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks



@dataclass
class Topic:
    """One of the possible Penetration Testing topics, used to choose
    a collection and to filter documents"""
    name: str

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Topic):
            return self.name == other.name
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


@dataclass
class Document:
    """Represents a processed data source such as HTML or PDF documents; it will
    be chunked and added to a Vector Database"""
    name: str
    content: str
    topic: Optional[Topic]

    def __str__(self):
        return f'{self.name} [{str(self.topic)}]\n{self.content}'


def chunk(document: Document) -> List[str]:
    """
    A basic chunking function for non-markdown documents.
    Currently, it returns the entire content as one chunk.
    Modify this function if you need more advanced chunking logic.
    """
    return [document.content]
    """One of the possible Penetration Testing topics, used to choose
    a collection and to filter documents"""
    name: str

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Topic):
            return self.name == other.name
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


@dataclass
class Document:
    """Represents a processed data source such as HTML or PDF documents; it will
    be chunked and added to a Vector Database"""
    name: str
    content: str
    topic: Optional[Topic]

    def __str__(self):
        return f'{self.name} [{str(self.topic)}]\n{self.content}'


@dataclass
class Collection:
    """Represents a Qdrant collection"""
    collection_id: int
    title: str
    documents: List[Document]
    topics: List[Topic]
    size: Optional[int] = 0  # points to the number of chunks in a Collection

    @staticmethod
    def from_json(path: str):
        """Create a collection from a JSON file in the following format:
        [
            {
                "title": "collection title",
                "content": "collection content",
                "category": "collection topics" | ["topic1", "topic2"]
            },
            ...,
        ]
        :param path: path to the json file.
        :return: Collection
        """
        # path string validation
        if not path or not isinstance(path, str):
            raise ValueError("Invalid parameter for path.")
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist.")
        if not os.path.isfile(path) or \
                not os.path.basename(path).endswith(".json"):
            raise ValueError(f"Path {path} is not JSON file")

        # load json file
        with open(path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)

        title = os.path.basename(path).split('.')[0]

        return Collection.from_dict(title, data)

    @staticmethod
    def from_dict(collection_title: str, data: list):
        # json scheme validation
        format_err_msg = f"Invalid format."
        if not isinstance(data, list):
            raise ValueError(format_err_msg + "Not a list.")

        valid_dict = [isinstance(item, dict) for item in data]
        if False in valid_dict:
            raise ValueError(format_err_msg + "Found not dict item.")

        all_keys = [list(item.keys()) for item in data]
        valid_keys = [
            'title' in keys and 'content' in keys and 'category' in keys
            for keys in all_keys
        ]
        if False in valid_keys:
            raise ValueError(format_err_msg + "Not found required keys.")

        documents: list[Document] = []
        all_topics: list[Topic] = []
        for item in data:
            title = item['title']
            content = item['content']
            category = item['category']
            if isinstance(category, list):
                topics = [Topic(topic) for topic in category]
                all_topics.extend([Topic(topic) for topic in category])
            else:
                topics = Topic(category)
                all_topics.append(Topic(category))

            documents.append(Document(
                name=title,
                content=content,
                topic=topics
            ))

        return Collection(
            collection_id=-1,
            title=collection_title,
            documents=documents,
            topics=all_topics,
            size=len(documents)
        )

    def to_json_metadata(self, path: str):
        """Saves the collection to the specified metadata file.
        ex. USER/.aiops/knowledge/collection_name.json
        {
            'id'
            'title'
            'documents': [
                {'name', 'topic'}
                ...
            ]
            'topics': [...]
        }"""
        print(f'[+] Saving {self.title} to {path}')
        print(self)
        collection_metadata = self.to_dict()

        with open(path, 'w+', encoding='utf-8') as fp:
            json.dump(collection_metadata, fp)

    def to_dict(self):
        """Convert collection to dictionary (strips out content)"""
        docs = []
        if len(self.documents) > 0:
            for document in self.documents:
                print(document.topic, type(document.topic))
                docs.append({
                    'name': document.name,
                    'content': '',  # document.content,
                    'topic': document.topic.name
                })

        collection_metadata = {
            'id': self.collection_id,
            'title': self.title,
            'documents': docs,
            'topics': list(set([topic.name for topic in self.topics]))
        }
        return collection_metadata

    def document_names(self) -> list:
        """The document names are used to filter queries to the
        Knowledge Database"""
        return [doc.name for doc in self.documents]

    def __str__(self):
        docs = "| - Documents\n"
        for doc in self.documents:
            docs += f'    | - {doc.name}\n'
        topics = ", ".join([topic.name for topic in set(self.topics)])
        return (f'Title: {self.title} \nID: {self.collection_id})\n'
                f'| - Topics: {topics}\n'
                f'{docs}')


if __name__ == "__main__":
    c = Collection.from_json('../../../data/json/owasp.json')
    print(c)
