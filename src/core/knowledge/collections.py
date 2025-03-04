"""RAG related data models with enhanced topic support"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set

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
    topics: List[Topic] = field(default_factory=list)
    source_type: str = "text"  # "markdown", "text", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        topics_str = ", ".join([str(topic) for topic in self.topics])
        return f'{self.name} [{topics_str}]\n{self.content}'
    
    @staticmethod
    def from_markdown(filename: str, content: str, topics: List[str]) -> 'Document':
        """Create document from markdown content with frontmatter parsing"""
        # Extract metadata from frontmatter if present
        metadata, cleaned_content = process_frontmatter(content)
        
        return Document(
            name=filename,
            content=cleaned_content,
            topics=[Topic(t) for t in topics],
            source_type="markdown",
            metadata=metadata
        )

def process_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """Extract YAML frontmatter from markdown content if present"""
    metadata = {}
    cleaned_content = content
    
    # Check if content has frontmatter (starts with ---)
    if content.startswith('---'):
        try:
            # Find the end of the frontmatter
            end_index = content.find('---', 3)
            if end_index != -1:
                frontmatter = content[3:end_index].strip()
                # Parse the frontmatter (simple key-value parsing)
                for line in frontmatter.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
                
                # Remove frontmatter from content
                cleaned_content = content[end_index + 3:].strip()
        except Exception:
            # If parsing fails, return the original content
            pass
            
    return metadata, cleaned_content

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

        documents: list[Document] = []
        all_topics: list[Topic] = []
        
        for item in data:
            # Validate minimal required fields
            if 'title' not in item or 'content' not in item:
                raise ValueError(format_err_msg + "Missing required fields.")
                
            title = item['title']
            content = item['content']
            
            # Handle both "category" and "topics" for backward compatibility
            topics_data = item.get('topics', item.get('category', []))
            
            # Extract metadata if present
            metadata = {k: v for k, v in item.items() 
                       if k not in ['title', 'content', 'topics', 'category']}
                       
            # Handle different topic formats
            if isinstance(topics_data, list):
                topics = [Topic(topic) for topic in topics_data]
                all_topics.extend(topics)
            else:
                topics = [Topic(topics_data)]
                all_topics.append(Topic(topics_data))

            documents.append(Document(
                name=title,
                content=content,
                topics=topics,
                metadata=metadata,
                source_type=item.get('source_type', 'text')
            ))

        return Collection(
            collection_id=-1,
            title=collection_title,
            documents=documents,
            topics=list(set(all_topics)),
            size=len(documents)
        )

    def to_json_metadata(self, path: str):
        """Saves the collection to the specified metadata file."""
        collection_metadata = self.to_dict()

        with open(path, 'w+', encoding='utf-8') as fp:
            json.dump(collection_metadata, fp, indent=2)

    def to_dict(self):
        """Convert collection to dictionary (strips out content)"""
        docs = []
        for document in self.documents:
            docs.append({
                'name': document.name,
                'content': '',  # Strip content to save space
                'topics': [topic.name for topic in document.topics],
                'source_type': document.source_type,
                'metadata': document.metadata
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
        
    def get_topics(self) -> Set[Topic]:
        """Returns all unique topics in this collection"""
        topics = set()
        for doc in self.documents:
            for topic in doc.topics:
                topics.add(topic)
        return topics
        
    def get_documents_by_topic(self, topic_name: str) -> List[Document]:
        """Returns all documents with the specified topic"""
        return [doc for doc in self.documents 
                if any(t.name.lower() == topic_name.lower() for t in doc.topics)]

    def __str__(self):
        docs = "| - Documents\n"
        for doc in self.documents:
            topic_names = ", ".join([t.name for t in doc.topics])
            docs += f'    | - {doc.name} [{topic_names}]\n'
        topics = ", ".join([topic.name for topic in set(self.topics)])
        return (f'Title: {self.title} \nID: {self.collection_id})\n'
                f'| - Topics: {topics}\n'
                f'{docs}')