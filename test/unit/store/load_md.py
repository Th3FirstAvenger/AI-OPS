import json
import os

def extract_markdown_info(md_path):
    """
    Extracts topics, title and content from a Markdown file.
    
    Args:
        md_path (str): Path to the Markdown file.
    
    Returns:
        dict: A dictionary with 'name', 'content', 'topics', 'source_type' and 'metadata'.
    """
    with open(md_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    topics = []
    title = ""
    content = ""
    for i, line in enumerate(lines):
        if line.startswith("::Topics::"):
            topics = line.split("::Topics::")[1].strip().split(", ")
        elif line.startswith("# "):
            title = line.strip("# ").strip()
            content = "".join(lines[i:])  # Everything from title onwards
            break
    
    if not title or not content:
        raise ValueError(f"No title or content found in {md_path}")
    
    return {
        "name": title,
        "content": content,
        "topics": topics,
        "source_type": "markdown",
        "metadata": {"source_type": "markdown"}
    }

def update_json_with_documents(json_path, md_files):
    """
    Updates the documentation JSON with new documents from Markdown files.
    
    Args:
        json_path (str): Path to the documentation JSON file.
        md_files (list): List of paths to Markdown files.
    """
    # Load existing JSON
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # Extract information from each Markdown and add it to the documents list
    for md_file in md_files:
        new_document = extract_markdown_info(md_file)
        data["documents"].append(new_document)
        
        # Update the general topics list
        current_topics = set(data["topics"])
        current_topics.update(new_document["topics"])
        data["topics"] = list(current_topics)
    
    # Save updated JSON
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

if __name__ == "__main__":
    # Usage example for testing get knowledge base 
    base_path = os.path.join(os.path.dirname(__file__), "../../data/")
    json_path = os.path.join(base_path, "json", "active_directory_exploitation_testing.json")
    md_files = [
        os.path.join(base_path, "md", "SUDO Misconfiguration Exploitation.md"),
        os.path.join(base_path, "md", "linux_kernel.md")
    ]  # Add your files here    

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file {json_path} does not exist.")
    
    # Ensure Markdown files exist
    for md_file in md_files:
        if not os.path.exists(md_file):
            raise FileNotFoundError(f"Markdown file {md_file} does not exist.")
    
    # Update JSON with Markdown documents
    update_json_with_documents(json_path, md_files)
    print("JSON successfully updated.")