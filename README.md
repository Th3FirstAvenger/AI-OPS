![pylint](https://img.shields.io/badge/PyLint-8.87-yellow?logo=python&logoColor=white)

# AI-OPS

### Table of Contents
1. [Overview](#-overview)
2. [Key Features](#key-features)
3. [Installation](#-install)
   - [End-User](#end-user)
   - [Development](#development)
4. [System Structure](#usage)
   - [Components](#components)
   - [Supported Models](#supported-ollama-models)
5. [Tools](#tools)
   - [Available Tools](#available-tools)
   - [Add a Tool](#add-a-tool)
6. [Knowledge](#-knowledge-)
   - [Available Collections](#available-collections)
   - [Add a Collection](#add-a-collection)
7. [Ethical and Legal Considerations](#-ethical-and-legal-considerations)

## 💡 Overview

**AI-OPS** is an AI-powered, Open-source Penetration testing Suite that leverages Large Language Models (LLMs) AI-Agent to assist and automate tasks such as reconnaissance, exploitation, and report generation. 

> **Note:** AI-OPS is designed to enhance, not replace, human penetration testers, similar to how AI enhances great programmers by making them more productive.

## Key Features

- 🎁 **Full Open-Source**: No need for third-party LLM providers; use any model you prefer with [Ollama](https://github.com/ollama/ollama).
- 🔧 **Tool Integration**: Execute common penetration testing tools or integrate new ones without needing to code in Python.
- 📚 **Up-to-date Knowledge**: Use the `RAG` system to keep the agent informed with the latest documents and data.
- ⚙️ **Scalability**: Independently deployable components allow you to utilize any hardware setup.

## 💻 Install
**Requirements**
- Ollama (see [Ollama](https://github.com/ollama/ollama))
- Docker (see [Docker Desktop](https://docs.docker.com/desktop/)) (*in development*)

### End-User
1. **Setup**
-  Clone Repository `git clone https://github.com/antoninoLorenzo/AI-OPS.git`

2. **Ollama**
- Launch Ollama **Locally**
  ```
  ./scripts/ollama_serve.* -i OLLAMA_HOST -o OLLAMA_ORIGINS
  ```
  *Note: there is ollama_serve.sh for Linux and ollama_serve.bat for Windows*
- As an alternative see my solution: [Ollama on Colab](https://github.com/antoninoLorenzo/Ollama-on-Colab-with-ngrok)

3. **Agent API**
- Build Docker Image
  ```
  docker build -t ai-ops:api-dev --build-arg ollama_endpoint=ENDPOINT ollama_model=MODEL .
  ```
- Run Docker Container
  ```
  docker run -p 8000:8000 ai-ops:api-dev
  ```

4. **CLI Client**
- Run Client
  ```
  python ai-ops-cli.py --api AGENT_API_ADDRESS
  ```
  
### Development

1. **Setup**
- Clone Repository `git clone https://github.com/antoninoLorenzo/AI-OPS.git`
- Install Python requirements `pip install -r requirements.txt`
- Install spacy model `python -m spacy download en_core_web_lg`


2. **Ollama**
- Set remote origins environment variable:  `OLLAMA_ORIGINS=1.2.3.4,...` *(Optional)*
- Set host environment variable: `OLLAMA_HOST=0.0.0.0:11434` *(Optional)*
- Run ollama: `ollama serve`


3. **Agent API**
- Launch Agent API (*in development*): `fastapi.exe dev ./src/api.py`
  
  -  Access from other machines: `fastapi.exe dev --host 0.0.0.0 ./src/api.py`
  -  Additional Settings in `.env` file:
  ```
  MODEL=model_name
  ENDPOINT=ollama_url
  ```
  *Note: the tools that require root would require also the API to be runned as root*

4. **Setup Tools** (*first time only*)
- move the content of `tools_settings` to `<user home>/.aiops/tools`.

5. **CLI Client**
- Run Client
  ```
  python ai-ops-cli.py --api AGENT_API_ADDRESS
  ```

## 📝Usage

### Components
![Deployment Diagram](static/images/deployment_diagram.svg)

| Component                                  | Description                                                             |
|--------------------------------------------|-------------------------------------------------------------------------|
| Frontend                                   | Web interface for the AI Agent built in `React`                         |
| AI Agent                                   | The implementation of the AI Agent exposed to `Frontend` with `FastAPI` |
| [Qdrant](https://github.com/qdrant/qdrant) | Vector Database                                                         |
| [Ollama](https://github.com/ollama/ollama) | LLM Provider                                                            | 


### Supported Ollama Models
| Name         | Implemented (prompts) |
|--------------|----------------------|
| **Gemma 7B** | &check;              |
| **Gemma2 9B**| &check;              |  
| **Mistral**  | &check;              |

<!--| **LLama 3**  | &cross;               | -->



## 🛠️Tools

### Available Tools

| Name        | Use Case                         | Implemented         |
|-------------|----------------------------------|---------------------|
| nmap        | Scanning/Network Exploitation    | &check;             |
| hashcat     | Password Cracking                | &check;             |
| SQLmap      | SQL Injection                    | &check;             | 
| gobuster    | Enumeration                      | &check;             |
| searchsploit| Research Vulnerabilities         | &check;             |
| Metasploit  | Exploitation                     | &cross;             |

*Note: virtually any tools that do not require additional code (such as Metasploit) can be executed*


### Add a Tool

Penetration Testing tools can be integrated using either JSON instructions or custom classes.

1. **JSON Instructions**: create a file with instructions for the Agent and add it to 
`/home/YOUR_USERNAME/.aiops/tools` (or `../Users/YOUR_USERNAME/.aiops/tools`); all available tools 
that use JSON Instructions are available in `tools_settings`.
```json
{
    "name": "...",
    "tool_description": "...",
    "args_description": [
        "Multiline JSON\n",
        "instructions\n",
        "..."
    ],
    "examples": [
        "<THOUGHT>reason to execute (thought process) </THOUGHT>\n",
        "<TOOL>command (ex. nmap -sV 127.0.0.1)</TOOL>\n"
    ]
}
```

2. **Custom Class**: tools that require more advanced usage can be implemented extending the class
`Tool` at `src.agent.tools.base`; you're welcome to **open an issue** for a tool request/proposal.


## 📚 Knowledge 

**TODO**

### Available Collections

**TODO**


### Add a Collection

**TODO**

## ⚖️ Ethical and Legal Considerations

**AI-OPS** is designed as a penetration testing tool intended for academic and educational purposes only. Its primary goal is to assist cybersecurity professionals and enthusiasts in enhancing their understanding and skills in penetration testing through the use of AI-driven automation and tools.

### Responsible Use

- **Legal Compliance**: Ensure that you comply with all relevant laws and regulations when using this tool. Unauthorized access to computer systems is illegal and unethical.
- **Permission**: Always obtain explicit permission from the system owner before performing any penetration testing. Unauthorized testing is illegal and can cause significant harm.
- **Academic Integrity**: Use this tool to support your learning and research in ethical hacking and cybersecurity. Do not use it for malicious purposes.

### Disclaimer

The creators and contributors of **AI-OPS** are not responsible for any misuse of this tool. By using **AI-OPS**, you agree to take full responsibility for your actions and to use the tool in a manner that is ethical, legal, and in accordance with the intended purpose.

> **Note**: This project is provided "as-is" without any warranties, express or implied. The creators are not liable for any damages or legal repercussions resulting from the use of this tool.

