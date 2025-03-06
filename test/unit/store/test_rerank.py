import logging
from src.core.knowledge.neural_reranker import NeuralReranker

# Configura el logging para ver mensajes informativos
logging.basicConfig(level=logging.INFO)

# Inicializa el reranker con Ollama
reranker = NeuralReranker(
    provider="ollama",
    model_name="qllama/bge-reranker-large",  # Modelo que descargaste
    endpoint="http://localhost:11434"  # Endpoint por defecto de Ollama
)

# Define una consulta y algunos resultados iniciales (simulados)
query = "Información sobre escaneo sigiloso con nmap"
initial_results = [
    {"text": "El escaneo sigiloso de Nmap usa paquetes SYN para evitar detección."},
    {"text": "SQL injection es un ataque que afecta bases de datos."},
    {"text": "Nmap puede realizar escaneos rápidos y silenciosos."}
]

# Reordena los resultados
reranked_results = reranker.rerank(query, initial_results, limit=3)

# Imprime los resultados reordenados
print("Resultados reordenados:")
for i, result in enumerate(reranked_results):
    print(f"{i+1}. {result['text'][:50]}... (score: {result['rerank_score']:.4f})")