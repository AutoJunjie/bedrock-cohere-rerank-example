import boto3
import json
region = "us-west-2"

bedrock_agent_runtime = boto3.client('bedrock-agent-runtime',region_name=region)

modelId = "cohere.rerank-v3-5:0"
model_package_arn = f"arn:aws:bedrock:{region}::foundation-model/{modelId}"

def rerank_text(text_query, text_sources, num_results, model_package_arn):
    response = bedrock_agent_runtime.rerank(
        queries=[
            {
                "type": "TEXT",
                "textQuery": {
                    "text": text_query
                }
            }
        ],
        sources=text_sources,
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "numberOfResults": num_results,
                "modelConfiguration": {
                    "modelArn": model_package_arn,
                }
            }
        }
    )
    return response['results']

example_query = "What emails have been about returning items?"

#在真实的RAG flow中，将从vector search中获取的documents整理成list传递给rerank_text函数

documents = [
    "Hola, llevo una hora intentando acceder a mi cuenta y sigue diciendo que mi contraseña es incorrecta. ¿Puede ayudarme, por favor?",
    "Hi, I recently purchased a product from your website but I never received a confirmation email. Can you please look into this for me?",
    "مرحبًا، لدي سؤال حول سياسة إرجاع هذا المنتج. لقد اشتريته قبل بضعة أسابيع وهو معيب",
    "Good morning, I have been trying to reach your customer support team for the past week but I keep getting a busy signal. Can you please help me?",
    "Hallo, ich habe eine Frage zu meiner letzten Bestellung. Ich habe den falschen Artikel erhalten und muss ihn zurückschicken.",
    "Hello, I have been trying to reach your customer support team for the past hour but I keep getting a busy signal. Can you please help me?",
    "Hi, I have a question about the return policy for this product. I purchased it a few weeks ago and it is defective.",
    "早上好，关于我最近的订单，我有一个问题。我收到了错误的商品",
    "Hello, I have a question about the return policy for this product. I purchased it a few weeks ago and it is defective."
]

text_sources = []
for text in documents:
    text_sources.append({
        "type": "INLINE",
        "inlineDocumentSource": {
            "type": "TEXT",
            "textDocument": {
                "text": text,
            }
        }
    })

response = rerank_text(example_query, text_sources, 3, model_package_arn)
print(response)