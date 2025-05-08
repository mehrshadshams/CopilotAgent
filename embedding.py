import math
import os
import aiohttp
import json
import traceback
from typing import List, Dict, Any

class EmbeddingService:
    model_gpt35 = "gpt-3.5-turbo"
    model_gpt4 = "gpt-4"
    model_embeddings = "text-embedding-ada-002"

    @staticmethod
    async def create_embedding(content: str, integration_id: str, api_token: str) -> List[float]:
        try:
            url = "https://api.githubcopilot.com/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_token}",
            }
            if integration_id:
                headers["Copilot-Integration-Id"] = integration_id

            payload = {
                "model": EmbeddingService.model_embeddings,
                "input": [content]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_message = await response.text()
                        raise RuntimeError(f"Unexpected status code: {response.status}, {error_message}")

                    response_data = await response.json()
                    if "data" in response_data and response_data["data"]:
                        return response_data["data"][0]["embedding"]

                    raise RuntimeError("No embeddings found in response")
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Unexpected error: {e}")

    @staticmethod
    async def generate_datasets(integration_id: str, api_token: str, filenames: List[str]) -> List[Dict[str, Any]]:
        datasets = []
        for filename in filenames:
            try:
                with open(filename, "r") as file:
                    file_content = file.read()

                embedding = await EmbeddingService.create_embedding(file_content, integration_id, api_token)
                datasets.append({
                    "embedding": embedding,
                    "filename": filename
                })
            except Exception as e:
                raise RuntimeError(f"Error processing file {filename}: {e}")

        return datasets

    @staticmethod
    def find_best_dataset(datasets: List[Dict[str, Any]], target_embedding: List[float]) -> Dict[str, Any]:
        best_dataset = None
        best_score = -1

        target_magnitude = math.sqrt(sum(x * x for x in target_embedding))

        for dataset in datasets:
            dataset_embedding = dataset["embedding"]
            dot_product = sum(t * d for t, d in zip(target_embedding, dataset_embedding))
            dataset_magnitude = math.sqrt(sum(d * d for d in dataset_embedding))

            if dataset_magnitude == 0 or target_magnitude == 0:
                continue

            similarity = dot_product / (target_magnitude * dataset_magnitude)
            if similarity > best_score:
                best_score = similarity
                best_dataset = dataset

        return best_dataset

embedding_service = EmbeddingService()