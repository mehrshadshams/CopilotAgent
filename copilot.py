import aiohttp
import json

class CopilotService:
    @staticmethod
    async def chat_completions(req, integration_id, api_key):
        try:
            body = json.dumps(req)
        except Exception as e:
            raise RuntimeError(f"Failed to marshal request: {e}")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if integration_id:
            headers["Copilot-Integration-Id"] = integration_id

        url = "https://api.githubcopilot.com/chat/completions"

        # print(f"Request body: {body}")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, data=body, headers=headers) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        print(error_body)
                        raise RuntimeError(f"Unexpected status code: {response.status}")

                    try:
                        chat_res = await response.json()
                    except Exception as e:
                        raise RuntimeError(f"Failed to unmarshal response body: {e}")

                    return chat_res

            except Exception as e:
                print(f"Error during chat_completion request: {e}")
                raise RuntimeError(f"Failed to send request: {e}")

    @staticmethod
    def get_function_call(res):
        if not res.get("choices"):
            return None

        if not res["choices"][0]["message"].get("tool_calls"):
            return None

        func_call = res["choices"][0]["message"]["tool_calls"][0].get("function")
        if not func_call:
            return None

        return func_call