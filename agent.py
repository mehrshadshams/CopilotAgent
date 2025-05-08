import os
import asyncio
import aiohttp

import json
from typing import Any
from datetime import datetime
from sse_starlette.sse import EventSourceResponse
from starlette.responses import JSONResponse, StreamingResponse
from embedding import EmbeddingService
from copilot import CopilotService as Copilot
from collections import OrderedDict

async def create_embedding(content, integration_id, api_token):
    return await EmbeddingService.create_embedding(content, integration_id, api_token)

async def generate_datasets(integration_id, api_token, filenames):
    return await EmbeddingService.generate_datasets(integration_id, api_token, filenames)

async def find_best_dataset(datasets, target_embedding):
    return EmbeddingService.find_best_dataset(datasets, target_embedding)


DONE = b"data: [DONE]\n\n"

class AgentService:
    def __init__(self):
        self.datasets = []
        self.datasets_initialized = False

        list_properties = OrderedDict()
        list_properties["repository_owner"] = {
            "type": "string",
            "description": "The owner of the repository",
        }
        list_properties["repository_name"] = {
            "type": "string",
            "description": "The type of the repository",
        }

        create_properties = OrderedDict()
        create_properties["repository_owner"] = {
            "type": "string",
            "description": "The owner of the repository",
        }
        create_properties["repository_name"] = {
            "type": "string",
            "description": "The name of the repository",
        }
        create_properties["issue_title"] = {
            "type": "string",
            "description": "The title of the issue being created",
        }
        create_properties["issue_body"] = {
            "type": "string",
            "description": "The content of the issue being created",
        }

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_issues",
                    "description": "Fetch a list of issues from github.com for a given repository. Users may specify the repository owner and the repository name separately, or they may specify it in the form {repository_owner}/{repository_name}, or in the form github.com/{repository_owner}/{repository_name}.",
                    "parameters": {
                        "type": "object",
                        "properties": list_properties,
                        "required": ["repository_owner", "repository_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_issue_dialog",
                    "description": "Creates a confirmation dialog in which the user can interact with in order to create an issue on a github.com repository. Only one dialog should be created for each issue/repository combination. Users may specify the repository owner and the repository name separately, or they may specify it in the form {repository_owner}/{repository_name}, or in the form github.com/{repository_owner}/{repository_name}.",
                    "parameters": {
                        "type": "object",
                        "properties": create_properties,
                        "required": ["repository_owner", "repository_name", "issue_title", "issue_body"],
                    },
                },
            },
        ]

    async def initialize_datasets(self, integration_id, api_token):
        if not self.datasets_initialized:
            try:
                data_dir = "data"
                filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
                self.datasets = await generate_datasets(integration_id, api_token, filenames)
                self.datasets_initialized = True
                print(f"Initialized datasets: {len(self.datasets)}")
            except Exception as e:
                raise RuntimeError(f"Error initializing datasets: {e}")

    async def generate_completion(self, request):
        body = await request.json()
        user_message = body.get("messages", [])

        if not isinstance(user_message, list):
            print("Invalid messages format: Expected a list of objects.")
            return JSONResponse({"error": "Invalid messages format: Expected a list of objects."}, status_code=400)

        integration_id = request.headers.get("Copilot-Integration-Id")
        api_token = request.headers.get("X-GitHub-Token")

        if not user_message:
            print("No message provided")
            return JSONResponse({"error": "No message provided"}, status_code=400)

        try:
            await self.initialize_datasets(integration_id, api_token)

            # print(f"User message: {user_message}")

            response_messages = []
            for message in reversed(user_message):
                if message.get("role") != "user":
                    continue

                if not message.get("content"):
                    continue

                message_content = message["content"]
                # print(f"Processing message: {message_content}")

                # Generate embedding for the user message
                embedding = await create_embedding(message_content, integration_id, api_token)
                # print(f"Generated embedding: {embedding}")
                best_dataset = await find_best_dataset(self.datasets, embedding)

                if not best_dataset:
                    return JSONResponse({"reply": "No suitable dataset found."})

                print(f"Best dataset found: {best_dataset['filename']}")

                with open(best_dataset['filename'], "r") as file:
                    context = file.read()

                response_messages.append({
                    "role": "system",
                    "content": "You are a helpful assistant that replies to user messages.  Use the following context when responding to a message. Ensure to give examples as markdown blocks. Don't use numbered list instead use bullet points.\n" +
                        "Context: " + context,
                })

                break

            response_messages.extend(user_message)
            chat_request = {
                "model": EmbeddingService.model_gpt35,
                "messages": response_messages,
                "stream": True,
            }

            return StreamingResponse(self.stream_chat_completions(integration_id, api_token, chat_request), media_type="text/plain")

        except Exception as e:
            print(f"Error processing request: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def function_calling(self, request):
        print("Function calling started")
        body = await request.json()
        integration_id = request.headers.get("Copilot-Integration-Id")
        api_token = request.headers.get("X-GitHub-Token")
        user_message = body.get("messages", [])

        if not isinstance(user_message, list):
            print("Invalid messages format: Expected a list of objects.")
            return JSONResponse({"error": "Invalid messages format: Expected a list of objects."}, status_code=400)

        print(f"User message: {body}")

        if not user_message:
            print("No message provided")
            return

        return EventSourceResponse(self._function_calling_internal(user_message, integration_id, api_token), media_type="text/plain")

    async def _function_calling_internal(self, user_message, integration_id, api_token):

        for conf in user_message[-1].get('confirmations', []):
            if conf['state'] != "accepted":
                continue

            try:
                print("Creating issue...")
                # await create_issue(ctx, api_token, conf['confirmation']['owner'], conf['confirmation']['repo'], conf['confirmation']['title'], conf['confirmation']['body'])
                await asyncio.sleep(1)  # Simulate issue creation
            except Exception as e:
                raise RuntimeError(f"Error creating issue: {e}")

            msg_json = json.dumps({
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": f"Created issue {conf['confirmation']['title']} on repository {conf['confirmation']['owner']}/{conf['confirmation']['repo']}",
                        },
                    },
                ],
            })
            yield dict(data=msg_json)

            return

        messages = user_message[:]
        confs = []

        for i in range(5):
            print(f"Function calling iteration {i}")
            chat_req = {
                "model": EmbeddingService.model_gpt35,
                "messages": messages,
            }

            if i < 4:
                chat_req["tools"] = self.tools

            try:
                # print(f"Chat request: {chat_req}")
                res = await Copilot.chat_completions(chat_req, integration_id, api_token)
            except Exception as e:
                raise RuntimeError(f"Failed to get chat completions stream: {e}")

            function = Copilot.get_function_call(res)

            if not function:
                print("No function call found, sending the choice")
                choices = [
                    {
                        "index": choice["index"],
                        "delta": {
                            "role": choice["message"]["role"],
                            "content": choice["message"]["content"],
                        },
                    }
                    for choice in res["choices"]
                ]

                msg_json = json.dumps({"choices": choices})
                yield dict(data=msg_json)
                yield dict(data="[DONE]")
                break

            print("Found function!", function["name"])

            if function["name"] == "list_issues":
                args = json.loads(function["arguments"])
                try:
                    msg = {
                        "role": "assistant",
                        "content": "Listing issues for a/b",
                    }
                    messages.append(msg)
                except Exception as e:
                    raise RuntimeError(f"Error in list_issues: {e}")

            elif function["name"] == "create_issue_dialog":
                args = json.loads(function["arguments"])
                try:
                    conf = {
                        "confirmation": {
                            "type": "action",
                            "title": "Create Issue",
                            "message": f"Are you sure you want to create an issue in repository {args['repository_owner']}/{args['repository_name']} with the title \"{args['issue_title']}\" and the content \"{args['issue_body']}\"",
                            "confirmation": {
                                "owner": args["repository_owner"],
                                "repo": args["repository_name"],
                                "title": args["issue_title"],
                                "body": args["issue_body"],
                            },
                        },
                    }

                    msg = {
                        "role": "system",
                        "content": "Issue dialog created: abc",
                    }

                    if not confs:
                        confs.append(conf["confirmation"])
                        print(f"Creating issue dialog: {json.dumps(confs[-1])}")

                        try:
                            yield dict(event="copilot_confirmation", data=json.dumps(confs[-1]))
                        except Exception as e:
                            raise RuntimeError(f"Failed to write event or data: {e}")

                        messages.append(msg)
                except Exception as e:
                    raise RuntimeError(f"Error in create_issue_dialog: {e}")

            else:
                raise RuntimeError(f"Unknown function call: {function['name']}")

    async def stream_chat_completions(self, integration_id: str, api_token: str, chat_req: dict):
        url = "https://api.githubcopilot.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_token}",
        }
        if integration_id:
            headers["Copilot-Integration-Id"] = integration_id

        # Uncomment the following lines to send the actual request to the API

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=chat_req) as response:
                if response.status != 200:
                    error_message = await response.text()
                    raise RuntimeError(f"Unexpected status code: {response.status}, {error_message}")

                async for line in response.content:
                    try:
                        yield line
                        yield b"\n"
                    except Exception as e:
                        raise RuntimeError(f"Failed to write to stream: {e}")



agent_service = AgentService()