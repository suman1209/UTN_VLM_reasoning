import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

class ClaudeModel():
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)

    def batch_process(self, messages):
        requests = []
        for i, message in enumerate(messages):
            requests.append(Request(
                custom_id=f"my-request-{i}",
                params=MessageCreateParamsNonStreaming(
                    model=self.model_name,
                    max_tokens=1024,
                    messages=[message]
                )
            ))
        message_batch = self.client.messages.batches.create(
            requests=requests
        )

        print(message_batch)
        self.batch_id = message_batch.id
        print(self.batch_id)

    def get_batch_status(self, batch_id=None):
        if batch_id is not None:
            self.batch_id = batch_id
        message_batch = self.client.messages.batches.retrieve(
            self.batch_id,
        )
        print(f"Batch {message_batch.id} processing status is {message_batch.processing_status}")
    
    def get_results(self, batch_id=None):
        if batch_id is not None:
            self.batch_id = batch_id
        reuslts = []
        for result in self.client.messages.batches.results(
            self.batch_id,
        ):
            match result.result.type:
                case "succeeded":
                    print(f"Success! {result.custom_id}")
                    print(result.result.message.content[0].text)
                    reuslts.append(result.result.message.content[0].text)
                case "errored":
                    if result.result.error.type == "invalid_request":
                        # Request body must be fixed before re-sending request
                        print(f"Validation error {result.custom_id}")
                    else:
                        # Request can be retried directly
                        print(f"Server error {result.custom_id}")
                case "expired":
                    print(f"Request expired {result.custom_id}")

        return reuslts