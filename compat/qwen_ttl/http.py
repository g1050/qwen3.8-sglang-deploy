"""HTTP-process side: request policy, TTL transport, and Kimi-shaped usage fields.

The engine reports the billing view only; this layer adds the write fields and headers.
"""
from contextvars import ContextVar
import json
import os
import time

from qwen_ttl.core import DEFAULT_TTL, PolicyError, encode_carrier, parse_options

APIS = {'/v1/chat/completions': 'chat', '/v1/completions': 'chat', '/v1/responses': 'responses',
        '/v1/messages': 'messages'}
CONTEXT = ContextVar('qwen_ttl_request', default=None)
# Acceptance only: lets a test supply the arrival time instead of waiting an hour.
TEST_CLOCK = os.environ.get('QWEN_TTL_TEST_CLOCK') == '1'


def capture(output, context):
    """Move the scheduler's bill out of meta_info so no physical detail reaches the client."""
    for item in output if isinstance(output, list) else (output,):
        meta = item.get('meta_info') if isinstance(item, dict) else None
        if not meta:
            continue
        details = meta.pop('cached_tokens_details', None)
        if details and 'ttl' in details:
            context['bills'][meta.get('id')] = details['ttl']


def total_bill(context):
    bills = list(context['bills'].values())
    if not bills:
        return None
    # n > 1 choices share one prompt; the first choice carries its bill.
    return bills[0]


def patch_usage(usage, api, bill):
    read, write_5m, write_1h, prompt = bill
    write = write_5m + write_1h
    if api == 'messages':
        usage.update(input_tokens=prompt - read - write, cache_read_input_tokens=read,
                     cache_creation_input_tokens=write,
                     cache_creation={'ephemeral_5m_input_tokens': write_5m, 'ephemeral_1h_input_tokens': write_1h})
        return
    name = 'input_tokens_details' if api == 'responses' else 'prompt_tokens_details'
    details = dict(usage.get(name) or {})
    details.update(cached_tokens=read, cache_write_tokens=write)
    usage[name] = details
    if api == 'chat':
        # Kimi omits the read fields on a miss and repeats them at the top level on a hit.
        usage.pop('cached_tokens', None)
        if read:
            usage['cached_tokens'] = read
        else:
            details.pop('cached_tokens')


def patch_document(document, api, bill):
    """Rewrite every usage object of one JSON response or SSE event; True if any was found."""
    found = False
    if isinstance(document, dict):
        for holder in (document, document.get('response'), document.get('message')):
            if isinstance(holder, dict) and isinstance(holder.get('usage'), dict):
                patch_usage(holder['usage'], api, bill)
                found = True
    return found


def patch_sse(chunk, api, bill):
    lines = chunk.split(b'\n')
    for index, line in enumerate(lines):
        if line.startswith(b'data:') and b'"usage"' in line:
            try:
                document = json.loads(line[5:])
            except ValueError:
                continue
            if patch_document(document, api, bill):
                lines[index] = b'data: ' + json.dumps(document, ensure_ascii=False, separators=(',', ':')).encode()
    return b'\n'.join(lines)


def error_body(api, message):
    if api == 'messages':
        return {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': message}}
    return {'error': {'message': message, 'type': 'invalid_request_error', 'code': 400}}


class TTLMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        api = APIS.get(scope.get('path')) if scope['type'] == 'http' and scope.get('method') == 'POST' else None
        if api is None:
            return await self.app(scope, receive, send)
        chunks = []
        while True:
            message = await receive()
            if message['type'] != 'http.request':
                return
            chunks.append(message.get('body', b''))
            if not message.get('more_body'):
                break
        raw = b''.join(chunks)
        ttl = DEFAULT_TTL
        try:
            body = json.loads(raw)
            if isinstance(body, dict):
                before = len(body), 'cache_control' in raw.decode('utf-8', 'ignore')
                ttl = parse_options(body, api)
                if before != (len(body), False):
                    raw = json.dumps(body, ensure_ascii=False).encode()
        except PolicyError as error:
            payload = json.dumps(error_body(api, str(error))).encode()
            await send({'type': 'http.response.start', 'status': 400,
                        'headers': [(b'content-type', b'application/json'),
                                    (b'content-length', str(len(payload)).encode())]})
            return await send({'type': 'http.response.body', 'body': payload})
        except ValueError:
            pass   # the engine reports malformed JSON itself
        context = {'ttl': ttl, 'bills': {}}
        if TEST_CLOCK:
            stamp = next((value for key, value in scope['headers'] if key == b'x-qwen-ttl-now'), None)
            if stamp is not None:
                context['now'] = float(stamp)
        token = CONTEXT.set(context)
        sent = False

        async def replay():
            nonlocal sent
            if sent:
                return await receive()
            sent = True
            return {'type': 'http.request', 'body': raw, 'more_body': False}

        length = next((value for key, value in scope['headers'] if key == b'content-length'), None)
        if length is not None and int(length) != len(raw):
            scope = {**scope, 'headers': [(key, str(len(raw)).encode() if key == b'content-length' else value)
                                          for key, value in scope['headers']]}
        start, streaming, buffered = None, False, []

        def with_bill_headers(message, bill, size=None):
            headers = [(k, v) for k, v in message['headers'] if size is None or k != b'content-length']
            if size is not None:
                headers.append((b'content-length', str(size).encode()))
            if bill:
                headers += [(b'msh-usage-cache-write-tokens-5m', str(bill[1]).encode()),
                            (b'msh-usage-cache-write-tokens-1h', str(bill[2]).encode())]
            return {**message, 'headers': headers}

        async def respond(message):
            nonlocal start, streaming
            if message['type'] == 'http.response.start':
                start = message
                streaming = any(k == b'content-type' and b'text/event-stream' in v for k, v in message['headers'])
                return
            if message['type'] != 'http.response.body':
                return await send(message)
            bill = total_bill(context)
            body = message.get('body', b'')
            if streaming:
                if start is not None:
                    # The first chunk follows admission, so the bill headers are already known.
                    await send(with_bill_headers(start, bill))
                    start = None
                if bill and b'"usage"' in body:
                    body = patch_sse(body, api, bill)
                return await send({**message, 'body': body})
            buffered.append(body)
            if message.get('more_body'):
                return
            body = b''.join(buffered)
            if bill and start['status'] == 200:
                try:
                    document = json.loads(body)
                    if patch_document(document, api, bill):
                        body = json.dumps(document, ensure_ascii=False).encode()
                except ValueError:
                    pass
            await send(with_bill_headers(start, bill if start['status'] == 200 else None, len(body)))
            await send({'type': 'http.response.body', 'body': body, 'more_body': False})

        try:
            await self.app(scope, replay, respond)
        finally:
            CONTEXT.reset(token)


def install(app):
    """Stamp every generate request with its TTL and arrival time; collect its bill."""
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    if getattr(TokenizerManager, '_qwen_ttl_hook', False):
        return
    original = TokenizerManager.generate_request

    async def generate_request(self, obj, request=None):
        context = CONTEXT.get()
        if isinstance(obj, GenerateReqInput):
            ttl = context['ttl'] if context else DEFAULT_TTL
            stamp = context.get('now') or time.time() if context else time.time()
            salt = obj.cache_salt
            obj.cache_salt = ([encode_carrier(ttl, stamp, item) for item in salt] if isinstance(salt, list)
                              else encode_carrier(ttl, stamp, salt))
        async for output in original(self, obj, request):
            if context is not None:
                capture(output, context)
            yield output

    TokenizerManager.generate_request = generate_request
    TokenizerManager._qwen_ttl_hook = True
    app.add_middleware(TTLMiddleware)
