import fcntl
import json
import logging
import os
import socket
import threading
import uuid
from datetime import datetime, timezone
from model_files import inspect_models
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROLE = os.environ.get('ROLE', 'coordinator')
PORT = int(os.environ.get('PORT', '7860' if ROLE == 'coordinator' else '8188'))
BASE = os.environ.get('PATH_BASE', '/').rstrip('/')
OUTPUT = Path(os.environ.get('OUTPUT_DIR', '/data/output'))
WORKER_COUNT = int(os.environ.get('WORKER_COUNT', '1'))
WORKER_URL = os.environ.get('WORKER_URL_TEMPLATE', 'http://worker:8188')
MODE = os.environ.get('MODE', 'standalone')
INSTANCE = os.environ.get('PLATFORM_INSTANCE_ID', '')
QUEUE = OUTPUT / '.queue' / (INSTANCE if MODE == 'platform' else 'standalone')
LOCK = threading.Lock()
WAKE = threading.Event()
LOG = logging.getLogger(__name__)
PIPELINE = None
MODELS = {'state': 'waiting', 'missing': [], 'error': None, 'loadCount': 0}
WORKER_ID = f'{socket.gethostname()}:{PORT}'


def recover_interrupted_jobs():
    for accepted in OUTPUT.glob('*/accepted'):
        try:
            marker = json.loads(accepted.read_text(encoding='utf-8'))
            context = json.loads(accepted.with_name('context.json').read_text(encoding='utf-8'))
            job = str(uuid.UUID(accepted.parent.name))
            if marker.get('worker') != WORKER_ID or context.get('instance') != INSTANCE:
                continue
            if json.loads(state_path(job).read_text(encoding='utf-8'))['status'] == 'running':
                save_state(job, 'failed', 'Worker restarted during generation; submit a new job to retry')
                report_usage(job, 'failed')
        except (OSError, ValueError, KeyError):
            LOG.warning('Unable to recover interrupted job %s; original files retained', accepted.parent.name)


def load_models():
    global PIPELINE, MODELS
    failed_revision = None
    while PIPELINE is None:
        try:
            missing, revision = inspect_models()
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            MODELS = {**MODELS, 'state': 'error', 'error': f'Invalid model files: {error}'}
            threading.Event().wait(1)
            continue
        if missing:
            MODELS = {**MODELS, 'state': 'waiting', 'missing': missing, 'error': None}
        elif revision != failed_revision:
            MODELS = {**MODELS, 'state': 'loading', 'missing': [], 'error': None}
            try:
                from skill_t2i import ResidentPipeline, parse_args
                PIPELINE = ResidentPipeline(parse_args([]))
            except (Exception, SystemExit) as error:
                LOG.exception('Model loading failed; waiting for corrected model files or configuration')
                failed_revision = revision
                MODELS = {**MODELS, 'state': 'error', 'error': str(error) or 'Model loading failed; inspect worker logs'}
            else:
                MODELS = {**MODELS, 'state': 'ready', 'loadCount': MODELS['loadCount'] + 1}
                return
        threading.Event().wait(1)


def model_status():
    workers = []
    for index in range(WORKER_COUNT):
        try:
            with build_opener(NoRedirect).open(WORKER_URL.format(index=index) + '/models', timeout=1) as response:
                state = json.load(response)
            workers.append({'index': index, **state})
        except (OSError, ValueError):
            workers.append({'index': index, 'state': 'unavailable', 'missing': [], 'error': 'Worker unavailable'})
    return {'workers': workers}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def report_usage(job, status):
    if MODE != 'platform':
        return
    try:
        context = json.loads((OUTPUT / job / 'context.json').read_text(encoding='utf-8'))
        event = {'eventId': f'skill-{INSTANCE}-{job}-final', 'requestId': context['request'],
                 'jobId': job, 'moduleUnits': '1' if status == 'completed' else '0',
                 'usageMode': 'delta', 'status': status, 'sequence': 1, 'isFinal': True}
        request = Request(os.environ['PLATFORM_USAGE_URL'], data=json.dumps(event).encode(),
            headers={'Content-Type': 'application/json',
                     'X-Module-Instance-Token': os.environ['PLATFORM_USAGE_TOKEN']})
        with build_opener(NoRedirect).open(request, timeout=5) as response:
            if not 200 <= response.status < 300:
                raise OSError('Usage rejected')
    except (OSError, ValueError, KeyError):
        LOG.warning('Usage delivery failed for job %s; no automatic retry', job)


def state_path(job):
    return OUTPUT / job / 'status.json'


def save_state(job, state, error=None):
    path = state_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'status': state, 'error': error}), encoding='utf-8')
    temporary.replace(path)


def execute(job, prompt):
    status = 'failed'
    try:
        folder = OUTPUT / job
        prompt_file = folder / 'prompt.txt'
        prompt_file.write_text(prompt + '\n', encoding='utf-8')
        PIPELINE.generate(prompt, folder)
        status = 'completed'
        save_state(job, status)
    except Exception:
        LOG.exception('Pipeline failed for job %s', job)
        save_state(job, 'failed', 'Worker failed; inspect worker logs')
    finally:
        LOCK.release()
        report_usage(job, status)


def request_worker(url, payload):
    req = Request(url + '/jobs', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    with build_opener(NoRedirect).open(req, timeout=5) as response:
        if response.status != 202:
            raise OSError('Unexpected worker response')


def enqueue(job, prompt, user, request_id):
    created_at = datetime.now(timezone.utc).isoformat()
    with (QUEUE / 'enqueue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        counter = QUEUE / 'counter'
        sequence = int(counter.read_text()) + 1 if counter.exists() else 1
        temporary = counter.with_suffix('.tmp')
        temporary.write_text(str(sequence), encoding='utf-8')
        temporary.replace(counter)
        folder = OUTPUT / job
        folder.mkdir()
        (folder / 'context.json').write_text(json.dumps({
            'user': user, 'request': request_id, 'instance': INSTANCE, 'queueSequence': sequence,
            'prompt': prompt, 'createdAt': created_at,
        }), encoding='utf-8')
        save_state(job, 'queued')
        entry = QUEUE / f'{sequence:020d}-{job}.json'
        temporary = entry.with_suffix('.tmp')
        temporary.write_text(json.dumps({'job': job, 'prompt': prompt}), encoding='utf-8')
        temporary.replace(entry)
    WAKE.set()
    return created_at


def history(user):
    jobs = []
    for folder in OUTPUT.iterdir():
        try:
            job = str(uuid.UUID(folder.name))
            context_file = folder / 'context.json'
            context = json.loads(context_file.read_text(encoding='utf-8'))
            if context.get('instance') != INSTANCE or context.get('user') != user:
                continue
            prompt = context.get('prompt')
            if prompt is None:
                prompt_file = folder / 'prompt.txt'
                if prompt_file.exists():
                    prompt = prompt_file.read_text(encoding='utf-8').strip()
                else:
                    entry = QUEUE / f"{context['queueSequence']:020d}-{job}.json"
                    if not entry.exists():
                        entry = entry.with_suffix('.unknown')
                    prompt = json.loads(entry.read_text(encoding='utf-8'))['prompt']
            created_at = context.get('createdAt') or datetime.fromtimestamp(
                context_file.stat().st_mtime, timezone.utc).isoformat()
            jobs.append({'job': job, 'prompt': prompt, 'createdAt': created_at, **job_state(job)})
        except (OSError, ValueError, KeyError):
            continue
    return sorted(jobs, key=lambda item: (item['createdAt'], item['job']))


def queue_entries():
    entries = {entry.stem: entry for entry in QUEUE.iterdir() if entry.suffix in ('.json', '.sending')}
    return sorted(entries.values())


def job_state(job):
    state = json.loads(state_path(job).read_text(encoding='utf-8'))
    if state['status'] == 'queued':
        context = json.loads((OUTPUT / job / 'context.json').read_text(encoding='utf-8'))
        name = f"{context['queueSequence']:020d}-{job}"
        if (QUEUE / (name + '.unknown')).exists():
            return {'status': 'submitting', 'error': 'Worker acceptance is unknown; inspect logs before retrying'}
        state['queuePosition'] = 1 + sum(entry.stem < name for entry in queue_entries())
    return state


def dispatch_pending():
    if WORKER_COUNT == 0:
        return
    with (QUEUE / 'dispatch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        # A previous dispatcher died after recording intent; its acceptance is ambiguous.
        for entry in QUEUE.glob('*.sending'):
            entry.replace(entry.with_suffix('.unknown'))
        for entry in sorted(QUEUE.glob('*.json')):
            payload = json.loads(entry.read_text(encoding='utf-8'))
            job = payload['job']
            if json.loads(state_path(job).read_text(encoding='utf-8'))['status'] != 'queued':
                entry.unlink()
                continue
            if (OUTPUT / job / 'accepted').exists():
                entry.replace(entry.with_suffix('.unknown'))
                continue
            sending = entry.with_suffix('.sending')
            entry.replace(sending)
            for index in range(WORKER_COUNT):
                try:
                    request_worker(WORKER_URL.format(index=index), payload)
                except HTTPError as error:
                    error.close()
                    if error.code == 503:
                        continue
                except URLError as error:
                    if isinstance(error.reason, (ConnectionRefusedError, socket.gaierror)):
                        continue
                except OSError:
                    pass
                else:
                    sending.unlink()
                    break
                # Never overwrite a worker's running/completed state or resend an uncertain job.
                sending.replace(sending.with_suffix('.unknown'))
                LOG.warning('Worker acceptance is unknown for job %s; no automatic retry', job)
                break
            else:
                sending.replace(entry)
                return


def dispatch_loop():
    while True:
        WAKE.clear()
        try:
            dispatch_pending()
        except (OSError, ValueError, KeyError):
            LOG.exception('Queue dispatch failed; persisted work retained')
        WAKE.wait(0.5)


class Handler(BaseHTTPRequestHandler):
    def identity(self, billable=False):
        if MODE == 'standalone':
            return None, None
        if self.headers.get('X-Platform-Instance-Id') != INSTANCE:
            raise PermissionError('Invalid Gateway instance context')
        try:
            user = str(uuid.UUID(self.headers.get('X-Platform-User-Id', '')))
            request = str(uuid.UUID(self.headers.get('X-Platform-Request-Id', ''))) if billable else None
        except ValueError:
            raise PermissionError('Missing Gateway identity') from None
        return user, request

    def respond(self, status, data, content_type='application/json', location=None):
        if isinstance(data, dict):
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        if location:
            self.send_header('Location', location)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == '/readyz':
            return self.respond(200, {'ready': True})
        if ROLE == 'worker' and path == '/models':
            return self.respond(200, MODELS)
        if ROLE != 'coordinator' or (BASE and not (path == BASE or path.startswith(BASE + '/'))):
            return self.respond(404, {'error': 'Not found'})
        try:
            user, _ = self.identity()
        except PermissionError:
            return self.respond(403, {'error': 'Trusted Gateway identity is required'})
        if BASE and path == BASE:
            return self.respond(308, {}, location=BASE + '/')
        path = path[len(BASE):] or '/'
        if path == '/':
            return self.respond(200, Path('index.html').read_bytes(), 'text/html; charset=utf-8')
        if path == '/api/jobs':
            return self.respond(200, {'jobs': history(user)})
        if path == '/api/models':
            return self.respond(200, model_status())
        parts = path.strip('/').split('/')
        if len(parts) in (3, 4) and parts[:2] == ['api', 'jobs']:
            try:
                job = str(uuid.UUID(parts[2]))
            except ValueError:
                return self.respond(404, {'error': 'Not found'})
            folder = OUTPUT / job
            if MODE == 'platform':
                try:
                    context = json.loads((folder / 'context.json').read_text(encoding='utf-8'))
                except FileNotFoundError:
                    return self.respond(404, {'error': 'Not found'})
                if context['user'] != user or context['instance'] != INSTANCE:
                    return self.respond(404, {'error': 'Not found'})
            if len(parts) == 3:
                try:
                    return self.respond(200, job_state(job))
                except FileNotFoundError:
                    return self.respond(404, {'error': 'Not found'})
            elif parts[3] in ('result_01_original.png', 'result_01_enhanced.png', 'result.json'):
                file = folder / parts[3]
                content_type = 'image/png' if file.suffix == '.png' else 'application/json'
            else:
                return self.respond(404, {'error': 'Not found'})
            try:
                return self.respond(200, file.read_bytes(), content_type)
            except FileNotFoundError:
                return self.respond(404, {'error': 'Not found'})
        return self.respond(404, {'error': 'Not found'})

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        expected = '/jobs' if ROLE == 'worker' else BASE + '/api/jobs'
        if path != expected:
            return self.respond(404, {'error': 'Not found'})
        if ROLE == 'coordinator':
            try:
                user, request_id = self.identity(billable=True)
            except PermissionError:
                return self.respond(403, {'error': 'Trusted Gateway identity is required'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 16384:
                raise ValueError('Invalid request size')
            payload = json.loads(self.rfile.read(length))
            prompt = payload['prompt']
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
                raise ValueError('Prompt must contain 1 to 4000 characters')
            if ROLE == 'worker':
                job = str(uuid.UUID(payload['job']))
            else:
                job = str(uuid.uuid4())
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return self.respond(400, {'error': 'Invalid request'})
        if ROLE == 'worker':
            if MODE == 'platform':
                try:
                    context = json.loads((OUTPUT / job / 'context.json').read_text(encoding='utf-8'))
                    if context['instance'] != INSTANCE:
                        raise ValueError('Wrong instance')
                    uuid.UUID(context['request'])
                    uuid.UUID(context['user'])
                except (OSError, ValueError, KeyError, TypeError):
                    return self.respond(403, {'error': 'Job does not belong to this instance'})
            if MODELS['state'] != 'ready':
                return self.respond(503, {'error': 'Models are not ready'})
            if not LOCK.acquire(blocking=False):
                return self.respond(503, {'error': 'Worker busy'})
            try:
                folder = OUTPUT / job
                folder.mkdir(parents=True, exist_ok=True)
                # Shared storage makes acceptance exclusive across worker processes and restarts.
                with (folder / 'accepted').open('x') as accepted:
                    json.dump({'worker': WORKER_ID}, accepted)
            except FileExistsError:
                LOCK.release()
                return self.respond(409, {'error': 'Job already accepted'})
            except OSError:
                LOCK.release()
                return self.respond(503, {'error': 'Worker storage is unavailable'})
            try:
                save_state(job, 'running')
                threading.Thread(target=execute, args=(job, prompt), daemon=True).start()
            except Exception:
                LOCK.release()
                raise
            return self.respond(202, {'job': job})
        try:
            created_at = enqueue(job, prompt, user, request_id)
        except (OSError, ValueError):
            LOG.exception('Unable to persist queued work')
            return self.respond(503, {'error': 'Queue storage is unavailable'})
        return self.respond(202, {'job': job, 'createdAt': created_at})


if __name__ == '__main__':
    if ROLE not in ('coordinator', 'worker') or MODE not in ('standalone', 'platform'):
        raise ValueError('ROLE must be coordinator/worker and MODE must be standalone/platform')
    if BASE and (not BASE.startswith('/') or any(value in BASE for value in ('//', '..', '?', '#', '\\'))):
        raise ValueError('PATH_BASE must be an absolute, traversal-free path')
    if ROLE == 'coordinator':
        if not 0 <= WORKER_COUNT <= 100:
            raise ValueError('WORKER_COUNT must be between 0 and 100')
        urls = [WORKER_URL.format(index=index) for index in range(WORKER_COUNT)]
        if len(set(urls)) != len(urls):
            raise ValueError('Multiple workers require unique URLs; use {index} in WORKER_URL_TEMPLATE')
        for url in urls:
            parsed = urlsplit(url)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username:
                raise ValueError('Worker URLs must be HTTP(S) URLs without credentials')
    if MODE == 'platform':
        uuid.UUID(INSTANCE)
        parsed = urlsplit(os.environ.get('PLATFORM_USAGE_URL', ''))
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or not os.environ.get('PLATFORM_USAGE_TOKEN'):
            raise ValueError('Platform mode requires a usage URL and instance credential')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if ROLE == 'coordinator':
        QUEUE.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=dispatch_loop, name='queue-dispatcher', daemon=True).start()
    server = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    if ROLE == 'worker':
        recover_interrupted_jobs()
        threading.Thread(target=load_models, name='model-loader', daemon=True).start()
    server.serve_forever()
